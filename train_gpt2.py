import math
import torch
from torch.nn import functional as F

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os
import time
import json

from model import GPT, GPTConfig
from dataset import DataLoaderLite
from checkpoint import save_checkpoint, load_checkpoint
from hellaswag import eval_hellaswag

import argparse

# ----------------------------------------------------------------------------------
# define global hyperparams

use_compile = True

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
block_size_train = T_train = 1024 # sequence length
vocab_size = 50304
n_layer: int = 12 # number of layers
n_head: int = 12 # number of heads
n_embd: int = 768  # embedding dimension

max_lr = 6e-4
min_lr = max_lr * 0.1 # go to 10% of the max_lr according to GPT-3
warmup_steps = 715
max_steps = 19073

learning_rate = 6e-4
weight_decay = 0.1 # 10%

val_step = 100 # validation loss every 100th step
val_loss_steps = 30
val_block_sizes = [512, 1024, 2048, 4096]

assert all([((B*T_train) % block_size == 0) for block_size in val_block_sizes]), f'{val_block_sizes} are not compatible with train - batch_size {B} and seq length {T_train}' 

eval_step = 250 # hellaswag evaluation every 250th step
sampling_step = 1000 # sample from the model every 200th step

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "log.json") # for logging losses etc.

checkpointing = True
checkpoint_step = 5000  # the checkpoint will be heavy, store 3-4 for the whole run
if checkpointing:
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

# load checkpoint path if passed
parser = argparse.ArgumentParser(description="Training")
parser.add_argument('--resume-from', type=str, required=False, default=None)
args = parser.parse_args()

resume_from = args.resume_from  # version of model
if resume_from:
    resume_from = os.path.join("checkpoint", resume_from)
    resume_from = os.path.join(log_dir, resume_from) # point it to the correct file

# updating the processing precision TF32 instead of FP32
torch.set_float32_matmul_precision('high')

seed = cuda_seed = 42

# load the tokenizer
import tiktoken
enc = tiktoken.get_encoding('gpt2')

# ----------------------------------------------------------------------------------
# run: torchrun --standalone --nproc_per_node=4 train_gpt2.py
# setup DDP (distributed data parallel)
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run

if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "as of now CUDA is needed for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing, etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")


assert total_batch_size % (B * T_train * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T_train * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T_train * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# seeding
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(cuda_seed)

# create model
# model = GPT(GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd))
model = GPT(GPTConfig(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd))
model.to(device)

# optimize, betas updated according to GPT-3
optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, device=device)

# load saved model if needed
start_step = 0
current_train_shard = None
current_train_pos = None
if resume_from:
    try:
        ckpt = load_checkpoint(resume_from, model, optimizer, device)
        start_step = ckpt['step']
        torch.set_rng_state(ckpt['rng_state'])
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"])

        current_train_shard, current_train_pos = ckpt['loader_pos']
        current_train_pos = current_train_pos + (B * T_train * ddp_rank)  # init the correct starting pos for each gpu
        if master_process:
            print(f"resuming training from: {resume_from}, step: {start_step}")
    except Exception as e:
        print(f"Error loading the checkpoint file - {e}")
        import sys; sys.exit(1)

assert start_step < max_steps, "starting step cannot be more than the max steps"

uncompiled_model = model # for processes where the shapes are changing we should use the uncompiled model

if use_compile:
    model = torch.compile(model) # dynamo + kernel fusion  <==>  this will optimise the GPU read and writes using the graph and the kernel fusion operation
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # always contains the "raw" model - ddp unwrapped

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr) 

# init data loader
train_loader = DataLoaderLite(B=B, T=T_train, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", current_shard=current_train_shard, current_pos=current_train_pos)
val_loader = DataLoaderLite(B=B, T=None, process_rank=ddp_rank, num_processes=ddp_world_size, split="val") # T has to be None for val, in case it is passed 


def val_loss(step, block_size=T_train):
    # validation also calculates the common sequence loss for different block_size's 

    global uncompiled_model, val_step, val_loader, device, val_loss_steps, ddp, master_process, T_train, B, val_block_sizes

    uncompiled_model.eval() # validation should not force recompile the model on each T update
    val_loader.reset(split="val") # call reset once every block_size 

    batch_size = (B * T_train) // block_size

    with torch.no_grad():
        val_loss_accum = 0.0

        for _ in range(val_loss_steps):
            # check the loss on validation set
            x, y = val_loader.next_batch(B=batch_size, T=block_size)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):  # do the forward pass in a lower precision
                # forward pass  # calculate logits and loss
                logits, loss = uncompiled_model(x, y) 
            
            loss = loss / val_loss_steps
            val_loss_accum += loss.detach()

    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

    val_loss_accum = val_loss_accum.item()
    if master_process:
        print(f"\n")
        print(f"step: {step}, block_size: {block_size}, validation loss: {val_loss_accum:.4f}")
        print("\n")

    return val_loss_accum


# ---------------------------------------------------------------------------------------
# run the training loop

for step in range(start_step, max_steps):

    accuracy = None
    avg_accuracy = None
    val_loss_accum = None

    # eval step, once in a while calculate the validation loss
    if step % val_step == 0:
        val_loss_accum = {f'val_loss_{block_size}' : val_loss(step, block_size) for block_size in val_block_sizes} # validation loss for different block sizes
   
    # run Hellaswag eval
    if step % eval_step == 0:
        uncompiled_model.eval()
        accuracy, avg_accuracy = eval_hellaswag(uncompiled_model, enc, device, ddp, ddp_rank, ddp_world_size, block_size=block_size_train)
        if master_process:
            print(f"Hellaswag Eval accuracy - {avg_accuracy*100:.2f}")

    # once in a while, generate from model - sampling
    if step > 0 and step % sampling_step == 0:
        model.eval()
        num_return_sequences = 2
        max_length = 64
        tokens = enc.encode("Hello, I'm a language model")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (4, len(tokens))
        xgen = tokens.to(device)

        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank) # to make this specific to gpu and have it different from training seed
        while xgen.size(1) < max_length:
            # forwarding the model to get the logits
            with torch.no_grad():
                logits, loss = uncompiled_model(xgen) # (B, T, vocab_size)  --  use uncompiled model because this will force recompile since the shape is not the same
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size) 
                # get the probabilities - for each batch
                probs = F.softmax(logits, dim=-1) 
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from top-k probabilities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # printing the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # save model if checkpointing is true
    if checkpointing and (step % checkpoint_step == 0 or step == max_steps-1) and step != 0 and master_process:
        path = os.path.join(checkpoint_dir, f"model_{step:05d}.pt")
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        save_checkpoint(path, uncompiled_model, optimizer, step, val_loss_accum, uncompiled_model.config, train_loader, cuda_rng_state=cuda_rng_state)
        print(f"Checkpoint saved at - {path}")

    # training loop
    t0 = time.time()

    model.train()

    # zero the gradients to avoid accumulation
    optimizer.zero_grad()

    loss_accum = 0.0
    for mini_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        if ddp:
            # the synchronization is not required after every mini step and is needed only at last
            model.require_backward_grad_sync = (mini_step == grad_accum_steps - 1)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):  # do the forward pass in a lower precision
            # forward pass  # calculate logits and loss
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps  # this is to cater for the mean reduction matching had the Batch dim was equal to desired Batch dim
        
        loss_accum += loss.detach()

        # backward pass == calculate grads
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # update params
    optimizer.step()

    if "cuda" in device:
        torch.cuda.synchronize()  # wait for the GPU to finish work

    t1 = time.time()
    dt = t1-t0  # time difference in seconds

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt

    mfu = raw_model.estimate_mfu(B * grad_accum_steps, train_loader.T, dt)

    if master_process:
        loss = loss_accum.item()
        norm = norm.item()

        print(f"step {step:4d} | loss: {loss:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | mfu: {mfu:.2f}")

        # log losses to file
        with open(log_file, 'a') as f:
            value = {
                'step': step, 
                'lr': lr, 
                'train': loss, 
                'norm': norm,
                'dt': dt,
                'mfu': mfu
            }

            if val_loss_accum is not None:
                value['val'] = val_loss_accum
            if accuracy is not None and avg_accuracy is not None:
                value['accuracy'] = accuracy
                value['avg_accuracy'] = avg_accuracy

            f.write(json.dumps(value) + "\n")

if ddp:
    destroy_process_group()


