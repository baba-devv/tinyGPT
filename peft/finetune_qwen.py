import torch

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import os
import sys
import math
import time
import json
import argparse

from model_setup import load_tokenizer, load_base_model, build_lora_config, wrap_with_lora, load_adapter, configure_optimizer, merge_and_save
from sft_data import DataLoaderSFT, T_MAX
from arc_ai import eval_arc_ai

# run from the repo root:
#   single device: uv run python peft/finetune_qwen.py
#   ddp:           uv run torchrun --standalone --nproc_per_node=4 peft/finetune_qwen.py
#   resume:        ... peft/finetune_qwen.py --resume-from log_peft/checkpoint/step_00100
# (run it as a file path, not `python -m peft.finetune_qwen` - see the note in model_setup.py)

parser = argparse.ArgumentParser(description="LoRA fine-tuning of a published HF base model with PEFT")
parser.add_argument('--model-id', type=str, default="Qwen/Qwen3-4B-Base")
parser.add_argument('--resume-from', type=str, required=False, default=None) # a checkpoint dir written by this script
args = parser.parse_args()

model_id = args.model_id

use_compile = True # every train / val batch is (B, T_MAX), so the compiled graph is reused, not rebuilt

# LoRA settings
rank = 16
alpha = 32  # scaling = alpha / rank = 2
lora_dropout = 0.05
load_in_4bit = False  # QLoRA, needs CUDA + bitsandbytes. bf16 4B fits fine on an A100 without it
gradient_checkpointing = True  # @TODO: What does this do ?

# same batching as finetune_gpt2.py: each row is one question+answer padded to T_MAX, so the
# batch size is counted in sequences
total_batch_size = 128 # sequences per optimizer step
B = 8 # micro batch size - lower than tinyGPT's 32 because the fp32 logits are B x T_MAX x 151936 here
num_epochs = 10 # max_steps is derived from the train split size once the loader is up

max_lr = 2e-4 # the usual LoRA lr, ~10x what full fine-tuning would use
min_lr = max_lr * 0.1
warmup_steps = 6
weight_decay = 0.0 # adapters are tiny and start at zero, decay mostly just fights the update
grad_clip = 1.0

val_step = 10 # validation loss (one full pass over the validation split) every 10th step
eval_step = 10 # ARC-Challenge evaluation every 10th step
sampling_step = 40
# same layout as the training rows: <|endoftext|> + question, the answer follows after a space
sample_prompt = "<|endoftext|>Which gas do plants absorb from the air for photosynthesis?"

log_dir = "log_peft"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.json")

checkpointing = True
checkpoint_step = 100
checkpoint_dir = os.path.join(log_dir, "checkpoint")
os.makedirs(checkpoint_dir, exist_ok=True)
save_merged = False # also write a merged full model (~8GB) at the end, in addition to the adapter (~tens of MB)

seed = 42

# ----------------------------------------------------------------------------------
# setup DDP (distributed data parallel)
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run

if ddp:
    assert torch.cuda.is_available(), "as of now CUDA is needed for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing, etc.
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

assert total_batch_size % (B * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * ddp_world_size"
grad_accum_steps = total_batch_size // (B * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size} sequences")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

# ----------------------------------------------------------------------------------
# model: published base weights (frozen) + LoRA adapters (trainable)

tokenizer = load_tokenizer(model_id)
model = load_base_model(model_id, device, load_in_4bit=load_in_4bit, gradient_checkpointing=gradient_checkpointing)

start_step = 0
trainer_state = None
# if args.resume_from:
#     # the checkpoint only holds the adapter + optimizer, the base weights come from model_id again
#     model = load_adapter(model, args.resume_from, is_trainable=True)
#     trainer_state = torch.load(os.path.join(args.resume_from, "trainer_state.pt"), map_location="cpu", weights_only=False)
#     start_step = trainer_state['next_step']
#     if master_process:
#         print(f"resuming training from: {args.resume_from} at step {start_step}")
# else:
model = wrap_with_lora(model, build_lora_config(rank, alpha, lora_dropout))

if master_process:
    model.print_trainable_parameters()

optimizer = configure_optimizer(model, learning_rate=max_lr, weight_decay=weight_decay, device=device)
if trainer_state is not None:
    optimizer.load_state_dict(trainer_state['optimizer'])
    torch.set_rng_state(trainer_state['rng_state'])
    # lora_dropout draws from the device generator, restore it too so the resumed run matches
    if trainer_state.get('device_rng_state') is not None:
        if "cuda" in device:
            torch.cuda.set_rng_state(trainer_state['device_rng_state'])
        elif device == "mps":
            torch.mps.set_rng_state(trainer_state['device_rng_state'])

# the plain PeftModel - for validation, ARC eval, generation and saving. It shares its weights with the
# compiled one, but runs eagerly: generate and the ARC eval change shapes on every call, which would force recompiles
uncompiled_model = model

if use_compile:
    model = torch.compile(model)
if ddp:
    # frozen params get no grads, but every trainable (LoRA) param is used each step,
    # so find_unused_parameters can stay off
    model = DDP(model, device_ids=[ddp_local_rank])

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

# ----------------------------------------------------------------------------------
# data

# same row-per-question loader as finetune_gpt2.py (built by: uv run python peft/sft_data.py)
train_loader = DataLoaderSFT(B=B, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", shuffle=True, seed=seed)
val_loader = DataLoaderSFT(B=B, process_rank=ddp_rank, num_processes=ddp_world_size, split="validation")
if trainer_state is not None:
    # jump the data stream to where the checkpoint left off, the shuffle is rebuilt from (seed, epoch)
    train_loader.load_state(trainer_state['loader_state'])

max_steps = num_epochs * train_loader.num_rows // total_batch_size # one step consumes total_batch_size sequences
# one pass over the validation split, every rank takes B questions per step
val_loss_steps = val_loader.num_rows // (B * ddp_world_size)
if master_process:
    print(f"train rows: {train_loader.num_rows}, val rows: {val_loader.num_rows}")
    print(f"=> max steps: {max_steps}, validation steps: {val_loss_steps}")


def to_device(batch):
    return {k: v.to(device) for k, v in batch.items()}

def count_supervised(batch):
    # labels are shifted inside the model, so the first position never gets scored
    return (batch["labels"][:, 1:] != -100).sum()


@torch.no_grad()
def evaluate_val_loss():
    uncompiled_model.eval()
    loss_sum = torch.zeros((), device=device)
    n_tokens = torch.zeros((), device=device)
    val_loader.reset()
    for _ in range(val_loss_steps):
        batch = to_device(val_loader.next_batch())
        n = count_supervised(batch)
        # num_items_in_batch switches the HF loss to sum / n, so n * loss is the summed token loss
        out = uncompiled_model(**batch, num_items_in_batch=n)
        loss_sum += out.loss * n
        n_tokens += n
    if ddp:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_tokens, op=dist.ReduceOp.SUM)
    return (loss_sum / n_tokens).item() # token-weighted mean over the completions


@torch.no_grad()
def sample(prompt, max_new_tokens=32):
    uncompiled_model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = uncompiled_model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, enc["input_ids"].size(1):], skip_special_tokens=True)


def save_training_checkpoint(step):
    path = os.path.join(checkpoint_dir, f"step_{step:05d}")
    uncompiled_model.save_pretrained(path) # adapter_config.json + adapter_model.safetensors (LoRA weights only)
    torch.save({
        'next_step': step, # saved before this step trains, so resume runs it
        'optimizer': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'device_rng_state': torch.cuda.get_rng_state() if "cuda" in device else (torch.mps.get_rng_state() if device == "mps" else None),
        'loader_state': train_loader.state(),
        'model_id': model_id,
    }, os.path.join(path, "trainer_state.pt"))
    return path

# ---------------------------------------------------------------------------------------
# run the training loop

for step in range(start_step, max_steps):
    last_step = step == max_steps - 1
    val_loss = None
    accuracy = None
    avg_accuracy = None

    # once in a while calculate the validation loss
    if step % val_step == 0 or last_step:
        val_loss = evaluate_val_loss()
        if master_process:
            print(f"\nstep: {step}, validation loss: {val_loss:.4f}\n")

    # once in a while, generate from the model (greedy, so runs are comparable)
    if master_process and ((step > 0 and step % sampling_step == 0) or last_step):
        print(f"sample: {sample_prompt!r} -> {sample(sample_prompt)!r}")

    # run ARC-AI (challenge) eval - every rank takes its share of the questions, so all ranks call it
    if step % eval_step == 0:
        accuracy, avg_accuracy = eval_arc_ai(uncompiled_model, tokenizer, device, ddp, ddp_rank, ddp_world_size, block_size=T_MAX)
        if master_process:
            print(f"\nARC-Challenge Eval accuracy - {accuracy*100:.2f}, avg accuracy - {avg_accuracy*100:.2f}\n")

    if checkpointing and master_process and step > start_step and step % checkpoint_step == 0:
        print(f"Checkpoint saved at - {save_training_checkpoint(step)}")

    # training step
    t0 = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # fetch the whole accumulation window up front so the loss can be normalised by the total
    # number of supervised tokens in it - otherwise short answers get weighted like long ones
    batches = [to_device(train_loader.next_batch()) for _ in range(grad_accum_steps)]
    n_tokens = sum(count_supervised(b) for b in batches)
    if ddp:
        # normalise by the global token count; DDP averages grads over ranks, so scale back up by world size
        dist.all_reduce(n_tokens, op=dist.ReduceOp.SUM)
    loss_scale = ddp_world_size

    loss_accum = torch.zeros((), device=device)
    for mini_step, batch in enumerate(batches):
        if ddp:
            # the synchronization is not required after every mini step and is needed only at last
            model.require_backward_grad_sync = (mini_step == grad_accum_steps - 1)
        out = model(**batch, num_items_in_batch=n_tokens)
        loss = out.loss * loss_scale # summed token loss / global tokens (x world size to undo DDP's mean)
        loss_accum += out.loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM) # per-rank partial sums -> global mean token loss

    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    if "cuda" in device:
        torch.cuda.synchronize()
    dt = time.time() - t0
    tokens_processed = B * T_MAX * grad_accum_steps * ddp_world_size # includes the padding tokens

    if master_process:
        loss_val = loss_accum.item()
        norm = norm.item()
        print(f"step {step:4d} | loss: {loss_val:.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_processed/dt:.2f}")

        with open(log_file, 'a') as f:
            value = {'step': step, 'lr': lr, 'train': loss_val, 'norm': norm, 'dt': dt}
            if val_loss is not None:
                value['val'] = val_loss
            if accuracy is not None and avg_accuracy is not None:
                value['accuracy'] = accuracy
                value['avg_accuracy'] = avg_accuracy
            f.write(json.dumps(value) + "\n")

    # run ARC-AI (challenge) eval on the final weights
    if last_step:
        accuracy, avg_accuracy = eval_arc_ai(uncompiled_model, tokenizer, device, ddp, ddp_rank, ddp_world_size, block_size=T_MAX)
        if master_process:
            print(f"Final ARC-Challenge Eval accuracy - {accuracy*100:.2f}, avg accuracy - {avg_accuracy*100:.2f}")

# ---------------------------------------------------------------------------------------
# final artifacts

if master_process:
    final_dir = os.path.join(log_dir, "adapter_final")
    uncompiled_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"LoRA adapter saved at - {final_dir}")

    if save_merged:
        if load_in_4bit:
            print("skipping merge: merge the adapter into a bf16 copy of the base model, not the 4-bit one")
        else:
            merged_dir = os.path.join(log_dir, "merged_final")
            merge_and_save(uncompiled_model, tokenizer, merged_dir)
            print(f"merged model saved at - {merged_dir}")

if ddp:
    destroy_process_group()
