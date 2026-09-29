import torch
from torch.nn import functional as F
import torch.distributed as dist
from datasets import load_dataset

# ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test[:4]")
ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

def render(data, tokenizer):
    # this is one question and then 3-5 options with label
    # same layout as the SFT data (finetuning/dataloader_ft.py): <|endoftext|> + question + " " + answer
    eot = tokenizer._special_tokens['<|endoftext|>']
    ques = [eot] + tokenizer.encode_ordinary(data["question"])
    options = data["choices"]["text"]
    labels = data["choices"]["label"] # "A".."E" 
    answer_key = data["answerKey"]
    if not answer_key or answer_key not in labels:
        return None # unlabelled question, skip it
    label = labels.index(answer_key)

    options_encoded = []
    max_seq = 0

    for opt in options:
        opt_encoded = tokenizer.encode_ordinary(" " + opt) # we need a space between question and answer
        if len(opt_encoded) > max_seq:
            max_seq = len(opt_encoded)
        options_encoded.append(opt_encoded)

    max_seq += len(ques)

    tokens = []
    targets = []

    for i, opt in enumerate(options_encoded):
        padding_len = max_seq - (len(ques) + len(opt))
        seq = ques + opt + [0] * padding_len

        # create targets / target
        mask_curr = [-1] * (len(ques) - 1) + opt + [-1] * (padding_len + 1)

        tokens.append(seq)
        targets.append(mask_curr)

    return tokens, targets, label

def eval_arc_ai(model, tokenizer, device, ddp, ddp_rank, ddp_world_size, block_size):
    #@NOTE: the model to be passed here should be raw in case ddp is not true

    label_correct = []  # this will hold either 0 or 1 based on whether the opt_pred == label
    label_correct_avg = []  # this will hold either 0 or 1 based on whether the opt_pred_avg == label

    model.eval()

    for i, item in enumerate(ds):

        if i % ddp_world_size != ddp_rank:
            continue # only process the sets which are multiple of your alloted rank

        rendered = render(item, tokenizer)
        if rendered is None:
            continue
        tokens, targets, label = rendered # (num_choices, T_full)
        assert len(tokens[0]) <= block_size, f"seq length can't be more than supported block size: {block_size}"

        # wrap to tensors and move to device
        tokens, targets = torch.tensor(tokens, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
        tokens, targets = tokens.to(device), targets.to(device)

        with torch.no_grad():
            logits, _ = model(tokens) # this loss is not usefull because it will give a CE over all the batches

        # logits shape (num_choices, T, C), targets (num_choices, T)
        losses = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1, # this will assign 0 loss to tokens where target is -1
            reduction='none',
        ).view(logits.size(0), -1) # (num_choices, T)

        sum_loss = losses.sum(dim=1) # sum the loss for each option, -1 tokens have already 0 loss
        counts = (targets != -1).sum(dim=1)   # (num_choices,)
        avg_loss = sum_loss / counts

        opt_pred, opt_pred_avg = sum_loss.argmin(), avg_loss.argmin()

        label_correct.append(int(opt_pred.item()==label))
        label_correct_avg.append(int(opt_pred_avg.item()==label))

    stats = torch.tensor([len(label_correct), sum(label_correct), sum(label_correct_avg)], dtype=torch.long, device=device)

    if ddp:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    accuracy = accuracy_avg = None
    if ddp_rank == 0:
        total, correct, correct_avg = stats.tolist()
        accuracy = correct / total
        accuracy_avg = correct_avg / total

    return accuracy, accuracy_avg


# test script
def main():
    from model import GPT, GPTConfig
    import tiktoken

    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    enc = tiktoken.get_encoding('gpt2')
    ddp_rank = 0  # for mps check
    ddp_world_size = 1 # for mps check
    block_size = 1024

    print(f"ARC-Easy evaluation on initialized tinygpt model")
    model = GPT(GPTConfig(block_size_max=block_size, vocab_size=50304))
    model = model.to(device)

    accuracy, accuracy_avg = eval_arc_ai(model, enc, device, ddp=False, ddp_rank=ddp_rank, ddp_world_size=ddp_world_size, block_size=block_size)
    print(f"ARC-Easy accuracy over total loss: {accuracy*100:.2f}% and over per-token avg loss: {accuracy_avg*100:.2f}%")

if __name__ == "__main__":
    main()
