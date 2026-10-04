import os
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Same design as finetuning/dataloader_ft.py, with the Qwen tokenizer: one right-padded
# question+answer per row, a mask that is 1 on the answer only, rows stored in .npy shards
# and read B whole rows per rank, reshuffled every epoch.
# run once before training (from the repo root): uv run python peft/sft_data.py

# -----------------------------------------------------------------
model_id = "Qwen/Qwen3-4B-Base" # shards are tokenizer specific, rebuild them for a different model family
local_dir = "ai2_arc_qwen3"
remote_name = "ARC-Challenge"
shard_size = int(1e8) # 100M uint32 values per shard, tokens and mask together

# with the Qwen tokenizer the longest train question+answer is 142 tokens and the longest test
# question+option is 175, so 256 leaves headroom for both
T_MAX = 256
# no +1 here unlike tinyGPT: the HF model shifts the labels itself, so a row is fed whole as input_ids
ROW_LEN = T_MAX
rows_per_shard = (shard_size // 2) // ROW_LEN

# every shard is (2, num_sequences, ROW_LEN) uint32: [TOKENS] holds one right-padded
# question+answer per row, [MASK] is 1 on the answer tokens and 0 on the question and padding
TOKENS, MASK = 0, 1
IGNORE_INDEX = -100 # transformers' cross entropy ignore index (tinyGPT uses -1)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)


def tokenize(doc, tokenizer):
    # tokenizes a single QA document into a (2, ROW_LEN) uint32 array - same layout as the tinyGPT fine-tune
    labels = doc["choices"]["label"]
    answer_key = doc["answerKey"]
    if not answer_key or answer_key not in labels:
        return None # unlabelled doc, drop it
    answer = doc["choices"]["text"][labels.index(answer_key)]

    eot = tokenizer.eos_token_id # <|endoftext|>, same delimiter string as GPT-2
    question_tokens = tokenizer(doc["question"], add_special_tokens=False)["input_ids"]
    answer_tokens = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]

    # the special <|endoftext|> token starts every sequence (Qwen adds no BOS of its own)
    tokens = [eot] + question_tokens + answer_tokens
    assert len(tokens) <= ROW_LEN, f"question+answer is {len(tokens)} tokens, raise T_MAX (currently {T_MAX})"
    mask = [0] * (1 + len(question_tokens)) + [1] * len(answer_tokens)

    # right pad with <|endoftext|>, the padding has mask 0 so it is never scored
    padding_len = ROW_LEN - len(tokens)
    tokens += [eot] * padding_len
    mask += [0] * padding_len

    # Qwen token ids go up to ~151k, past the uint16 range the GPT-2 shards use
    return np.stack([np.array(tokens), np.array(mask)]).astype(np.uint32)


def write_datafile(filename, shard_np):
    np.save(filename, shard_np)


def main(split="train"):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ds = load_dataset("allenai/ai2_arc", name=remote_name, split=split)

    # ARC is ~1k docs, a plain loop takes a second - no process pool (HF fast tokenizers and fork don't mix well)
    shard_index = 0
    all_np = np.empty((2, rows_per_shard, ROW_LEN), dtype=np.uint32)
    row_count = 0
    for doc in tqdm(ds, unit="seqs", desc=f"{split}"):
        doc_np = tokenize(doc, tokenizer)
        if doc_np is None:
            continue
        all_np[:, row_count] = doc_np
        row_count += 1

        if row_count == rows_per_shard:
            write_datafile(os.path.join(DATA_CACHE_DIR, f"arc_ai_{split}_{shard_index:06d}"), all_np)
            shard_index += 1
            row_count = 0

    if row_count != 0:
        write_datafile(os.path.join(DATA_CACHE_DIR, f"arc_ai_{split}_{shard_index:06d}"), all_np[:, :row_count])
        print(f"wrote final shard {shard_index} with {row_count} sequences")


# -----------------------------------------------------------------
# loading side

def load_tokens(filename):
    shard = np.load(filename)
    tokens = torch.tensor(shard[TOKENS].astype(np.int64), dtype=torch.long) # (num_sequences, ROW_LEN)
    loss_mask = torch.tensor(shard[MASK], dtype=torch.bool) # (num_sequences, ROW_LEN)
    return tokens, loss_mask


class DataLoaderSFT:
    """DataLoaderFT for HF models: every row is one right-padded question+answer, a batch is B whole
    rows, and the batch comes back as {input_ids, labels} with the non-answer labels set to -100.
    Labels are NOT shifted - the HF model shifts them inside its loss."""

    def __init__(self, B, process_rank, num_processes, split, data_root=DATA_CACHE_DIR, shuffle=False, seed=42):
        self.B = B
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'validation'}, "split should be either train or validation"
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        shards = [s for s in os.listdir(data_root) if f"_{split}_" in s]
        shards = sorted(os.path.join(data_root, s) for s in shards)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}, run: uv run python peft/sft_data.py"
        if self.process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")

        self.current_shard = 0
        self.load_shard()
        # total rows over all shards - used to size epochs and a full validation pass
        self.num_rows = sum(np.load(s, mmap_mode='r').shape[1] for s in shards)

        needed = B * num_processes
        assert len(self.tokens) >= needed, (
            f"shard has {len(self.tokens)} sequences but B*world = {needed} are needed for one batch; "
            f"lower B for this dataset"
        )
        self.current_position = self.B * self.process_rank # positions are in rows (sequences)

    def load_shard(self):
        self.tokens, self.loss_mask = load_tokens(self.shards[self.current_shard])
        if self.shuffle:
            # seeded by (epoch, shard) and NOT by rank: every rank must hold the same permutation,
            # so that their strided slices of it stay disjoint
            g = torch.Generator().manual_seed(self.seed + self.epoch * len(self.shards) + self.current_shard)
            perm = torch.randperm(len(self.tokens), generator=g)
            self.tokens, self.loss_mask = self.tokens[perm], self.loss_mask[perm]

    def reset(self):
        self.epoch = 0
        self.current_shard = 0
        self.load_shard()
        self.current_position = self.B * self.process_rank

    def state(self):
        return {'epoch': self.epoch, 'shard': self.current_shard, 'position': self.current_position}

    def load_state(self, state):
        # restore a checkpointed position - the permutation is rebuilt from (seed, epoch, shard)
        self.epoch, self.current_shard = state['epoch'], state['shard']
        self.load_shard()
        self.current_position = state['position']

    def next_batch(self, B=None):
        B = self.B if B is None else B

        lo, hi = self.current_position, self.current_position + B
        input_ids = self.tokens[lo:hi] # (B, T_MAX)
        labels = input_ids.clone() # cloned because we are about to stamp -100 into it
        # unshifted: labels[:, t] is the token AT t. The model scores logits[:, t-1] against it, which is
        # the same pairing as tinyGPT's y[:, t-1] = tokens[:, t] with the mask shifted by one
        labels[~self.loss_mask[lo:hi]] = IGNORE_INDEX # question / padding positions do not contribute to the loss

        # advance the position in the rows
        self.current_position += B * self.num_processes
        # move to the next shard when the next step doesn't fit for ALL ranks, so all ranks switch together
        step_start = self.current_position - B * self.process_rank
        if step_start + B * self.num_processes > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            if self.current_shard == 0:
                self.epoch += 1 # wrapped past the last shard, the next pass gets a fresh shuffle
            self.load_shard()
            self.current_position = B * self.process_rank

        # no attention_mask: padding is on the right and attention is causal, so no real token
        # ever attends to a pad - same reasoning as the tinyGPT fine-tune
        return {"input_ids": input_ids, "labels": labels}


if __name__ == "__main__":
    main(split="train")
    main(split="validation")
