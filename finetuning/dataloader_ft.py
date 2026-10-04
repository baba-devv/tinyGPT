import os
import multiprocessing as mp
import numpy as np
import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------------------------------------------
local_dir = "ai2_arc"
remote_name = "ARC-Easy"
shard_size = int(1e8) # 100M uint16 values per shard, tokens and mask together

# model sequence length, same for training and the ARC-Easy eval: the longest train question+answer
# is 120 tokens and the longest test question+option is 156, so 256 leaves headroom for both
T_MAX = 256
# each stored row is one token wider than T_MAX so that x = row[:-1] and y = row[1:] are both T_MAX long
ROW_LEN = T_MAX + 1
# the mask is stored next to the tokens, so a shard only holds shard_size // 2 tokens,
# and it is cut on whole rows so a question+answer sequence never spans two shards
rows_per_shard = (shard_size // 2) // ROW_LEN

# every shard is (2, num_sequences, ROW_LEN) uint16: [TOKENS] holds one right-padded
# question+answer per row, [MASK] is 1 on the answer tokens and 0 on the question and padding
TOKENS, MASK = 0, 1

# create the local cache directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    # tokenizes a single QA document into a (2, ROW_LEN) uint16 array. The mask row is what lets the
    # fine-tune pay a loss for predicting the answer only, and not for reciting the prompt back.
    labels = doc["choices"]["label"]
    answer_key = doc["answerKey"]
    if not answer_key or answer_key not in labels:
        return None # unlabelled doc, drop it
    answer = doc["choices"]["text"][labels.index(answer_key)]

    # the document text is the question with the correct choice appended
    question_tokens = enc.encode_ordinary(doc["question"])
    answer_tokens = enc.encode_ordinary(" " + answer)

    # the special <|endoftext|> token starts every sequence
    tokens = [eot] + question_tokens + answer_tokens
    assert len(tokens) <= ROW_LEN, f"question+answer is {len(tokens)} tokens, raise T_MAX (currently {T_MAX})"
    # built from the piece lengths, so the two rows line up by construction
    mask = [0] * (1 + len(question_tokens)) + [1] * len(answer_tokens)

    # right pad with <|endoftext|>, the padding has mask 0 so it becomes a -1 target and is never scored
    padding_len = ROW_LEN - len(tokens)
    tokens += [eot] * padding_len
    mask += [0] * padding_len

    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2 **16).all(), "token dictionary total tokens exceed the limit"
    return np.stack([tokens_np, np.array(mask)]).astype(np.uint16)

def write_datafile(filename, shard_np):
    # writes the (2, num_sequences, ROW_LEN) token/mask array to a .npy file
    np.save(filename, shard_np)

# tokenize all documents and write output shards, each of at most rows_per_shard sequences
def main(split="train"):
    # stream the dataset so downloading overlaps tokenization instead of blocking on the full download
    fw = load_dataset("allenai/ai2_arc", name=remote_name, split=split, streaming=True)

    nproc = max(1, os.cpu_count() // 2)
    with mp.Pool(nproc) as pool: # instantiating n copy processes
        shard_index = 0
        # preallocate buffer to hold current shard
        all_np = np.empty((2, rows_per_shard, ROW_LEN), dtype=np.uint16)
        row_count = 0
        progress_bar = None

        for doc_np in pool.imap(tokenize, fw, chunksize=16):
            if doc_np is None:
                continue
            # every doc is exactly one row, so it either fits whole or the shard was already flushed
            all_np[:, row_count] = doc_np
            row_count += 1
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=rows_per_shard, unit="seqs", desc=f"Shard {shard_index}: ")
            progress_bar.update(1)

            if row_count == rows_per_shard:
                # shard is full, write it and start a new one
                filename = os.path.join(DATA_CACHE_DIR, f"arc_ai_{split}_{shard_index:06d}")
                write_datafile(filename, all_np)
                shard_index += 1
                progress_bar = None
                row_count = 0

        # write any remaining sequences as the last shard
        if row_count != 0:
            filename = os.path.join(DATA_CACHE_DIR, f"arc_ai_{split}_{shard_index:06d}")
            write_datafile(filename, all_np[:, :row_count])
            print(f"wrote final shard {shard_index} with {row_count} sequences")


# -----------------------------------------------------------------
# loading side

def load_tokens(filename):
    shard = np.load(filename)
    tokens = torch.tensor(shard[TOKENS], dtype=torch.long) # (num_sequences, ROW_LEN)
    loss_mask = torch.tensor(shard[MASK], dtype=torch.bool) # (num_sequences, ROW_LEN)
    return tokens, loss_mask

class DataLoaderFT:
    """Every row is one right-padded question+answer, so a batch is B whole rows and a sequence
    never attends to another one. The targets of every non-answer position are set to -1 so
    model.forward's cross_entropy(ignore_index=-1) skips them."""

    def __init__(self, B, process_rank, num_processes, split, data_root=DATA_CACHE_DIR, shuffle=False, seed=42):
        self.B = B
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'validation'}, "split should be either train or validation"
        # shuffle the row order of every shard each epoch, so the model doesn't see the same batch
        # sequence every epoch, and the rows dropped at the end of an epoch differ each time
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # get the shard filenames
        shards = [s for s in os.listdir(data_root) if f"_{split}_" in s]
        shards = sorted(os.path.join(data_root, s) for s in shards)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if self.process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")

        # state, init at shard zero
        self.current_shard = 0
        self.load_shard()

        # ARC is tiny, so check that a single shard holds at least one batch across all processes
        needed = B * num_processes
        assert len(self.tokens) >= needed, (
            f"shard has {len(self.tokens)} sequences but B*world = {needed} are needed for one batch; "
            f"lower B for this dataset"
        )
        self.current_position = self.B * self.process_rank # positions are in rows (sequences) now

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

    def next_batch(self, B=None):
        B = self.B if B is None else B

        # B whole rows, each one question+answer of ROW_LEN = T_MAX + 1 tokens
        lo, hi = self.current_position, self.current_position + B
        buf = self.tokens[lo:hi] # (B, T_MAX + 1)
        x = buf[:, :-1] # inputs (B, T_MAX)
        y = buf[:, 1:].clone() # targets (B, T_MAX), cloned because we are about to stamp -1 into it
        # y[:, t] is tokens[:, t+1], so the mask that lines up with y is the mask shifted by one as well
        supervised = self.loss_mask[lo:hi, 1:]
        y[~supervised] = -1 # question / padding positions do not contribute to the loss

        # advance the position in the rows
        self.current_position += B * self.num_processes
        # move to the next shard when the next step doesn't fit for ALL ranks. Checked on the step's start
        # (same on every rank) rather than this rank's own slice, so all ranks switch shard - and permutation - together
        step_start = self.current_position - B * self.process_rank
        if step_start + B * self.num_processes > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            if self.current_shard == 0:
                self.epoch += 1 # wrapped past the last shard, the next pass gets a fresh shuffle
            self.load_shard()
            self.current_position = B * self.process_rank

        return x, y

if __name__ == "__main__":
    main(split="train")
    main(split="validation")
