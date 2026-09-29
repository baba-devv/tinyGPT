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
shard_size = int(1e8) # 100M tokens per shard

# every array here is (2, N) uint16: row TOKENS is the token stream, row MASK is 1 on the
# answer tokens and 0 elsewhere.
TOKENS, MASK = 0, 1

# create the local cache directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    # tokenizes a single QA document into a (2, N) uint16 array. The mask row is what lets the
    # fine-tune pay a loss for predicting the answer only, and not for reciting the prompt back.
    labels = doc["choices"]["label"]
    answer_key = doc["answerKey"]
    if not answer_key or answer_key not in labels:
        return None # test split ships unlabelled docs, drop them
    answer = doc["choices"]["text"][labels.index(answer_key)]

    # the document text is the question with the correct choice appended
    question_tokens = enc.encode_ordinary(doc["question"])
    answer_tokens = enc.encode_ordinary(" " + answer)

    # the special <|endoftext|> token delimits all documents
    tokens = [eot] + question_tokens + answer_tokens
    # built from the piece lengths, so the two rows line up by construction
    mask = [0] * (1 + len(question_tokens)) + [1] * len(answer_tokens)

    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2 **16).all(), "token dictionary total tokens exceed the limit"
    return np.stack([tokens_np, np.array(mask)]).astype(np.uint16)

def write_datafile(filename, shard_np):
    # writes the (2, N) token/mask array to a .npy file
    np.save(filename, shard_np)

# tokenize all documents and write output shards, each of shard_size tokens
def main(split="train"):
    # stream the dataset so downloading overlaps tokenization instead of blocking on the full download
    fw = load_dataset("allenai/ai2_arc", name=remote_name, split=split, streaming=True)

    nproc = max(1, os.cpu_count() // 2)
    with mp.Pool(nproc) as pool: # instantiating n copy processes
        shard_index = 0
        # preallocate buffer to hold current shard
        all_np = np.empty((2, shard_size), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for doc_np in pool.imap(tokenize, fw, chunksize=16):
            if doc_np is None:
                continue
            n = doc_np.shape[1]
            # is there enough space in the current shard for the new tokens ?
            if token_count + n < shard_size:
                # simply append tokens to current shard
                all_np[:, token_count:token_count+n] = doc_np
                token_count += n
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}: ")
                progress_bar.update(n)
            else:
                # write the current shard and start a new one
                filename = os.path.join(DATA_CACHE_DIR, f"arc_ai_{split}_{shard_index:06d}")
                # split the document into whatever fits into this shard, the remainder goes to next shard
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_np[:, token_count:token_count+remainder] = doc_np[:, :remainder]
                write_datafile(filename, all_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with leftovers of the current doc
                all_np[:, 0:n-remainder] = doc_np[:, remainder:]
                token_count = n - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            filename = os.path.join(DATA_CACHE_DIR, f"arc_ai_{split}_{shard_index:06d}")
            write_datafile(filename, all_np[:, :token_count])
            print(f"wrote final shard {shard_index} with {token_count} tokens")


# -----------------------------------------------------------------
# loading side

def load_tokens(filename):
    shard = np.load(filename)
    tokens = torch.tensor(shard[TOKENS], dtype=torch.long)
    loss_mask = torch.tensor(shard[MASK], dtype=torch.bool)
    return tokens, loss_mask

class DataLoaderFT:
    """Same flat-stream slicing as DataLoaderLite, but the targets of every non-answer
    position are set to -1 so model.forward's cross_entropy(ignore_index=-1) skips them."""

    def __init__(self, B, T, process_rank, num_processes, split, data_root=DATA_CACHE_DIR, current_shard=None, current_pos=None):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'validation'}, "split should be either train or validation"

        # get the shard filenames
        shards = [s for s in os.listdir(data_root) if f"_{split}_" in s]
        shards = sorted(os.path.join(data_root, s) for s in shards)
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if self.process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")

        # state, init at shard zero
        self.current_shard = current_shard if current_shard is not None else 0  # if loading from checkpoint
        self.tokens, self.loss_mask = load_tokens(shards[self.current_shard])

        if T is not None:
            # ARC is tiny next to the pretraining corpus, so a single shard can be smaller than one batch
            needed = B * T * num_processes + 1
            assert len(self.tokens) >= needed, (
                f"shard has {len(self.tokens)} tokens but B*T*world+1 = {needed} are needed for one batch; "
                f"lower B or T for this dataset"
            )
            self.current_position = current_pos if current_pos is not None else self.B * self.T * self.process_rank
        else:
            self.current_position = None

    def reset(self, split='train'):
        self.current_shard = 0
        self.tokens, self.loss_mask = load_tokens(self.shards[self.current_shard])
        if split != 'validation':
            self.current_position = self.B * self.T * self.process_rank
        else:
            self.current_position = None

    def next_batch(self, B=None, T=None):
        B, T = self.B if B is None else B, self.T if T is None else T

        if self.current_position is None:
            # initialize the position if not set already
            self.current_position = B * T * self.process_rank

        lo, hi = self.current_position, self.current_position + B*T + 1
        buf = self.tokens[lo:hi]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T).clone() # targets, cloned because we are about to stamp -1 into it
        # y[i] is tokens[i+1], so the mask that lines up with y is the mask shifted by one as well
        supervised = self.loss_mask[lo+1:hi].view(B, T)
        y[~supervised] = -1 # question / delimiter positions do not contribute to the loss

        # advance the position in the tensor
        self.current_position += B * T * self.num_processes  # we're not doing B * T + 1 here because in training the consumption was only till B * T, the +1 was used for target hence there is no overlap
        # if loading next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens, self.loss_mask = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank  # 1 epoch completed (almost)

        return x, y

if __name__ == "__main__":
    main(split="train")
    main(split="validation")
