# peft — LoRA fine-tuning of published HF base models

LoRA fine-tuning of `Qwen/Qwen3-4B-Base` (or any HF causal LM via `--model-id`) with Hugging Face PEFT.
The training loop has the same shape as `finetune_gpt2.py` (DDP, grad accumulation, warmup + cosine lr, json logs),
but the model, the LoRA injection, and the adapter save/merge come from `transformers` + `peft` instead of `finetuning/lora.py`.

| file | what |
|---|---|
| `model_setup.py` | tokenizer / base model loading (bf16 or 4-bit QLoRA), `LoraConfig`, `get_peft_model`, optimizer, merge |
| `sft_data.py` | prompt/completion → `input_ids` + `labels` (prompt masked with -100), padding collator, DDP sampler |
| `finetune_qwen.py` | the training script |

## Setup

```bash
uv add peft               # (+ bitsandbytes on CUDA if load_in_4bit = True)
```

## Run (from the repo root)

```bash
uv run python peft/finetune_qwen.py
uv run torchrun --standalone --nproc_per_node=4 peft/finetune_qwen.py
uv run python peft/finetune_qwen.py --resume-from log_peft/checkpoint/step_00100
```

Run it as a file path. This folder has no `__init__.py` on purpose: it's a namespace package, so `import peft`
still resolves to the installed library. Don't add an `__init__.py`, and don't run it with `python -m peft...`.

## Data contract

`sft_data.load_examples(split)` returns `[{"prompt": str, "completion": str}, ...]`. Loss is only paid on
`completion + <|endoftext|>`. It raises `NotImplementedError` until the ARC-Challenge formatting is written.

## Outputs (`log_peft/`)

- `log.json` — per-step train loss / lr / norm, val loss every `val_step`
- `checkpoint/step_XXXXX/` — adapter weights + `trainer_state.pt` (optimizer, rng, data position), resumable
- `adapter_final/` — the trained adapter (~tens of MB); load it with `PeftModel.from_pretrained(base, path)`
- `merged_final/` — full merged model, only if `save_merged = True`
