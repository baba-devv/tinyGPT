import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

# Qwen3 decoder block linear layers - attention (q, k, v, o) and the SwiGLU MLP (gate, up, down).
# Targeting all of them (QLoRA paper recipe) beats attention-only LoRA at the same param budget.
# embed_tokens / lm_head are tied in Qwen3-4B and deliberately left frozen.
QWEN3_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Qwen3 base ships pad_token == eos_token == <|endoftext|>.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # training pads on the right, generation sets left itself
    return tokenizer


def load_base_model(model_id, device, dtype=torch.bfloat16, load_in_4bit=False, gradient_checkpointing=True):
    if load_in_4bit:
        # QLoRA: frozen base stored in 4-bit NF4, LoRA adapters trained in higher precision. CUDA only.
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        # quantized weights can't be moved with .to(), they have to be placed at load time
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, dtype=dtype, device_map={"": device})
        # casts norms to fp32, enables input grads so checkpointing works through the frozen embeddings
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa")
        model.to(device)
        if gradient_checkpointing:
            # enable gradient checkpointing, this will trade of compute for memory
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # the kv cache is useless in training and conflicts with gradient checkpointing
    model.config.use_cache = False
    return model


def build_lora_config(rank, alpha, dropout, target_modules=QWEN3_TARGET_MODULES):
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def wrap_with_lora(model, lora_config):
    # get_peft_model freezes every base parameter and injects the A/B matrices into the target - the PEFT equivalent of inject_lora_layers + freeze_param_for_lora
    return get_peft_model(model, lora_config)


def load_adapter(model, adapter_dir, is_trainable):
    # attaches a saved adapter onto a freshly loaded base model. is_trainable=True to resume training,
    # False for inference (the adapter is then loaded frozen)
    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=is_trainable)


def configure_optimizer(model, learning_rate, weight_decay, device):
    # only the LoRA params are trainable, everything else was frozen by get_peft_model
    params = [p for p in model.parameters() if p.requires_grad]
    # same split as GPT.configure_optimizers: decay the 2D matrices, not the 1D tensors (if any)
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    use_fused = "cuda" in str(device)
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, fused=use_fused)


def merge_and_save(peft_model, tokenizer, out_dir):
    # folds B @ A * scaling into the base weights and drops the LoRA modules
    merged = peft_model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return merged
