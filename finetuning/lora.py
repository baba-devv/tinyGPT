import torch
import torch.nn as nn


class LinearLoRA(nn.Linear):

    def __init__(self, fan_in, fan_out, rank, alpha=16, device=None):
        super().__init__(fan_in, fan_out, device=device)

        self.scaling = alpha / rank

        self.merged = False

        fan_out_individual = fan_out // 3 # distribute into q, k and v vectors, because the base model uses fused qkv mat

        # B maps rank -> fan_out (Expansion) - pytorch representation of weights 
        self.B_q = nn.Parameter(torch.zeros(fan_out_individual, rank, device=device))  # n_embd, rank
        self.B_k = nn.Parameter(torch.zeros(fan_out_individual, rank, device=device))  # n_embd, rank
        self.B_v = nn.Parameter(torch.zeros(fan_out_individual, rank, device=device))  # n_embd, rank
        # A maps fan_in -> rank (Compression) - pytorch representation of weights
        self.A_q = nn.Parameter(torch.zeros(rank, fan_in, device=device))  # rank, n_embd
        self.A_k = nn.Parameter(torch.zeros(rank, fan_in, device=device))  # rank, n_embd
        self.A_v = nn.Parameter(torch.zeros(rank, fan_in, device=device))  # rank, n_embd

        nn.init.kaiming_uniform_(self.A_q, a=5**0.5)
        nn.init.kaiming_uniform_(self.A_k, a=5**0.5)
        nn.init.kaiming_uniform_(self.A_v, a=5**0.5)
        nn.init.zeros_(self.B_q)
        nn.init.zeros_(self.B_k)
        nn.init.zeros_(self.B_v)

    def forward(self, x):
        out = super().forward(x)

        # check the merge status in the forward pass
        if not self.merged:
            lora_update_q = (x @ self.A_q.T @ self.B_q.T) * self.scaling  # n_embd, n_embd
            lora_update_k = (x @ self.A_k.T @ self.B_k.T) * self.scaling  # n_embd, n_embd
            lora_update_v = (x @ self.A_v.T @ self.B_v.T) * self.scaling  # n_embd, n_embd

            lora_update = torch.cat((lora_update_q, lora_update_k, lora_update_v), dim=-1)  # n_embd, 3 * n_embd

            # lora_update = (x @ self.A.T @ self.B.T) * self.scaling
            out = out + lora_update
        
        return out
    
    def merge(self):
        # don't remerge
        if self.merged:
            return

        with torch.no_grad():
            # (out, rank) @ (rank, in)
            fused_q = (self.B_q @ self.A_q) * self.scaling
            fused_k = (self.B_k @ self.A_k) * self.scaling
            fused_v = (self.B_v @ self.A_v) * self.scaling

            fused = torch.cat((fused_q, fused_k, fused_v), dim=0) # 3C, C
            # in place merge
            self.weight.add_(fused)

        self.merged = True

    def unmerge(self):
        # unmerge only if already merged
        if self.merged:
            with torch.no_grad():
                # (out, rank) @ (rank, in)
                fused_q = (self.B_q @ self.A_q) * self.scaling
                fused_k = (self.B_k @ self.A_k) * self.scaling
                fused_v = (self.B_v @ self.A_v) * self.scaling

                fused = torch.cat((fused_q, fused_k, fused_v), dim=0) # 3C, C
                # in place merge
                self.weight.sub_(fused)

            self.merged = False


# ---------------------------------------------------------------------------------------
# initialisation for PEFT (LoRA)

def inject_lora_layers(model, rank, alpha):

    # target_layer_names = ["q_attn", "k_attn", "v_attn"]
    target_layer_names = ["c_attn"]
    layers_to_replace = []

    for name, module in model.named_modules():
        # Check if the name ends with one of our targets (e.g., "c_attn")
        if any(name.endswith(target) for target in target_layer_names):
            layers_to_replace.append((name, module))

    for name, old_layer in layers_to_replace:
        # inject the new layer
        new_layer = LinearLoRA(
            fan_in=old_layer.in_features,
            fan_out=old_layer.out_features,
            rank=rank,
            alpha=alpha,
            device=old_layer.weight.device
        )
        # copy the existing pre-trained weights
        with torch.no_grad():
            new_layer.weight.copy_(old_layer.weight)
            if old_layer.bias is not None:
                new_layer.bias.copy_(old_layer.bias)

            # in case checkpoint is present for SFT
            # if old_layer.B is not None:
            #     new_layer.B.copy_(old_layer.B)
            # if old_layer.A is not None:
            #     new_layer.A.copy_(old_layer.A)

        # overwrite the old layer attribute with the new LoRA layer
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent_module = model.get_submodule(parent_name)
        else:
            parent_module = model
            attr_name = name

        # Swap the old layer with the new one on the correct parent module
        setattr(parent_module, attr_name, new_layer)


def freeze_param_for_lora(model):

    # freeze every param in the model
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze only LoRA layers params
    for module in model.modules():
        if isinstance(module, LinearLoRA):
            module.A_q.requires_grad = True
            module.B_q.requires_grad = True            
            module.A_k.requires_grad = True
            module.B_k.requires_grad = True
            module.A_v.requires_grad = True
            module.B_v.requires_grad = True

