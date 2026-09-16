"""
Qwazon LoRA — ziemniak-friendly fine-tuning.
Trenuj 1.2B na 6GB RAM zamiast 24GB.

Użycie:
  from qwazon.lora import apply_lora, LoRAConfig
  model = QwazonForCausalLM(config)
  model = apply_lora(model, LoRAConfig(r=16, alpha=32, target_modules=["q_proj","v_proj"]))

Zainspiruj się QLoRA: 4-bit base + LoRA adapters.
Na ziemniaku: możesz fine-tunować nano 39M na CPU w <1GB RAM!
"""
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LoRAConfig:
    r: int = 16  # rank
    alpha: int = 32  # scaling
    dropout: float = 0.05
    target_modules: List[str] = None  # ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    bias: str = "none"  # none, all, lora_only

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "gate_proj", "up_proj"]

class LoRALinear(nn.Module):
    """
    Wrap nn.Linear z LoRA adapterem: W + BA * (alpha/r)
    A: (r, in), B: (out, r) — tylko BA jest trenowane, W zamrożone
    """
    def __init__(self, base_layer: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base_layer
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        in_f = base_layer.in_features
        out_f = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0

        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, in_f) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_f, r))
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.dropout = nn.Identity()

    def forward(self, x):
        result = self.base(x)
        if self.r > 0 and self.lora_A is not None:
            # x: [..., in], A: [r, in], B: [out, r]
            # lora = (dropout(x) @ A.T @ B.T) * scaling
            dropped = self.dropout(x)
            # (..., in) @ (in, r) = (..., r)
            lora_mid = dropped @ self.lora_A.T
            lora_out = lora_mid @ self.lora_B.T
            result = result + lora_out * self.scaling
        return result

def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """
    Podmień wszystkie Linear z target_modules na LoRALinear.
    Zwraca model z zamrożonym base + trenowalnymi LoRA.
    """
    replaced = 0
    for name, module in model.named_modules():
        # Sprawdź czy to target
        short_name = name.split(".")[-1]
        if short_name in config.target_modules and isinstance(module, nn.Linear):
            # Znajdź parent i podmień
            parent_name = ".".join(name.split(".")[:-1])
            parent = model.get_submodule(parent_name) if parent_name else model
            lora_layer = LoRALinear(module, r=config.r, alpha=config.alpha, dropout=config.dropout)
            setattr(parent, short_name, lora_layer)
            replaced += 1

    # Zlicz trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Podmieniono {replaced} warstw | Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total*100:.2f}%)")
    print(f"[LoRA] r={config.r}, alpha={config.alpha}, dropout={config.dropout}, target={config.target_modules}")
    if total > 0:
        # Estymacja RAM: base 4-bit (0.5B) + LoRA fp16
        print(f"[LoRA] Est. RAM QLoRA: ~{total*0.5/1e9:.2f}GB base 4-bit + {trainable*2/1e9:.3f}GB LoRA = ~{total*0.5/1e9 + trainable*2/1e9:.2f}GB total")
    return model

def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

def get_lora_state_dict(model: nn.Module):
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}

def load_lora_state_dict(model: nn.Module, state: dict):
    model.load_state_dict(state, strict=False)
    print(f"[LoRA] Załadowano {len(state)} adapterów")

# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from qwazon.config import get_config
    from qwazon.model import QwazonForCausalLM
    cfg = get_config("qwazon-nano")
    model = QwazonForCausalLM(cfg)
    print(f"Przed LoRA: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    model = apply_lora(model, LoRAConfig(r=8, alpha=16))
    # forward test
    import torch
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    out = model(ids)
    print("Forward OK, logits:", out["logits"].shape)
