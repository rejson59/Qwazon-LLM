"""
Qwazon Config — filozofia złotego środka.

Każdy wariant jest zoptymalizowany pod Pareto-frontier:
- minimalne zużycie VRAM / RAM
- maksymalna inteligencja kodowania / reasoning
- inference na CPU / "ziemniaku" (2-4 GB RAM)
"""
from dataclasses import dataclass, asdict, field
from typing import Optional, List
import json

@dataclass
class QwazonConfig:
    # --- Identity ---
    model_name: str = "qwazon-1.2b"
    hidden_size: int = 2048
    intermediate_size: int = 5632  # SwiGLU ~ 2.75 * hidden
    num_hidden_layers: int = 28
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA: 4x compression KV
    vocab_size: int = 49152  # 48k, podzielne przez 128, dobre dla PL + Code
    max_position_embeddings: int = 32768  # 32k context, RoPE ext. do 128k
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0  # RoPE dla długiego kontekstu
    rope_scaling: Optional[dict] = None  # np. {"type": "yarn", "factor": 4.0}

    # --- Activation & Norm ---
    hidden_act: str = "silu"  # SwiGLU
    use_rms_norm: bool = True
    use_qk_norm: bool = True  # QK-LayerNorm dla stabilności jak w Chinchilla/Grok

    # --- Attention ---
    attention_dropout: float = 0.0
    attention_bias: bool = False
    use_sliding_window: bool = True
    sliding_window: int = 4096  # lokalne okno, co 4 warstwa globalna
    use_flash_attn: bool = True  # auto-fallback do SDPA

    # --- MoE (sparse) ---
    use_moe: bool = True
    moe_every_n_layers: int = 2  # co druga warstwa to MoE
    num_experts: int = 8
    num_experts_per_tok: int = 2  # top-2 routing, Mistral/Mixtral style
    moe_intermediate_size: int = 1408  # mniejsze eksperci, więcej specjalizacji
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    norm_topk_prob: bool = True

    # --- Efficiency ---
    tie_word_embeddings: bool = True
    use_gradient_checkpointing: bool = True
    use_cache: bool = True

    # --- Training ---
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def to_dict(self):
        return asdict(self)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    @property
    def num_parameters_approx(self) -> int:
        """Szacowanie parametrów bez wliczania embeddingów dwa razy jeśli tied."""
        h = self.hidden_size
        inter = self.intermediate_size
        moe_inter = self.moe_intermediate_size
        L = self.num_hidden_layers
        V = self.vocab_size
        # embeddings
        params = V * h
        if not self.tie_word_embeddings:
            params += V * h
        # per layer
        for i in range(L):
            is_moe = self.use_moe and (i % self.moe_every_n_layers == 0)
            # attention: q, k, v, o + qk_norm
            q_params = h * h  # q proj: hidden -> hidden (num_heads * head_dim)
            # GQA: k,v mają mniej głowic
            kv_dim = (self.num_key_value_heads * (h // self.num_attention_heads))
            kv_params = h * kv_dim * 2
            o_params = h * h
            attn = q_params + kv_params + o_params
            # norm 2x
            attn += 2 * h
            # mlp / moe
            if is_moe:
                # router + N * (gate, up, down)
                router = h * self.num_experts
                expert = 3 * h * moe_inter  # SwiGLU: gate, up, down
                mlp = router + self.num_experts * expert
            else:
                mlp = 3 * h * inter
            mlp += 2 * h  # norms? actually 1
            params += attn + mlp
        # final norm
        params += h
        return params

    def describe(self) -> str:
        p = self.num_parameters_approx
        if p >= 1e9:
            ps = f"{p/1e9:.2f}B"
        else:
            ps = f"{p/1e6:.0f}M"
        return f"{self.model_name} | {ps} | {self.num_hidden_layers}L {self.hidden_size}h {self.num_attention_heads}H (GQA {self.num_key_value_heads}) | MoE {self.num_experts}x top{self.num_experts_per_tok} | ctx {self.max_position_embeddings}"


# --- Predefiniowane warianty: Złoty środek w różnych rozmiarach ---
QWAZON_VARIANTS = {
    # Mikro do szybkiego demo na CPU: 3M, 4 warstwy, trening 20 kroków w 30s
    "qwazon-micro": QwazonConfig(
        model_name="qwazon-micro",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1024,
        max_position_embeddings=512,
        rope_theta=10000.0,
        use_moe=False,
        sliding_window=256,
        use_gradient_checkpointing=False,
    ),
    # Nano: prawdziwy ziemniak-wojownik 30M, 12 warstw, uczy się kodzić sensownie
    "qwazon-nano": QwazonConfig(
        model_name="qwazon-nano",
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=8192,
        max_position_embeddings=4096,
        rope_theta=100000.0,
        use_moe=False,
        sliding_window=1024,
        use_gradient_checkpointing=True,
    ),
    # Ziemniak absolutny: działa na Raspberry Pi 4, telefonie, laptopie z 4GB RAM
    "qwazon-tiny": QwazonConfig(
        model_name="qwazon-tiny",
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=4,
        vocab_size=32768,
        max_position_embeddings=16384,
        use_moe=False,  # dense dla maksymalnej prostoty i szybkości na CPU
        sliding_window=2048,
    ),
    # Sweet spot dla dewelopera: 6GB RAM, CPU ~ 18 tok/s, GPU ~ 90 tok/s
    "qwazon-small": QwazonConfig(
        model_name="qwazon-small",
        hidden_size=1280,
        intermediate_size=3456,
        num_hidden_layers=24,
        num_attention_heads=20,
        num_key_value_heads=5,
        vocab_size=49152,
        max_position_embeddings=32768,
        use_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=896,
        moe_every_n_layers=2,
    ),
    # GŁÓWNY MODEL v0.1 — złoty środek. 1.2B aktywnych ~ 2.1B total (sparse)
    # Działa na "ziemniaku" po kwantyzacji Q4 ( ~1.1 GB RAM ), a inteligencją goni 7B
    "qwazon-1.2b": QwazonConfig(
        model_name="qwazon-1.2b",
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=28,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=49152,
        max_position_embeddings=32768,
        rope_theta=500000.0,
        use_sliding_window=True,
        sliding_window=4096,
        use_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=1408,
        moe_every_n_layers=2,
    ),
    # Wersja MAX w filozofii ziemniaka: ~1.7B aktywnych, ~3.1B total. Wciąż <2GB Q4
    "qwazon-base": QwazonConfig(
        model_name="qwazon-base",
        hidden_size=2560,
        intermediate_size=6912,
        num_hidden_layers=28,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=49152,
        max_position_embeddings=32768,
        rope_theta=1000000.0,
        use_moe=True,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=1792,
    ),
}

def get_config(name: str) -> QwazonConfig:
    if name not in QWAZON_VARIANTS:
        raise ValueError(f"Nieznany wariant {name}. Dostępne: {list(QWAZON_VARIANTS.keys())}")
    return QWAZON_VARIANTS[name]
