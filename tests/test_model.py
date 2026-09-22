import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from qwazon.config import get_config
from qwazon.model import QwazonForCausalLM

def test_forward():
    cfg = get_config("qwazon-tiny")
    cfg.vocab_size = 1024  # zmniejsz dla testu
    model = QwazonForCausalLM(cfg)
    ids = torch.randint(0, 1024, (2, 16))
    out = model(ids, labels=ids)
    assert out["logits"].shape == (2, 16, 1024)
    assert out["loss"] is not None
    print("✅ forward OK, loss", out["loss"].item())

def test_generate():
    cfg = get_config("qwazon-tiny")
    cfg.vocab_size = 1024
    cfg.num_hidden_layers = 4
    cfg.hidden_size = 256
    cfg.intermediate_size = 512
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.max_position_embeddings = 512
    model = QwazonForCausalLM(cfg)
    model.eval()
    ids = torch.randint(0, 1024, (1, 8))
    gen = model.generate(ids, max_new_tokens=5, do_sample=False)
    assert gen.shape[1] == 13
    print("✅ generate OK", gen.shape)

def test_yarn_rope():
    """v0.4: YaRN scaling musi wydłużyć kontekst (inv_freq / factor) i przeskalować cache."""
    from qwazon.model import RotaryEmbedding
    dim = 64
    base_emb = RotaryEmbedding(dim, max_position_embeddings=128, base=10000.0)
    yarn_emb = RotaryEmbedding(dim, max_position_embeddings=128, base=10000.0,
                               rope_scaling={"type": "yarn", "factor": 4.0})
    # inv_freq podzielone przez factor => dłuższy kontekst
    ratio = (base_emb.inv_freq / yarn_emb.inv_freq).mean().item()
    assert abs(ratio - 4.0) < 1e-4, f"YaRN factor nie zastosowany: {ratio}"
    # mscale = 0.1*ln(factor)+1
    import math
    expected_mscale = 0.1 * math.log(4.0) + 1.0
    assert abs(yarn_emb.mscale - expected_mscale) < 1e-6, yarn_emb.mscale
    assert abs(base_emb.mscale - 1.0) < 1e-9
    # forward musi działać i zwrócić cache o właściwym kształcie
    x = torch.randn(1, 4, 16, dim)
    cos, sin = yarn_emb(x, seq_len=32)
    assert cos.shape == (32, dim) and sin.shape == (32, dim)
    # pełny model z YaRN w configu musi zrobić forward
    cfg = get_config("qwazon-micro")
    cfg.rope_scaling = {"type": "yarn", "factor": 4.0}
    model = QwazonForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    out = model(ids, labels=ids)
    assert out["loss"] is not None
    print(f"✅ YaRN OK: factor 4.0 → inv_freq x{ratio:.2f}, mscale {yarn_emb.mscale:.4f}, forward loss {out['loss'].item():.4f}")

def test_synthetic_v4():
    """v0.4: trainer musi używać 50 zadań z data/synthetic_v4.py, nie 25 z v0.2."""
    from data.synthetic_v4 import SYNTHETIC_V4
    from qwazon.trainer import build_synthetic_texts
    assert len(SYNTHETIC_V4) == 50, f"oczekiwano 50 zadań, jest {len(SYNTHETIC_V4)}"
    texts = build_synthetic_texts(300)
    assert len(texts) == 300
    uniq = set(texts)
    assert len(uniq) >= 50, f"za mało unikalnych: {len(uniq)}"
    # v4 ma zadania których nie było w v0.2 (bugfix, review, overfitting)
    joined = "\n".join(SYNTHETIC_V4)
    for marker in ["Popraw błąd", "Przejrzyj kod", "overfitting", "Zrefaktoryzuj"]:
        assert marker in joined, f"brak markera v4: {marker}"
    print(f"✅ Synthetic v4 OK: {len(SYNTHETIC_V4)} zadań bazowych, {len(uniq)} unikalnych w próbce 300")

def test_quantize():
    """v0.4: realna kwantyzacja int8 + symulacja Q4 muszą działać i zmniejszać model."""
    from qwazon.quantize import quantize_dynamic_int8, quantize_4bit_simulate
    cfg = get_config("qwazon-micro")
    model = QwazonForCausalLM(cfg)
    stats = quantize_4bit_simulate(model)
    assert stats["q4_gb"] < stats["fp32_gb"] / 7, stats  # 4bit ~8x mniejszy niż fp32
    q = quantize_dynamic_int8(model)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out = q(ids)  # forward na kwantyzowanym musi działać
    assert out["logits"].shape[0] == 1
    print(f"✅ Quantize OK: Q4 {stats['q4_gb']*1000:.2f}MB vs FP32 {stats['fp32_gb']*1000:.2f}MB, MSE {stats['mse']:.6f}, int8 forward działa")

if __name__ == "__main__":
    test_forward()
    test_generate()
    test_yarn_rope()
    test_synthetic_v4()
    test_quantize()
    print("\n🎉 Wszystkie testy v0.4 przeszły")
