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

if __name__ == "__main__":
    test_forward()
    test_generate()
