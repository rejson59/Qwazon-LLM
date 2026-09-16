"""
Qwazon Quantize — GGUF / AWQ / GPTQ / bitsandbytes.

Złoty środek = kwantyzacja jest first-class obywatelem, nie afterthought.
Model od początku trenowany z myślą o Q4 (QAT-ready, RMSNorm stabilny).
"""
import torch
from pathlib import Path

def estimate_sizes(config):
    params = config.num_parameters_approx
    print(f"Model {config.model_name}: {params/1e9:.3f}B parametrów")
    for bits, name in [(16, "FP16"), (8, "INT8"), (4, "Q4_K_M"), (3, "Q3_K_M"), (2, "Q2_K")]:
        size_gb = params * bits / 8 / 1e9
        print(f"  {name:8s} ({bits}bit): {size_gb:.2f} GB")
    print("  → Q4_K_M to sweet spot: ~0.5 byte/param, 4x mniejszy niż FP16, <2% degradacji")

def export_gguf_fake(checkpoint_dir: str, out_path: str, qtype: str = "Q4_K_M"):
    """
    Placeholder dla prawdziwego exportu GGUF (wymaga llama.cpp convert).
    W produkcji: python -m llama_cpp.convert --outtype q4_k_m
    Tu: symulujemy i pokazujemy co by się stało.
    """
    print(f"[Quantize] Export GGUF {qtype} z {checkpoint_dir} -> {out_path}")
    print("[Quantize] W prawdziwym środowisku uruchom:")
    print(f"  python scripts/export_gguf.py --checkpoint {checkpoint_dir} --out {out_path} --type {qtype}")
    # fake file
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # nie tworzymy prawdziwego GGUF, tylko info
    with open(out_path + ".info.txt", "w") as f:
        f.write(f"Qwazon GGUF export placeholder\nCheckpoint: {checkpoint_dir}\nType: {qtype}\n")
    print(f"[Quantize] Placeholder info zapisany: {out_path}.info.txt")

if __name__ == "__main__":
    from .config import get_config
    for name in ["qwazon-tiny", "qwazon-small", "qwazon-1.2b", "qwazon-base"]:
        print()
        estimate_sizes(get_config(name))
