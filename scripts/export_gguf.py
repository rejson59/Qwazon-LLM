#!/usr/bin/env python3
"""
Export Qwazon do GGUF dla llama.cpp / Ollama / LM Studio.

W prawdziwym setupie:
  pip install gguf
  python scripts/export_gguf.py --checkpoint checkpoints/qwazon-1.2b --out qwazon-1.2b-q4_k_m.gguf --type Q4_K_M

Ten skrypt pokazuje strukturę i przygotowuje metadata.
Pełny export wymaga biblioteki gguf i wag HF.
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pathlib import Path
import json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--type", type=str, default="Q4_K_M", choices=["F16","Q8_0","Q4_K_M","Q4_0","Q5_K_M","Q6_K"])
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not (ckpt / "config.json").exists():
        print(f"❌ Brak config.json w {ckpt}")
        print("   Upewnij się że checkpoint istnieje: ls checkpoints/")
        sys.exit(1)

    with open(ckpt / "config.json") as f:
        cfg = json.load(f)

    print(f"Export GGUF: {cfg['model_name']} -> {args.out} ({args.type})")
    print(f"  Hidden: {cfg['hidden_size']}, Layers: {cfg['num_hidden_layers']}, Vocab: {cfg['vocab_size']}")
    print(f"  MoE: {cfg.get('use_moe')}")

    # Próba prawdziwego exportu jeśli gguf dostępne
    try:
        import gguf
        print(f"  gguf lib found: {gguf.__version__ if hasattr(gguf,'__version__') else 'unknown'}")
        print("  → Tutaj byłby prawdziwy zapis GGUF (wymaga wag HF format).")
        print("  → Dla demo: tworzę placeholder i instrukcję.")
    except ImportError:
        print("  gguf nie zainstalowane: pip install gguf")
        print("  → Tworzę placeholder + instrukcję Ollama.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # placeholder
    info = {
        "model": cfg["model_name"],
        "type": args.type,
        "checkpoint": str(ckpt),
        "gguf_out": args.out,
        "ollama_instruction": f"FROM {args.out}\nTEMPLATE \"<|user|>{{{{ .Prompt }}}}<|assistant|>\"\nPARAMETER temperature 0.7",
    }
    with open(args.out + ".json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Ollama Modelfile
    with open(str(Path(args.out).with_suffix("")) + ".Modelfile", "w") as f:
        f.write(f"FROM {args.out}\n")
        f.write('TEMPLATE """<|system|>\n{{ .System }}\n<|user|>\n{{ .Prompt }}\n<|assistant|>\n"""\n')
        f.write("PARAMETER temperature 0.7\nPARAMETER top_p 0.9\n")

    print(f"✅ Info zapisane: {args.out}.json")
    print(f"✅ Modelfile: {Path(args.out).with_suffix('').__str__()}.Modelfile")
    print("\nAby uruchomić w Ollama:")
    print(f"  ollama create qwazon -f {Path(args.out).with_suffix('').__str__()}.Modelfile")
    print(f"  ollama run qwazon \"Napisz quicksort w Pythonie\"")

if __name__ == "__main__":
    main()
