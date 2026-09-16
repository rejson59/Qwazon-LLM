#!/usr/bin/env python3
"""
Qwazon Train Entry Point

Użycie:
  python scripts/train.py --config configs/qwazon_tiny.yaml
  python scripts/train.py --variant qwazon-tiny --steps 500 --demo
  python scripts/train.py --variant qwazon-1.2b --data data/train.jsonl

Dla szybkiego testu na ziemniaku:
  python scripts/train.py --variant qwazon-tiny --steps 100 --batch 2
"""
import argparse, yaml, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qwazon.config import get_config, QwazonConfig
from qwazon.trainer import TrainArgs, train
from qwazon.tokenizer import QwazonTokenizer

def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser(description="Trenuj Qwazona")
    p.add_argument("--config", type=str, help="ścieżka do YAML (configs/...)")
    p.add_argument("--variant", type=str, default="qwazon-tiny", choices=["qwazon-tiny","qwazon-small","qwazon-1.2b","qwazon-base"], help="wariant jeśli brak YAML")
    p.add_argument("--steps", type=int, default=None, help="nadpisz max_steps")
    p.add_argument("--batch", type=int, default=None, help="nadpisz batch size")
    p.add_argument("--seq", type=int, default=None, help="nadpisz max_seq_length")
    p.add_argument("--data", type=str, default=None, help="ścieżka do train.jsonl")
    p.add_argument("--output", type=str, default=None, help="output_dir")
    p.add_argument("--demo", action="store_true", help="demo na syntetycznych danych (bez pliku)")
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()

    # config
    if args.config:
        cfg_yaml = load_yaml(args.config)
        m = cfg_yaml.get("model", {})
        # zbuduj QwazonConfig z YAML
        # tylko pola które istnieją
        base = get_config(m.get("model_name", args.variant))
        # nadpisz
        for k, v in m.items():
            if hasattr(base, k):
                setattr(base, k, v)
        config = base
        t = cfg_yaml.get("training", {})
        train_args = TrainArgs(
            output_dir=t.get("output_dir", f"checkpoints/{config.model_name}"),
            max_seq_length=t.get("max_seq_length", 2048),
            learning_rate=t.get("learning_rate", 3e-4),
            max_steps=t.get("max_steps", 5000),
            per_device_train_batch_size=t.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
            bf16=t.get("bf16", True),
        )
        if args.data:
            train_args.train_file = args.data
        print(f"[Train] Wczytano YAML: {args.config}")
    else:
        config = get_config(args.variant)
        train_args = TrainArgs(
            output_dir=f"checkpoints/{config.model_name}",
            max_seq_length=1024 if config.model_name=="qwazon-tiny" else 2048,
            learning_rate=3e-4,
            max_steps=100 if args.demo else 5000,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            bf16=False,
        )
        if args.data:
            train_args.train_file = args.data

    # CLI overrides
    if args.steps is not None:
        train_args.max_steps = args.steps
    if args.batch is not None:
        train_args.per_device_train_batch_size = args.batch
    if args.seq is not None:
        train_args.max_seq_length = args.seq
    if args.output is not None:
        train_args.output_dir = args.output
    if args.lr is not None:
        train_args.learning_rate = args.lr
    if args.demo:
        train_args.train_file = None

    print("="*60)
    print(f"QWAZON TRENING — {config.model_name}")
    print(f"  {config.describe()}")
    print(f"  Params: {config.num_parameters_approx/1e6:.0f}M total")
    print(f"  Output: {train_args.output_dir}")
    print(f"  Steps: {train_args.max_steps}, seq: {train_args.max_seq_length}, batch: {train_args.per_device_train_batch_size} x {train_args.gradient_accumulation_steps}")
    print("="*60)

    tokenizer = QwazonTokenizer(vocab_size=config.vocab_size)
    train(config, train_args, tokenizer=tokenizer)

if __name__ == "__main__":
    main()
