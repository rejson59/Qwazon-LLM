#!/usr/bin/env python3
"""
Qwazon Tokenizer Training — własny BPE 48k dla PL + Code.

Dlaczego własny?
- HF Qwen tokenizer: 152k, dużo nieużywanych tokenów dla PL, wolniejszy na ziemniaku
- Nasz: 32k-48k, zoptymalizowany pod PL diakrytyki + Python/JS/Rust + markdown
- Mniej tokenów = szybszy ziemniak (mniej forwardów)

Użycie:
  python scripts/train_tokenizer.py --input data/train.jsonl --vocab 32768 --out tokenizer-qwazon
  # lub z HF datasets:
  python scripts/train_tokenizer.py --hf --limit 10000 --vocab 49152

Wymaga: tokenizers
  pip install tokenizers

Jeśli brak danych HF, użyje syntetyku v2.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def train_tokenizer(texts, vocab_size=32768, out_dir="tokenizer-qwazon"):
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    except ImportError:
        print("Instaluj: pip install tokenizers")
        print("Fallback: używam byte-level (każdy bajt = token) — nie trenuję BPE")
        # fallback: zapisz dummy
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
            json.dump({"vocab_size": vocab_size, "type": "byte-level", "note": "pip install tokenizers dla BPE"}, f)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({"vocab_size": vocab_size, "tokenizer_type": "byte"}, f)
        return

    print(f"Trenuję BPE tokenizer vocab={vocab_size} na {len(texts)} tekstach...")

    # BPE
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"],
        min_frequency=2,
        show_progress=True,
    )

    # texts to iterator
    def iterator():
        for t in texts:
            yield t

    tok.train_from_iterator(iterator(), trainer=trainer, length=len(texts))

    os.makedirs(out_dir, exist_ok=True)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    # zapisz config dla qwazon (tokenizer_class potrzebny żeby AutoTokenizer mógł to wczytać)
    tok_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "vocab_size": vocab_size,
        "bos_token": "<bos>", "bos_token_id": 1,
        "eos_token": "<eos>", "eos_token_id": 2,
        "pad_token": "<pad>", "pad_token_id": 0,
        "unk_token": "<unk>", "unk_token_id": 3,
        "type": "bpe-qwazon",
        "pre_tokenizer": "ByteLevel",
    }
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok_config, f, indent=2, ensure_ascii=False)

    # test
    test = "Cześć! Napisz funkcję quicksort w Pythonie. def quicksort(arr):"
    enc = tok.encode(test)
    dec = tok.decode(enc.ids)
    print(f"Test: {test!r}")
    print(f"  ids ({len(enc.ids)}): {enc.ids[:20]}...")
    print(f"  dec: {dec!r}")
    print(f"  Kompresja: {len(test.encode('utf-8'))} bytes -> {len(enc.ids)} tokens = {len(test.encode('utf-8'))/len(enc.ids):.2f} bytes/tok (byte-level =1.0, BPE lepsze >2.0)")
    print(f"✅ Zapisano do {out_dir}/tokenizer.json ({vocab_size} vocab)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default=None, help="jsonl z {text}")
    p.add_argument("--hf", action="store_true", help="pobierz próbkę z HF (fineweb)")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--out", type=str, default="tokenizer-qwazon")
    args = p.parse_args()

    texts = []
    if args.input and os.path.exists(args.input):
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                try:
                    texts.append(json.loads(line)["text"])
                except: pass
        print(f"Wczytano {len(texts)} z {args.input}")
    elif args.hf:
        try:
            from datasets import load_dataset
            print("Pobieram fineweb-edu sample...")
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
            for i, ex in enumerate(ds):
                if i >= args.limit: break
                t = ex.get("text", "")
                if len(t) > 100:
                    texts.append(t[:4000])
            print(f"Pobrano {len(texts)} z HF")
        except Exception as e:
            print(f"HF failed: {e}, fallback syntetyk")

    if not texts:
        # syntetyk v4
        from qwazon.trainer import build_synthetic_texts
        texts = build_synthetic_texts(args.limit)
        print(f"Używam syntetyku v4: {len(texts)} tekstów")

    # dodatkowo PL wiki sample jeśli mało
    # ensure at least limit
    texts = texts[:args.limit]

    train_tokenizer(texts, vocab_size=args.vocab, out_dir=args.out)

if __name__ == "__main__":
    main()
