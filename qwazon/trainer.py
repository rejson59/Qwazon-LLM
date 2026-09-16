"""
Qwazon Trainer — pipeline treningowy złotego środka.

Filozofia: nie trenujemy od zera 100B tokenów jak giganci.
Używamy:
1. Distillation z nauczyciela (Claude/GPT/Qwen-72B) — uczeń uczy się rozkładów
2. Wysokiej jakości dane: 40% kod, 25% polski, 20% angielski STEM, 15% math/reasoning
3. Curriculum + 8-bit AdamW + gradient checkpointing — trening na 1x RTX 4090 możliwy
4. DPO na końcu dla preferencji

Ten plik pozwala odpalić trening na ziemniaku dla wersji tiny/small i na GPU dla 1.2B.
"""
import os, math, time, json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .config import QwazonConfig
from .model import QwazonForCausalLM

@dataclass
class TrainArgs:
    output_dir: str = "checkpoints/qwazon-1.2b"
    # data
    train_file: Optional[str] = None  # jsonl z {text: ...} lub {input_ids: ...}
    eval_file: Optional[str] = None
    max_seq_length: int = 2048  # start 2k, potem 8k, potem 32k (curriculum)
    # optim
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    max_steps: int = 10000
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 2
    bf16: bool = True
    grad_clip: float = 1.0
    # distillation
    use_distillation: bool = False
    teacher_model: Optional[str] = None
    distillation_alpha: float = 0.5
    distillation_temp: float = 2.0

class TextDataset(Dataset):
    """Prosty dataset: listy token ids already tokenized, lub surowe teksty."""
    def __init__(self, texts_or_ids, tokenizer, max_len=2048, is_ids=False):
        self.data = texts_or_ids
        self.tok = tokenizer
        self.max_len = max_len
        self.is_ids = is_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_ids:
            ids = self.data[idx][:self.max_len]
        else:
            ids = self.tok.encode(self.data[idx])[:self.max_len]
        # labels = ids copy
        # padding handled by collator
        return torch.tensor(ids, dtype=torch.long)

def collate_fn(batch, pad_token_id=0):
    # batch: list[Tensor variable len]
    max_len = max(x.size(0) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        L = x.size(0)
        input_ids[i, :L] = x
        attention_mask[i, :L] = 1
        labels[i, :L] = x
        # mask pad w labels = -100 (ignore)
        # ale dla input_ids już pad_token_id
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(config: QwazonConfig, args: TrainArgs, tokenizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Device: {device}")
    print(f"[Trainer] Config: {config.describe()}")
    print(f"[Trainer] Params: {config.num_parameters_approx/1e9:.3f}B total, aktywne ~ {config.num_parameters_approx * (0.45 if config.use_moe else 1.0)/1e9:.3f}B")

    model = QwazonForCausalLM(config)
    model.to(device)
    if torch.cuda.is_available() and args.bf16:
        model = model.to(torch.bfloat16)

    # optionally compile — tylko na GPU z dostępnym kompilatorem, na CPU/ziemniaku wyłączone
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            # quick probe czy inductor ma Python.h
            import subprocess, textwrap
            has_header = os.path.exists("/usr/include/python3.11/Python.h") or os.path.exists("/usr/include/python3.10/Python.h")
            if has_header:
                model = torch.compile(model)
                print("[Trainer] torch.compile włączone")
            else:
                print("[Trainer] torch.compile skip — brak Python.h (ziemniak CPU), używam eager")
        except Exception as e:
            print(f"[Trainer] torch.compile failed: {e}")

    # dummy data jeśli brak pliku — pozwala przetestować pipeline
    if args.train_file is None or not os.path.exists(args.train_file or ""):
        print("[Trainer] Brak train_file, generuję syntetyczne dane demo (polski + kod)")
        demo_texts = [
            "Cześć! Jestem Qwazon, polski LLM zoptymalizowany pod ziemniaka. Jak mogę pomóc?",
            "Napisz funkcję quicksort w Pythonie:\n```python\ndef quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```",
            "Wyjaśnij różnicę między `list` a `tuple` w Pythonie. Lista jest mutowalna, tupla nie.",
            "Zadanie: suma dwóch liczb. Input: 2 3, Output: 5. Rozwiązanie w O(1).",
            "Polska to kraj w Europie Środkowej. Stolica to Warszawa.",
            "System sortowania: implementacja mergesort z rekurencją i dzieleniem na pół.",
            "Dla danego stringu odwróć kolejność słów: 'ala ma kota' -> 'kota ma ala'",
            "SELECT * FROM users WHERE age > 18 ORDER BY name;",
            "Oblicz silnię rekurencyjnie i iteracyjnie. Porównaj złożoność.",
            "Co to jest Mixture of Experts? To architektura gdzie tylko podzbiór ekspertów jest aktywny na token.",
        ] * 200  # 2000 przykładów
        from .tokenizer import QwazonTokenizer
        if tokenizer is None:
            tokenizer = QwazonTokenizer(vocab_size=config.vocab_size)
        train_ds = TextDataset(demo_texts, tokenizer, max_len=args.max_seq_length, is_ids=False)
    else:
        # load jsonl
        with open(args.train_file) as f:
            lines = [json.loads(l) for l in f]
        # zakładamy {"text": "..."} lub {"input_ids": [...]}
        if "input_ids" in lines[0]:
            ids_list = [x["input_ids"] for x in lines]
            train_ds = TextDataset(ids_list, None, max_len=args.max_seq_length, is_ids=True)
        else:
            texts = [x["text"] for x in lines]
            from .tokenizer import QwazonTokenizer
            if tokenizer is None:
                tokenizer = QwazonTokenizer(vocab_size=config.vocab_size)
            train_ds = TextDataset(texts, tokenizer, max_len=args.max_seq_length, is_ids=False)

    train_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_token_id=config.pad_token_id))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    total_steps = args.max_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.bf16))

    model.train()
    global_step = 0
    accum = args.gradient_accumulation_steps
    os.makedirs(args.output_dir, exist_ok=True)

    # log file
    log_path = Path(args.output_dir) / "train_log.jsonl"
    # save config
    config.to_json(os.path.join(args.output_dir, "config.json"))

    print(f"[Trainer] Start treningu: {total_steps} kroków, accum {accum}, batch {args.per_device_train_batch_size}, seq {args.max_seq_length}")
    print(f"[Trainer] Efektywny batch: {args.per_device_train_batch_size * accum} seq, ~{args.per_device_train_batch_size * accum * args.max_seq_length / 1000:.1f}k tokenów/krok")

    # mixed precision context
    autocast_dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float16 if device.type == "cuda" else None

    iter_loader = iter(train_loader)
    start_time = time.time()
    losses = []

    while global_step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        for micro in range(accum):
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # forward
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = out["loss"] / accum
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out["loss"] / accum

            # backward
            if device.type == "cuda" and not args.bf16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item()

        # clip & step
        if device.type == "cuda" and not args.bf16:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        scheduler.step()

        global_step += 1
        losses.append(accum_loss)
        avg_loss = sum(losses[-50:]) / min(len(losses), 50)
        lr = scheduler.get_last_lr()[0]

        if global_step % args.logging_steps == 0:
            elapsed = time.time() - start_time
            tok_per_sec = (args.per_device_train_batch_size * accum * args.max_seq_length * args.logging_steps) / elapsed if elapsed>0 else 0
            ppl = math.exp(min(avg_loss, 10))
            print(f"step {global_step:5d}/{total_steps} | loss {accum_loss:.4f} avg {avg_loss:.4f} ppl {ppl:.2f} lr {lr:.2e} | {tok_per_sec:.0f} tok/s")
            # log
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": global_step, "loss": accum_loss, "avg_loss": avg_loss, "lr": lr, "ppl": ppl}) + "\n")
            start_time = time.time()

        if global_step % args.save_steps == 0:
            # save
            ckpt_dir = Path(args.output_dir) / f"step-{global_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # unwrapped model if compiled
            m = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(m.state_dict(), ckpt_dir / "pytorch_model.bin")
            config.to_json(ckpt_dir / "config.json")
            print(f"[Trainer] Checkpoint zapisany: {ckpt_dir}")

    # final save
    final_dir = Path(args.output_dir)
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(m.state_dict(), final_dir / "pytorch_model.bin")
    print(f"[Trainer] Trening zakończony. Model w {final_dir}")
    print(f"[Trainer] Aby uruchomić inference: python scripts/generate.py --checkpoint {final_dir} --prompt 'Napisz quicksort'")
    return model

if __name__ == "__main__":
    from .config import get_config
    cfg = get_config("qwazon-tiny")
    args = TrainArgs(max_steps=100, per_device_train_batch_size=2, max_seq_length=512)
    train(cfg, args)
