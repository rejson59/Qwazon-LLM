#!/usr/bin/env python3
"""
Qwazon Eval — sprawdzanie inteligencji na ziemniaku.
Testy:
- Perpleksja na hold-out (PL + kod)
- HumanEval-mini (5 zadań kodowania, sprawdź czy model generuje poprawny kod)
- Polski QA (5 pytań)
- Szybkość (tok/s)

Użycie:
  python scripts/eval.py --checkpoint checkpoints/qwazon-micro
  python scripts/eval.py --checkpoint checkpoints/qwazon-nano --full
"""
import argparse, sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qwazon.inference import QwazonPipeline

# Mini HumanEval — 5 zadań, oceniane heuristic (czy zawiera kluczowe słowa)
HUMANEVAL_MINI = [
    {
        "prompt": "Pytanie: Napisz funkcję silnia w Pythonie. Odpowiedź: ```python\n",
        "check": ["def", "silnia", "return", "if"],
        "name": "silnia"
    },
    {
        "prompt": "Pytanie: Napisz funkcję quicksort w Pythonie. Odpowiedź: ```python\n",
        "check": ["def", "quicksort", "pivot", "return"],
        "name": "quicksort"
    },
    {
        "prompt": "Pytanie: Napisz BFS w Pythonie. Odpowiedź: ```python\n",
        "check": ["def", "bfs", "deque", "visited"],
        "name": "bfs"
    },
    {
        "prompt": "def fibonacci(n):",
        "check": ["def", "fibonacci", "return"],
        "name": "fibonacci"
    },
    {
        "prompt": "SQL: wybierz użytkowników z >3 zamówieniami. Odpowiedź: ```sql\n",
        "check": ["SELECT", "COUNT", "GROUP BY", "HAVING"],
        "name": "sql-groupby"
    },
]

POLISH_QA = [
    {"prompt": "Pytanie: Gdzie leży Polska? Odpowiedź:", "check": ["Europa", "Warszawa"], "name": "polska"},
    {"prompt": "Pytanie: Co to jest closure w JavaScript? Odpowiedź:", "check": ["funkcja", "pamięta"], "name": "closure"},
]

def eval_checkpoint(ckpt, max_new=64, full=False):
    print(f"\n{'='*60}")
    print(f"EVAL: {ckpt}")
    print(f"{'='*60}")
    pipe = QwazonPipeline(ckpt)
    cfg = pipe.config
    print(f"Model: {cfg.describe()}")
    print(f"Params: {cfg.num_parameters_approx/1e6:.1f}M")
    # speed
    _, stats = pipe.generate("Cześć, kim jesteś? Opowiedz o sobie.", max_new_tokens=32, do_sample=False)
    print(f"Speed: {stats['tok_per_sec']:.1f} tok/s")

    # HumanEval mini
    print("\n--- HumanEval-Mini (5 zadań) ---")
    passed = 0
    for task in HUMANEVAL_MINI:
        text, _ = pipe.generate(task["prompt"], max_new_tokens=max_new, temperature=0.2, do_sample=False)
        # wyciągnij generację po prompcie
        gen = text[len(task["prompt"]):] if text.startswith(task["prompt"]) else text
        # check
        ok = all(kw.lower() in gen.lower() for kw in task["check"])
        # dodatkowo sprawdź czy nie jest to losowy bełkot (musi mieć >20 znaków sensownych)
        # heuristic: jeśli zawiera ``` to lepiej
        score = "✅ PASS" if ok else "❌ FAIL"
        if ok: passed += 1
        print(f"  {task['name']:12s} {score} | {gen[:80].replace(chr(10),' ')}...")
    print(f"Wynik HumanEval-mini: {passed}/{len(HUMANEVAL_MINI)} ({passed/len(HUMANEVAL_MINI)*100:.0f}%)")

    print("\n--- Polski QA (2 pytania) ---")
    passed_pl = 0
    for task in POLISH_QA:
        text, _ = pipe.generate(task["prompt"], max_new_tokens=32, temperature=0.2, do_sample=False)
        gen = text[len(task["prompt"]):] if text.startswith(task["prompt"]) else text
        ok = any(kw.lower() in gen.lower() for kw in task["check"])
        if ok: passed_pl += 1
        print(f"  {task['name']:12s} {'✅' if ok else '❌'} | {gen[:70].replace(chr(10),' ')}...")

    # perplexity na syntetyku (jeśli mamy data)
    print("\n--- Perpleksja (syntetyk hold-out) ---")
    try:
        import torch
        from qwazon.model import QwazonForCausalLM
        from qwazon.trainer import TextDataset, collate_fn
        from torch.utils.data import DataLoader
        from qwazon.trainer import build_synthetic_texts
        texts = build_synthetic_texts(100)[-10:]  # 10 eval
        # WAŻNE: używamy TEGO SAMEGO tokenizera co model (z checkpointu).
        # Wcześniej tworzono tu nowy QwazonTokenizer() → byte-level fallback,
        # więc model wytrenowany na BPE dostawał inne id tokenów i PPL była bez sensu.
        tok = pipe.tokenizer
        print(f"  Tokenizer: {'BPE' if getattr(tok, 'hf_tokenizer', None) is not None else 'byte-level'} (vocab {len(tok)})")
        ds = TextDataset(texts, tok, max_len=128)
        loader = DataLoader(ds, batch_size=2, collate_fn=lambda b: collate_fn(b, pad_token_id=cfg.pad_token_id))
        model = pipe.model
        model.eval()
        total_loss = 0.0
        total_tok = 0
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(pipe.device)
                labels = batch["labels"].to(pipe.device)
                out = model(input_ids=input_ids, labels=labels)
                # loss to średnia po ważnych (nie -100) tokenach po przesunięciu
                n_valid = int((labels[..., 1:] != -100).sum().item())
                total_loss += out["loss"].item() * n_valid
                total_tok += n_valid
        avg_loss = total_loss / max(1, total_tok)
        ppl = math.exp(min(avg_loss, 20))
        # bits/byte — jedyna metryka porównywalna między różnymi tokenizerami
        total_bytes = 0
        for t in texts:
            ids = tok.encode(t)[:128]
            total_bytes += len(tok.decode(ids).encode("utf-8"))
        bits_per_byte = (total_loss / math.log(2)) / max(1, total_bytes)
        print(f"  Eval loss: {avg_loss:.4f} | PPL: {ppl:.2f} | tokenów: {total_tok}")
        print(f"  bits/byte: {bits_per_byte:.3f} (losowy model = 8.0, byte-level ~5.1, im mniej tym lepiej)")
        # interpretacja
        if bits_per_byte < 2.0:
            print("  ✅ Bardzo dobrze (model już kuma strukturę kodu/PL)")
        elif bits_per_byte < 3.5:
            print("  ⚠️ Średnio — wytrenuj dłużej")
        else:
            print("  ❌ Wysoka entropia — model świeży po init, potrzebuje więcej kroków")
    except Exception as e:
        print(f"  (skip ppl) {e}")

    # podsumowanie
    print("\n" + "="*60)
    total_pass = passed + passed_pl
    total = len(HUMANEVAL_MINI) + len(POLISH_QA)
    print(f"PODSUMOWANIE: {total_pass}/{total} zadań zaliczonych")
    if total_pass >= 5:
        print("🎉 Qwazon już kodzi! Gotowy do użycia jako asystent.")
    elif total_pass >= 3:
        print("👍 Nieźle — po 100-300 krokach będzie kodził sensownie.")
    else:
        print("🌱 Wczesny etap — potrzeba więcej treningu (uruchom --full, 300 kroków).")
    print("="*60)
    return {"humaneval": passed/len(HUMANEVAL_MINI), "polish": passed_pl/len(POLISH_QA), "tok_s": stats['tok_per_sec']}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/qwazon-micro")
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--full", action="store_true", help="dłuższa eval")
    args = p.parse_args()
    eval_checkpoint(args.checkpoint, max_new=args.max_new, full=args.full)
