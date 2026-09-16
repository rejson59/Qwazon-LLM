#!/usr/bin/env python3
"""
Qwazon Generate — szybkie inference na ziemniaku.
Użycie:
  python scripts/generate.py --checkpoint checkpoints/qwazon-tiny --prompt "Napisz quicksort"
  python scripts/generate.py --checkpoint checkpoints/qwazon-1.2b --benchmark
  echo "Wyjaśnij rekurencję" | python scripts/generate.py --checkpoint checkpoints/qwazon-tiny --stdin
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from qwazon.inference import QwazonPipeline

def main():
    p = argparse.ArgumentParser(description="Qwazon Generate")
    p.add_argument("--checkpoint", type=str, default="checkpoints/qwazon-tiny")
    p.add_argument("--prompt", type=str, default="Cześć! Napisz funkcję quicksort w Pythonie i wyjaśnij jej działanie krok po kroku.")
    p.add_argument("--stdin", action="store_true", help="czytaj prompt z stdin")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--chat", action="store_true", help="tryb chat (multi-turn)")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.stdin:
        args.prompt = sys.stdin.read().strip()

    pipe = QwazonPipeline(args.checkpoint, device=args.device)

    if args.benchmark:
        pipe.benchmark_potato()
        return

    if args.chat:
        print("Qwazon Chat — wpisz 'exit' aby wyjść")
        history = []
        while True:
            try:
                user = input("\n👤 Ty: ")
            except EOFError:
                break
            if user.strip().lower() in ("exit", "quit", "q"):
                break
            history.append({"role": "user", "content": user})
            text, stats = pipe.chat(history, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            # wyciągnij tylko ostatnią odpowiedź
            # nasz chat zwraca całość, więc wycinamy prefix
            # prosty split po ostatnim <|assistant|>
            if "<|assistant|>" in text:
                text = text.split("<|assistant|>")[-1].strip()
            print(f"\n🤖 Qwazon: {text}")
            print(f"   ({stats['tok_per_sec']:.1f} tok/s)")
            history.append({"role": "assistant", "content": text})
        return

    text, stats = pipe.generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    print("\n" + "="*60)
    print(text)
    print("="*60)
    print(f"\n✅ {stats['tokens']} tokenów w {stats['time']:.2f}s → {stats['tok_per_sec']:.1f} tok/s")

if __name__ == "__main__":
    main()
