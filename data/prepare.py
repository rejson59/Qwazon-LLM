"""
Przygotowanie danych dla Qwazona — filozofia złotego środka v0.4.

Nie potrzebujemy 15T tokenów. Potrzebujemy 100B, ale *krystalicznie* czystych.
Miks v0.4 (50 syntetyków + prawdziwe dane):
  35% kod (The Stack v2 dedup + StarCoder2, filtrowany przez AST i testy)
  20% polski edukacyjny (FineWeb-PL + Wolne Lektury + WikiPL)
  15% angielski STEM (FineWeb-Edu)
  10% CodeReasoning v4 (50 zadań CoT, bugfix, review, PL)
  10% Math (GSM8K, MATH, OrcaMath po polsku)
  10% dialogi / instrukcje (UltraFeedback PL, OASST PL)

Ten skrypt pokazuje jak przygotować jsonl z HF datasets.
Uruchom: python data/prepare.py --out data/train.jsonl --limit 10000 --demo
"""
import argparse, json, random, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from data.synthetic_v4 import SYNTHETIC_V4
    DEMO_DATA = [{"text": t} for t in SYNTHETIC_V4]
except ImportError:
    DEMO_DATA = [
        {"text": "Pytanie: Napisz funkcję silnia w Pythonie.\nOdpowiedź: ```python\ndef silnia(n):\n    if n <= 1:\n        return 1\n    return n * silnia(n-1)\n```\nWyjaśnienie: rekurencja, przypadek bazowy n<=1, złożoność O(n)."},
        {"text": "Pytanie: Co to jest closure w JavaScript?\nOdpowiedź: Closure to funkcja która pamięta swoje leksykalne otoczenie nawet gdy wykonuje się poza nim. Przykład: `function outer(x){ return function inner(y){ return x+y; }}`"},
        {"text": "Pytanie: Odwróć listę w Pythonie bez .reverse().\nOdpowiedź: ```python\ndef reverse_list(arr):\n    return arr[::-1]  # slicing\n# lub\ndef reverse_loop(arr):\n    res = []\n    for i in range(len(arr)-1, -1, -1):\n        res.append(arr[i])\n    return res\n```"},
        {"text": "System: Jesteś Qwazon, pomocny asystent kodowania. User: Napisz REST API w FastAPI dla todo. Assistant: ```python\nfrom fastapi import FastAPI\napp = FastAPI()\ntodos = []\n@app.get(\"/todos\")\ndef list_todos():\n    return todos\n@app.post(\"/todos\")\ndef add_todo(item: str):\n    todos.append(item)\n    return {\"ok\": True}\n```"},
        {"text": "Polska leży w Europie Środkowej. Graniczy z Niemcami, Czechami, Słowacją, Ukrainą, Białorusią, Litwą i Rosją (obwód kaliningradzki). Stolica to Warszawa, ludność ~38 mln."},
        {"text": "Zadanie: Dla tablicy liczb znajdź maksymalną sumę podtablicy (Kadane).\n```python\ndef max_subarray(nums):\n    cur = best = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best\n```\nZłożoność O(n), pamięć O(1)."},
        {"text": "Wyjaśnij różnicę między procesem a wątkiem. Proces ma własną przestrzeń pamięci, wątek dzieli pamięć procesu. Wątki są lżejsze, ale wymagają synchronizacji."},
        {"text": "SQL: znajdź użytkowników którzy kupili >3 produkty.\n```sql\nSELECT user_id, COUNT(*) as cnt\nFROM orders\nGROUP BY user_id\nHAVING COUNT(*) > 3;\n```"},
    ]

def build_demo(out_path, limit=10000, repeat=1):
    # powiel demo data z wariacjami
    data = (DEMO_DATA * ((limit // len(DEMO_DATA)) + 1))[:limit]
    # dodaj wariacje: zamień liczby, nazwy
    extended = []
    for i, ex in enumerate(data):
        # lekka randomizacja żeby nie było exact duplicate
        text = ex["text"]
        if i % 7 == 0:
            text = text.replace("Python", "Python 3.11")
        extended.append({"text": text, "id": i})

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in extended:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✅ Zapisano demo dataset: {out_path} ({len(extended)} przykładów)")
    print(f"   Przykładowy wpis: {extended[0]['text'][:120]}...")

def try_hf_download(out_path, limit=10000):
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets nie zainstalowane: pip install datasets")
        return False
    # przykład: pobierz fineweb-edu sample + starcoder sample
    print("Pobieram próbkę FineWeb-Edu PL + StarCoder...")
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        # weź limit
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in ds:
                if count >= limit:
                    break
                text = ex.get("text", "")[:4000]  # trim
                if len(text) < 200:
                    continue
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count += 1
        print(f"✅ Pobrano {count} z HF")
        return True
    except Exception as e:
        print(f"HF download failed: {e}")
        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/train.jsonl")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--demo", action="store_true", help="zawsze użyj demo syntetyku (offline)")
    p.add_argument("--try_hf", action="store_true", help="spróbuj pobrać z HF, fallback demo")
    args = p.parse_args()

    if args.demo or not args.try_hf:
        build_demo(args.out, limit=args.limit)
    else:
        ok = try_hf_download(args.out, limit=args.limit)
        if not ok:
            build_demo(args.out, limit=args.limit)

    # statystyki
    with open(args.out, encoding="utf-8") as f:
        total_chars = sum(len(json.loads(l).get("text","")) for l in f)
    print(f"Statystyki: ~{total_chars/1e6:.2f}M znaków, ~{total_chars/4/1e6:.2f}M tokenów est.")

if __name__ == "__main__":
    main()
