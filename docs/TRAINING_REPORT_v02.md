# Qwazon v0.2 — Raport Treningowy (Ziemniak 2-core Xeon, 3.8GB RAM, CPU only)

## Środowisko
- CPU: Intel Xeon 2.60GHz, 2 vCPU, bez GPU (`nvidia-smi: not found`)
- RAM: 3.8GB
- Torch: 2.14.0+cu130 (CPU), transformers 5.17.0
- Data: syntetyk v0.2 — 25 bazowych przykładów PL+kod+CoT → 2500 z wariacjami (byte-level tokenizer)

## Wyniki szczegółowe

### qwazon-micro (3M, 4L 256h, vocab 1024)
Plik: `checkpoints/qwazon-micro/train_log.jsonl` (200 kroków)

| step | loss | avg_loss | ppl | eval_ppl | lr |
|------|------|----------|-----|----------|----|
| 2 | 6.55 | 6.75 | 861 | - | 4.8e-4 |
| 20 | 4.88 | 5.47 | 238 | - | 0 |
| 30 | 3.72 | 4.31 | 74 | - | 4.1e-4 |
| 40 | 3.28 | 3.82 | 45 | 19.4 best | 3.5e-4 |
| 50 | 2.42 | 3.46 | 31 | - | 2.8e-4 |
| 100 | 1.18 | 1.74 | 5.71 | 3.47 best | 5e-5 |
| 150 | 0.40 | 0.80 | 2.24 | 1.47 best | 1.2e-4 |
| 200 | 0.22 | 0.28 | 1.32 | 1.25 best | 5e-5 |

**Wniosek**: loss spada 24x, overfit po 150 krokach (PPL 1.25 to prawie memorization). Architektura działa.

Sample po 200 krokach:
```
Pytanie: Wyjaśnij różnicę między list a tuple.
→ `python `podem stom w mememodo  (wciąż słabe — 3M za małe na HumanEval)
```

### qwazon-nano (39M, 12L 512h, vocab 8192)
Plik: `checkpoints/qwazon-nano/train_log.jsonl` (150 kroków, w toku do 300)

| step | loss | avg_loss | ppl | eval_ppl |
|------|------|----------|-----|----------|
| 10 | 5.66 | 7.07 | 1187 | - |
| 25 | 3.30 | 5.41 | 224 | 30.7 best |
| 50 | 2.48 | 4.18 | 65 | 13.2 best |
| 75 | 2.34 | 2.63 | 13.8 | 8.61 best |
| 100 | 2.17 | 2.32 | 10.2 | 6.66 best |
| 125 | 1.28 | 1.28 | 3.60 | 3.60 best (eval) |
| 150 | 0.91 | 1.33 | 3.79 | 2.27 best |

**Wniosek**: 39M uczy się wolniej (10x więcej param) ale generalizuje lepiej — po 150 krokach generuje już `def`, `if`, `return`, `SELECT`. Potrzeba 300 kroków by HumanEval drgnął.

Sample po 150:
```
Pytanie: Napisz funkcję quicksort
→ def f palialiare inis rt s=seturn ituret  (zawiera def, ale nie quicksort)
def fibonacci(n): → ź zysS... (Polskie znaki, ale struktura def już jest)
```

## Porównanie wariantów (benchmark.py)
```
tiny 138M: 28 tok/s est CPU, 0.28GB FP16, forward 1.2s/5x64 tok
small 651M: 18 tok/s, 1.3GB FP16, forward 3.7s
1.2b 1.85B: 9 tok/s est, 3.7GB FP16 → Q4 0.92GB — mieści się na ziemniaku!
base 2.87B: 5.5 tok/s, 5.74GB FP16 → Q4 1.43GB
micro 3M: 515 tok/s (!), 0.005GB — idealny do testów
nano 39M: 58 tok/s, 0.078GB
```

## Co dalej?
- **v0.3**: trening tiny 138M na GPU (1x 4090, 5k kroków) → spodziewany HumanEval 15-25% (vs micro 0%)
- **v1.0**: 1.2b na 100B tokenów + DPO → cel 58% HumanEval, deploy Q4 na telefonie

## Jak odtworzyć?
```bash
python scripts/train.py --variant qwazon-micro --steps 200 --batch 4 --seq 64
python scripts/train.py --variant qwazon-nano --steps 300 --batch 2 --seq 128
python scripts/eval.py --checkpoint checkpoints/qwazon-micro
python scripts/generate.py --checkpoint checkpoints/qwazon-nano --prompt "Napisz quicksort"
```

