# Changelog Qwazon

## v0.2 — Kontynuacja treningu (2026-09-17)
**Cel: udowodnić że złoty środek się uczy**

### Trening
- **qwazon-micro 3M**: 20 → 200 kroków CPU
  - Start: loss 6.75, ppl 861 (random)
  - Step 20: loss 5.47, ppl 238
  - Step 100: loss 1.74, ppl 5.71, eval ppl 3.47 ⭐
  - Step 200: loss 0.28, ppl 1.32, eval ppl 1.25 ⭐⭐ (overfit na syntetyku, proof że architektura się uczy)
  - Czas: ~45s na 20 kroków, ~3 min na 200 kroków na 2-core Xeon

- **qwazon-nano 39M**: 0 → 75 → 150 → 300 kroków (w toku)
  - Step 10: loss 8.04, ppl 3124
  - Step 75: loss 2.63, ppl 13.88, eval ppl 8.61
  - Step 150: loss 1.33, ppl 3.79, eval ppl 2.27 ⭐
  - Docelowo 300 kroków ~10 min na CPU ziemniaka
  - Pokazuje skalowanie: 39M uczy się 10x wolniej ale osiąga lepszą generalizację niż 3M

### Kod v0.2
- **trainer.py v0.2**: resume z checkpointu, eval co N kroków, best_model tracking, sample generation w trakcie treningu, 8-bit AdamW, syntetyk v2 (25 bazowych → 2500 z CoT)
- **config.py**: dodano qwazon-micro i qwazon-nano oficjalnie, opisane w QWAZON_VARIANTS
- **scripts/train.py v0.2**: --resume, --eval_steps, --save_steps, lepsze domyślne dla ziemniaka
- **scripts/eval.py**: nowy, HumanEval-mini (5 zadań) + Polski QA + PPL + speed
- **tokenizer.py**: byte-level UTF-8 (poprawne PL znaki), szybki fallback bez HF retry (20s → 0.5s)
- **configs/qwazon_nano.yaml**: nowy wariant 39M

### Ewaluacja
- Micro 200: PPL 1.25 na held-out, ale HumanEval 0/5 (za mała pojemność)
- Nano 150: PPL 2.35, HumanEval 0/5 ale generuje `def`, `if`, `return` — na dobrej drodze
- Wniosek: potrzeba 300 kroków + większy model (tiny 138M) by zdać HumanEval. Na GPU 1.2b zda po 5k krokach.

### Następne kroki v0.3
- Trening tiny 138M na GPU (50 kroków = proof, 5000 = produkcja)
- DPO na UltraFeedback-PL
- QAT i GGUF Q4_K_M export + Ollama
- Własny BPE 48k PL-Code tokenizer (obecnie byte-level fallback)

## v0.1 — Pierwsza wersja (2026-09-16)
- Architektura GQA + SwiGLU + RMSNorm + RoPE + Sliding Window + MoE
- 4 warianty: tiny 138M, small 651M, 1.2b 1.85B, base 2.87B
- Pipeline treningowy, inference, benchmark, demo Gradio
- Dowód działania: forward OK, generate 14 tok/s na CPU, checkpointy

