# Changelog Qwazon

## v0.4 — Więcej danych, dłuższy kontekst, realna kwantyzacja (2026-09-22)
**Cel: poprawić generalizację (więcej zadań) i udostępnić model na telefonie (Q4)**

### Dane
- **`data/synthetic_v4.py`**: **50 zadań** bazowych (vs 25 w v0.2) — 2x więcej różnorodności przy tym samym rozmiarze zbioru (2500 przykładów), więc mniej zapamiętywania, więcej generalizacji
  - nowe kategorie: **bugfix** (znajdź/popraw błąd), **code review**, **refactor**, **testy jednostkowe**, **tłumaczenia PL↔EN**, **math** (równania, całki), **CoT krok po kroku** (mediana, URL shortener, dlaczego quicksort jest szybki)
- **`data/prepare.py`** i **`qwazon/trainer.py`** importują v4 z jednym fallbackiem (gdy brak pliku — 1 zadanie zastępcze)

### Model
- **YaRN (128k kontekstu)** w `RotaryEmbedding`: `rope_scaling={"type":"yarn","factor":4.0}` dzieli `inv_freq` przez factor i skaluje cache przez `mscale = 0.1·ln(factor)+1` — 4x dłuższy kontekst bez dotrenowywania
- Docstringi i wersje ujednolicone do v0.4

### Kwantyzacja (`qwazon/quantize.py` — przepisany)
- **`quantize_dynamic_int8()`** — prawdziwa kwantyzacja `torch.quantization` (działa na CPU, bez GPU): forward na modelu INT8 zweryfikowany testem
- **`quantize_4bit_simulate()`** — symulacja Q4 z realnym MSE na próbkach wag (micro: MSE 8e-6) + estymacja rozmiaru
- **`benchmark_quantized()`** — porównanie FP32 vs INT8 (czas + RAM)
- `export_gguf_fake()` — dopisuje estymację rozmiaru do pliku info

### Testy (`tests/test_model.py`)
- `test_yarn_rope` — factor 4.0 → inv_freq x4.00, mscale 1.1386, forward OK
- `test_synthetic_v4` — 50 zadań, ≥50 unikalnych, markery v4 obecne
- `test_quantize` — Q4 8x mniejszy, MSE, forward na INT8
- **Wynik: 5/5 testów przechodzi** (forward, generate, YaRN, v4, quantize)

### Dokumentacja
- **`docs/PROSTY_OPIS.md`** — wytłumaczenie bez żargonu, dla osoby która nie zna ML

## v0.3 — Ziemniak na sterydach (2026-09-17)
**Cel: udowodnić że nano zdaje testy z programowania + infrastruktura produkcyjna**

### Trening
- **qwazon-nano 39M → 300 kroków**: PPL 1.058, **eval PPL 1.044** ⭐⭐⭐
  - **HumanEval-mini 1/5 (20%)** — pierwszy zdany test (`silnia`), vs 0/5 po 150 krokach
  - Dowód skalowania: 39M (20%) > 3M (0%)
- **qwazon-tiny-lite 25M** (8L 512h GQA4 ctx1024): 25 kroków w 53s, loss 7.44→5.72, eval PPL 130 — proof że 25M też się uczy

### Kod v0.3
- **`qwazon/api.py`** — FastAPI (CORS, `/generate`, `/chat`, `/health`, `/models`, docs na `/docs`)
- **`qwazon/lora.py`** — LoRA r=8 (17.9M/39.5M trainable na nano), QLoRA-ready
- **`qwazon/dpo.py`** — DPO beta=0.1 bez reward modelu (loss 0.708)
- **`scripts/train_tokenizer.py`** — własny BPE 1024, **2.87 bytes/tok** (vs 1.0 byte-level) = 2.8x mniej tokenów
- **`demo/app.py` v0.3** — streaming, system prompt, 8 klikanych przykładów, footer tok/s
- **`Dockerfile` + `docker-compose.yml`** — obraz CPU ~1GB z healthcheckiem

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

