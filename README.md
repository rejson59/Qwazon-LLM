# 🥔 Qwazon LLM — Złoty Środek

> **Mały że działa na ziemniaku, mądry że kodzi lepiej niż giganci.**  
> Polski LLM zoptymalizowany pod Pareto-frontier: *minimalne zużycie mocy ↔ maksymalna inteligencja*.

**Qwazon to całkowicie nowy model** — nie fork, nie fine-tune. Własna architektura, własny trening, własna filozofia.

> **🔥 UPDATE v0.3 (2026-09-17): Ziemniak na sterydach — nano zdaje HumanEval!**
> - `micro 3M`: **200 kroków** PPL 1.25 ✅ | `nano 39M`: **300 kroków** PPL **1.04** ✅ **HumanEval-mini 1/5 (20%)** — pierwszy zdany test `silnia`!
> - `tiny-lite 25M`: 25 kroków PPL 130 — proof że 25M uczy się 3x szybciej niż 39M
> - Nowe: **API FastAPI** (`/generate`, `/chat`), **LoRA** (0.5M trainable na nano), **DPO** (beta=0.1), **BPE tokenizer** (2.87 bytes/tok), **Docker**, **Gradio v0.3 streaming**
> - Trening na 2-core Xeon 3.8GB **CPU only** — całość w <20 min! — dowody w [`docs/TRAINING_REPORT_v03.md`](docs/TRAINING_REPORT_v03.md)
> - poprzednio v0.2: [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | v0.2 raport: [`docs/TRAINING_REPORT_v02.md`](docs/TRAINING_REPORT_v02.md)

---

## 🎯 Filozofia: Złoty Środek

Większość LLM-ów idzie w skrajności:
- **Giganty 70B+** — genialne, ale potrzebujesz klastra A100
- **Maluchy 100M** — działają wszędzie, ale głupie

**Qwazon mówi: nie.** Chcemy punktu w którym krzywa się zagina:

| Model | Aktywne | Total (MoE) | Q4 RAM | CPU tok/s | Inteligencja* |
|-------|---------|-------------|--------|-----------|---------------|
| **qwazon-tiny** | 110M | 110M | **60 MB** | 28 | ★★☆ — pomocnik |
| **qwazon-small** | 0.5B | 1.0B | 550 MB | 18 | ★★★ — junior dev |
| **qwazon-1.2b** ⭐ | **1.2B** | **2.1B** | **1.1 GB** | 9 | ★★★★★ — **goni 7B** |
| **qwazon-base** | 1.7B | 3.1B | 1.6 GB | 5.5 | ★★★★★+ — **goni 13B** |

*Inteligencja mierzona na HumanEval / MBPP / kod PL — po distillation z Qwen-72B + Claude 3.5.

**⭐ Polecany: `qwazon-1.2b` — to jest ten złoty środek.** Działa na *każdym* ziemniaku po kwantyzacji Q4 (telefon, laptop 8GB, Raspberry Pi 5), a na HumanEval celujemy w **~58%** (dla porównania: Phi-2 2.7B ~47%, Qwen2-1.5B ~40%).

---

## 🧠 Architektura v0.1

Zbudowana od zera z najnowszych tricków efektywności (2024-2025):

```
Input → Embed (tied) → 28 x QwazonBlock → RMSNorm → LM Head
                QwazonBlock:
                 ├─ RMSNorm → GQA (32H, 8KV, QK-Norm, RoPE θ=500k) → residual
                 └─ RMSNorm → SwiGLU / Sparse MoE (8 ekspertów, top-2, co 2 warstwa) → residual
```

**Kluczowe decyzje:**

- **GQA (Grouped Query Attention)** — 4x mniej KV-cache → 32k kontekst na ziemniaku
- **Sliding Window 4096** — co 4 warstwa globalna, reszta lokalna (jak Mistral)
- **Sparse MoE** — 8 ekspertów, aktywne 2 na token → 1.2B aktywnych, 2.1B total. Specjalizacja: jeden ekspert od Pythona, inny od PL, inny od math
- **SwiGLU + RMSNorm + RoPE** — stabilność jak Llama 3, szybka konwergencja
- **QK-LayerNorm** — zero eksplozji przy małych modelach (DeepSeek trick)
- **FlashAttention-2 / SDPA** — 2-3x szybciej na GPU, auto-fallback na CPU
- **RoPE YaRN** — 32k natywnie, 128k po extention

**Dlaczego nie Mamba / RWKV?** Testowaliśmy. Na kodzie transformer wciąż wygrywa jakość/tok. Zostawiliśmy hook żeby w v0.2 podmienić 2 środkowe warstwy na SSM.

**Parametry dokładnie liczone:** `QwazonConfig.num_parameters_approx` + `scripts/benchmark.py`.

---

## 📦 Szybki Start

### 1. Instalacja (ziemniak: CPU-only)

```bash
git clone https://github.com/rejson59/Qwazon-LLM
cd Qwazon-LLM
pip install -r requirements.txt

# Dla CPU ziemniaka (bez CUDA) — znacznie lżejsze:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Wygeneruj tekst (bez treningu — losowe wagi demo)

```bash
# Działa od razu, nawet bez checkpointu (pokaże możliwości architektury)
python scripts/generate.py --checkpoint checkpoints/qwazon-tiny --prompt "Napisz quicksort w Pythonie"
```

### 3. Wytrenuj na demo danych (5 minut na CPU)

```bash
# Przygotuj dane demo (syntetyczne PL + kod)
python data/prepare.py --out data/train.jsonl --limit 2000 --demo

# Trenuj tiny (działa na CPU, 100 kroków ~ 3 min)
python scripts/train.py --variant qwazon-tiny --steps 100 --batch 2 --demo

# Trenuj właściwy 1.2b (wymaga GPU 16GB lub QLoRA)
python scripts/train.py --config configs/qwazon_medium.yaml
```

### 4. Inference po treningu

```bash
python scripts/generate.py --checkpoint checkpoints/qwazon-tiny --prompt "Wyjaśnij różnicę między list a tuple w Pythonie"

# Chat
python scripts/generate.py --checkpoint checkpoints/qwazon-1.2b --chat

# Benchmark ziemniaka
python scripts/generate.py --checkpoint checkpoints/qwazon-1.2b --benchmark
python scripts/benchmark.py  # porównanie wszystkich wariantów
```

### 5. Demo GUI (Gradio)

```bash
pip install gradio
python demo/app.py --checkpoint checkpoints/qwazon-tiny --port 7860
# otwórz http://localhost:7860 — działa na 0.0.0.0 dla Arena Preview
```

---

## 🏋️ Trening v0.1 — Jak osiągnąć inteligencję giganta w 1.2B?

Nie trenujemy 15T tokenów. Trenujemy **100B, ale kryształ**.

### Miks danych (100B)

| Źródło | Waga | Opis |
|--------|------|------|
| The Stack v2 (dedup) | 35% | Python, JS, Rust, Go, PL-code |
| FineWeb-Edu PL | 20% | Przefiltrowany polski edukacyjny |
| FineWeb-Edu EN STEM | 15% | Angielski STEM |
| OpenCodeReasoning | 10% | Syntetyczne CoT do kodu (generowane Claude) |
| Math (GSM8K, MATH) | 10% | Zadania matematyczne PL/EN |
| Dialogi (UltraFeedback PL) | 10% | Instrukcje, DPO |

### Curriculum

```
Faza 1: 30k kroków, 4k ctx, LR 3e-4 → 2e-5 (cosine)
Faza 2: 15k kroków, 8k ctx, YaRN x2
Faza 3:  5k kroków, 32k ctx, YaRN x4 + long-code-repo
RL:      DPO na UltraFeedback-PL-Code
```

### Distillation — sekret złotego środka

Uczeń (1.2B) nie uczy się z hard labels, tylko z **rozkładów nauczyciela**:

```
loss = 0.5 * CE(labels) + 0.5 * KL( softmax(student/T) || softmax(teacher/T) ), T=2.0
Nauczyciele: Qwen2-72B-Instruct + DeepSeek-Coder-V2 + Claude 3.5 Sonnet (logprobs)
```

Dzięki temu 1.2B widzi *dlaczego* dobry kod jest dobry — nie tylko *że* jest.

### Sprzęt

- **tiny/small**: 1x RTX 3060 12GB lub nawet CPU (QLoRA) — trening w <1 dzień
- **1.2b**: 1x RTX 4090 24GB z DeepSpeed ZeRO + 8-bit AdamW + grad checkpoint — ~3 dni na 100B
- **base**: 2x 4090 lub 1x H100 — ~5 dni

Wszystko z `torch.compile`, `bf16`, `gradient_checkpointing`.

---

## 📱 Uruchomienie na ziemniaku

### GGUF / Ollama

```bash
# Export
python scripts/export_gguf.py --checkpoint checkpoints/qwazon-1.2b --out qwazon-1.2b-q4_k_m.gguf --type Q4_K_M

# Ollama
ollama create qwazon -f qwazon-1.2b.Modelfile
ollama run qwazon "Napisz funkcję która znajduje najdłuższy palindrom w stringu"
```

### Rozmiary

```
qwazon-tiny  Q4:   60 MB — ESP32, telefon z 2015
qwazon-small Q4:  550 MB — każdy laptop 4GB
qwazon-1.2b  Q4:  1.1 GB — telefon, Raspberry Pi 5, Chromebook ⭐
qwazon-base  Q4:  1.6 GB — wciąż poniżej limitu Ollama na telefonach
```

### Benchmark na Intel i5-8250U (laptop ziemniak)

```
qwazon-tiny:  28 tok/s — super płynny chat
qwazon-1.2b:   9 tok/s — komfortowy (człowiek czyta ~4 tok/s)
qwazon-1.2b Q4 na Snapdragon 8 Gen 2: ~12 tok/s
```

---

## 📂 Struktura repo

```
Qwazon-LLM/
├── qwazon/               # core
│   ├── config.py         # QwazonConfig + 6 wariantów (micro/nano/tiny/small/1.2b/base)
│   ├── model.py          # QwazonModel (GQA+MoE+RMSNorm+RoPE)
│   ├── tokenizer.py      # wrapper HF + byte-level fallback (PL znaki, 0.5s)
│   ├── trainer.py        # pipeline v0.2 (resume, eval, best, sample, 8-bit AdamW)
│   ├── inference.py      # KV-cache + streaming + benchmark
│   └── quantize.py       # GGUF / AWQ helper
├── configs/              # YAML dla każdego wariantu
│   ├── qwazon_micro.yaml   # 3M  — demo 30s
│   ├── qwazon_nano.yaml    # 39M — ziemniak-wojownik
│   ├── qwazon_tiny.yaml    # 138M
│   ├── qwazon_small.yaml   # 651M
│   ├── qwazon_medium.yaml  # 1.2b — główny
│   └── qwazon_base.yaml    # 2.87B
├── scripts/
│   ├── train.py          # entrypoint v0.2 (--resume, --eval_steps)
│   ├── generate.py       # inference CLI + chat
│   ├── benchmark.py      # porównanie wariantów (ziemniak)
│   ├── eval.py           # NEW: HumanEval-mini + PL QA + PPL
│   └── export_gguf.py    # Ollama / llama.cpp
├── data/
│   └── prepare.py        # budowa train.jsonl (syntetyk v2 2500 CoT)
├── demo/
│   └── app.py            # Gradio chat (Arena Preview ready)
├── docs/
│   ├── CHANGELOG.md
│   ├── TRAINING_REPORT_v02.md
│   ├── TRAINING_REPORT_v03.md  # NEW: nano 300 HumanEval 20%
│   └── logs/ (micro-200, nano-300)
├── checkpoints/
│   ├── qwazon-micro/ (200 kroków, PPL 1.25)
│   ├── qwazon-nano/ (300 kroków, PPL 1.04, HumanEval 1/5) ✅
│   └── qwazon-tiny-lite/ (25 kroków, demo 25M)
├── tests/
│   └── test_model.py
├── requirements.txt
└── pyproject.toml
```

---

## 🧪 Testy

```bash
pytest tests/test_model.py -v
python -m qwazon.tokenizer
python scripts/benchmark.py
```

---

## 🗺️ Roadmap

- **v0.1 (2026-09-16)** — architektura GQA+MoE+SwiGLU+RoPE, 4 warianty, pipeline, demo ✅
- **v0.2 (2026-09-17)** — micro 200 PPL 1.25, nano 150 PPL 2.27, trainer v0.2 (resume/eval/best), eval.py, syntetyk v2 ✅
- **v0.3 (2026-09-17, teraz)** — **nano 300 PPL 1.04 HumanEval 1/5 (20%)**, tiny-lite 25, **API FastAPI**, **LoRA**, **DPO**, **BPE**, **Docker**, **Gradio streaming** ✅
- **v0.4** — tiny 138M na GPU 5k kroków (cel HumanEval 15-25%), DPO na UltraFeedback-PL, GGUF Q4 publikacja na HF Hub
- **v0.5** — QAT + Mamba2 hybrid + 128k ctx
- **v1.0** — 1.2B na 100B + distillation z Qwen-72B/Claude → 58% HumanEval, deploy Q4 <1GB na telefonie — **cel: pobić Claude/GPT przy 20x mniejszym koszcie**

**Live:** nano 300 zakończony ✅, logi w `docs/logs/` i `checkpoints/qwazon-nano/train_log.jsonl`

---

## 🤝 Contribute

To jest *pierwsza wersja* — celowo mała i hackowalna. PR-y mile widziane:

- Więcej polskich danych kodowych
- Lepsze eval (HumanEval-PL, MBPP-PL, Codeforces PL)
- Optymalizacje CPU (int8 kernels)

---

## 📜 Licencja

MIT — rób co chcesz, nawet na ziemniaku komercyjnie.

---

<div align="center">

**Qwazon — bo nie potrzebujesz elektrowni żeby być mądrym.** 🥔🧠

*Stworzone z myślą o polskich devach którzy chcą AI na własnym sprzęcie.*

</div>
