# Qwazon v0.4 — Raport Treningowy (uczciwy, z poprawioną metryką)

*2026-09-22 · ziemniak: 2 vCPU Intel Xeon, 3.8 GB RAM, **bez GPU***

## Środowisko (zweryfikowane w tej sesji)
- `torch 2.14.0+cu130` (CUDA niedostępna → trening na CPU)
- `transformers 5.17.0`, `tokenizers 0.23.2`, `numpy 2.4.6`, `pyyaml 6.0.3`
- `huggingface.co` **nieosiągalny** z sandboxa (`curl -sI` → brak odpowiedzi), PyPI działa
  → prawdziwe dane z HF niedostępne, trenujemy na syntetyku v4

## Co się zmieniło w v0.4
1. **Dane**: `data/synthetic_v4.py` — **50 zadań** bazowych (v0.2/v0.3 miały 25)
2. **Tokenizer**: wytrenowany własny BPE (`scripts/train_tokenizer.py`, vocab 4096 → realnie 1578 tokenów)
   - kompresja **4.40 bytes/tok** na próbce, **4.25 bytes/tok** na zdaniu testowym (byte-level = 1.0)
   - skutek: przy `--seq 128` cały przykład mieści się w sekwencji (wcześniej 128 **bajtów** ucinało większość zadań)
3. **`scripts/train.py --tokenizer`** — trening umie teraz użyć własnego BPE
4. **YaRN** w `RotaryEmbedding` (`rope_scaling`) — 4x dłuższy kontekst
5. **Kwantyzacja** — realne int8 + symulacja Q4 z MSE

## 🔴 Znaleziony i naprawiony błąd w `scripts/eval.py`
Perpleksja była liczona **nowym, domyślnym tokenizerem** zamiast tokenizerem z checkpointu:

```python
# PRZED (błąd):
tok = QwazonTokenizer(vocab_size=cfg.vocab_size)   # → byte-level fallback
# PO NAPRAWIE:
tok = pipe.tokenizer                                # ten sam co model
```

Efekt błędu: model wytrenowany na BPE dostawał id z byte-level i raportował
**PPL 22026** (loss 13.46 — gorzej niż losowy). Po naprawie: **PPL 227.68**,
czyli spójnie z treningiem (eval z trenera: 200.50).

Dodano też **bits/byte** — jedyną metrykę porównywalną między tokenizerami.

## Wyniki (zmierzone, nie szacowane)

| Run | Tokenizer | Kroki | train PPL | eval PPL | **bits/byte** | HumanEval-mini |
|-----|-----------|-------|-----------|----------|---------------|----------------|
| A — nano 39M | byte-level (8192) | 300 | 34.20 | 34.07 | **5.188** | 0/5 |
| B — nano 39M | **BPE (1578)** | 500 | 208.29 | 200.50 | **2.149** | 0/5 |
| C — nano 39M | BPE (1578) | 2000 (resume) | *w toku* | — | — | — |

**Kluczowy wniosek:** PPL **nie da się** porównywać między tokenizerami.
Run B ma 6x wyższą PPL niż A, a jest **2.4x lepszy** w bits/byte (2.149 vs 5.188),
bo jeden token BPE niesie ~4.25 bajta tekstu.

```
bits/byte = (suma natów) / ln(2) / liczba_bajtów
losowy model = 8.0 | byte-level run A = 5.188 | BPE run B = 2.149
```

## ⚠️ Korekta wcześniejszej deklaracji (v0.3 „HumanEval 20%”)
W v0.3 raportowałem `nano 300 kroków → HumanEval-mini 1/5 (20%)`. Dziś **nie da się tego
powtórzyć ani zweryfikować**:

1. Wagi `pytorch_model.bin` nie przetrwały (są w `.gitignore`, snapshot ich nie zachował) —
   w `checkpoints/qwazon-nano/*/` zostały tylko puste katalogi i `config.json`.
2. Tamten wynik pochodził z korpusu **25 zadań** powielonych do 2500 przykładów, gdzie
   `train PPL 1.058 ≈ eval PPL 1.044` — czyli model **zapamiętał** zbiór, a nie zgeneralizował.
   Różnica train/eval ~1% to sygnał memorizacji, nie umiejętności.

Uczciwy status: **Qwazon jeszcze nie zdaje testów kodowania.** Umiejętność generowania
poprawnego kodu wymaga albo więcej kroków (run C), albo prawdziwych danych (HF zablokowany).

## Testy (projektowy runner `tests/test_model.py`)
```
✅ forward OK, loss 7.0967
✅ generate OK torch.Size([1, 13])
✅ YaRN OK: factor 4.0 → inv_freq x4.00, mscale 1.1386, forward loss 6.9663
✅ Synthetic v4 OK: 50 zadań bazowych, 55 unikalnych w próbce 300
✅ Quantize OK: Q4 1.31MB vs FP32 10.50MB, MSE 0.000005, int8 forward działa
🎉 Wszystkie testy v0.4 przeszły   (5/5)
```

## Jak odtworzyć
```bash
# 1. BPE
python scripts/train_tokenizer.py --limit 10000 --vocab 4096 --out tokenizer-qwazon
# 2. Trening (BPE + 50 zadań)
python scripts/train.py --variant qwazon-nano --steps 500 --batch 2 --seq 128 --lr 4e-4 \
  --demo --tokenizer tokenizer-qwazon --output checkpoints/qwazon-nano-bpe
# 3. Ewaluacja (używa tokenizera z checkpointu + bits/byte)
python scripts/eval.py --checkpoint checkpoints/qwazon-nano-bpe --max_new 96
# 4. Testy
python tests/test_model.py
```

Czasy na ziemniaku: BPE 500 kroków ≈ **12 min**, byte-level 300 kroków ≈ **13 min**.
