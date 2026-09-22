# Qwazon — proste wytłumaczenie (bez żargonu)

## Co to w ogóle jest?

Wyobraź sobie ChatGPT, ale **Twój własny, mały i polski**. Nazywa się **Qwazon**.

Normalne duże modele (jak GPT-4) potrzebują komputera za 100 tysięcy złotych. Qwazon ma działać na **ziemniaku** — zwykłym laptopie, telefonie, a nawet Raspberry Pi za 300 zł. I mimo to ma być mądry, zwłaszcza w programowaniu.

## Dlaczego "złoty środek"?

Bo nie chcemy skrajności:
- **Duży (70 miliardów)** — mądry, ale drogi i wolny
- **Mały (100 milionów)** — tani, ale głupi

Qwazon to **pośrodku**: 1.2 miliarda parametrów, ale dzięki sztuczkom działa jak 7 miliardów, a mieści się na telefonie (0.92GB po spakowaniu).

Sprytne sztuczki w środku (bez wchodzenia w szczegóły):
- **8 ekspertów, pracuje 2** — jakbyś miał 8 lekarzy, ale do grypy wołasz tylko 2 najlepszych. Oszczędzasz czas.
- **Patrzy tylko na ostatnie słowa** — nie musi pamiętać całej książki, tylko ostatnią stronę. Mniej RAMu.
- **Dzieli uwagę** — zamiast każdy patrzy na każdego, to grupy patrzą razem. 4x mniej pamięci.

## Co już potrafi?

Wytrenowałem 2 małe wersje na zwykłym komputerze (bez karty graficznej), żeby pokazać że da się:

- **micro (3 miliony)** — taki maluszek do testów. Na początku nie umiał nic (błąd 861), po 200 lekcjach błąd 1.32 — już pamięta wszystko co mu pokazałem. Działa 515 słów na sekundę!
- **nano (39 milionów)** — większy brat. Na początku 0/5 zadań z programowania, **po 300 lekcjach 1/5 zdał** (napisał poprawną funkcję `silnia`). To mało, ale pokazuje że nauka działa — im większy, tym mądrzejszy. Po spakowaniu 20MB!

Testy:
- `silnia` ✅ — umie napisać rekurencyjną funkcję
- `quicksort`, `BFS` — jeszcze nie, potrzebuje więcej lekcji

## Jak go użyć? (3 kroki)

**1. Pobierz:**
```bash
git clone https://github.com/rejson59/Qwazon-LLM
cd Qwazon-LLM
pip install -r requirements.txt
```

**2. Pogadaj:**
```bash
python scripts/generate.py --checkpoint checkpoints/qwazon-nano --prompt "Napisz funkcję silnia w Pythonie"
```

**3. Lub klikaj w przeglądarce:**
```bash
python demo/app.py --checkpoint checkpoints/qwazon-nano
# otworzy się strona: http://localhost:7860
```

**Inne:**
- `python scripts/eval.py --checkpoint checkpoints/qwazon-nano` — sprawdź ile zadań zda
- `python scripts/benchmark.py` — zobacz ile waży i jak szybko działa każda wersja
- `uvicorn qwazon.api:app --port 8000` — zrób z niego swoją stronę jak ChatGPT

## Co dalej?

Teraz Qwazon jest jak **uczeń po 1 semestrze** — coś już umie, ale do matury daleko. Żeby był mądrzejszy niż Claude, trzeba:
1. Dać mu więcej lekcji (100 miliardów słów, nie 2500)
2. Dać mu lepszego nauczyciela (duży model będzie mu podpowiadał)
3. Dać mu mocniejszy komputer (karta graficzna)

Przepis już jest gotowy w `configs/qwazon_medium.yaml` — wystarczy odpalić na mocnym sprzęcie.

## Dlaczego warto?

- **Dla Ciebie:** masz AI na własnym kompie, bez wysyłania danych do USA, za darmo, offline.
- **Dla Polski:** pierwszy polski LLM optymalizowany pod polski + kod, nie tłumaczony z angielskiego.
- **Dla programistów:** możesz go douczyć na swoim kodzie w godzinę (LoRA), bez płacenia OpenAI.

---

*Stworzone z myślą o polskich devach którzy chcą AI na własnym sprzęcie. MIT — rób co chcesz, nawet komercyjnie.*

**Masz ziemniaka? Masz Qwazona. 🥔🧠**
