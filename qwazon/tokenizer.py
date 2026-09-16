"""
Qwazon Tokenizer — lekki wrapper.

- Domyślnie próbuje załadować HF tokenizer (Qwen2 / Llama) dla kompatybilności.
- Fallback: char-level BPE proxy dla offline demo (działa bez internetu).
- W produkcji: trenuj własny BPE 48k na PL + Code (The Stack PL, Wolne Lektury, FineWeb-PL).

Złoty środek: vocab 48k = mniej tokenów na polski i kod, szybszy ziemniak.
"""
import os, json
from typing import List, Optional
from pathlib import Path

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class QwazonTokenizer:
    def __init__(self, pretrained: str = "Qwen/Qwen2-0.5B", vocab_size: int = 49152, use_fast: bool = True):
        self.vocab_size = vocab_size
        self.pretrained = pretrained
        self.hf_tokenizer = None
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.unk_token_id = 3

        if HAS_TRANSFORMERS:
            # Najpierw próba lokalnie bez sieci — szybki fallback bez retry
            try:
                self.hf_tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True, use_fast=use_fast, local_files_only=True)
                print(f"[QwazonTokenizer] Załadowano HF tokenizer lokalnie: {pretrained}")
            except Exception as e_local:
                # W env offline (Arena) nie próbuj sieci — od razu fallback, bo HF retry = 20s+ timeout
                # Jeśli chcesz pobrać z HF, ustaw env QWAZON_ALLOW_HF_DOWNLOAD=1
                if os.environ.get("QWAZON_ALLOW_HF_DOWNLOAD") == "1":
                    try:
                        self.hf_tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True, use_fast=use_fast, local_files_only=False)
                        print(f"[QwazonTokenizer] Załadowano HF tokenizer z HF Hub: {pretrained}")
                    except Exception as e:
                        print(f"[QwazonTokenizer] Brak HF tokenizer ({e}), fallback na prosty tokenizer.")
                        self.hf_tokenizer = None
                else:
                    # szybki fallback offline
                    # print(f"[QwazonTokenizer] Lokalny tokenizer nie znaleziony ({e_local}), używam fallback char-level (offline). Ustaw QWAZON_ALLOW_HF_DOWNLOAD=1 aby pobrać z HF.")
                    self.hf_tokenizer = None
        else:
            print("[QwazonTokenizer] transformers nie zainstalowane, fallback char-level.")

        if self.hf_tokenizer is not None:
            # ujednolicamy special tokens
            if self.hf_tokenizer.pad_token_id is not None:
                self.pad_token_id = self.hf_tokenizer.pad_token_id
            if self.hf_tokenizer.eos_token_id is not None:
                self.eos_token_id = self.hf_tokenizer.eos_token_id
            if self.hf_tokenizer.bos_token_id is not None:
                self.bos_token_id = self.hf_tokenizer.bos_token_id
            self.vocab_size = len(self.hf_tokenizer)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        if self.hf_tokenizer is not None:
            ids = self.hf_tokenizer.encode(text, add_special_tokens=False)
            if add_bos and self.bos_token_id is not None:
                ids = [self.bos_token_id] + ids
            if add_eos:
                ids = ids + [self.eos_token_id]
            return ids
        else:
            # byte-level fallback (jak GPT-2 / Llama): UTF-8 bytes + offset
            # preserves Polish diacritics (ś, ć, ą, etc.) poprawnie
            byte_ids = list(text.encode('utf-8'))
            # offset by 4 to reserve special tokens, modulo to fit vocab
            ids = [ (b % (self.vocab_size - 4)) + 4 for b in byte_ids ]
            if add_bos:
                ids = [self.bos_token_id] + ids
            if add_eos:
                ids = ids + [self.eos_token_id]
            return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if self.hf_tokenizer is not None:
            return self.hf_tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        else:
            # fallback: odwróć byte-level -> bytes -> utf-8
            byte_vals = []
            for i in ids:
                if skip_special_tokens and i in (self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id):
                    continue
                b = (i - 4) % 256
                byte_vals.append(b)
            # decode bytes, ignoring errors for partial sequences (random weights may produce invalid utf-8)
            try:
                return bytes(byte_vals).decode('utf-8', errors='replace')
            except:
                return "".join(chr(b) for b in byte_vals)

    def __len__(self):
        if self.hf_tokenizer is not None:
            return len(self.hf_tokenizer)
        return self.vocab_size

    def save_pretrained(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        if self.hf_tokenizer is not None:
            self.hf_tokenizer.save_pretrained(path)
        else:
            with open(os.path.join(path, "tokenizer.json"), "w") as f:
                json.dump({"vocab_size": self.vocab_size, "type": "fallback-char"}, f)

    @classmethod
    def from_pretrained(cls, path: str):
        tok = cls(pretrained=path)
        return tok

# Test
if __name__ == "__main__":
    tok = QwazonTokenizer()
    text = "Cześć! Napisz funkcję quicksort w Pythonie."
    ids = tok.encode(text)
    print(ids[:20], len(ids))
    print(tok.decode(ids))
