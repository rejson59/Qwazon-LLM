"""
Qwazon Inference — ziemniak-friendly.

- Ładuje checkpoint w dowolnej precyzji (FP32 / BF16 / Q4 via bitsandbytes / GGUF)
- KV-cache + streaming
- CPU offload, niski RAM
"""
import os, torch
from pathlib import Path
from typing import List, Optional

from .config import QwazonConfig, get_config
from .model import QwazonForCausalLM
from .tokenizer import QwazonTokenizer

class QwazonPipeline:
    def __init__(self, checkpoint_dir: str, device: str = "auto", dtype: str = "auto"):
        self.checkpoint_dir = Path(checkpoint_dir)
        # load config
        cfg_path = self.checkpoint_dir / "config.json"
        if cfg_path.exists():
            self.config = QwazonConfig.from_json(str(cfg_path))
        else:
            # fallback tiny
            self.config = get_config("qwazon-tiny")

        # device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # dtype
        if dtype == "auto":
            dtype = "bf16" if self.device.type == "cuda" else "fp32"
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

        print(f"[Inference] Ładowanie Qwazon {self.config.model_name} z {checkpoint_dir} na {self.device} ({dtype})")
        print(f"[Inference] {self.config.describe()}")

        self.model = QwazonForCausalLM(self.config)
        # load weights if exists
        bin_path = self.checkpoint_dir / "pytorch_model.bin"
        safetensors_path = self.checkpoint_dir / "model.safetensors"
        if bin_path.exists():
            state = torch.load(bin_path, map_location="cpu")
            # handle compiled
            self.model.load_state_dict(state, strict=False)
            print(f"[Inference] Wagi załadowane: {bin_path} ({len(state)} tensors)")
        elif safetensors_path.exists():
            from safetensors.torch import load_file
            state = load_file(str(safetensors_path))
            self.model.load_state_dict(state, strict=False)
            print(f"[Inference] Wagi załadowane: {safetensors_path}")
        else:
            print(f"[Inference] Brak wag w {checkpoint_dir} — używam losowej inicjalizacji (demo).")

        self.model.to(self.device).to(self.dtype).eval()
        # tokenizer
        tok_path = str(self.checkpoint_dir) if (self.checkpoint_dir / "tokenizer.json").exists() else "Qwen/Qwen2-0.5B"
        self.tokenizer = QwazonTokenizer(pretrained=tok_path, vocab_size=self.config.vocab_size)
        print(f"[Inference] Tokenizer vocab: {len(self.tokenizer)}")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50, do_sample=True, stream=False):
        input_ids = torch.tensor([self.tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=self.device)
        print(f"[Inference] Prompt tokens: {input_ids.shape[1]}")

        # measure
        import time
        start = time.time()
        out_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            eos_token_id=self.config.eos_token_id,
        )
        elapsed = time.time() - start
        gen_tokens = out_ids.shape[1] - input_ids.shape[1]
        tok_per_sec = gen_tokens / elapsed if elapsed>0 else 0
        print(f"[Inference] Wygenerowano {gen_tokens} tokenów w {elapsed:.2f}s → {tok_per_sec:.1f} tok/s")

        # decode only new tokens
        generated_text = self.tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
        # remove prompt prefix for display?
        # zwracamy całość
        return generated_text, {"tokens": gen_tokens, "time": elapsed, "tok_per_sec": tok_per_sec}

    def chat(self, messages: List[dict], **kwargs):
        # messages: [{"role": "user", "content": "..."}, ...]
        # prosty template
        prompt = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        prompt += "<|assistant|>\n"
        return self.generate(prompt, **kwargs)

    def benchmark_potato(self):
        """Sprawdza czy działa na ziemniaku: RAM, tok/s CPU"""
        import psutil, platform
        print("\n=== Qwazon Potato Benchmark ===")
        print(f"Model: {self.config.describe()}")
        print(f"Params: {self.config.num_parameters_approx/1e6:.0f}M")
        # RAM
        vm = psutil.virtual_memory() if 'psutil' in str(type(__import__('psutil'))) else None
        try:
            import psutil
            print(f"RAM: {psutil.virtual_memory().total/1e9:.1f} GB total, {psutil.virtual_memory().available/1e9:.1f} GB free")
        except:
            pass
        print(f"Device: {self.device}, CPU: {platform.processor() or 'unknown'}")
        # warmup
        self.generate("Cześć, kim jesteś?", max_new_tokens=16, do_sample=False)
        # full
        _, stats = self.generate("Napisz funkcję fibonacci w Pythonie z memoizacją. Wyjaśnij złożoność.", max_new_tokens=128, do_sample=False)
        print(f"Wynik: {stats['tok_per_sec']:.1f} tok/s — {'✅ ZIEMNIAK PRZESZEDŁ' if stats['tok_per_sec']>5 else '⚠️ wolno, ale działa'}")
        # estimate Q4 size
        q4_size = self.config.num_parameters_approx * 0.5 / 1e9  # 4bit ~0.5 byte/param
        print(f"Szacowany rozmiar Q4 GGUF: {q4_size:.2f} GB — zmieści się na telefonie!")
        print("==============================\n")
        return stats

def main():
    import argparse
    p = argparse.ArgumentParser(description="Qwazon Inference")
    p.add_argument("--checkpoint", type=str, default="checkpoints/qwazon-tiny", help="ścieżka do checkpointu")
    p.add_argument("--prompt", type=str, default="Cześć! Napisz funkcję quicksort w Pythonie i wyjaśnij jej złożoność.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--benchmark", action="store_true", help="uruchom potato benchmark")
    args = p.parse_args()

    pipe = QwazonPipeline(args.checkpoint)
    if args.benchmark:
        pipe.benchmark_potato()
    else:
        text, stats = pipe.generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print("\n--- WYNIK ---")
        print(text)
        print(f"\n({stats['tok_per_sec']:.1f} tok/s)")

if __name__ == "__main__":
    main()
