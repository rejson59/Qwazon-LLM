"""
Qwazon API — FastAPI server dla ziemniaka.
Działa na CPU, 0.0.0.0, gotowy na Arena Preview, Raspberry Pi, Docker.

Uruchom:
  pip install fastapi uvicorn
  python -m qwazon.api --checkpoint checkpoints/qwazon-nano --port 8000
  # lub: uvicorn qwazon.api:app --host 0.0.0.0 --port 8000

Endpoints:
  GET  /health
  POST /generate
  POST /chat
  GET  /models
"""
import time
from typing import List, Optional
from pathlib import Path

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("[Qwazon API] FastAPI nie zainstalowane: pip install fastapi uvicorn")

from .inference import QwazonPipeline
from .config import get_config, QWAZON_VARIANTS

# Models
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True

class ChatMessage(BaseModel):
    role: str  # system, user, assistant
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

# Global pipeline (lazy)
_pipeline: Optional[QwazonPipeline] = None
_checkpoint: str = "checkpoints/qwazon-nano"

def get_pipeline():
    global _pipeline, _checkpoint
    if _pipeline is None:
        # Use tiny as fallback if nano not trained
        import os
        ckpt = _checkpoint
        if not Path(ckpt).exists():
            ckpt = "checkpoints/qwazon-micro"
        if not Path(ckpt).exists():
            ckpt = "checkpoints/qwazon-nano"
        _pipeline = QwazonPipeline(ckpt)
    return _pipeline

if HAS_FASTAPI:
    app = FastAPI(
        title="Qwazon API 🥔",
        description="Złoty środek LLM — działa na ziemniaku, kodzi jak duży",
        version="0.3.0",
    )

    # CORS for Arena Preview
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        pipe = get_pipeline()
        return {
            "status": "ok",
            "model": pipe.config.model_name,
            "params": pipe.config.num_parameters_approx,
            "device": str(pipe.device),
            "vocab": len(pipe.tokenizer),
        }

    @app.get("/models")
    def models():
        return {
            "variants": [
                {"name": k, "desc": v.describe(), "params": v.num_parameters_approx}
                for k, v in QWAZON_VARIANTS.items()
            ]
        }

    @app.post("/generate")
    def generate(req: GenerateRequest):
        pipe = get_pipeline()
        start = time.time()
        text, stats = pipe.generate(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            do_sample=req.do_sample,
        )
        return {
            "prompt": req.prompt,
            "generated": text,
            "stats": stats,
            "time": time.time() - start,
        }

    @app.post("/chat")
    def chat(req: ChatRequest):
        pipe = get_pipeline()
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        text, stats = pipe.chat(msgs, max_new_tokens=req.max_new_tokens, temperature=req.temperature, top_p=req.top_p)
        # Wyciągnij ostatnią odpowiedź
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>")[-1].strip()
        if "<|user|>" in text:
            text = text.split("<|user|>")[0].strip()
        return {
            "response": text,
            "stats": stats,
        }

    @app.get("/")
    def root():
        return {
            "message": "Qwazon API 🥔 — złoty środek LLM",
            "docs": "/docs",
            "health": "/health",
            "generate": "POST /generate {prompt}",
            "chat": "POST /chat {messages}",
        }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/qwazon-nano")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    global _checkpoint
    _checkpoint = args.checkpoint

    if not HAS_FASTAPI:
        print("Instaluj: pip install fastapi uvicorn")
        return

    import uvicorn
    print(f"🚀 Qwazon API start na http://{args.host}:{args.port} — model {args.checkpoint}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    print(f"   Arena Preview: https://{args.port}-{{sandboxId}}.e2b.app")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
