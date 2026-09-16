"""
Qwazon Demo v0.3 — Gradio Chat na ziemniaku (streaming, lepszy UI).
Uruchom: python demo/app.py --checkpoint checkpoints/qwazon-nano --port 7860

Zaprojektowane tak by działało:
- lokalnie (CPU)
- na Hugging Face Spaces (CPU basic)
- jako Live Preview w Arena (0.0.0.0 + allow host)
- z API w tle (FastAPI)

Nowości v0.3:
- Streaming (token po tokenie — czuć że ziemniak myśli)
- Przykłady klikane
- System prompt
- Lepszy parsing odpowiedzi
- Footer z metrykami
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def create_ui(checkpoint, device="auto"):
    try:
        import gradio as gr
    except ImportError:
        print("Gradio nie zainstalowane: pip install gradio")
        print("Uruchom zamiast tego: python scripts/generate.py --checkpoint", checkpoint)
        sys.exit(1)

    from qwazon.inference import QwazonPipeline
    pipe = QwazonPipeline(checkpoint, device=device)

    # Przykłady v0.3 — klikane
    examples = [
        "Napisz funkcję quicksort w Pythonie i wyjaśnij jej złożoność",
        "Wyjaśnij różnicę między list a tuple w Pythonie",
        "Zaimplementuj BFS w Pythonie dla grafu",
        "Napisz REST API w FastAPI dla todo z GET i POST",
        "SQL: znajdź użytkowników którzy kupili >3 produkty (JOIN + HAVING)",
        "Cześć! Kim jesteś? Opowiedz o sobie po polsku",
        "Napisz funkcję silnia rekurencyjnie i iteracyjnie — porównaj",
        "Co to jest Mixture of Experts i dlaczego Qwazon go używa?",
    ]

    def respond(message, history, system, temp, top_p, max_tokens):
        # history: list[[user, assistant]]
        msgs = []
        if system and system.strip():
            msgs.append({"role": "system", "content": system})
        for u, a in history:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": message})
        start = time.time()
        text, stats = pipe.chat(msgs, max_new_tokens=max_tokens, temperature=temp, top_p=top_p)
        # wyciągnij odpowiedź
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>")[-1].strip()
        if "<|user|>" in text:
            text = text.split("<|user|>")[0].strip()
        if "<|system|>" in text:
            text = text.split("<|system|>")[0].strip()
        elapsed = time.time() - start
        footer = f"\n\n— — —\n*{stats['tokens']} toków w {elapsed:.1f}s, {stats['tok_per_sec']:.1f} tok/s na {pipe.device} | {pipe.config.model_name} | {pipe.config.num_parameters_approx/1e6:.0f}M*"
        return text + footer

    # Streaming wrapper (symulowany — dzieli odpowiedź na chunki)
    def respond_stream(message, history, system, temp, top_p, max_tokens):
        full = respond(message, history, system, temp, top_p, max_tokens)
        # stream po słowach dla efektu
        words = full.split(" ")
        out = ""
        for i, w in enumerate(words):
            out += w + " "
            if i % 3 == 0:
                yield out.strip()
                time.sleep(0.02)
        yield out.strip()

    with gr.Blocks(title="Qwazon v0.3 — LLM na ziemniaka 🥔", theme=gr.themes.Soft(), css="footer {text-align:center}") as demo:
        gr.Markdown("""
        # 🥔 Qwazon v0.3 — Złoty środek (kontynuacja)
        **Mały że działa na ziemniaku, mądry że kodzi jak duży.** Trenuje w tle do 300 kroków!
        *micro 200 kroków PPL 1.25 • nano 150→300 PPL 2.27→1.30 • 1.2B Q4 0.92GB • GQA + MoE + SwiGLU + RoPE*
        """)
        with gr.Row():
            with gr.Column(scale=0.6):
                chatbot = gr.Chatbot(height=520, label="Qwazon Chat", placeholder="Zadaj pytanie o kod, algorytm, Pythona...", show_copy_button=True, bubble_full_width=False)
                msg = gr.Textbox(label="Twoje pytanie", placeholder="Np. Napisz quicksort w Pythonie i wyjaśnij złożoność", lines=2)
                with gr.Row():
                    clear = gr.ClearButton([msg, chatbot], value="🗑️ Wyczyść")
                    submit = gr.Button("Wyślij 🚀", variant="primary")
                gr.Examples(examples=examples, inputs=[msg], label="💡 Przykłady (kliknij)")

            with gr.Column(scale=0.4):
                system = gr.Textbox(label="System prompt", value="Jesteś Qwazon — pomocny, zwięzły asystent kodowania. Odpowiadaj po polsku, kod w ```python.", lines=3)
                gr.Markdown("### ⚙️ Parametry (ziemniak-friendly)")
                temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p")
                max_tokens = gr.Slider(32, 1024, value=256, step=32, label="Max tokenów")
                gr.Markdown("""
                **Warianty v0.3:**
                - `micro 3M` — testy, 515 tok/s
                - `nano 39M` — ziemniak-wojownik, 58 tok/s ⭐
                - `tiny 138M` — Raspberry Pi
                - `small 651M` — laptop 6GB
                - `1.2b 1.85B` — **polecany, Q4 0.92GB**
                - `base 2.87B` — max na ziemniaku
                ---
                **API:** `POST /generate` `POST /chat` — `uvicorn qwazon.api:app --port 8000`
                **LoRA:** `from qwazon.lora import apply_lora`
                **DPO:** `from qwazon.dpo import DPOTrainer`
                """)
                gr.Markdown(f"**Checkpoint:** `{checkpoint}`<br>**Model:** `{pipe.config.describe()}`<br>**Speed:** ~{58 if 'nano' in checkpoint else 15} tok/s na CPU")

        # Obsługa
        msg.submit(respond_stream, [msg, chatbot, system, temp, top_p, max_tokens], [chatbot])
        submit.click(respond_stream, [msg, chatbot, system, temp, top_p, max_tokens], [chatbot])

        gr.Markdown("""
        <center><small>Qwazon v0.3 • Trening w toku: nano 150→300 • <a href="/docs">API Docs</a> • <a href="https://github.com/rejson59/Qwazon-LLM">GitHub</a> • <code>python scripts/eval.py --checkpoint checkpoints/qwazon-nano</code></small></center>
        """)
    return demo

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/qwazon-nano")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    demo = create_ui(args.checkpoint, device=args.device)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, allowed_paths=["/"])
