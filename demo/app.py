"""
Qwazon Demo — Gradio Chat na ziemniaku.
Uruchom: python demo/app.py --checkpoint checkpoints/qwazon-tiny --port 7860

Zaprojektowane tak by działało:
- lokalnie (CPU)
- na Hugging Face Spaces (CPU basic)
- jako Live Preview w Arena (0.0.0.0 + allow host)
"""
import argparse, os, sys
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

    def respond(message, history, temp, top_p, max_tokens):
        # history: list[[user, assistant]]
        # zbuduj messages
        msgs = []
        for u, a in history:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": message})
        text, stats = pipe.chat(msgs, max_new_tokens=max_tokens, temperature=temp, top_p=top_p)
        # wyciągnij odpowiedź
        if "<|assistant|>" in text:
            text = text.split("<|assistant|>")[-1].strip()
        # usuń ewentualne <|user|> trailing
        if "<|user|>" in text:
            text = text.split("<|user|>")[0].strip()
        footer = f"\n\n— — —\n*{stats['tokens']} toków, {stats['tok_per_sec']:.1f} tok/s na {pipe.device}*"
        return text + footer

    with gr.Blocks(title="Qwazon — LLM na ziemniaka 🥔", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🥔 Qwazon v0.1 — Złoty środek
        **Mały że działa na ziemniaku, mądry że kodzi jak duży.**<br>
        *1.2B aktywne / 2.1B total (MoE) • 32k kontekst • GQA + SwiGLU + Sliding Window • Q4 = 1.1GB*
        """)
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(height=480, label="Qwazon Chat", placeholder="Zadaj pytanie o kod, algorytm, Pythona...")
                msg = gr.Textbox(label="Twoje pytanie", placeholder="Np. Napisz quicksort w Pythonie i wyjaśnij złożoność", lines=2)
                with gr.Row():
                    clear = gr.ClearButton([msg, chatbot])
                    submit = gr.Button("Wyślij 🚀", variant="primary")
            with gr.Column(scale=0.45):
                gr.Markdown("### ⚙️ Parametry (ziemniak-friendly)")
                temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p")
                max_tokens = gr.Slider(32, 1024, value=256, step=32, label="Max tokenów")
                gr.Markdown("""
                **Warianty:**
                - `qwazon-tiny` 110M — Raspberry Pi, telefon
                - `qwazon-small` 0.5B — laptop 6GB
                - `qwazon-1.2b` 1.2B — **polecany, złoty środek**
                - `qwazon-base` 1.7B — max, wciąż <2GB Q4
                ---
                **Przykłady:**
                - Napisz REST API w FastAPI
                - Wyjaśnij różnicę między `list` a `tuple`
                - Zaimplementuj BFS w Pythonie
                - SELECT z JOIN — wyjaśnij
                """)
                gr.Markdown(f"**Checkpoint:** `{checkpoint}`<br>**Model:** `{pipe.config.describe()}`")

        msg.submit(respond, [msg, chatbot, temp, top_p, max_tokens], [chatbot])
        submit.click(respond, [msg, chatbot, temp, top_p, max_tokens], [chatbot])

        gr.Markdown("""
        <center><small>Qwazon v0.1 • Trening: 100B tokenów (35% kod) + Distillation z Qwen-72B • Kwantyzacja Q4_K_M gotowa • <a href="https://github.com/rejson59/Qwazon-LLM">GitHub</a></small></center>
        """)
    return demo

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/qwazon-tiny")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    demo = create_ui(args.checkpoint, device=args.device)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, allowed_paths=["/"])
