# Matnchi — o'zbekcha va ruscha audio/videoni matnga aylantiruvchi dastur
import os, subprocess, tempfile, gc
import torch
import gradio as gr
from transformers import pipeline

# Modellar (xohlasangiz boshqasiga almashtiring)
MODELS = {
    "O'zbekcha": "navai-uz/whisper-medium-uzbek",   # o'zbekchaga o'qitilgan Whisper
    "Ruscha": "openai/whisper-large-v3-turbo",      # ruschada juda yaxshi
}
LANG_CODE = {"O'zbekcha": "uz", "Ruscha": "ru"}

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
_pipe, _loaded = None, None


def get_pipe(lang):
    """Kerakli modelni yuklaydi (6 GB videokartaga sig'ishi uchun bittadan)."""
    global _pipe, _loaded
    if _loaded != lang:
        _pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        _pipe = pipeline(
            "automatic-speech-recognition",
            model=MODELS[lang],
            torch_dtype=DTYPE,
            device=DEVICE,
        )
        _loaded = lang
    return _pipe


def to_wav(path):
    """Video yoki audiodan 16 kHz mono WAV ajratib oladi (ffmpeg)."""
    out = os.path.join(tempfile.gettempdir(), "matnchi_audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", out],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def ts(sec):
    sec = sec or 0
    h, m, s = int(sec // 3600), int(sec % 3600 // 60), sec % 60
    return f"{h:02}:{m:02}:{int(s):02},{int((s % 1) * 1000):03}"


def transcribe(file, lang, progress=gr.Progress()):
    if not file:
        return "Avval fayl yuklang.", None, None
    progress(0.1, desc="Ovoz ajratilmoqda...")
    wav = to_wav(file)
    progress(0.3, desc="Model yuklanmoqda...")
    pipe = get_pipe(lang)
    progress(0.5, desc="Matnga aylantirilmoqda...")
    res = pipe(
        wav,
        chunk_length_s=30,
        batch_size=8,
        return_timestamps=True,
        generate_kwargs={"language": LANG_CODE[lang], "task": "transcribe"},
    )
    text = res["text"].strip()

    # .txt va .srt fayllar
    base = os.path.splitext(os.path.basename(file))[0]
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "natijalar")
    os.makedirs(outdir, exist_ok=True)
    txt_path = os.path.join(outdir, base + ".txt")
    srt_path = os.path.join(outdir, base + ".srt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, ch in enumerate(res.get("chunks", []), 1):
            a, b = ch["timestamp"]
            f.write(f"{i}\n{ts(a)} --> {ts(b if b else a)}\n{ch['text'].strip()}\n\n")
    return text, txt_path, srt_path


with gr.Blocks(title="Matnchi") as app:
    gr.Markdown("# 🎙️ Matnchi\nVideo yoki audio yuklang — o'zbekcha yoki ruscha matn oling.")
    with gr.Row():
        with gr.Column():
            f = gr.File(label="Video / audio fayl", type="filepath")
            lang = gr.Radio(list(MODELS), value="O'zbekcha", label="Til")
            btn = gr.Button("Boshlash", variant="primary")
            gr.Markdown(f"Qurilma: **{'GPU (' + torch.cuda.get_device_name(0) + ')' if DEVICE != 'cpu' else 'CPU (sekin)'}**")
        with gr.Column():
            out = gr.Textbox(label="Matn", lines=18, buttons=["copy"])
            txt = gr.File(label="Matn fayli (.txt)")
            srt = gr.File(label="Subtitr (.srt)")
    btn.click(transcribe, [f, lang], [out, txt, srt])

if __name__ == "__main__":
    app.launch(inbrowser=True)
