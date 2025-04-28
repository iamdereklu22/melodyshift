# generate.py
import os
import torch
import soundfile as sf
import numpy as np
import librosa
from datasets import audio_to_mel, mel_to_audio, SR
from models import TransformerNet

CHECKPOINT = "checkpoints/stylizer_best.pth"
CONTENT_DIR = "data/eval/"
OUT_DIR     = "outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stylizer = TransformerNet().to(device)
stylizer.load_state_dict(torch.load(CHECKPOINT, map_location=device))
stylizer.eval()

for fname in os.listdir(CONTENT_DIR):
    if not fname.endswith(".wav"):
        continue
    path = os.path.join(CONTENT_DIR, fname)

    # get normalized spectrogram + original dB range
    spec, orig_min, orig_max = audio_to_mel(path)   # spec: (80, T)

    # prepare input tensor
    tensor = torch.from_numpy(spec)[None]           # (1, 80, T)
    tensor = tensor.repeat(3, 1, 1).to(device)      # (3, 80, T)

    # stylize
    with torch.no_grad():
        out_rgb = stylizer(tensor.unsqueeze(0)).cpu().squeeze(0).numpy()  # (3, H, T)
    out = out_rgb.mean(axis=0)  # collapse channels → (H, T)

    # reconstruct audio (higher Griffin-Lim iterations)
    y = mel_to_audio(out, orig_min, orig_max)

    # ensure mono shape
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    # normalize and cast to 16-bit PCM
    y = y / np.max(np.abs(y))
    y_int16 = (y * 32767).astype(np.int16)

    # save clean PCM-16 WAV
    sf.write(
        os.path.join(OUT_DIR, fname),
        y_int16,
        SR,
        subtype="PCM_16"
    )
    print(f"Wrote stylized → {os.path.join(OUT_DIR, fname)}")