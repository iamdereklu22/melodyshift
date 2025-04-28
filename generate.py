# generate.py
import os
import torch
import soundfile as sf
from datasets import audio_to_mel, mel_to_audio
from models import TransformerNet

SR = 16000
CHECKPOINT = "checkpoints/stylizer_epoch30.pth"
CONTENT_DIR = "data/contents/"
OUT_DIR     = "outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stylizer = TransformerNet().to(device)
stylizer.load_state_dict(torch.load(CHECKPOINT, map_location=device))
stylizer.eval()

for fname in os.listdir(CONTENT_DIR):
    if not fname.endswith(".wav"): continue
    path = os.path.join(CONTENT_DIR, fname)
    spec = audio_to_mel(path)             # (80, T)
    tensor = torch.from_numpy(spec)[None] # (1, 80, T)
    tensor = tensor.repeat(3, 1, 1).to(device)  # (3, 80, T)
    with torch.no_grad():
        out = stylizer(tensor.unsqueeze(0)).cpu().squeeze(0)[0].numpy()  # (80, T)
    y = mel_to_audio(out)
    sf.write(os.path.join(OUT_DIR, fname), y, SR)
    print(f"Wrote stylized → {os.path.join(OUT_DIR, fname)}")