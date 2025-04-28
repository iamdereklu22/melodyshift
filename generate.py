# =============================
# generate.py (residual inference with padding)
# =============================
import os
import torch
import soundfile as sf
import numpy as np
import librosa
from datasets import audio_to_mel, SR
from models import TransformerNet

# load residual model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stylizer = TransformerNet().to(device)
stylizer.load_state_dict(torch.load('checkpoints/stylizer_best.pth', map_location=device))
stylizer.eval()

# ensure output directory
OUT_DIR = 'outputs_residual'
os.makedirs(OUT_DIR, exist_ok=True)

for fname in os.listdir('data/eval'):
    if not fname.endswith('.wav'): continue
    path = os.path.join('data/eval', fname)

    # compute mel spectrogram
    spec, orig_min, orig_max = audio_to_mel(path)  # (n_mels, T)
    T = spec.shape[1]
    # pad time dimension to multiple of 4 for conv/down-up consistency
    pad_T = ((T + 3) // 4) * 4
    if pad_T != T:
        spec = np.pad(spec, ((0,0),(0,pad_T - T)), mode='constant')

    # prepare input tensor (1,3,H,pad_T)
    content = torch.from_numpy(spec).unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(device)

    # stylize: predict residual + add skip connection
    with torch.no_grad():
        residual = stylizer(content)
    styled = content + residual  # (1,3,H,pad_T)

    # collapse channels and trim to original length
    out_rgb = styled.cpu().squeeze(0).numpy()    # (3,H,pad_T)
    out     = out_rgb.mean(axis=0)               # (H,pad_T)
    out     = out[:, :T]                         # (H,T)

    # invert spectrogram → waveform
    mel_db = out * (orig_max - orig_min) + orig_min
    S      = librosa.db_to_power(mel_db)
    y      = librosa.feature.inverse.mel_to_audio(
        S, sr=SR, n_fft=1024, hop_length=160, win_length=400, n_iter=800
    )

    # normalize & save
    y = y / np.max(np.abs(y))
    sf.write(os.path.join(OUT_DIR, fname), (y * 32767).astype(np.int16), SR)
