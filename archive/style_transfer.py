#!/usr/bin/env python3
"""
style_transfer.py

Neural style transfer on audio via spectrograms + Griffin–Lim inversion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import librosa
import numpy as np
import soundfile as sf

# Paths & hyperparameters
CONTENT_PATH      = "data/content.wav"
STYLE_PATH        = "data/style.wav"
OUTPUT_PATH       = "stylized.wav"
SR                = 16000
N_MELS            = 80
N_FFT             = 1024
HOP_LENGTH        = 160
WIN_LENGTH        = 400

ALPHA             = 1e3
BETA              = 1e9
LBFGS_ITERS       = 600
GRIFFIN_LIM_ITERS = 60

def audio_to_mel(path):
    """Load audio as mono 16 kHz, compute log-mel spectrogram in dB."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
    )
    log_S = librosa.power_to_db(S)
    return log_S.astype(np.float32)  # shape (n_mels, T)

def gram_matrix(feat):
    """Compute Gram matrix from a feature map (B, C, H, W)."""
    B, C, H, W = feat.size()
    f = feat.view(B, C, H * W)
    G = torch.bmm(f, f.transpose(1,2))
    return G / (C * H * W)

class VGGFeatures(nn.Module):
    """Extract specified layers from VGG19."""
    def __init__(self, content_layers, style_layers):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.content_layers = content_layers
        self.style_layers   = style_layers
        self.target_layers  = {**content_layers, **style_layers}

    def forward(self, x):
        feats = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.target_layers:
                feats[self.target_layers[i]] = x
        return feats

def prep_image(spec, device):
    """Turn (H, W) spectrogram into a 3×H×W tensor for VGG input."""
    t = torch.from_numpy(spec)[None]      # (1, H, W)
    t = t.repeat(1, 3, 1, 1)              # (1, 3, H, W)
    return t.to(device)

def mel_to_audio(mel_spec):
    """Invert log-mel dB back to waveform via Griffin–Lim."""
    S = librosa.db_to_power(mel_spec)
    stft = librosa.feature.inverse.mel_to_stft(
        S, sr=SR, n_fft=N_FFT, power=1.0
    )
    y = librosa.griffinlim(
        stft,
        n_iter=GRIFFIN_LIM_ITERS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )
    return y

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load & preprocess
    content_spec = audio_to_mel(CONTENT_PATH)
    style_spec   = audio_to_mel(STYLE_PATH)
    content_img  = prep_image(content_spec, device)
    style_img    = prep_image(style_spec, device)

    # 2) Setup feature extractor
    content_layers = {10: "conv4_2"}
    style_layers = {
        1:  "conv1_1",
        6:  "conv2_1",
        11: "conv3_1",
        20: "conv4_1",
        21: "conv4_2",   # extra
        22: "conv5_1",
        29: "conv5_2",   # extra
    }
    extractor = VGGFeatures(content_layers, style_layers).to(device)

    # 3) Extract targets
    with torch.no_grad():
        content_feats = extractor(content_img)
        style_feats   = extractor(style_img)
        style_targets = {
            name: gram_matrix(style_feats[name])
            for name in style_feats
        }

    # 4) Initialize generated spectrogram (starts as content)
    generated = content_img.clone().requires_grad_(True)

    # 5) Define optimizer & loss
    optimizer = optim.LBFGS([generated], max_iter=LBFGS_ITERS)
    mse = nn.MSELoss()

    def closure():
        optimizer.zero_grad()
        gen_feats = extractor(generated)
        # content loss
        c_loss = mse(gen_feats["conv4_2"], content_feats["conv4_2"])
        # style loss
        s_loss = 0
        for name, target in style_targets.items():
            Gg = gram_matrix(gen_feats[name])
            s_loss += mse(Gg, target)
        (ALPHA * c_loss + BETA * s_loss).backward()
        return ALPHA * c_loss + BETA * s_loss

    print("Optimizing…")
    optimizer.step(closure)
    print("Done.")

    # 6) Pull out the final spectrogram & invert
    out_spec = generated.detach().cpu().squeeze(0)[0].numpy()
    y = mel_to_audio(out_spec)

    # 7) Save
    sf.write(OUTPUT_PATH, y, SR)
    print(f"Wrote {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
