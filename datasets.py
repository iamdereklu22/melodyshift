# datasets.py
import os
import random
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

# Audio / spectrogram params
SR = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 400
# increase for cleaner inversion
GRIFFIN_LIM_ITERS = 200

def audio_to_mel(path):
    """Load audio, compute log-mel spectrogram and return normalization range."""
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
    orig_min, orig_max = log_S.min(), log_S.max()
    norm_S = (log_S - orig_min) / (orig_max - orig_min)
    return norm_S.astype(np.float32), orig_min, orig_max

def mel_to_audio(mel_spec, orig_min, orig_max):
    """Convert mel spectrogram back to audio using original dB scale and Griffin-Lim."""
    mel_db = mel_spec * (orig_max - orig_min) + orig_min
    S = librosa.db_to_power(mel_db)
    # invert mel→audio with higher iterations
    y = librosa.feature.inverse.mel_to_audio(
        S,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_iter=GRIFFIN_LIM_ITERS
    )
    return y

class SpectrogramDataset(Dataset):
    """
    Returns paired (content, style) spectrograms as 3×H×W tensors.
    """
    def __init__(self, content_dir, style_dir, max_length=1000):
        self.content_paths = [
            os.path.join(content_dir, f)
            for f in os.listdir(content_dir) if f.endswith(".wav")
        ]
        self.style_paths = [
            os.path.join(style_dir, f)
            for f in os.listdir(style_dir) if f.endswith(".wav")
        ]
        self.max_length = max_length

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, idx):
        content_path = self.content_paths[idx]
        style_path = random.choice(self.style_paths)

        content_spec, _, _ = audio_to_mel(content_path)
        style_spec, _, _ = audio_to_mel(style_path)

        def pad_or_truncate(spec):
            if spec.shape[1] > self.max_length:
                return spec[:, :self.max_length]
            return np.pad(
                spec,
                ((0, 0), (0, self.max_length - spec.shape[1])),
                mode='constant'
            )

        content_spec = pad_or_truncate(content_spec)
        style_spec   = pad_or_truncate(style_spec)

        content_tensor = torch.from_numpy(content_spec).unsqueeze(0).repeat(3, 1, 1)
        style_tensor   = torch.from_numpy(style_spec).unsqueeze(0).repeat(3, 1, 1)

        return content_tensor, style_tensor
