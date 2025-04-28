# datasets.py

import os
import random
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

# Audio / spectrogram params
SR = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 400

def audio_to_mel(path):
    """Load audio, compute log-mel spectrogram."""
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
    # Normalize to [0, 1]
    log_S = (log_S - log_S.min()) / (log_S.max() - log_S.min())
    return log_S.astype(np.float32)  # (n_mels, T)

def mel_to_audio(mel_spec):
    """Convert mel spectrogram back to audio."""
    # Denormalize from [0, 1] to original scale
    mel_spec = mel_spec * (mel_spec.max() - mel_spec.min()) + mel_spec.min()
    
    # Convert from dB to power
    S = librosa.db_to_power(mel_spec)
    
    # Inverse mel spectrogram
    y = librosa.feature.inverse.mel_to_audio(
        S,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )
    return y

class SpectrogramDataset(Dataset):
    """
    Returns paired (content, style) spectrograms as 3×H×W tensors.
    Assumes .wav files in two folders.
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
        
        # Load and process spectrograms
        content_spec = audio_to_mel(content_path)
        style_spec = audio_to_mel(style_path)
        
        # Pad or truncate to fixed length
        def pad_or_truncate(spec):
            if spec.shape[1] > self.max_length:
                return spec[:, :self.max_length]
            else:
                return np.pad(spec, ((0, 0), (0, self.max_length - spec.shape[1])), mode='constant')
        
        content_spec = pad_or_truncate(content_spec)
        style_spec = pad_or_truncate(style_spec)
        
        # Convert to tensor and add channel dimension
        content_tensor = torch.from_numpy(content_spec).unsqueeze(0)
        style_tensor = torch.from_numpy(style_spec).unsqueeze(0)
        
        # Replicate to 3 channels for VGG
        content_tensor = content_tensor.repeat(3, 1, 1)  # (3, H, W)
        style_tensor = style_tensor.repeat(3, 1, 1)
        
        return content_tensor, style_tensor
