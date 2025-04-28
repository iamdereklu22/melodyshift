# neural_vocoder.py

import torch
import numpy as np

# ─── PATCH librosa for MelGAN hubconf ────────────────────────────────────────────
import librosa
import librosa.filters
import librosa.core.spectrum

# Hub’s MelGAN code does:
#   from librosa.core.spectrum import mel as librosa_mel_fn
# but in newer librosa releases that import gives you a no-arg stub.
# So we reassign it to the real filters.mel function:
librosa.core.spectrum.mel = librosa.filters.mel


class Vocoder:
    """
    Neural vocoder wrapper using a pretrained MelGAN generator via torch.hub.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        # Load the MelGAN-NeurIPS checkpoint via torch.hub
        self.vocoder = torch.hub.load(
            'descriptinc/melgan-neurips',
            'load_melgan',
            model_name='multi_speaker',
            trust_repo=True
        )
        # try moving to device / eval if supported
        try:
            self.vocoder = self.vocoder.to(self.device).eval()
        except Exception:
            pass

    def invert(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Invert a normalized mel spectrogram (n_mels×T in [0,1]) to audio.

        Returns a mono waveform float32 array in [-1,1].
        """
        # Reconstruct dB: range was [-80, 0]
        mel_db = mel_spec * 80.0 - 80.0
        # power spectrogram
        S = 10.0 ** (mel_db / 10.0)
        # batch it
        m = torch.from_numpy(S).unsqueeze(0).to(self.device)
        with torch.no_grad():
            audio = self.vocoder(m)   # (1, L)
        audio = audio.squeeze(0).cpu().numpy()
        # normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        return audio
