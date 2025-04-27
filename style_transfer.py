import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
import soundfile as sf


import librosa
import numpy as np

CONTENT_PATH = "data/content.wav"
STYLE_PATH   = "data/style.wav"
OUTPUT_PATH  = "stylized.wav"

# 1. Audio → log-mel-spectrogram
def audio_to_mel(audio_path, sr=16000, n_mels=80, n_fft=1024,
                 hop_length=160, win_length=400):
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels
    )
    log_S = librosa.power_to_db(S)
    log_S -= log_S.min()
    log_S /= log_S.max()
    return log_S.astype(np.float32)


# 2. Gram matrix for style
def gram_matrix(feat):
    # feat: (B, C, H, W)
    B, C, H, W = feat.size()
    f = feat.view(B, C, H*W)
    G = torch.bmm(f, f.transpose(1,2))  # (B, C, C)
    return G / (C * H * W)

# 3. Build a VGG feature extractor
class VGGFeatures(nn.Module):
    def __init__(self, layer_names):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.layer_names = layer_names
        self.layers = {str(i): name for i,name in enumerate(vgg)}

    def forward(self, x):
        feats = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            name = self.layer_names.get(i, None)
            if name:
                feats[name] = x
        return feats

# 4. Prepare content, style, and generated “images”
def prep_image(spec, device):
    # spec: (n_mels, T)
    # we need 3×H×W tensor
    im = torch.from_numpy(spec)[None]          # (1, H, W)
    im = im.repeat(1,3,1,1)                    # (1,3,H,W)
    return im.to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load your audio and style audio
content_spec = audio_to_mel(CONTENT_PATH)
style_spec   = audio_to_mel(STYLE_PATH)

content_img = prep_image(content_spec, device)
style_img   = prep_image(style_spec, device)

# 5. Extract targets
# pick which layers you want for content & style
content_layers = {10: 'conv4_2'}
style_layers   = {1: 'conv1_1', 6: 'conv2_1', 11: 'conv3_1', 20: 'conv4_1', 29: 'conv5_1'}
feat_extractor = VGGFeatures({**content_layers, **style_layers}).to(device)

with torch.no_grad():
    content_feats = feat_extractor(content_img)
    style_feats   = feat_extractor(style_img)
    style_targets = {name: gram_matrix(style_feats[name]) for name in style_feats}

# 6. Initialize generated image
generated = content_img.clone().requires_grad_(True)

# 7. Define losses and optimizer
alpha, beta = 1e4, 1e2
optimizer = optim.LBFGS([generated], max_iter=300)

mse = nn.MSELoss()

def closure():
    optimizer.zero_grad()
    gen_feats = feat_extractor(generated)

    # content loss
    c_loss = mse(gen_feats['conv4_2'], content_feats['conv4_2'])

    # style loss
    s_loss = 0
    for name in style_targets:
        Gg = gram_matrix(gen_feats[name])
        Gt = style_targets[name]
        s_loss += mse(Gg, Gt)

    loss = alpha * c_loss + beta * s_loss
    loss.backward()
    return loss

# 8. Run style transfer
print("Optimizing...")
optimizer.step(closure)
print("Done.")

# 9. Grab the final spectrogram, de-normalize
out = generated.detach().cpu().squeeze(0)[0].numpy()  # (H, W)
# reverse normalization
out = out * (content_spec.max() - content_spec.min()) + content_spec.min()

# 10. Invert mel→audio via Griffin–Lim
def mel_to_audio(mel_spec, sr=16000, n_fft=1024, hop_length=160, win_length=400, n_iter=60):
    # power from dB
    S = librosa.db_to_power(mel_spec)
    # invert mel-filterbank
    inv = librosa.feature.inverse.mel_to_stft(S, sr=sr,
                                              n_fft=n_fft,
                                              power=1.0)
    # griffin–lim
    y = librosa.griffinlim(inv,
                          n_iter=n_iter,
                          hop_length=hop_length,
                          win_length=win_length)
    return y

y = mel_to_audio(out)
sf.write("stylized.wav", y, 16000)
print("Wrote stylized.wav")
