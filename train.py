# =============================
# train.py (residual style transfer)
# =============================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import SpectrogramDataset
from models import TransformerNet, VGGFeatures, gram_matrix

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # — data loader —
    ds = SpectrogramDataset(args.content_dir, args.style_dir)
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # — models & losses —
    stylizer = TransformerNet().to(device)
    content_layers = {10: "conv4_2"}
    style_layers   = {1: "conv1_1", 6: "conv2_1", 11: "conv3_1",
                      20: "conv4_1", 21: "conv4_2", 22: "conv5_1", 29: "conv5_2"}
    vgg       = VGGFeatures(content_layers, style_layers).to(device).eval()
    mse       = nn.MSELoss()
    optimizer = optim.Adam(stylizer.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # ImageNet normalization for VGG
    im_mean = torch.tensor([0.485,0.456,0.406], device=device)[None,:,None,None]
    im_std  = torch.tensor([0.229,0.224,0.225], device=device)[None,:,None,None]

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs+1):
        stylizer.train()
        running_loss = 0.0
        for i, (content, style) in enumerate(train_loader, 1):
            content, style = content.to(device), style.to(device)
            # stylizer now predicts a residual to add to content
            residual = stylizer(content)
            out = content + residual

            # normalize for VGG
            c_norm = (content - im_mean) / im_std
            s_norm = (style   - im_mean) / im_std
            o_norm = (out     - im_mean) / im_std

            # features
            f_out     = vgg(o_norm)
            f_cont    = vgg(c_norm)
            f_sty     = vgg(s_norm)

            # content & style losses on 'out'
            c_loss = mse(f_out['conv4_2'], f_cont['conv4_2'])
            s_loss = sum(mse(gram_matrix(f_out[n]), gram_matrix(f_sty[n]))
                         for n in style_layers.values())

            # reconstruction encourages small residuals
            r_loss = mse(residual, torch.zeros_like(residual))

            # weighted sum
            loss = args.alpha*c_loss + args.beta*s_loss + args.gamma*r_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running_loss += loss.item()

        train_avg = running_loss/len(train_loader)
        print(f"Epoch {epoch}/{args.epochs} — Train Loss: {train_avg:.4f}")

        # validation
        stylizer.eval(); val_loss=0
        with torch.no_grad():
            for content, style in val_loader:
                content, style = content.to(device), style.to(device)
                residual = stylizer(content)
                out      = content + residual
                c_norm = (content - im_mean) / im_std
                s_norm = (style   - im_mean) / im_std
                o_norm = (out     - im_mean) / im_std
                f_out     = vgg(o_norm)
                f_cont    = vgg(c_norm)
                f_sty     = vgg(s_norm)
                c_loss = mse(f_out['conv4_2'], f_cont['conv4_2'])
                s_loss = sum(mse(gram_matrix(f_out[n]), gram_matrix(f_sty[n]))
                             for n in style_layers.values())
                r_loss = mse(residual, torch.zeros_like(residual))
                val_loss += (args.alpha*c_loss + args.beta*s_loss + args.gamma*r_loss).item()
        val_avg = val_loss/len(val_loader)
        print(f"Validation Loss: {val_avg:.4f}")
        scheduler.step(val_avg)

        # save best
        if val_avg < best_val_loss:
            best_val_loss=val_avg
            torch.save(stylizer.state_dict(), os.path.join(args.checkpoint_dir,'stylizer_best.pth'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--content_dir', default='data/contents')
    p.add_argument('--style_dir',   default='data/styles')
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--epochs',      type=int, default=5)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--alpha',       type=float, default=1e3)
    p.add_argument('--beta',        type=float, default=1e9)
    p.add_argument('--gamma',       type=float, default=1e0)
    p.add_argument('--checkpoint_dir', default='checkpoints')
    args=p.parse_args(); train(args)