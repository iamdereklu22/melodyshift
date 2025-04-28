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
    # Split into train and validation
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # — models & losses —
    stylizer = TransformerNet().to(device)
    content_layers = {10: "conv4_2"}
    style_layers   = {
        1: "conv1_1", 6: "conv2_1", 11: "conv3_1",
        20: "conv4_1", 21: "conv4_2", 22: "conv5_1", 29: "conv5_2"
    }
    vgg = VGGFeatures(content_layers, style_layers).to(device).eval()
    mse = nn.MSELoss()
    optimizer = optim.Adam(stylizer.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # precompute ImageNet mean/std on the correct device & dtype
    im_mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    im_std  = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    # — training loop —
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        stylizer.train()
        running_loss = 0.0
        for i, (content, style) in enumerate(train_loader, 1):
            content, style = content.to(device), style.to(device)
            out = stylizer(content)

            # ImageNet-style normalization for VGG
            content_norm = (content - im_mean) / im_std
            style_norm   = (style   - im_mean) / im_std
            out_norm     = (out     - im_mean) / im_std

            # Feature extraction
            feats_out     = vgg(out_norm)
            feats_content = vgg(content_norm)
            feats_style   = vgg(style_norm)

            # Content loss
            c_loss = mse(feats_out["conv4_2"], feats_content["conv4_2"])
            
            # Style loss
            s_loss = 0.0
            for name in style_layers.values():
                Gg = gram_matrix(feats_out[name])
                Gt = gram_matrix(feats_style[name])
                s_loss += mse(Gg, Gt)
            
            # Reconstruction loss
            r_loss = mse(out, content)

            # Total loss
            loss = args.alpha * c_loss + args.beta * s_loss + args.gamma * r_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % args.log_interval == 0:
                # print(f"[Epoch {epoch}/{args.epochs}] Batch {i}/{len(train_loader)} — Loss: {loss.item():.4f}")
                pass

        train_avg = running_loss / len(train_loader)

        # Validation
        stylizer.eval()
        val_loss = 0.0
        with torch.no_grad():
            for content, style in val_loader:
                content, style = content.to(device), style.to(device)
                out = stylizer(content)

                # ImageNet-style normalization for VGG
                content_norm = (content - im_mean) / im_std
                style_norm   = (style   - im_mean) / im_std
                out_norm     = (out     - im_mean) / im_std

                feats_out     = vgg(out_norm)
                feats_content = vgg(content_norm)
                feats_style   = vgg(style_norm)

                c_loss = mse(feats_out["conv4_2"], feats_content["conv4_2"])

                s_loss = 0.0
                for name in style_layers.values():
                    Gg = gram_matrix(feats_out[name])
                    Gt = gram_matrix(feats_style[name])
                    s_loss += mse(Gg, Gt)
                
                r_loss = mse(out, content)

                loss = args.alpha * c_loss + args.beta * s_loss + args.gamma * r_loss
                val_loss += loss.item()
        
        val_avg = val_loss / len(val_loader)

        print(f"Epoch {epoch} complete. \t Train Avg Loss: {train_avg:.4f} \t Validation Loss: {val_avg:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_avg)
        
        # Save best model
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            ckpt = os.path.join(args.checkpoint_dir, "stylizer_best.pth")
            torch.save(stylizer.state_dict(), ckpt)
            # print(f"→ Saved best model to {ckpt}")
        
        # Save checkpoint
        ckpt = os.path.join(args.checkpoint_dir, f"stylizer_epoch{epoch}.pth")
        torch.save(stylizer.state_dict(), ckpt)
        # print(f"→ Saved {ckpt}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train fast audio style-transfer")
    p.add_argument("--content_dir", default="data/contents")
    p.add_argument("--style_dir",   default="data/styles")
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--epochs",      type=int, default=5)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--alpha",       type=float, default=1e3)  # content weight
    p.add_argument("--beta",        type=float, default=1e9)  # style weight
    p.add_argument("--gamma",       type=float, default=1e0)  # reconstruction weight
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_interval", type=int, default=10)
    args = p.parse_args()
    train(args)
