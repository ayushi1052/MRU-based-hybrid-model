# -*- coding: utf-8 -*-
import argparse
import datetime
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from torch.cuda.amp import autocast, GradScaler


# ===============================
# MRU BLOCK
# ===============================
class LightMRU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mask = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.feat = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def forward(self, x):
        m = torch.sigmoid(self.mask(x))
        fx = self.feat(x)
        return m * fx + (1 - m) * x


# ===============================
# GENERATOR
# ===============================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.e1 = LightMRU(3, 16)
        self.e2 = LightMRU(16, 32)
        self.e3 = LightMRU(32, 64)

        self.b = LightMRU(64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d1 = LightMRU(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d2 = LightMRU(64, 32)

        self.up3 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.d3 = LightMRU(32, 16)

        self.out = nn.Conv2d(16, 3, 1)


    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))

        b = self.b(self.pool(s3))

        d1 = self.up1(b)
        d1 = self.d1(torch.cat([d1, s3], dim=1))

        d2 = self.up2(d1)
        d2 = self.d2(torch.cat([d2, s2], dim=1))

        d3 = self.up3(d2)
        d3 = self.d3(torch.cat([d3, s1], dim=1))

        return torch.tanh(self.out(d3))


# ===============================
# DISCRIMINATOR
# ===============================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 16, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(16, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


# ===============================
# DATASET (PAIRING)
# ===============================
class MRUDataset(Dataset):
    def __init__(self, root, size=128):
        self.root = Path(root)

        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

        def normalize(p):
            return p.stem.replace("-1", "").replace("_1", "")

        photos = {normalize(f): f for f in self.root.glob("photo/**/*.jpg")}
        sketches = {normalize(f): f for f in self.root.glob("sketch/**/*.png")}

        self.keys = sorted(set(photos) & set(sketches))
        self.photos = photos
        self.sketches = sketches

        print(f"✅ Found {len(self.keys)} matched pairs")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        sketch = Image.open(self.sketches[k]).convert("RGB")
        photo = Image.open(self.photos[k]).convert("RGB")

        return {
            "sketch": self.transform(sketch),
            "image": self.transform(photo)
        }

#=============================
#save
#===============================


def save_images(self, x, fake, y, step):
    # Denormalize
    x = (x + 1) / 2
    fake = (fake + 1) / 2
    y = (y + 1) / 2

    x = x[0].cpu()
    fake = fake[0].cpu()
    y = y[0].cpu()

    grid = torch.cat([x, fake, y], dim=2)

    from torchvision.transforms.functional import to_pil_image
    img = to_pil_image(grid)

    save_path = os.path.join(self.output_dir, f"step_{step}.png")
    img.save(save_path)


# ===============================
# TRAINER (WITH AMP)
# ===============================
class MRUTrainer:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=config.lr, betas=(0.5, 0.999))

        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()

        self.scaler = GradScaler()

        self.dataloader = DataLoader(
            MRUDataset(config.data_root, config.image_size),
            batch_size=config.batch_size,
            shuffle=True
        )

        self.config = config

    def train(self):
        step = 0
        for epoch in range(1, self.config.epochs + 1):
            step += 1
            for batch in self.dataloader:
                x = batch["sketch"].to(self.device)
                y = batch["image"].to(self.device)

                # ---- Train D ----
                with autocast():
                    fake = self.G(x)

                    D_real = self.D(x, y)
                    D_fake = self.D(x, fake.detach())

                    loss_D = (
                        self.criterion_GAN(D_real, torch.ones_like(D_real)) +
                        self.criterion_GAN(D_fake, torch.zeros_like(D_fake))
                    ) * 0.5

                self.opt_D.zero_grad()
                self.scaler.scale(loss_D).backward()
                self.scaler.step(self.opt_D)

                # ---- Train G ----
                with autocast():
                    fake = self.G(x)
                    D_fake = self.D(x, fake)

                    loss_G = (
                        self.criterion_GAN(D_fake, torch.ones_like(D_fake)) +
                        100 * self.criterion_L1(fake, y)
                    )

                self.opt_G.zero_grad()
                self.scaler.scale(loss_G).backward()
                self.scaler.step(self.opt_G)

                self.scaler.update()
                if step % 100 == 0:
                    print(f"Step {step}: Saving images...")
                    self.save_images(x, fake, y, step)

            print(f"[Epoch {epoch}] G={loss_G.item():.4f}, D={loss_D.item():.4f}")


# ===============================
# ARGS
# ===============================
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    return parser.parse_args()


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    config = args()
    trainer = MRUTrainer(config)
    trainer.train()