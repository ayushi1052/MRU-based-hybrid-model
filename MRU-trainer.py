# -*- coding: utf-8 -*-
import argparse, datetime, json, os
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ===============================
# WEIGHT INIT
# ===============================
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)


# ===============================
# FiLM
# ===============================
class FiLM(nn.Module):
    def __init__(self, embed_dim, num_features):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_features * 2)

    def forward(self, x, embed):
        gamma, beta = self.linear(embed).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta  = beta[:, :, None, None]
        return gamma * x + beta


# ===============================
# MRU BLOCK
# ===============================
class CondLightMRU(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()
        self.mask = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.feat = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        self.film = FiLM(embed_dim, out_ch)

    def forward(self, x, embed):
        m = torch.sigmoid(self.mask(x))
        out = m * self.feat(x) + (1 - m) * self.skip(x)
        return self.film(out, embed)


# ===============================
# GENERATOR
# ===============================
class CondGenerator(nn.Module):
    def __init__(self, num_classes, embed_dim=64):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.pool = nn.MaxPool2d(2)

        self.e1 = CondLightMRU(3, 16, embed_dim)
        self.e2 = CondLightMRU(16, 32, embed_dim)
        self.e3 = CondLightMRU(32, 64, embed_dim)

        self.b = CondLightMRU(64, 128, embed_dim)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d1 = CondLightMRU(128, 64, embed_dim)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d2 = CondLightMRU(64, 32, embed_dim)

        self.up3 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.d3 = CondLightMRU(32, 16, embed_dim)

        self.out = nn.Conv2d(16, 3, 1)

    def forward(self, x, label):
        embed = self.class_embed(label)

        s1 = self.e1(x, embed)
        s2 = self.e2(self.pool(s1), embed)
        s3 = self.e3(self.pool(s2), embed)

        b = self.b(self.pool(s3), embed)

        d1 = self.d1(torch.cat([self.up1(b), s3], dim=1), embed)
        d2 = self.d2(torch.cat([self.up2(d1), s2], dim=1), embed)
        d3 = self.d3(torch.cat([self.up3(d2), s1], dim=1), embed)

        return torch.tanh(self.out(d3))


# ===============================
# DISCRIMINATOR
# ===============================
class CondDiscriminator(nn.Module):
    def __init__(self, num_classes, embed_dim=64):
        super().__init__()
        self.features = nn.Sequential(
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
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.proj = nn.Linear(embed_dim, 1)

    def forward(self, x, y, label):
        feat = self.features(torch.cat([x, y], dim=1))
        bias = self.proj(self.class_embed(label))[:, :, None, None]
        return feat + bias


# ===============================
# DATASET (same as yours)
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
            return p.stem.replace('-1','').replace('_1','')

        photos = {normalize(f):(f,f.parent.name) for f in self.root.glob('photo/**/*.jpg')}
        sketches = {normalize(f):(f,f.parent.name) for f in self.root.glob('sketch/**/*.png')}

        self.keys = sorted(set(photos) & set(sketches))
        self.photos = photos
        self.sketches = sketches

        classes = sorted({photos[k][1] for k in self.keys})
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.num_classes = len(classes)

        print(f"[DATASET] {len(self.keys)} pairs, {self.num_classes} classes")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        sketch_path, class_name = self.sketches[k]
        photo_path,_ = self.photos[k]

        sketch = Image.open(sketch_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")

        return {
            "sketch": self.transform(sketch),
            "image": self.transform(photo),
            "label": torch.tensor(self.class_to_idx[class_name])
        }


# ===============================
# TRAINER (FULL FEATURES)
# ===============================
class MRUTrainer:
    def __init__(self, config):
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            cpu=config.force_cpu
        )

        dataset = MRUDataset(config.data_root, config.image_size)

        self.G = CondGenerator(dataset.num_classes, config.embed_dim)
        self.D = CondDiscriminator(dataset.num_classes, config.embed_dim)

        self.G.apply(init_weights)
        self.D.apply(init_weights)

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=config.lr, betas=(0.5,0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=config.lr, betas=(0.5,0.999))

        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()

        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers
        )

        self.G, self.D, self.dataloader, self.opt_G, self.opt_D = self.accelerator.prepare(
            self.G, self.D, self.dataloader, self.opt_G, self.opt_D
        )

        self.best_loss = float("inf")
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.config = config

    def train(self):
        iterator = iter(self.dataloader)

        for step in range(1, self.config.steps + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.dataloader)
                batch = next(iterator)

            x, y, label = batch["sketch"], batch["image"], batch["label"]

            # ---- D ----
            with self.accelerator.autocast():
                fake = self.G(x, label)
                real = self.D(x, y, label)
                fake_pred = self.D(x, fake.detach(), label)

                real_labels = torch.ones_like(real) * 0.9
                fake_labels = torch.zeros_like(fake_pred)

                loss_D = (
                    self.criterion_GAN(real, real_labels) +
                    self.criterion_GAN(fake_pred, fake_labels)
                ) * 0.5

            self.opt_D.zero_grad()
            self.accelerator.backward(loss_D)
            self.accelerator.clip_grad_norm_(self.D.parameters(), 1.0)
            self.opt_D.step()

            # ---- G ----
            with self.accelerator.autocast():
                fake = self.G(x, label)
                pred = self.D(x, fake, label)

                loss_G = (
                    self.criterion_GAN(pred, torch.ones_like(pred)) +
                    self.config.lambda_l1 * self.criterion_L1(fake, y)
                )

            self.opt_G.zero_grad()
            self.accelerator.backward(loss_G)
            self.accelerator.clip_grad_norm_(self.G.parameters(), 1.0)
            self.opt_G.step()

            self.log(step, loss_G, loss_D, [x, fake, y])

    def log(self, step, loss_G, loss_D, images):
        base = os.path.join(self.config.output_root, self.timestamp)
        img_dir = os.path.join(base, "images")
        os.makedirs(img_dir, exist_ok=True)

        G = self.accelerator.unwrap_model(self.G)
        D = self.accelerator.unwrap_model(self.D)

        # best model
        if loss_G.item() < self.best_loss:
            self.best_loss = loss_G.item()
            torch.save(G.state_dict(), os.path.join(base,"best_G.pth"))
            torch.save(D.state_dict(), os.path.join(base,"best_D.pth"))

        # periodic save
        if step % self.config.output_freq == 0:
            print(f"[STEP {step}] G={loss_G.item():.4f}, D={loss_D.item():.4f}")

            grid = []
            for img in images:
                img = (img.detach().cpu() / 2 + 0.5).clamp(0,1)
                grid.append(make_grid(img, nrow=1)[:3])
            grid = torch.cat(grid, dim=2)

            T.ToPILImage()(grid).save(os.path.join(img_dir, f"{step}.png"))


# ===============================
# ARGS (UNCHANGED)
# ===============================
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_precision', type=str, default='no')
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--lambda_l1', type=float, default=100.0)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--output_freq', type=int, default=100)
    parser.add_argument('--output_root', type=str, default='./output')
    return parser.parse_args()


if __name__ == "__main__":
    config = args()
    trainer = MRUTrainer(config)
    trainer.train()