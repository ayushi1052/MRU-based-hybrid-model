# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import os
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
# FILM — Feature-wise Linear Modulation
# ===============================
class FiLM(nn.Module):
    """
    Projects a class embedding into per-channel scale (γ) and shift (β),
    then applies: output = γ * x + β
    (γ, β) are broadcast over the spatial dimensions of x.
    """
    def __init__(self, embed_dim, num_features):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_features * 2)

    def forward(self, x, embed):
        # embed: (b, embed_dim)
        gamma, beta = self.linear(embed).chunk(2, dim=1)   # each (b, num_features)
        gamma = gamma[:, :, None, None]                     # (b, c, 1, 1)
        beta  = beta[:, :, None, None]
        return gamma * x + beta


# ===============================
# CONDITIONAL MRU BLOCK
# ===============================
class CondLightMRU(nn.Module):
    """
    MRU selective fusion followed by FiLM class conditioning:
      1. MRU: out = sigmoid(mask(x)) * feat(x) + (1 - sigmoid(mask(x))) * skip(x)
      2. FiLM: out = γ(embed) * out + β(embed)
    """
    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()
        self.mask = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.feat = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        self.film = FiLM(embed_dim, out_ch)

    def forward(self, x, embed):
        m   = torch.sigmoid(self.mask(x))
        out = m * self.feat(x) + (1 - m) * self.skip(x)   # MRU selective fusion
        return self.film(out, embed)                        # FiLM modulation


# ===============================
# CONDITIONAL GENERATOR — MRU U-Net
# ===============================
class CondGenerator(nn.Module):
    """
    U-Net generator with MRU blocks.
    The class label is embedded and injected into every Cond MRU block via FiLM.
    """
    def __init__(self, num_classes, embed_dim=64):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.pool        = nn.MaxPool2d(2)

        # Encoder
        self.e1 = CondLightMRU(3,   16,  embed_dim)
        self.e2 = CondLightMRU(16,  32,  embed_dim)
        self.e3 = CondLightMRU(32,  64,  embed_dim)

        # Bottleneck
        self.b  = CondLightMRU(64,  128, embed_dim)

        # Decoder  (in_ch = upsample_ch + skip_ch)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d1  = CondLightMRU(128, 64,  embed_dim)   # 64 (up) + 64 (skip E3)

        self.up2 = nn.ConvTranspose2d(64,  32, 2, 2)
        self.d2  = CondLightMRU(64,  32,  embed_dim)   # 32 (up) + 32 (skip E2)

        self.up3 = nn.ConvTranspose2d(32,  16, 2, 2)
        self.d3  = CondLightMRU(32,  16,  embed_dim)   # 16 (up) + 16 (skip E1)

        self.out = nn.Conv2d(16, 3, 1)

    def forward(self, x, label):
        embed = self.class_embed(label)                         # (b, embed_dim)

        # Encoder
        s1 = self.e1(x,             embed)
        s2 = self.e2(self.pool(s1), embed)
        s3 = self.e3(self.pool(s2), embed)

        # Bottleneck
        b  = self.b(self.pool(s3),  embed)

        # Decoder with skip connections
        d1 = self.d1(torch.cat([self.up1(b),  s3], dim=1), embed)
        d2 = self.d2(torch.cat([self.up2(d1), s2], dim=1), embed)
        d3 = self.d3(torch.cat([self.up3(d2), s1], dim=1), embed)

        return torch.tanh(self.out(d3))


# ===============================
# CONDITIONAL DISCRIMINATOR — PatchGAN + Projection
# ===============================
class CondDiscriminator(nn.Module):
    """
    PatchGAN discriminator with projection conditioning.
    The class embedding is projected to a scalar bias and added to the patch output:
      score = patch_features(sketch ‖ image) + Linear(embed)
    This lets the discriminator ask "does this image look like class C?" per patch.
    """
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

            nn.Conv2d(64, 1, 4, 1, 1)                         # (b, 1, h', w')
        )
        # Projection: class embed → scalar bias added to every patch
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.proj        = nn.Linear(embed_dim, 1)

    def forward(self, x, y, label):
        feat = self.features(torch.cat([x, y], dim=1))         # (b, 1, h', w')
        bias = self.proj(self.class_embed(label))               # (b, 1)
        bias = bias[:, :, None, None]                           # (b, 1, 1, 1) → broadcast
        return feat + bias


# ===============================
# DATASET
# ===============================
class MRUDataset(Dataset):
    """
    Expects data_root with two subdirectories:
        photo/  <class_name>/  *.jpg
        sketch/ <class_name>/  *.png

    The class label is derived from the parent folder name.
    Sketchy dataset structure works out of the box.
    """
    def __init__(self, root, size=128):
        self.root = Path(root)
        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

        def normalize(p):
            return p.stem.replace('-1', '').replace('_1', '')

        # Value: (path, class_name) — class_name = parent folder
        photos   = {normalize(f): (f, f.parent.name) for f in self.root.glob('photo/**/*.jpg')}
        sketches = {normalize(f): (f, f.parent.name) for f in self.root.glob('sketch/**/*.png')}

        self.keys    = sorted(set(photos) & set(sketches))
        self.photos  = photos
        self.sketches = sketches

        # Build a sorted, stable class index from matched pairs only
        classes           = sorted({photos[k][1] for k in self.keys})
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes  = len(classes)

        print(f'[DATASET] {len(self.keys)} matched pairs across {self.num_classes} classes: {classes}')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        sketch_path, class_name = self.sketches[k]
        photo_path,  _          = self.photos[k]

        sketch = Image.open(sketch_path).convert('RGB')
        photo  = Image.open(photo_path).convert('RGB')
        label  = self.class_to_idx[class_name]

        return {
            'sketch': self.transform(sketch),
            'image':  self.transform(photo),
            'label':  torch.tensor(label, dtype=torch.long)
        }


# ===============================
# TRAINER
# ===============================
class MRUTrainer:
    def __init__(self, config):
        self.accelerator = Accelerator(mixed_precision=config.mixed_precision, cpu=config.force_cpu)

        # Build dataset first — num_classes comes from the data
        dataset     = MRUDataset(config.data_root, config.image_size)
        num_classes = dataset.num_classes

        self.G = CondGenerator(num_classes, embed_dim=config.embed_dim)
        self.D = CondDiscriminator(num_classes, embed_dim=config.embed_dim)
        self.G.requires_grad_(True)
        self.D.requires_grad_(True)

        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1  = nn.L1Loss()

        self.dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers
        )

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=config.lr, betas=(0.5, 0.999))

        self.G, \
        self.D, \
        self.dataloader, \
        self.opt_G, \
        self.opt_D = self.accelerator.prepare(
            self.G,
            self.D,
            self.dataloader,
            self.opt_G,
            self.opt_D
        )

        self.best_iter   = 0
        self.best_loss_G = float('inf')
        self.timestamp   = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.config      = config
        self.config.num_classes = num_classes    # persist for logging

    def train(self):
        dataloader_iterator = iter(self.dataloader)
        dataloader_index    = 0

        for iteration in range(1, self.config.steps + 1):
            batch = next(dataloader_iterator)
            x     = batch['sketch']               # condition  (sketch)
            y     = batch['image']                # target     (photo)
            label = batch['label']                # class label

            # ---- Train D ----
            with self.accelerator.autocast():
                fake   = self.G(x, label)
                D_real = self.D(x, y,             label)
                D_fake = self.D(x, fake.detach(), label)
                loss_D = (
                    self.criterion_GAN(D_real, torch.ones_like(D_real)) +
                    self.criterion_GAN(D_fake, torch.zeros_like(D_fake))
                ) * 0.5

            self.opt_D.zero_grad(set_to_none=True)
            self.accelerator.backward(loss_D)
            self.opt_D.step()

            # ---- Train G ----
            with self.accelerator.autocast():
                fake   = self.G(x, label)
                D_fake = self.D(x, fake, label)
                loss_G = (
                    self.criterion_GAN(D_fake, torch.ones_like(D_fake)) +
                    self.config.lambda_l1 * self.criterion_L1(fake, y)
                )

            self.opt_G.zero_grad(set_to_none=True)
            self.accelerator.backward(loss_G)
            self.opt_G.step()

            self.log(iteration, loss_G, loss_D, [x, fake, y])

            dataloader_index = (dataloader_index + 1) % len(self.dataloader)
            if dataloader_index == 0:
                dataloader_iterator = iter(self.dataloader)

    def log(self, iteration, loss_G, loss_D, images):
        w_iter            = len(str(self.config.steps))
        output_dir        = os.path.join(self.config.output_root, self.timestamp)
        images_dir        = os.path.join(output_dir, 'images')
        config_path       = os.path.join(output_dir, 'config.json')
        last_state_path   = os.path.join(output_dir, 'last_state.pt')
        best_state_path   = os.path.join(output_dir, 'best_state.pt')
        best_checkpoint_G = os.path.join(output_dir, 'generator.pth')
        best_checkpoint_D = os.path.join(output_dir, 'discriminator.pth')

        for directory in [output_dir, images_dir]:
            if not os.path.isdir(directory):
                os.makedirs(directory)

        if not os.path.isfile(config_path):
            with open(config_path, 'w') as fp:
                json.dump(vars(self.config), fp, indent=2)

        # Save best checkpoint when G loss improves
        if iteration == 1 or loss_G.item() < self.best_loss_G:
            self.best_iter   = iteration
            self.best_loss_G = loss_G.item()

            torch.save({
                'G_state_dict':     self.G.state_dict(),
                'D_state_dict':     self.D.state_dict(),
                'opt_G_state_dict': self.opt_G.state_dict(),
                'opt_D_state_dict': self.opt_D.state_dict(),
                'step':             iteration,
                'loss_G':           loss_G.item(),
                'loss_D':           loss_D.item()
            }, best_state_path)

            torch.save(self.G.state_dict(), best_checkpoint_G)
            torch.save(self.D.state_dict(), best_checkpoint_D)

            # Save grid: sketch | fake | real photo
            grid_cols = []
            for img in images:
                img = img.detach().cpu().float()
                img = (img / 2 + 0.5).clamp(0, 1)
                img = make_grid(img, nrow=1)
                grid_cols.append(img[:3, :, :])
            grid_image = T.ToPILImage()(torch.cat(grid_cols, dim=2))
            grid_image.save(os.path.join(
                images_dir,
                f'iter_{str(iteration).zfill(w_iter)}_lossG_{loss_G.item():.4f}.jpg'
            ))

        # Periodic checkpoint
        if iteration == 1 or iteration == self.config.steps or iteration % self.config.output_freq == 0:
            torch.save({
                'G_state_dict':     self.G.state_dict(),
                'D_state_dict':     self.D.state_dict(),
                'opt_G_state_dict': self.opt_G.state_dict(),
                'opt_D_state_dict': self.opt_D.state_dict(),
                'step':             iteration,
                'loss_G':           loss_G.item(),
                'loss_D':           loss_D.item()
            }, last_state_path)

            print(
                f'[TRAIN] ITER: {iteration:{w_iter}d}/{self.config.steps} |',
                f'LOSS_G: {loss_G.item():.4f} | LOSS_D: {loss_D.item():.4f} |',
                f'BEST_ITER: {self.best_iter:{w_iter}d} | BEST_LOSS_G: {self.best_loss_G:.4f} |'
            )


# ===============================
# ARGS
# ===============================
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_precision', type=str,   default='no',    choices=['no', 'fp16', 'bf16', 'fp8'], help='Mixed precision training')
    parser.add_argument('--force_cpu',       action='store_true',                                                 help='Force execution on CPU')
    parser.add_argument('--data_root',       type=str,   default='./data',                                       help='Dataset root (expects photo/ and sketch/ subdirs)')
    parser.add_argument('--image_size',      type=int,   default=128,                                            help='Image size')
    parser.add_argument('--batch_size',      type=int,   default=8,                                              help='Batch size')
    parser.add_argument('--shuffle',         action='store_true',                                                 help='Reshuffle data each epoch')
    parser.add_argument('--num_workers',     type=int,   default=0,                                              help='DataLoader worker count')
    parser.add_argument('--lr',              type=float, default=2e-4,                                           help='Learning rate')
    parser.add_argument('--embed_dim',       type=int,   default=64,                                             help='Class embedding dimension')
    parser.add_argument('--lambda_l1',       type=float, default=100.0,                                          help='L1 reconstruction loss weight')
    parser.add_argument('--steps',           type=int,   default=1000,                                              help='Total training steps')
    parser.add_argument('--output_freq',     type=int,   default=100,                                            help='Log/checkpoint frequency (steps)')
    parser.add_argument('--output_root',     type=str,   default='./output',                                     help='Output root directory')
    return parser.parse_args()


# ===============================
# MAIN
# ===============================
if __name__ == '__main__':
    config = args()
    trainer = MRUTrainer(config)
    trainer.train()