# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import os

import torch
import torch.nn as nn
from accelerate import Accelerator
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ===============================
# MRU BLOCK
# ===============================
class LightMRU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mask = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.feat = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        m = torch.sigmoid(self.mask(x))
        fx = self.feat(x)
        x_skip = self.skip(x)
        return m * fx + (1 - m) * x_skip


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

        d1 = self.d1(torch.cat([self.up1(b),  s3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d1), s2], dim=1))
        d3 = self.d3(torch.cat([self.up3(d2), s1], dim=1))

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
# DATASET
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
            return p.stem.replace('-1', '').replace('_1', '')

        photos   = {normalize(f): f for f in self.root.glob('photo/**/*.jpg')}
        sketches = {normalize(f): f for f in self.root.glob('sketch/**/*.png')}

        self.keys    = sorted(set(photos) & set(sketches))
        self.photos  = photos
        self.sketches = sketches
        print(f'[DATASET] Found {len(self.keys)} matched pairs')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        sketch = Image.open(self.sketches[k]).convert('RGB')
        photo  = Image.open(self.photos[k]).convert('RGB')
        return {
            'sketch': self.transform(sketch),
            'image':  self.transform(photo)
        }


# ===============================
# TRAINER
# ===============================
class MRUTrainer:
    def __init__(self, config):
        self.accelerator = Accelerator(mixed_precision=config.mixed_precision, cpu=config.force_cpu)

        self.G = Generator()
        self.D = Discriminator()
        self.G.requires_grad_(True)
        self.D.requires_grad_(True)


        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1  = nn.L1Loss()

        self.dataloader = DataLoader(
            MRUDataset(config.data_root, config.image_size),
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

    def train(self):
        dataloader_iterator = iter(self.dataloader)
        dataloader_index    = 0

        for iteration in range(1, self.config.steps + 1):
            batch = next(dataloader_iterator)
            x = batch['sketch']   # condition (sketch)
            y = batch['image']    # target (photo)

            # ---- Train D ----
            with self.accelerator.autocast():
                fake   = self.G(x)
                D_real = self.D(x, y)
                D_fake = self.D(x, fake.detach())
                loss_D = (
                    self.criterion_GAN(D_real, torch.ones_like(D_real)) +
                    self.criterion_GAN(D_fake, torch.zeros_like(D_fake))
                ) * 0.5

            self.opt_D.zero_grad(set_to_none=True)
            self.accelerator.backward(loss_D)
            self.opt_D.step()

            # ---- Train G ----
            with self.accelerator.autocast():
                fake   = self.G(x)
                D_fake = self.D(x, fake)
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
        best_chechpoint_full_G = os.path.join(output_dir, 'full_generator.pth')

        for directory in [output_dir, images_dir]:
            if not os.path.isdir(directory):
                os.makedirs(directory)

        if not os.path.isfile(config_path):
            with open(config_path, 'w') as fp:
                json.dump(vars(self.config), fp, indent=2)

        # Save best state when generator loss improves
        if iteration == 1 or loss_G.item() < self.best_loss_G:
            self.best_iter   = iteration
            self.best_loss_G = loss_G.item()

            torch.save({
                'G_state_dict':     self.G.state_dict(),
                'D_state_dict':     self.D.state_dict(),
                'opt_G_state_dict': self.opt_G.state_dict(),
                'opt_D_state_dict': self.opt_D.state_dict(),
                'G_full':           self.G,   # ✅ FULL GENERATOR ADDED
                'step':             iteration,
                'loss_G':           loss_G.item(),
                'loss_D':           loss_D.item()
            }, best_state_path)

            torch.save(self.G.state_dict(), best_checkpoint_G)
            torch.save(self.D.state_dict(), best_checkpoint_D)
            torch.save(self.G, best_chechpoint_full_G)

            # Save image grid: [sketch | fake | real]
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
                'G_full':           self.G,   # ✅ FULL GENERATOR ADDED
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
    parser.add_argument('--mixed_precision', type=str,   default='no', choices=['no', 'fp16', 'bf16', 'fp8'], help='Mixed precision training')
    parser.add_argument('--force_cpu',       action='store_true',                                              help='Force execution on CPU')
    parser.add_argument('--data_root',       type=str,   default='./data',                                    help='Dataset directory')
    parser.add_argument('--image_size',      type=int,   default=128,                                         help='Image size')
    parser.add_argument('--batch_size',      type=int,   default=8,                                           help='Batch size')
    parser.add_argument('--shuffle',         action='store_true',                                              help='Reshuffle data at every epoch')
    parser.add_argument('--num_workers',     type=int,   default=0,                                           help='Number of subprocesses for data loading')
    parser.add_argument('--lr',              type=float, default=2e-4,                                        help='Learning rate')
    parser.add_argument('--lambda_l1',       type=float, default=100.0,                                       help='L1 loss weight')
    parser.add_argument('--steps',           type=int,   default=1,                                           help='Number of training steps')
    parser.add_argument('--output_freq',     type=int,   default=100,                                         help='Log/save frequency (steps)')
    parser.add_argument('--output_root',     type=str,   default='./output',                                  help='Output directory')
    return parser.parse_args()


# ===============================
# MAIN
# ===============================
if __name__ == '__main__':
    config = args()
    trainer = MRUTrainer(config)
    trainer.train()