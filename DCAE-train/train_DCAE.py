# -*- coding: utf-8 -*-
"""
Train the DCAE (see DC_AE.py) on a folder of images.

Usage:
    python train_DCAE.py --data /path/to/images

Input : --data  (folder of images; jpg/jpeg/png/bmp, searched recursively)
Output: written to --save_dir (default checkpoint_dcae/) --
        dcae_latest.pt / dcae_best.pt   training checkpoints (for --resume)
        train_log.csv / train_curve.png per-epoch metrics
        DCAE.pth                        plain state_dict exported after
                                        training; copy it to
                                        model_weight/DCAE.pth for main.py

Run `python train_DCAE.py --help` for the full list of options.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import math
import os
import random
from typing import NamedTuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from DC_AE import DCAE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
LOG_COLUMNS = ("epoch", "steps", "loss", "psnr_db", "ssim", "timestamp")


class EpochStats(NamedTuple):
    """Per-epoch averages produced by :func:`train_one_epoch`."""

    loss: float
    psnr: float  # dB
    ssim: float


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    """Seed every RNG used by this script (python / numpy / torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker init: derive per-worker seeds deterministically."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class FlatImageDataset(Dataset):
    """Flat-folder image dataset with deterministic random subsampling.

    Preprocessing: short-side resize -> center crop -> (optional) random
    horizontal flip -> ToTensor (pixels in [0, 1]).
    """

    def __init__(
        self,
        root: str,
        img_size: int,
        n_select: int = 10_000,
        seed: int = 2025,
        flip: bool = True,
    ) -> None:
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset directory not found: {root}")

        paths: list[str] = []
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        if not paths:
            raise FileNotFoundError(f"No images found in {root}")
        paths = sorted(set(paths))

        if 0 < n_select < len(paths):
            paths = random.Random(seed).sample(paths, n_select)
        self.paths = paths

        tfs = [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
        if flip:
            tfs.append(transforms.RandomHorizontalFlip())
        tfs.append(transforms.ToTensor())
        self.transform = transforms.Compose(tfs)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Metrics: differentiable SSIM and PSNR
# ---------------------------------------------------------------------------


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """2-D Gaussian window, normalised to sum 1."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Mean SSIM of two [B, C, H, W] batches in [0, data_range] (differentiable)."""
    c = x.shape[1]
    win = _gaussian_window(window_size, sigma).to(device=x.device, dtype=x.dtype)
    win = win.expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu_x = F.conv2d(x, win, padding=pad, groups=c)
    mu_y = F.conv2d(y, win, padding=pad, groups=c)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_xx = F.conv2d(x * x, win, padding=pad, groups=c) - mu_xx
    sigma_yy = F.conv2d(y * y, win, padding=pad, groups=c) - mu_yy
    sigma_xy = F.conv2d(x * y, win, padding=pad, groups=c) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    )
    return ssim_map.mean()


def psnr_from_mse(mse: float | torch.Tensor) -> float:
    """PSNR (dB) for images normalised to [0, 1]; ``inf`` when MSE is 0."""
    if torch.is_tensor(mse):
        mse = mse.item()
    return float("inf") if mse <= 0.0 else 10.0 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


class CompositeLoss(nn.Module):
    """loss = pixel_w * Pixel(x_hat, x) + ssim_w * (1 - SSIM(x_hat, x)).

    ``pixel`` is "l1" or "mse"; ``ssim_w = 0`` disables the SSIM term.
    """

    def __init__(self, pixel: str = "l1", pixel_w: float = 1.0, ssim_w: float = 1.0) -> None:
        super().__init__()
        if pixel not in ("l1", "mse"):
            raise ValueError(f"pixel must be 'l1' or 'mse', got {pixel!r}")
        self.pixel = pixel
        self.pixel_w = float(pixel_w)
        self.ssim_w = float(ssim_w)

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pixel_term = F.l1_loss(x_hat, x) if self.pixel == "l1" else F.mse_loss(x_hat, x)
        loss = self.pixel_w * pixel_term
        if self.ssim_w > 0:
            loss = loss + self.ssim_w * (1.0 - ssim(x_hat, x, data_range=1.0))
        return loss


def build_criterion(args: argparse.Namespace) -> CompositeLoss:
    """Map the ``--loss`` choice onto a :class:`CompositeLoss` instance."""
    table = {
        "l1_ssim": ("l1", args.ssim_w),
        "mse_ssim": ("mse", args.ssim_w),
        "l1": ("l1", 0.0),
        "mse": ("mse", 0.0),
    }
    pixel, ssim_w = table[args.loss]
    return CompositeLoss(pixel=pixel, pixel_w=args.pixel_w, ssim_w=ssim_w)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> EpochStats:
    """Run one training pass and return the epoch's average statistics."""
    model.train()

    loss_sum = mse_sum = ssim_sum = 0.0
    n_batch = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", ncols=100)
    for x in pbar:
        x = x.to(device, non_blocking=True)

        x_hat = model(x)
        loss = criterion(x_hat, x)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_sum += loss.item()
            mse_sum += torch.mean((x_hat - x) ** 2).item()
            ssim_sum += ssim(x_hat, x).item()
        n_batch += 1
        pbar.set_postfix(
            loss=f"{loss_sum / n_batch:.5f}",
            psnr=f"{psnr_from_mse(mse_sum / n_batch):.2f}dB",
            ssim=f"{ssim_sum / n_batch:.4f}",
        )

    n = max(n_batch, 1)
    return EpochStats(loss_sum / n, psnr_from_mse(mse_sum / n), ssim_sum / n)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_psnr: float,
    args: argparse.Namespace,
) -> None:
    """Write a checkpoint dict (model + optimizer + progress + config)."""
    state = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr,
        "args": vars(args),
    }
    torch.save(state, path)


def export_plain_state_dict(best_path: str, model: nn.Module, export_path: str) -> None:
    """Export a plain state_dict (the format main.py loads) to export_path.

    Uses the best-PSNR checkpoint when available, else the current weights.
    """
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        state = ckpt["model"]
        source = best_path
    else:
        state = model.state_dict()
        source = "current weights (no best checkpoint found)"
    torch.save(state, export_path)
    print(f"[Export] plain state_dict ({source}) -> {export_path}")


def append_log_row(save_dir: str, epoch: int, steps: int, stats: EpochStats) -> None:
    """Append one row to ``train_log.csv``, writing the header if needed."""
    csv_path = os.path.join(save_dir, "train_log.csv")
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        if is_new:
            f.write(",".join(LOG_COLUMNS) + "\n")
        f.write(
            f"{epoch},{steps},{stats.loss:.6f},{stats.psnr:.4f},{stats.ssim:.6f},"
            f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
        )


def plot_training_curves(save_dir: str) -> None:
    """Render ``train_curve.png`` from ``train_log.csv``.

    Silently skipped when matplotlib is unavailable or the log is too short.
    """
    csv_path = os.path.join(save_dir, "train_log.csv")
    if not os.path.exists(csv_path):
        return
    try:
        import csv

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 2:
            return

        ep = [int(r["epoch"]) for r in rows]
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        axes[0].plot(ep, [float(r["loss"]) for r in rows], color="tab:red")
        axes[0].set_ylabel("loss")
        axes[0].grid(True)

        axes[1].plot(ep, [float(r["psnr_db"]) for r in rows], color="tab:blue")
        axes[1].set_ylabel("PSNR (dB)")
        axes[1].grid(True)

        axes[2].plot(ep, [float(r["ssim"]) for r in rows], color="tab:green")
        axes[2].set_ylabel("SSIM")
        axes[2].set_xlabel("epoch")
        axes[2].grid(True)

        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "train_curve.png"), dpi=120)
        plt.close(fig)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DCAE trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Training image folder (e.g., COCO train2017)")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Input size; must be divisible by 8 (3 stages of x2 pooling)")
    parser.add_argument("--n_select", type=int, default=10000,
                        help="Number of images randomly selected from the folder")
    parser.add_argument("--seed", type=int, default=2025,
                        help="Seed for image selection and weight initialisation")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--loss", choices=["l1_ssim", "mse_ssim", "l1", "mse"],
                        default="l1_ssim",
                        help="Loss: l1_ssim (default), mse_ssim, l1, or mse")
    parser.add_argument("--pixel_w", type=float, default=1.0,
                        help="Weight of the pixel loss term")
    parser.add_argument("--ssim_w", type=float, default=1.0,
                        help="Weight of the SSIM loss term")
    parser.add_argument("--no_flip", action="store_true",
                        help="Disable random horizontal flip")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers; 4 is recommended on Windows")
    parser.add_argument("--save_dir", type=str, default="checkpoint_dcae",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--export", type=str, default="",
                        help="Where to write the exported plain state_dict "
                             "(default: <save_dir>/DCAE.pth)")
    parser.add_argument("--resume", type=str, default="",
                        help="Checkpoint path to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args)

    # The DCAE encoder pools x2 three times, so the input side must be a
    # multiple of 8.
    if args.img_size % 8 != 0:
        raise ValueError(f"--img_size must be divisible by 8, got {args.img_size}")

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] torch {torch.__version__}, device: {device}")

    # ----- data -----
    dataset = FlatImageDataset(
        args.data, args.img_size,
        n_select=args.n_select, seed=args.seed, flip=not args.no_flip,
    )
    print(f"[Data] {len(dataset)} images loaded from {args.data}")

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.workers > 0,
        worker_init_fn=_seed_worker,
        generator=loader_gen,
    )

    # ----- model / optimizer / objective -----
    model = DCAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = build_criterion(args)
    print(f"[Loss] {criterion.pixel_w}*{criterion.pixel} + {criterion.ssim_w}*(1-SSIM)")

    # ----- optional resume -----
    start_epoch, best_psnr = 1, 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = float(ckpt.get("best_psnr", 0.0))
        print(f"[Resume] starting from epoch {start_epoch}, best PSNR {best_psnr:.2f} dB")

    os.makedirs(args.save_dir, exist_ok=True)
    latest_path = os.path.join(args.save_dir, "dcae_latest.pt")
    best_path = os.path.join(args.save_dir, "dcae_best.pt")

    # ----- training -----
    for epoch in range(start_epoch, args.epochs + 1):
        stats = train_one_epoch(model, loader, criterion, optimizer, device,
                                epoch, args.epochs)
        print(f"[Epoch {epoch}] loss={stats.loss:.5f}  "
              f"PSNR={stats.psnr:.2f} dB  SSIM={stats.ssim:.4f}")

        append_log_row(args.save_dir, epoch, epoch * len(loader), stats)
        plot_training_curves(args.save_dir)
        save_checkpoint(latest_path, model, optimizer, epoch,
                        max(best_psnr, stats.psnr), args)
        if stats.psnr > best_psnr:
            best_psnr = stats.psnr
            save_checkpoint(best_path, model, optimizer, epoch, best_psnr, args)
            print(f"[Save] new best PSNR {best_psnr:.2f} dB -> {os.path.basename(best_path)}")

    # ----- export a plain state_dict for main.py -----
    export_path = args.export or os.path.join(args.save_dir, "DCAE.pth")
    export_plain_state_dict(best_path, model, export_path)
    print("[Tip] copy it to model_weight/DCAE.pth to run main.py with it, "
          "or pass --export model_weight/DCAE.pth next time")

    print(f"[Done] best training PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
