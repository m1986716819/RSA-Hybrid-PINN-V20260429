from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models.gating_transformer import GatingTransformer, GatingTransformerConfig


def _weighted_mse_loss(
    pred: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    loss = (pred - targets) ** 2
    return torch.sum(loss * sample_weight) / (torch.sum(sample_weight) + 1e-12)


def _regression_metrics(pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    err = pred - y_true
    mae = float(torch.mean(torch.abs(err)).item())
    rmse = float(torch.sqrt(torch.mean(err**2)).item())
    var = torch.var(y_true)
    r2 = float(1.0 - (torch.mean(err**2) / (var + 1e-12)).item())
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _split_with_failed_ratio(
    failed: np.ndarray,
    seed: int,
    train_frac: float = 0.8,
    val_failed_frac: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(failed.shape[0])
    all_idx = np.arange(n, dtype=np.int64)
    failed_idx = all_idx[failed.astype(bool)]
    ok_idx = all_idx[~failed.astype(bool)]
    rng.shuffle(failed_idx)
    rng.shuffle(ok_idx)

    n_val = max(1, n - int(train_frac * n))
    n_val_failed = min(failed_idx.shape[0], int(round(val_failed_frac * n_val)))
    n_val_ok = min(ok_idx.shape[0], max(0, n_val - n_val_failed))
    if n_val_failed + n_val_ok < n_val:
        remaining = np.setdiff1d(all_idx, np.concatenate([failed_idx[:n_val_failed], ok_idx[:n_val_ok]]), assume_unique=False)
        rng.shuffle(remaining)
        extra = remaining[: n_val - (n_val_failed + n_val_ok)]
        idx_val = np.concatenate([failed_idx[:n_val_failed], ok_idx[:n_val_ok], extra], axis=0)
    else:
        idx_val = np.concatenate([failed_idx[:n_val_failed], ok_idx[:n_val_ok]], axis=0)
    idx_val = np.unique(idx_val)
    mask = np.ones(n, dtype=bool)
    mask[idx_val] = False
    idx_train = all_idx[mask]
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    return idx_train, idx_val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=os.path.join("data", "gating_dataset.npz"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--step_lr_step", type=int, default=10)
    ap.add_argument("--step_lr_gamma", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val_failed_frac", type=float, default=0.3)
    ap.add_argument("--save", type=str, default=os.path.join("checkpoints", "gating_transformer_best.pth"))
    args = ap.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}. Run collect_gating_data.py first.")

    data = np.load(args.data)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    weight_arr = data["weight"].astype(np.float32) if "weight" in data else np.ones((y.shape[0],), dtype=np.float32)
    bucket = data["bucket"].astype(np.int64) if "bucket" in data else np.zeros((y.shape[0],), dtype=np.int64)
    failed = data["failed"].astype(np.int64) if "failed" in data else np.zeros((y.shape[0],), dtype=np.int64)

    idx_train, idx_val = _split_with_failed_ratio(
        failed=failed,
        seed=int(args.seed),
        train_frac=0.8,
        val_failed_frac=float(args.val_failed_frac),
    )

    X_train = torch.from_numpy(X[idx_train])
    y_train = torch.from_numpy(y[idx_train]).to(dtype=torch.float32)
    w_train = torch.from_numpy(weight_arr[idx_train]).to(dtype=torch.float32)
    X_val = torch.from_numpy(X[idx_val])
    y_val = torch.from_numpy(y[idx_val]).to(dtype=torch.float32)
    w_val = torch.from_numpy(weight_arr[idx_val]).to(dtype=torch.float32)

    dev = torch.device(args.device)
    cfg = GatingTransformerConfig(input_dim=int(X.shape[2]), seq_len=int(X.shape[1]), dropout=float(args.dropout))
    model = GatingTransformer(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=int(args.step_lr_step), gamma=float(args.step_lr_gamma))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train, w_train),
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val, w_val),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for xb, yb, wb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            wb = wb.to(dev)
            pred = model(xb)
            loss = _weighted_mse_loss(pred=pred, targets=yb, sample_weight=wb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * int(xb.shape[0])
            n_seen += int(xb.shape[0])
        train_loss = train_loss / max(1, n_seen)

        model.eval()
        val_loss = 0.0
        n_val = 0
        pred_all: list[torch.Tensor] = []
        y_all: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                wb = wb.to(dev)
                pred = model(xb)
                loss = _weighted_mse_loss(pred=pred, targets=yb, sample_weight=wb)
                val_loss += float(loss.item()) * int(xb.shape[0])
                n_val += int(xb.shape[0])
                pred_all.append(pred.detach().cpu())
                y_all.append(yb.detach().cpu())
        val_loss = val_loss / max(1, n_val)
        pred_cat = torch.cat(pred_all, dim=0)
        y_cat = torch.cat(y_all, dim=0)
        m = _regression_metrics(pred_cat, y_cat)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"mae={m['mae']:.3f} rmse={m['rmse']:.3f} r2={m['r2']:.3f} | "
            f"target_mean={float(torch.mean(y_train).item()):.3f} "
            f"bucket_train={float(np.mean(bucket[idx_train] == 0)):.2f}/"
            f"{float(np.mean(bucket[idx_train] == 1)):.2f}/"
            f"{float(np.mean(bucket[idx_train] == 2)):.2f} "
            f"val_failed_frac={float(np.mean(failed[idx_val])):.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": asdict(cfg),
                    "best_val_loss": float(best_val),
                    "task": "value_regression",
                },
                args.save,
            )

        sch.step()

    print(f"Best | val_loss={best_val:.4f} saved={args.save}")


if __name__ == "__main__":
    main()
