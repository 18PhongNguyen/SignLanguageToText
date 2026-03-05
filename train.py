"""Script huấn luyện Bi-LSTM + CTC cho nhận diện ngôn ngữ ký hiệu.

Chạy:
    python train.py
    python train.py --epochs 200 --batch-size 32
"""
from __future__ import annotations

import argparse
import os
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset, random_split

import config as cfg
from pipeline.model import BiLSTMCTC


# =====================================================================
# DATASET
# =====================================================================
class SignLanguageDataset(Dataset):
    """Đọc các file .npy (features) + labels.csv → sample (features, target)."""

    def __init__(
        self,
        features_dir: str,
        label_file: str,
        char_to_idx: dict[str, int],
    ) -> None:
        super().__init__()
        self.char_to_idx = char_to_idx
        self.samples: list[tuple[str, list[int]]] = []

        df = pd.read_csv(label_file)
        for _, row in df.iterrows():
            npy_path = os.path.join(features_dir, str(row["filename"]))
            text = str(row["text"])
            if not os.path.exists(npy_path):
                logger.warning(f"File không tồn tại, bỏ qua: {npy_path}")
                continue

            indices = self._encode(text)
            if not indices:
                logger.warning(f"Text rỗng sau encode, bỏ qua: {npy_path}")
                continue

            self.samples.append((npy_path, indices))

        logger.info(f"Loaded {len(self.samples)} samples từ {label_file}")

    def _encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text.lower())
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[idx]
        features = np.load(path)  # (T, feature_dim)
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.long),
        )


# =====================================================================
# COLLATE — gộp batch có chiều dài khác nhau
# =====================================================================
def ctc_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad features theo max time trong batch, nối targets thành 1-D.

    Returns:
        (padded_features, all_targets, input_lengths, target_lengths)
    """
    features_list, targets_list = zip(*batch)

    input_lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets_list], dtype=torch.long)

    # Lọc bỏ sample có T < target_length (CTC constraint: T >= L)
    valid_mask = input_lengths >= target_lengths
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum().item()
        logger.warning(f"Bỏ {n_invalid} sample(s) vi phạm CTC constraint (T < L)")
        features_list = tuple(f for f, v in zip(features_list, valid_mask) if v)
        targets_list = tuple(t for t, v in zip(targets_list, valid_mask) if v)
        input_lengths = input_lengths[valid_mask]
        target_lengths = target_lengths[valid_mask]

    if len(features_list) == 0:
        # Trả batch rỗng — training loop sẽ skip
        dummy = torch.zeros(1, 1, features_list[0].size(1) if features_list else 1)
        return dummy, torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)

    max_T = int(input_lengths.max().item())
    feat_dim = features_list[0].size(1)

    padded = torch.zeros(len(features_list), max_T, feat_dim)
    for i, f in enumerate(features_list):
        padded[i, : f.size(0)] = f

    all_targets = torch.cat(targets_list)

    return padded, all_targets, input_lengths, target_lengths


# =====================================================================
# TRAINING LOOP
# =====================================================================
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Dataset ────────────────────────────────────────────────
    dataset = SignLanguageDataset(cfg.FEATURES_DIR, cfg.LABEL_FILE, cfg.CHAR_TO_IDX)
    if len(dataset) == 0:
        logger.error("Không có dữ liệu! Hãy thu thập dữ liệu bằng data_collector.py trước.")
        return

    # Train/val split (80/20)
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    logger.info(f"Train: {n_train} | Val: {n_val}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=0,
    )

    # ── Model ──────────────────────────────────────────────────
    model = BiLSTMCTC(
        feature_dim=cfg.FEATURE_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_classes=cfg.NUM_CLASSES,
        num_layers=cfg.NUM_LSTM_LAYERS,
        dropout=cfg.DROPOUT,
    ).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    ctc_loss = nn.CTCLoss(blank=cfg.BLANK_IDX, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.SCHEDULER_FACTOR,
        patience=cfg.SCHEDULER_PATIENCE,
    )

    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    best_val_loss = float("inf")

    # ── Epochs ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for padded, targets, inp_len, tgt_len in train_loader:
            if padded.size(0) == 0:
                continue

            padded = padded.to(device)
            targets = targets.to(device)

            log_probs = model(padded)          # (N, T, C)
            log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) — CTC yêu cầu time-first

            loss = ctc_loss(log_probs, targets, inp_len, tgt_len)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train = train_loss_sum / max(train_batches, 1)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for padded, targets, inp_len, tgt_len in val_loader:
                if padded.size(0) == 0:
                    continue
                padded = padded.to(device)
                targets = targets.to(device)

                log_probs = model(padded).permute(1, 0, 2)
                loss = ctc_loss(log_probs, targets, inp_len, tgt_len)
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val = val_loss_sum / max(val_batches, 1)
        scheduler.step(avg_val)

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # --- Save best ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), cfg.TRAINED_MODEL_PATH)
            logger.info(f"  ✓ Saved best model (val_loss={avg_val:.4f})")

    logger.info(f"Training hoàn tất. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Model đã lưu tại: {cfg.TRAINED_MODEL_PATH}")


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện Bi-LSTM + CTC")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
