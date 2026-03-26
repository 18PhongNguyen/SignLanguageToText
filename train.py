"""Script huấn luyện Bi-LSTM + CTC cho nhận diện ngôn ngữ ký hiệu.

Chế độ CTC word-level: mỗi nhãn (ví dụ "bạn khỏe không") được tách
thành chuỗi word indices [idx("bạn"), idx("khỏe"), idx("không")].
Model output per-frame log-probs, CTC tự căn chỉnh chuỗi.

Chạy:
    python train.py
    python train.py --epochs 200 --batch-size 32
"""
from __future__ import annotations

import argparse
import math
import os
import random
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

import config as cfg
from pipeline.model import BiLSTMCTC
from pipeline.decoder import decode_to_text
from vocab import build_vocab, load_vocab, vocab_to_dicts, text_to_word_indices

# Model conformer — import trễ để không bắt buộc khi dùng bilstm
_CONFORMER_AVAILABLE = True
try:
    from pipeline.model_conformer import SplitConformerCTC
except ImportError:
    _CONFORMER_AVAILABLE = False


# =====================================================================
# FOCAL CTC LOSS — tăng trọng số cho token khó, giảm cho token dễ
# =====================================================================
class FocalCTCLoss(nn.Module):
    """CTC Loss với focal weighting theo token rarity.

    Ý tưởng: token hiếm (chỉ xuất hiện trong 1-2 phrases) cần được model
    học kỹ hơn. Thay vì đặt weight trên loss toàn batch, ta scale loss
    theo mức "hiếm" trung bình của target tokens trong mỗi sample.

    weight_i = (max_freq / freq_i) ^ alpha
    sample_weight = mean(weight cho từng token trong target)

    alpha=0.5: căn bậc hai — không quá mạnh, đủ để đẩy rare token lên.
    """

    def __init__(
        self,
        blank: int = 0,
        token_counts: dict[int, int] | None = None,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)
        self.alpha = alpha

        # Tính weight cho mỗi token dựa trên tần suất xuất hiện
        if token_counts:
            max_count = max(token_counts.values())
            self.token_weights: dict[int, float] = {
                idx: (max_count / count) ** alpha
                for idx, count in token_counts.items()
            }
            logger.info(f"[FocalCTC] Token weights (alpha={alpha}):")
            for idx in sorted(self.token_weights, key=self.token_weights.get, reverse=True):
                logger.info(f"  token {idx:2d}: count={token_counts[idx]:5d}  weight={self.token_weights[idx]:.3f}")
        else:
            self.token_weights = {}

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        # CTC loss per sample (reduction="none")
        losses = self.ctc(log_probs, targets, input_lengths, target_lengths)  # (B,)

        if not self.token_weights:
            return losses.mean()

        # Tính weight cho mỗi sample dựa trên rare-token score
        device = losses.device
        B = losses.size(0)
        weights = torch.ones(B, device=device)
        offset = 0
        for i in range(B):
            tlen = int(target_lengths[i].item())
            tgt = targets[offset:offset + tlen].tolist()
            offset += tlen
            if tgt:
                w = sum(self.token_weights.get(t, 1.0) for t in tgt) / len(tgt)
                weights[i] = w

        # Weighted mean loss
        return (losses * weights).sum() / weights.sum()


# =====================================================================
# DATA AUGMENTATION
# =====================================================================
def augment_batch(
    padded: torch.Tensor,
    input_lengths: torch.Tensor,
) -> torch.Tensor:
    """Augment batch features — áp dụng trong training only.

    Kỹ thuật:
    1. Gaussian noise  — thêm nhiễu nhỏ vào toàn bộ frame
    2. Feature masking — zero out một số chiều đặc trưng liên tiếp
    3. Time masking    — zero out một đoạn frame liên tiếp
    """
    B, _T, D = padded.shape
    aug = padded.clone()

    for i in range(B):
        L = int(input_lengths[i].item())

        # 1. Gaussian noise (50%, σ=0.01)
        if random.random() < 0.5:
            aug[i, :L] += torch.randn(L, D, device=padded.device) * 0.01

        # 2. Feature masking: zero out 5% feature dims (50%)
        if random.random() < 0.5:
            n_feat = max(1, int(D * 0.05))
            f_start = random.randint(0, D - n_feat)
            aug[i, :L, f_start : f_start + n_feat] = 0.0

        # 3. Time masking: zero out ≤12.5% frames (50%)
        if random.random() < 0.5 and L > 20:
            n_time = random.randint(1, max(1, L // 8))
            t_start = random.randint(0, L - n_time)
            aug[i, t_start : t_start + n_time, :] = 0.0

    return aug


# =====================================================================
# DATASET
# =====================================================================
class SignLanguageDataset(Dataset):
    """Đọc các file .npy (features) + labels.csv → sample (features, word_indices)."""

    def __init__(
        self,
        features_dir: str,
        label_file: str,
        word2idx: dict[str, int],
    ) -> None:
        super().__init__()
        self.word2idx = word2idx
        self.samples: list[tuple[str, list[int]]] = []

        df = pd.read_csv(label_file)
        for _, row in df.iterrows():
            npy_path = os.path.join(features_dir, str(row["filename"]))
            text = str(row["text"])
            if not os.path.exists(npy_path):
                logger.warning(f"File không tồn tại, bỏ qua: {npy_path}")
                continue

            indices = text_to_word_indices(text, word2idx)
            if not indices:
                logger.warning(f"Text không encode được, bỏ qua: {npy_path} → '{text}'")
                continue

            self.samples.append((npy_path, indices))

        logger.info(f"Loaded {len(self.samples)} samples từ {label_file}")

        # CTC constraint: T >= 2*L - 1 (tối thiểu để CTC loss tính được)
        valid = []
        for path, indices in self.samples:
            T = np.load(path, mmap_mode='r').shape[0]
            if T >= 2 * len(indices) - 1:
                valid.append((path, indices))
            else:
                logger.warning(f"Removed (CTC constraint violated): {path} T={T} < {2*len(indices)-1}")
        self.samples = valid
        logger.info(f"After CTC filter: {len(self.samples)} valid samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target_indices = self.samples[idx]
        features = np.load(path)  # (T, raw_feature_dim)

        # Velocity augmentation: nối delta liên frame → (T, D*2)
        if cfg.USE_VELOCITY:
            from pipeline.extractor import augment_sequence_with_velocity
            features = augment_sequence_with_velocity(features)

        # Tương thích ngược: slice/pad nếu feature_dim khác config
        expected = cfg.FEATURE_DIM
        if features.shape[1] > expected:
            features = features[:, :expected]
        elif features.shape[1] < expected:
            pad = np.zeros((features.shape[0], expected - features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, pad], axis=1)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(target_indices, dtype=torch.long),
        )


# =====================================================================
# COLLATE — gộp batch cho CTC
# =====================================================================
def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad features và targets cho CTC loss.

    Returns:
        (padded_features, targets_cat, input_lengths, target_lengths)
    """
    features_list, targets_list = zip(*batch)

    # Input lengths (thực tế, trước khi pad)
    input_lengths = torch.tensor([f.size(0) for f in features_list], dtype=torch.long)

    # Pad features → (batch, max_T, feat_dim)
    max_T = max(f.size(0) for f in features_list)
    feat_dim = features_list[0].size(1)
    padded = torch.zeros(len(features_list), max_T, feat_dim)
    for i, f in enumerate(features_list):
        padded[i, : f.size(0)] = f

    # Concatenate targets + lengths cho CTC
    target_lengths = torch.tensor([t.size(0) for t in targets_list], dtype=torch.long)
    targets_cat = torch.cat(targets_list)  # 1-D tensor

    return padded, targets_cat, input_lengths, target_lengths


# =====================================================================
# TRAINING LOOP
# =====================================================================
def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Build vocab trước khi train ───────────────────────────
    vocab = build_vocab()
    word2idx, idx2word = vocab_to_dicts(vocab)
    num_classes = len(vocab)
    logger.info(f"Vocab: {num_classes} tokens (1 blank + {num_classes - 1} words)")
    logger.info(f"Words: {vocab[1:]}")

    # ── Dataset ────────────────────────────────────────────────
    dataset = SignLanguageDataset(cfg.FEATURES_DIR, cfg.LABEL_FILE, word2idx)
    if len(dataset) == 0:
        logger.error("Không có dữ liệu! Hãy thu thập dữ liệu bằng data_collector.py trước.")
        return

    # Stratified train/val split (80/20 per label)
    label_to_indices: dict[tuple, list[int]] = defaultdict(list)
    for i, (_, tgt) in enumerate(dataset.samples):
        label_to_indices[tuple(tgt)].append(i)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for tgt_key, idxs in label_to_indices.items():
        random.shuffle(idxs)
        n_v = max(1, int(len(idxs) * 0.2))
        val_indices.extend(idxs[:n_v])
        train_indices.extend(idxs[n_v:])

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    logger.info(f"Train: {len(train_indices)} | Val: {len(val_indices)} (stratified)")
    logger.info("Val distribution per label:")
    for tgt_key, idxs in sorted(label_to_indices.items(), key=lambda x: -len(x[1])):
        label_str = " ".join(idx2word.get(j, "?") for j in tgt_key)
        n_v = max(1, int(len(idxs) * 0.2))
        logger.info(f"  {label_str!r:40s}  total={len(idxs):4d}  val={n_v:3d}  train={len(idxs)-n_v:4d}")

    # Weighted sampler: token-aware — phrase chứa rare token được sample nhiều hơn
    # Bước 1: Đếm tần suất mỗi token TRÊN TOÀN dataset (không chỉ phrase count)
    token_counter: Counter[int] = Counter()
    for i in train_indices:
        _, tgt = dataset.samples[i]
        token_counter.update(tgt)
    logger.info(f"Token frequency in train set: {dict(token_counter.most_common())}")

    # Bước 2: Score mỗi sample = mean(1/freq cho mỗi token) → phrase chứa
    # rare token ("quê", "ở", "đâu") có score cao hơn nhiều
    max_token_count = max(token_counter.values()) if token_counter else 1
    sample_weights_list: list[float] = []
    for i in train_indices:
        _, tgt = dataset.samples[i]
        # Rare-token score: token càng hiếm → score càng cao
        token_scores = [
            (max_token_count / token_counter.get(t, 1)) ** 0.5
            for t in tgt
        ]
        # Mean score cho sample — phrase có nhiều rare token được boost nhiều hơn
        sample_weights_list.append(sum(token_scores) / max(len(token_scores), 1))

    sample_weights = torch.tensor(sample_weights_list, dtype=torch.float32)
    logger.info(f"[SAMPLER] Token-aware weights: min={sample_weights.min():.3f} "
                f"max={sample_weights.max():.3f} mean={sample_weights.mean():.3f}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(sample_weights, len(sample_weights)),
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # ── Chọn model theo flag --model ────────────────────────────
    # Checkpoint path: bilstm dùng path gốc, conformer dùng path riêng
    if args.model == "conformer":
        if not _CONFORMER_AVAILABLE:
            logger.error("pipeline/model_conformer.py không tồn tại! Hãy tạo file trước.")
            return
        model = SplitConformerCTC(
            feature_dim=cfg.FEATURE_DIM,
            num_classes=num_classes,
            use_aux_loss=args.aux_loss,
        ).to(device)
        _ckpt_path = cfg.CONFORMER_MODEL_PATH  # models/sltt/conformer/conformer_ctc.pt
        logger.info(
            f"[MODEL] SplitConformerCTC | params={sum(p.numel() for p in model.parameters()):,} "
            f"| aux_loss={args.aux_loss}"
        )
    else:
        model = BiLSTMCTC(
            feature_dim=cfg.FEATURE_DIM,
            hidden_dim=cfg.HIDDEN_DIM,
            num_classes=num_classes,
            num_layers=cfg.NUM_LSTM_LAYERS,
            dropout=cfg.DROPOUT,
            input_dropout=cfg.INPUT_DROPOUT,
        ).to(device)
        _ckpt_path = cfg.TRAINED_MODEL_PATH
        logger.info(f"[MODEL] BiLSTMCTC | params={sum(p.numel() for p in model.parameters()):,}")

    os.makedirs(cfg.BILSTM_DIR if args.model == "bilstm" else cfg.CONFORMER_DIR, exist_ok=True)
    best_val_wer = float("inf")
    best_val_loss = float("inf")
    no_improve_count = 0

    # ── Auto-resume from checkpoint ─────────────────────────────
    if not args.no_resume and os.path.exists(_ckpt_path):
        try:
            ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
            # Hỗ trợ cả checkpoint dict mới lẫn state_dict thuần (backward compat)
            state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            model.load_state_dict(state)
            logger.info(f"[RESUME] Tìm thấy model cũ, tiếp tục train từ {_ckpt_path}")
        except RuntimeError as e:
            logger.warning(f"[RESUME] Checkpoint không tương thích (kiến trúc thay đổi), train từ đầu: {e}")
    elif args.no_resume:
        logger.info("[RESUME] Bỏ qua checkpoint, bắt đầu train từ đầu (--no-resume).")
    else:
        logger.info("[RESUME] Chưa có checkpoint, bắt đầu train mới.")

    # ── Criterion / Optimizer / Scheduler ──────────────────────
    criterion = FocalCTCLoss(
        blank=0,
        token_counts=dict(token_counter),
        alpha=0.5,
    )
    val_criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    logger.info("[LOSS] FocalCTCLoss (alpha=0.5) — rare token upweighted")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # ── Helper: word-level Levenshtein distance ─────────────────
    def _levenshtein(a: list, b: list) -> int:
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                prev, dp[j] = dp[j], prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
        return dp[n]

    # ── Epochs ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for padded, targets, input_lengths, target_lengths in train_loader:
            padded = padded.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            # Augmentation (training only)
            padded = augment_batch(padded, input_lengths)

            # ── Forward pass (conformer hỗ trợ aux_loss) ─────────
            if args.model == "conformer" and args.aux_loss:
                logits, aux_logits = model(padded, training=True)  # (B,T,C), (B,C)
            elif args.model == "conformer":
                logits = model(padded, training=True)               # (B,T,C)
            else:
                logits = model(padded)                              # (B,T,C)

            log_probs = F.log_softmax(logits, dim=-1)     # (B, T, C)
            log_probs = log_probs.permute(1, 0, 2)        # (T, B, C) — CTC format

            ctc = criterion(log_probs, targets, input_lengths, target_lengths)

            # Aux loss: CE trên từ đầu tiên của mỗi sample làm proxy label
            if args.model == "conformer" and args.aux_loss:
                # Lấy word index đầu tiên của từng sample từ targets flat
                first_words: list[int] = []
                _offset = 0
                for _i in range(len(target_lengths)):
                    first_words.append(int(targets[_offset].item()))
                    _offset += int(target_lengths[_i].item())
                first_word_tensor = torch.tensor(first_words, dtype=torch.long, device=device)
                aux_loss_val = F.cross_entropy(aux_logits, first_word_tensor)
                loss = ctc + 0.3 * aux_loss_val
            else:
                loss = ctc

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
        val_edit_dist = 0
        val_ref_words = 0
        val_correct = 0
        val_total = 0
        sample_pairs: list[tuple[str, str]] = []  # (pred, ref) để in mẫu
        with torch.no_grad():
            for padded, targets, input_lengths, target_lengths in val_loader:
                padded = padded.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                logits = model(padded) if args.model != "conformer" else model(padded, training=False)
                # Với conformer + aux_loss: model trả về tuple, lấy logits đầu
                if isinstance(logits, tuple):
                    logits = logits[0]
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs_t = log_probs.permute(1, 0, 2)

                loss = val_criterion(log_probs_t, targets, input_lengths, target_lengths)
                val_loss_sum += loss.item()
                val_batches += 1

                # Beam search decode để tính WER và Acc
                offset = 0
                for i in range(padded.size(0)):
                    tgt_len = target_lengths[i].item()
                    tgt = targets[offset:offset + tgt_len].tolist()
                    offset += tgt_len

                    # Chuyển target indices → words (tránh re-map ngược dễ lỗi Unicode)
                    ref_words = [idx2word.get(j, "") for j in tgt]

                    results = decode_to_text(
                        logits[i].unsqueeze(0).cpu(),
                        idx2word,
                        blank_idx=0,
                        beam_width=5,    # beam nhỏ cho train (nhanh hơn)
                        min_frames=0,    # không lọc frame khi validation
                        confidence_threshold=0.0,  # không lọc confidence khi validation
                    )
                    decoded_str = results[0][0] if results and results[0][0] else ""
                    # Lowercase khi so sánh vì normalize_vietnamese() viết hoa chữ đầu
                    # nhưng vocab/ref_words đều lowercase
                    decoded_words = decoded_str.lower().split() if decoded_str else []

                    # WER: so sánh word strings trực tiếp (không re-map indices)
                    val_edit_dist += _levenshtein(decoded_words, ref_words)
                    val_ref_words += max(len(ref_words), 1)

                    # Exact match (val_acc)
                    val_correct += int(decoded_words == ref_words)
                    val_total += 1

                    # Lưu 1 mẫu cho mỗi nhãn khác nhau (tối đa 5 nhãn)
                    ref_str = " ".join(ref_words)
                    if len(sample_pairs) < 5 and ref_str not in {r for _, r in sample_pairs}:
                        sample_pairs.append((decoded_str or "<empty>", ref_str))

        avg_val = val_loss_sum / max(val_batches, 1)
        val_wer = val_edit_dist / max(val_ref_words, 1)
        val_acc = val_correct / max(val_total, 1)

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
            f"Val WER: {val_wer:.1%} | Val Acc: {val_acc:.1%} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # In mẫu dự đoán mỗi 10 epoch (và epoch đầu tiên)
        if epoch % 10 == 0 or epoch == 1:
            logger.info("  ── Sample predictions ──")
            for pred, ref in sample_pairs:
                match = "✓" if pred == ref else "✗"
                logger.info(f"    [{match}] REF:  {ref}")
                logger.info(f"         PRED: {pred}")
            logger.info("  ────────────────────────")

        # --- Save best / Early stopping ---
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            best_val_loss = avg_val
            no_improve_count = 0
            # Lưu checkpoint dict đầy đủ để recognizer.py auto-detect model_type
            checkpoint = {
                "model_state": model.state_dict(),
                "model_type": args.model,           # 'bilstm' hoặc 'conformer'
                "use_aux_loss": getattr(args, "aux_loss", False),
                "num_classes": num_classes,
                "feature_dim": cfg.FEATURE_DIM,
                "val_wer": val_wer,
                "epoch": epoch,
            }
            torch.save(checkpoint, _ckpt_path)
            logger.info(f"  ✓ Saved best model (val_wer={val_wer:.1%}, val_loss={avg_val:.4f}) → {_ckpt_path}")
        else:
            no_improve_count += 1
            if args.patience > 0 and no_improve_count >= args.patience:
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(no WER improvement for {args.patience} epochs)"
                )
                break

        scheduler.step()  # CosineAnnealingLR: step each epoch

    logger.info(f"Training hoàn tất. Best val WER: {best_val_wer:.1%} | Best val loss: {best_val_loss:.4f}")
    logger.info(f"Model đã lưu tại: {_ckpt_path}")


# =====================================================================
# MAIN
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện Bi-LSTM + CTC / SplitConformer CTC")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE)
    parser.add_argument("--no-resume", action="store_true",
                        help="Bắt đầu train từ đầu, bỏ qua checkpoint cũ (mặc định: tự động load nếu có)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Số epoch không cải thiện val WER trước khi dừng sớm (0 = tắt)")
    parser.add_argument(
        "--model", type=str, default="bilstm", choices=["bilstm", "conformer"],
        help="Kiến trúc model: 'bilstm' (mặc định) hoặc 'conformer' (SplitConformerCTC)"
    )
    parser.add_argument(
        "--aux-loss", action="store_true",
        help="Bật auxiliary loss head cho conformer (CE trên token đầu tiên). Chỉ dùng với --model conformer"
    )
    args = parser.parse_args()

    if args.aux_loss and args.model != "conformer":
        parser.error("--aux-loss chỉ dùng được khi --model conformer")

    train(args)


if __name__ == "__main__":
    main()
