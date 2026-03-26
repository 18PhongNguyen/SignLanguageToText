"""Wrapper chạy 3 experiment liên tiếp và ghi kết quả vào experiments/results.csv.

Experiment A — SplitConformer, không aux loss:
    python train.py --model conformer --epochs 150 --no-resume

Experiment B — SplitConformer + aux loss:
    python train.py --model conformer --aux-loss --epochs 150 --no-resume

Experiment C — BiLSTM baseline (control):
    python train.py --model bilstm --epochs 150 --no-resume

Sau mỗi experiment, đọc best checkpoint và ghi vào experiments/results.csv:
    timestamp, model_type, aux_loss, best_val_wer, best_val_acc, epochs_trained

Chạy:
    python train_experiment.py
    python train_experiment.py --epochs 150 --batch-size 16
    python train_experiment.py --only A        # chỉ chạy experiment A
    python train_experiment.py --only A B      # chạy A và B
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR  = PROJECT_ROOT / "experiments"
RESULTS_CSV  = RESULTS_DIR / "results.csv"
SLTT_DIR     = PROJECT_ROOT / "models" / "sltt"

EXPERIMENTS: dict[str, dict] = {
    "A": {
        "label":     "Conformer (no aux)",
        "model":     "conformer",
        "aux_loss":  False,
        "ckpt_name": "conformer_ctc.pt",
    },
    "B": {
        "label":    "Conformer + aux loss",
        "model":    "conformer",
        "aux_loss": True,
        "ckpt_name": "conformer_ctc.pt",   # sẽ overwrite A nếu chạy cùng lúc
    },
    "C": {
        "label":    "BiLSTM baseline",
        "model":    "bilstm",
        "aux_loss": False,
        "ckpt_name": "bilstm_ctc.pt",
    },
}


def _build_cmd(exp: dict, epochs: int, batch_size: int) -> list[str]:
    """Xây dựng command cho một experiment."""
    cmd = [
        sys.executable, "train.py",
        "--model",      exp["model"],
        "--epochs",     str(epochs),
        "--batch-size", str(batch_size),
        "--no-resume",
    ]
    if exp["aux_loss"]:
        cmd.append("--aux-loss")
    return cmd


def _read_best_from_checkpoint(ckpt_name: str) -> dict:
    """Đọc best_val_wer và thông tin từ checkpoint sau khi train xong."""
    ckpt_path = SLTT_DIR / ckpt_name
    result = {"best_val_wer": "N/A", "best_val_acc": "N/A", "epochs_trained": "N/A"}
    if not ckpt_path.exists():
        return result
    try:
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            wer = raw.get("val_wer")
            epoch = raw.get("epoch")
            if wer is not None:
                result["best_val_wer"] = f"{wer:.4f}"
            if epoch is not None:
                result["epochs_trained"] = str(epoch)
    except Exception as e:
        print(f"  [WARN] Không đọc được checkpoint {ckpt_name}: {e}")
    return result


def _write_csv_row(row: dict) -> None:
    """Ghi một dòng vào results.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_CSV.exists()
    fieldnames = [
        "timestamp", "experiment", "label",
        "model_type", "aux_loss",
        "best_val_wer", "best_val_acc", "epochs_trained",
        "duration_min",
    ]
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  [CSV] Đã ghi kết quả vào {RESULTS_CSV}")


def run_experiment(
    exp_id: str,
    exp: dict,
    epochs: int,
    batch_size: int,
) -> None:
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT {exp_id}: {exp['label']}")
    print(f"  model={exp['model']}  aux_loss={exp['aux_loss']}  epochs={epochs}")
    print(f"{'='*60}\n")

    cmd = _build_cmd(exp, epochs, batch_size)
    print(f"  CMD: {' '.join(cmd)}\n")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    duration_min = (time.time() - t0) / 60.0

    if result.returncode != 0:
        print(f"\n  [ERROR] Experiment {exp_id} thất bại (returncode={result.returncode})")

    # Đọc kết quả từ checkpoint
    ckpt_info = _read_best_from_checkpoint(exp["ckpt_name"])

    row = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment":     exp_id,
        "label":          exp["label"],
        "model_type":     exp["model"],
        "aux_loss":       str(exp["aux_loss"]),
        "best_val_wer":   ckpt_info["best_val_wer"],
        "best_val_acc":   ckpt_info["best_val_acc"],
        "epochs_trained": ckpt_info["epochs_trained"],
        "duration_min":   f"{duration_min:.1f}",
    }
    _write_csv_row(row)
    print(f"\n  Kết quả: WER={row['best_val_wer']}  epochs={row['epochs_trained']}  "
          f"time={row['duration_min']} phút")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chạy experiment A/B/C và ghi kết quả vào experiments/results.csv"
    )
    parser.add_argument("--epochs", type=int, default=150,
                        help="Số epoch cho mỗi experiment (mặc định: 150)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (mặc định: 16)")
    parser.add_argument(
        "--only", nargs="+", choices=list(EXPERIMENTS.keys()),
        metavar="ID",
        help="Chỉ chạy subset experiment, vd: --only A C"
    )
    args = parser.parse_args()

    to_run = args.only if args.only else list(EXPERIMENTS.keys())

    print(f"\nSẽ chạy experiments: {to_run}")
    print(f"Epochs={args.epochs}  BatchSize={args.batch_size}")
    print(f"Kết quả sẽ lưu vào: {RESULTS_CSV}\n")

    for exp_id in to_run:
        run_experiment(exp_id, EXPERIMENTS[exp_id], args.epochs, args.batch_size)

    print(f"\n{'='*60}")
    print(f"  HOÀN TẤT tất cả experiment.")
    print(f"  Xem kết quả: {RESULTS_CSV}")
    print(f"{'='*60}\n")

    # In bảng kết quả nếu có
    if RESULTS_CSV.exists():
        try:
            with open(RESULTS_CSV, encoding="utf-8") as f:
                print(f.read())
        except Exception:
            pass


if __name__ == "__main__":
    main()
