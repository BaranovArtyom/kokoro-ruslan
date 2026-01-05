#!/usr/bin/env python3
"""
Checkpoint management utilities
Compatible with PyTorch 1.13.x and 2.x
"""

import os
import torch
import pickle
from pathlib import Path
from typing import Optional, Tuple
import logging

from config import TrainingConfig
from russian_phoneme_processor import RussianPhonemeProcessor

logger = logging.getLogger(__name__)


# ---------------------------
# Phoneme processor helpers
# ---------------------------

def save_phoneme_processor(processor: RussianPhonemeProcessor, output_dir: str):
    processor_path = os.path.join(output_dir, "phoneme_processor.pkl")
    with open(processor_path, "wb") as f:
        pickle.dump(processor.to_dict(), f)
    logger.info(f"Phoneme processor saved: {processor_path}")


def load_phoneme_processor(output_dir: str) -> RussianPhonemeProcessor:
    processor_path = os.path.join(output_dir, "phoneme_processor.pkl")
    with open(processor_path, "rb") as f:
        processor_data = pickle.load(f)
    processor = RussianPhonemeProcessor.from_dict(processor_data)
    logger.info(f"Phoneme processor loaded: {processor_path}")
    return processor


# ---------------------------
# Checkpoint save
# ---------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    config: TrainingConfig,
    output_dir: str
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "config": config,
    }

    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


# ---------------------------
# Checkpoint load
# ---------------------------

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    output_dir: str,
) -> Tuple[int, float, RussianPhonemeProcessor]:

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # 🔒 Safe globals — ТОЛЬКО если поддерживается (torch >= 2.0)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            TrainingConfig,
            RussianPhonemeProcessor,
        ])

    def _load_and_process_state_dict(ckpt_state, current_model):
        model_keys = current_model.state_dict().keys()
        new_state = {}

        for k, v in ckpt_state.items():
            if k not in model_keys:
                logger.warning(f"Skipping unexpected key in checkpoint: {k}")
                continue

            cur_shape = current_model.state_dict()[k].shape

            if "positional_encoding.pe" in k:
                if v.shape == cur_shape:
                    new_state[k] = v
                elif v.dim() == 3 and v.shape[0] == cur_shape[0] and v.shape[2] == cur_shape[2]:
                    if v.shape[1] > cur_shape[1]:
                        logger.warning(
                            f"Truncating positional encoding {k} from {v.shape} → {cur_shape}"
                        )
                        new_state[k] = v[:, :cur_shape[1], :]
                    else:
                        logger.error(
                            f"Positional encoding too short for {k}: {v.shape} vs {cur_shape}"
                        )
                else:
                    logger.error(
                        f"Skipping incompatible positional encoding {k}: {v.shape} vs {cur_shape}"
                    )
            else:
                if v.shape == cur_shape:
                    new_state[k] = v
                else:
                    logger.error(
                        f"Shape mismatch for {k}: checkpoint {v.shape}, model {cur_shape}"
                    )

        return new_state

    torch_version_major = int(torch.__version__.split(".")[0])

    # ---------------------------
    # PyTorch 2.x path
    # ---------------------------
    if torch_version_major >= 2:
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as e:
            logger.warning(f"weights_only=True failed: {e}")
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
    else:
        # ---------------------------
        # PyTorch 1.13 path
        # ---------------------------
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    # Load model
    filtered_state = _load_and_process_state_dict(
        checkpoint["model_state_dict"], model
    )
    model.load_state_dict(filtered_state, strict=False)

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["loss"]

    if "phoneme_processor" in checkpoint:
        phoneme_processor = checkpoint["phoneme_processor"]
    else:
        phoneme_processor = load_phoneme_processor(output_dir)

    logger.info(f"Resumed from epoch {start_epoch} with loss {best_loss:.4f}")
    return start_epoch, best_loss, phoneme_processor


# ---------------------------
# Find latest checkpoint
# ---------------------------

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    checkpoint_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None

    checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
    latest = checkpoints[-1]
    logger.info(f"Found latest checkpoint: {latest}")
    return str(latest)


# ---------------------------
# Save final model
# ---------------------------

def save_final_model(model: torch.nn.Module, config: TrainingConfig, output_dir: str):
    final_path = os.path.join(output_dir, "kokoro_russian_final.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        final_path,
    )
    logger.info(f"Final model saved: {final_path}")
