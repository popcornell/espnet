#!/usr/bin/env python3
"""Debug training script with XEUS SSL model."""

import sys
from pathlib import Path
import torch
import lightning as L
from torch.utils.data import DataLoader

# Add paths
recipe_dir = Path(__file__).parent
sys.path.insert(0, str(recipe_dir))
sys.path.insert(0, str(recipe_dir.parent.parent.parent))

# Imports
from src.dataset import DiarizationDataset, collate_fn
from espnet3.components.diarization.segmentation_model import PowersetDiarizationModel


class DiarizationLightningModule(L.LightningModule):
    """Lightning module for diarization training."""

    def __init__(self, model_config, optimizer_config, scheduler_config):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model = PowersetDiarizationModel(**model_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

    def forward(self, waveform, waveform_lengths=None):
        return self.model(waveform, waveform_lengths)

    def training_step(self, batch, batch_idx):
        waveform = batch["speech"]
        labels = batch["labels"]
        lengths = batch["speech_lengths"]

        # Forward pass
        logits, out_lengths = self.model(waveform, lengths)

        # Compute loss
        loss, stats = self.model.compute_loss(logits, labels, out_lengths)

        # Log
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/accuracy", stats["accuracy"], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        waveform = batch["speech"]
        labels = batch["labels"]
        lengths = batch["speech_lengths"]

        # Forward pass
        logits, out_lengths = self.model(waveform, lengths)

        # Compute loss
        loss, stats = self.model.compute_loss(logits, labels, out_lengths)

        # Log
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/accuracy", stats["accuracy"], prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # Get SSL parameters vs non-SSL parameters
        ssl_params = []
        non_ssl_params = []

        for name, param in self.model.named_parameters():
            if "ssl_frontend" in name:
                ssl_params.append(param)
            else:
                non_ssl_params.append(param)

        # If SSL is frozen, only optimize non-SSL params
        if self.model.ssl_frontend.freeze or len(ssl_params) == 0:
            optimizer = torch.optim.AdamW(non_ssl_params, **self.optimizer_config)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **self.scheduler_config
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            # Dual optimizer (not used when frozen)
            optimizer = torch.optim.AdamW(non_ssl_params, **self.optimizer_config)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **self.scheduler_config
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }


def main():
    # Config for XEUS
    data_dir = Path("data/ami_debug")
    exp_dir = Path("exp/xeus_debug_run")
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AMI Diarization Training with XEUS SSL")
    print("=" * 80)

    # Model config - XEUS
    # IMPORTANT: You need to download XEUS checkpoint first!
    # Download from: https://huggingface.co/espnet/xeus/tree/main
    # Or set XEUS_CHECKPOINT environment variable to your checkpoint path
    import os
    xeus_checkpoint = os.environ.get("XEUS_CHECKPOINT", None)

    if xeus_checkpoint is None:
        print("\n" + "=" * 80)
        print("ERROR: XEUS checkpoint not found!")
        print("=" * 80)
        print("\nTo use XEUS, you need to:")
        print("1. Download XEUS checkpoint from: https://huggingface.co/espnet/xeus")
        print("2. Set environment variable: export XEUS_CHECKPOINT=/path/to/checkpoint.pth")
        print("\nAlternatively, use WavLM for debugging:")
        print("  python3 train_debug.py")
        print("\n" + "=" * 80)
        sys.exit(1)

    model_config = {
        "ssl_model_name": "xeus",  # XEUS SSL model
        "ssl_model_path": xeus_checkpoint,  # Path to checkpoint
        "ssl_freeze": True,  # Freeze for faster debug
        "ssl_num_layers": None,  # Auto-detect from model
        "ssl_hidden_size": None,  # Auto-detect from model
        "ssl_layer_weights": True,
        "ssl_feature_grad_mult": 1.0,
        "projection_size": 128,
        "conformer_num_blocks": 2,
        "conformer_attention_heads": 4,
        "conformer_ffn_units": 512,
        "conformer_kernel_size": 31,
        "conformer_dropout": 0.1,
        "num_speakers": 4,
        "max_speakers_per_frame": 2,
        "loss_type": "nll",
        "cardinality_weight_type": "uniform",
        "use_pit": True,  # Permutation Invariant Training
    }

    optimizer_config = {
        "lr": 0.001,
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
    }

    scheduler_config = {
        "T_max": 3,
        "eta_min": 0.0001,
    }

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = DiarizationDataset(
        manifest_path=str(data_dir / "train_cuts.jsonl.gz"),
        chunk_duration=4.0,
        chunk_shift=3.0,
        frame_shift=0.02,
        max_speakers=4,
        sample_rate=16000,
    )

    val_dataset = DiarizationDataset(
        manifest_path=str(data_dir / "dev_cuts.jsonl.gz"),
        chunk_duration=None,
        chunk_shift=None,
        frame_shift=0.02,
        max_speakers=4,
        sample_rate=16000,
    )

    print(f"Train dataset: {len(train_dataset)} chunks")
    print(f"Val dataset: {len(val_dataset)} chunks")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Create model
    print("\nCreating model with XEUS...")
    print("  Loading XEUS from espnet/xeus...")
    model = DiarizationLightningModule(model_config, optimizer_config, scheduler_config)

    # Create trainer
    print("\nSetting up trainer...")
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=exp_dir / "checkpoints",
        filename="epoch{epoch:03d}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        gradient_clip_val=5.0,
        val_check_interval=1.0,
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        precision="32",
        limit_train_batches=20,
        limit_val_batches=5,
        callbacks=[checkpoint_callback],
        default_root_dir=exp_dir,
    )

    # Train
    print("\nStarting training with XEUS...")
    print("=" * 80)
    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
