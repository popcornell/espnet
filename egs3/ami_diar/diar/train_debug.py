#!/usr/bin/env python3
"""Simple debug training script for diarization."""

import sys
from pathlib import Path
import torch
import lightning as L
from torch.utils.data import DataLoader

# Add paths
recipe_dir = Path(__file__).parent
sys.path.insert(0, str(recipe_dir))
sys.path.insert(0, str(recipe_dir.parent.parent.parent))  # espnet root

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

        # If SSL is frozen, no need for SSL optimizer
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
            # Dual optimizer (not used in debug mode since SSL is frozen)
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
    # Config (matching debug.yaml)
    data_dir = Path("data/ami_debug")
    exp_dir = Path("exp/debug_run")
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("AMI Diarization Debug Training")
    print("=" * 80)

    # Model config
    model_config = {
        "ssl_model_name": "wavlm_base",
        "ssl_freeze": True,
        "ssl_num_layers": 13,
        "ssl_hidden_size": 768,
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
        chunk_duration=None,  # Whole recordings
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

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
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
        accelerator="cpu",
        devices=1,
        gradient_clip_val=5.0,
        val_check_interval=1.0,
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        precision="16-mixed",
        limit_train_batches=200,  # Only 20 batches per epoch
        limit_val_batches=50,     # Only 5 val batches
        callbacks=[checkpoint_callback],
        default_root_dir=exp_dir,
    )

    # Train
    print("\nStarting training...")
    print("=" * 80)
    trainer.fit(model, train_loader, val_loader)

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
