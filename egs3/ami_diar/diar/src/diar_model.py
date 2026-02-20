"""Diarization model wrapper for ESPnet3 Lightning (loss, stats, weight interface)."""

import torch
from typing import Dict, Tuple

from espnet3.components.diarization.segmentation_model import PowersetDiarizationModel


class DiarizationTrainingModel(torch.nn.Module):
    """Wraps PowersetDiarizationModel so that forward(**batch) returns (loss, stats, weight)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.model = PowersetDiarizationModel(**kwargs)

    def forward(
        self,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor,
        labels: torch.Tensor,
        labels_lengths: torch.Tensor,
        **_
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        logits, lengths = self.model(waveform, waveform_lengths)
        loss, stats = self.model.compute_loss(logits, labels, lengths=lengths)
        weight = labels_lengths.sum().float().to(loss.device)
        if weight <= 0:
            weight = torch.tensor(1.0, device=loss.device)
        return loss, stats, weight
