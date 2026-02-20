"""Callbacks for ESPnet3 trainer."""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import torch
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from typeguard import typechecked


@typechecked
class AverageCheckpointsCallback(Callback):
    """A custom callback for weight averaging over the top-K checkpoints.

    This can be useful to smooth out fluctuations in weights across the best-performing
    models and can lead to improved generalization performance at inference time.

    Behavior:
        - Loads the state_dict from each of the top-K checkpoints saved by given
          ModelCheckpoint callbacks.
        - Averages the model parameters (keys starting with `model.`).
        - Ignores or simply accumulates integer-type parameters
          (e.g., BatchNorm's `num_batches_tracked`).
        - Saves the averaged model as a `.ckpt` file in `output_dir` (Lightning-style
          with state_dict + hyper_parameters from one source) so the same loader works.

    Args:
        output_dir (str or Path):
            The directory where the averaged model will be saved.
        best_ckpt_callbacks (List[ModelCheckpoint]):
            A list of ModelCheckpoint callbacks whose top-K checkpoints will be used
            for averaging. Each callback must have `best_k_models` populated.

    Notes:
        - Only keys that start with `model.` are included in the averaging.
        - The final filename will be:
            `{monitor_name}.ave_{K}best.ckpt`
        - This callback only runs on the global rank 0 process
            (for distributed training).

    Example:
        >>> avg_ckpt_cb = AverageCheckpointsCallback(
        ...     output_dir="checkpoints/",
        ...     best_ckpt_callbacks=[val_loss_ckpt_cb, acc_ckpt_cb]
        ... )
        >>> trainer = Trainer(callbacks=[avg_ckpt_cb])
    """

    def __init__(self, output_dir, best_ckpt_callbacks):
        """Initialize AverageCheckpointsCallback object."""
        self.output_dir = output_dir
        self.best_ckpt_callbacks = best_ckpt_callbacks

    def on_validation_end(self, trainer, pl_module):
        """At the end of validation, average the top-K checkpoints and save."""
        if trainer.is_global_zero:
            for ckpt_callback in self.best_ckpt_callbacks:
                checkpoints = list(ckpt_callback.best_k_models.keys())
                if not checkpoints:
                    continue

                avg_state_dict = None
                reference_keys = None
                source_ckpt_full = None  # keep first full ckpt for hyper_parameters
                for ckpt_path in checkpoints:
                    ckpt_full = torch.load(
                        ckpt_path, map_location="cpu", weights_only=False
                    )
                    if source_ckpt_full is None:
                        source_ckpt_full = ckpt_full
                    state_dict = ckpt_full

                    # for deepspeed checkpoints
                    if "module" in state_dict:
                        state_dict = state_dict["module"]
                    # for PytorchLightning checkpoints
                    if "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]

                    if avg_state_dict is None:
                        avg_state_dict = state_dict
                        reference_keys = set(state_dict.keys())
                    else:
                        # Check key consistency
                        current_keys = set(state_dict.keys())
                        if current_keys != reference_keys:
                            raise KeyError(
                                f"Mismatch in keys between checkpoints.\n"
                                f"Expected: {reference_keys}\n"
                                f"Got: {current_keys} (from {ckpt_path})"
                            )
                        for k in avg_state_dict:
                            avg_state_dict[k] = avg_state_dict[k] + state_dict[k]

                for k in avg_state_dict:
                    if str(avg_state_dict[k].dtype).startswith("torch.int"):
                        # For int type, not averaged, but only accumulated.
                        # e.g. BatchNorm.num_batches_tracked
                        logging.info(
                            "The following parameters were only accumulated, "
                            f"not averaged: {k}"
                        )
                        pass
                    else:
                        avg_state_dict[k] = avg_state_dict[k] / len(checkpoints)

                # Keep only model keys (same prefix as in source .ckpt for loader compatibility)
                avg_state_dict = {
                    k: v for k, v in avg_state_dict.items() if k.startswith("model.")
                }

                # Save as .ckpt (Lightning-style) so the same inference loader works
                monitor_name = ckpt_callback.monitor.replace("/", ".")
                avg_filename = f"{monitor_name}.ave_{len(checkpoints)}best.ckpt"
                avg_ckpt_path = Path(self.output_dir) / avg_filename
                out = {"state_dict": avg_state_dict}
                if isinstance(source_ckpt_full, dict) and "hyper_parameters" in source_ckpt_full:
                    out["hyper_parameters"] = source_ckpt_full["hyper_parameters"]
                torch.save(out, avg_ckpt_path)


@typechecked
def get_default_callbacks(
    exp_dir: str = "./exp",
    log_interval: int = 500,
    best_model_criterion: Union[List[Tuple[str, int, str]], List[List]] = [
        ("valid/loss", 3, "min")
    ],
) -> List[Callback]:
    """Return a list of callbacks tailored for most training workflows.

    Includes:
        - `ModelCheckpoint` for saving the last model checkpoint (`save_last`)
        - One or more `ModelCheckpoint`s for saving the top-K checkpoints according to
            specific metrics
        - `AverageCheckpointsCallback` to compute and save the average model from top-K
            checkpoints
        - `LearningRateMonitor` to track and log learning rates during training
        - `TQDMProgressBar` to show a rich progress bar during training

    Args:
        exp_dir (str): Directory to store checkpoints and logs.
        log_interval (int): Frequency (in training steps) to refresh the progress bar.
        best_model_criterion (List[Tuple[str, int, str]]): A list of criteria for
            saving top-K checkpoints.
            Each item is a tuple: (name, top_k, mode), where:
            - `name` (str): The name of the validation value to monitor
                (e.g., "val/loss").
            - `top_k` (int): Number of best models to keep.
            - `mode` (str): "min" to keep models with lowest value, "max" for highest.

    Returns:
        List[Callback]: A list of callbacks to be passed to the PyTorch Lightning
            Trainer.

    Example:
        >>> from default_callbacks import get_default_callbacks
        >>> callbacks = get_default_callbacks(
        ...     exp_dir="./exp",
        ...     log_interval=100,
        ...     best_model_criterion=[("val/loss", 5, "min"), ("val/acc", 3, "max")]
        ... )
        >>> trainer = Trainer(callbacks=callbacks, ...)
    """
    last_ckpt_callback = ModelCheckpoint(
        dirpath=exp_dir,
        save_last="link",
        filename="step{step}",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        save_weights_only=False,
    )

    best_ckpt_callbacks = []
    for monitor, nbest, mode in best_model_criterion:
        # Include metric value in filename (e.g. epoch3_step100_valid.loss_0.5134)
        # Use monitor with "/" for placeholder so it matches callback_metrics key
        monitor_safe = monitor.replace("/", ".")
        filename = "epoch{epoch}_step{step}_" + monitor_safe + "_{" + monitor + ":.4f}"
        best_ckpt_callbacks.append(
            ModelCheckpoint(
                save_top_k=nbest,
                monitor=monitor,
                mode=mode,  # "min" or "max"
                dirpath=exp_dir,
                save_last=False,
                filename=filename,
                auto_insert_metric_name=False,
                save_on_train_epoch_end=False,
                save_weights_only=True,
                enable_version_counter=False,  # just overwrite
            )
        )
    ave_ckpt_callback = AverageCheckpointsCallback(
        output_dir=exp_dir, best_ckpt_callbacks=best_ckpt_callbacks
    )

    # Monitor learning rate
    lr_callback = LearningRateMonitor()

    # Progress bar
    progress_bar_callback = TQDMProgressBar(refresh_rate=log_interval)

    return [
        last_ckpt_callback,
        *best_ckpt_callbacks,  # unpack list to add them to the list of callbacks.
        ave_ckpt_callback,
        lr_callback,
        progress_bar_callback,
    ]
