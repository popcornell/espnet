"""Diarization system implementation.

This module adds diarization-specific stages on top of the base system.
"""

import logging
import time
from importlib import import_module
from pathlib import Path

from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


def load_function(path):
    """Load a callable from a dotted module path.

    Args:
        path: Dotted module path (e.g., ``package.module.function``).

    Returns:
        Callable referenced by the path.

    Example:
        >>> fn = load_function("math.sqrt")
        >>> fn(9)
        3.0
    """
    module_path, func_name = path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, func_name)


class DiarizationSystem(BaseSystem):
    """Diarization-specific system.

    This system handles:
      - Dataset creation with lhotse manifests
      - Training diarization models
      - Inference with speaker embeddings
      - DER metric computation
    """

    def train_tokenizer(self, *args, **kwargs):
        """No-op for diarization (no tokenizer)."""
        self._reject_stage_args("train_tokenizer", args, kwargs)
        logger.info("DiarizationSystem.train_tokenizer(): no-op (no tokenizer).")

    def create_dataset(self, *args, **kwargs):
        """Create datasets using the configured helper function.

        The callable is resolved from ``train_config.create_dataset.func`` and
        invoked with the remaining configuration values.

        Raises:
            RuntimeError: If the configuration does not specify a function.
        """
        self._reject_stage_args("create_dataset", args, kwargs)
        logger.info("DiarizationSystem.create_dataset(): starting dataset creation")
        start = time.perf_counter()

        config = getattr(self.train_config, "create_dataset", None)
        if config is None or not getattr(config, "func", None):
            raise RuntimeError(
                "train_config.create_dataset.func must be set to run create_dataset"
            )

        fn = load_function(config.func)
        extra = {k: v for k, v in config.items() if k != "func"}

        logger.info("Creating dataset with function %s", config.func)
        result = fn(**extra)

        logger.info(
            "Dataset creation completed in %.2fs using %s",
            time.perf_counter() - start,
            config.func,
        )
        return result

    def get_stage_log_dir(self, stage: str) -> Path:
        """Return stage-specific log directories when configured.

        The diarization system routes logs to artifact directories:
          - ``create_dataset``: ``train_config.create_dataset.dataset_dir`` or
            ``train_config.dataset_dir`` or ``train_config.data_dir``.
          - ``collect_stats``: ``train_config.stats_dir``.
          - ``train``/``publish``: ``train_config.exp_dir``.
          - ``infer``: ``infer_config.infer_dir``.
          - ``measure``: ``metric_config.infer_dir`` or ``infer_config.infer_dir``.

        Args:
            stage: Stage name being executed.

        Returns:
            Path: Directory where the stage log should be placed.
        """
        if stage == "create_dataset":
            cfg = getattr(self.train_config, "create_dataset", None)
            if cfg is not None:
                dataset_dir = getattr(cfg, "dataset_dir", None)
                if dataset_dir:
                    return Path(dataset_dir)

            dataset_dir = getattr(self.train_config, "dataset_dir", None)
            if dataset_dir:
                return Path(dataset_dir)

            data_dir = getattr(self.train_config, "data_dir", None)
            if data_dir:
                return Path(data_dir)

        elif stage == "collect_stats":
            stats_dir = getattr(self.train_config, "stats_dir", None)
            if stats_dir:
                return Path(stats_dir)

        elif stage in {"train", "publish", "pack_model", "upload_model"}:
            exp_dir = getattr(self.train_config, "exp_dir", None)
            if exp_dir:
                return Path(exp_dir)

        elif stage == "infer":
            infer_dir = getattr(self.infer_config, "infer_dir", None)
            if infer_dir:
                return Path(infer_dir)

        elif stage in {"metric", "measure"}:
            infer_dir = getattr(self.metric_config, "infer_dir", None)
            if infer_dir:
                return Path(infer_dir)
            infer_dir = getattr(self.infer_config, "infer_dir", None)
            if infer_dir:
                return Path(infer_dir)

        return super().get_stage_log_dir(stage)
