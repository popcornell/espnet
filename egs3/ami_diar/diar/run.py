#!/usr/bin/env python3
"""Run script for AMI diarization recipe.

This script provides a command-line interface for running the diarization pipeline.
Use: source path.sh && ./run.sh
"""

import os
import sys
from pathlib import Path

# Add parent directories to path (recipe dir first so "import src" finds this recipe's src)
recipe_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(recipe_dir))
sys.path.insert(0, str(recipe_dir.parent.parent.parent))  # espnet root

# Import template utilities (load_config, main, run_stages)
sys.path.insert(0, str(recipe_dir.parent.parent / "TEMPLATE" / "asr"))
from run import DEFAULT_STAGES, build_parser, main, parse_cli_and_stage_args

from espnet3.systems.diarization.system import DiarizationSystem


class AMIDiarizationSystem(DiarizationSystem):
    """Diarization system with recipe-specific infer and metric (RTTM + DER)."""

    def train(self, *args, **kwargs):
        """Train; inject XEUS_CHECKPOINT into model.ssl_model_path when using XEUS frontend."""
        if self.train_config is not None and hasattr(self.train_config, "model"):
            model_cfg = self.train_config.model
            if (
                getattr(model_cfg, "ssl_model_name", None) == "xeus"
                and (getattr(model_cfg, "ssl_model_path", None) is None or model_cfg.ssl_model_path == "")
            ):
                env_path = os.environ.get("XEUS_CHECKPOINT")
                if env_path:
                    from omegaconf import OmegaConf
                    OmegaConf.update(model_cfg, "ssl_model_path", env_path)
        return super().train(*args, **kwargs)

    def infer(self, *args, **kwargs):
        self._reject_stage_args("infer", args, kwargs)
        # Ensure recipe dir is first so this recipe's src is used (not TEMPLATE's)
        _recipe = Path(__file__).resolve().parent
        sys.path.insert(0, str(_recipe))
        from src.diar_stages import run_diar_infer
        return run_diar_infer(self)

    def metric(self, *args, **kwargs):
        self._reject_stage_args("metric", args, kwargs)
        _recipe = Path(__file__).resolve().parent
        sys.path.insert(0, str(_recipe))
        from src.diar_stages import run_diar_metric
        return run_diar_metric(self)


if __name__ == "__main__":
    parser = build_parser(DEFAULT_STAGES, default_stages=DEFAULT_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=AMIDiarizationSystem, stages=DEFAULT_STAGES)
