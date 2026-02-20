"""Symlink AMI IHM lhotse manifests for the recipe."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def symlink_ami_manifests(source_dir: str, output_dir: str) -> None:
    source_path = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for link_name, target_name in [
        ("train_cuts.jsonl.gz", "ami-ihm_cuts_train.jsonl.gz"),
        ("dev_cuts.jsonl.gz", "ami-ihm_cuts_dev.jsonl.gz"),
        ("test_cuts.jsonl.gz", "ami-ihm_cuts_test.jsonl.gz"),
    ]:
        src_file = source_path / target_name
        link_file = out_path / link_name
        if not src_file.exists():
            raise FileNotFoundError("AMI manifest not found: " + str(src_file))
        if link_file.exists():
            link_file.unlink()
        link_file.symlink_to(src_file.resolve())
        logger.info("Symlinked %s -> %s", link_file, src_file)
