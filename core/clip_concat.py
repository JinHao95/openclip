"""Concatenate multiple video clips into a single compilation video."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def concat_clips(clip_paths: List[str], output_path: str) -> bool:
    """Concatenate clips using ffmpeg concat demuxer (no re-encoding)."""
    if not clip_paths:
        return False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
        list_file = f.name

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        Path(list_file).unlink(missing_ok=True)
        if result.returncode != 0:
            logger.error(f"ffmpeg concat failed: {result.stderr[-500:]}")
            return False
        logger.info(f"Compilation saved to {output_path}")
        return True
    except Exception as e:
        Path(list_file).unlink(missing_ok=True)
        logger.error(f"concat_clips error: {e}")
        return False
