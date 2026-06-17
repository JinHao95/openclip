"""Extract keyframes from video segments for vision-based analysis."""

import base64
import subprocess
import tempfile
from pathlib import Path
from typing import List


def extract_keyframes(
    video_path: str, start: float, end: float, count: int = 3, width: int = 480
) -> List[str]:
    """Extract keyframes from a video segment, return as base64 JPEG strings."""
    duration = end - start
    frames_b64 = []

    for i in range(count):
        t = start + duration * (i + 1) / (count + 1)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            cmd = [
                "ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", video_path,
                "-vframes", "1", "-vf", f"scale={width}:-1",
                "-q:v", "8", tmp.name,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0 and Path(tmp.name).stat().st_size > 0:
                frames_b64.append(base64.b64encode(Path(tmp.name).read_bytes()).decode())

    return frames_b64
