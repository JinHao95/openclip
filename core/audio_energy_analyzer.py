"""
Audio Energy Analyzer
Detects high-energy segments in video audio using EBU R128 momentary loudness.
Used for "Long Video Acceleration" to skip full ASR and only transcribe exciting moments.
"""

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import AUDIO_ENERGY_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class AudioEnergyResult:
    segments: List[Dict[str, Any]] = field(default_factory=list)
    total_duration: float = 0.0
    coverage_ratio: float = 0.0
    analysis_time: float = 0.0
    fell_back_to_full: bool = False


class AudioEnergyAnalyzer:
    """Detects high-energy audio segments via EBU R128 momentary loudness."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = {**AUDIO_ENERGY_CONFIG, **(config or {})}
        self.threshold_k: float = cfg["threshold_k"]
        self.rolling_window_seconds: int = cfg["rolling_window_seconds"]
        self.min_segment_seconds: float = cfg["min_segment_seconds"]
        self.merge_gap_seconds: float = cfg["merge_gap_seconds"]
        self.context_padding_seconds: float = cfg["context_padding_seconds"]
        self.min_coverage_ratio: float = cfg["min_coverage_ratio"]
        self.max_coverage_ratio: float = cfg["max_coverage_ratio"]
        self.fallback_top_n: int = cfg["fallback_top_n"]

    # PLACEHOLDER_ANALYZE_AND_HELPERS

    def _get_duration(self, video_path: str) -> float:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "compact=print_section=0:nokey=1",
            "-show_entries", "format=duration", video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"ffprobe duration failed: {e}")
            return 0.0

    def _extract_loudness(self, video_path: str):
        """Extract EBU R128 momentary loudness time series via ffmpeg."""
        cmd = [
            "ffmpeg", "-i", video_path, "-af",
            "ebur128=metadata=1,ametadata=print:key=lavfi.r128.M",
            "-f", "null", "-",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            stderr = result.stderr
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg loudness extraction timed out")
            return np.array([]), np.array([])
        except Exception as e:
            logger.error(f"ffmpeg loudness extraction failed: {e}")
            return np.array([]), np.array([])

        timestamps = []
        values = []
        ts_pattern = re.compile(r"pts_time:([\d.]+)")
        val_pattern = re.compile(r"lavfi\.r128\.M=(-?[\d.]+)")

        current_ts = None
        for line in stderr.splitlines():
            ts_match = ts_pattern.search(line)
            if ts_match:
                current_ts = float(ts_match.group(1))
            val_match = val_pattern.search(line)
            if val_match and current_ts is not None:
                timestamps.append(current_ts)
                values.append(float(val_match.group(1)))
                current_ts = None

        return np.array(timestamps), np.array(values)

    def _detect_segments(
        self, timestamps, loudness, duration, k_override=None
    ) -> List[Dict[str, Any]]:
        """Identify segments where loudness exceeds adaptive threshold."""
        k = k_override if k_override is not None else self.threshold_k
        window_sec = min(self.rolling_window_seconds, duration / 3)
        window_samples = max(3, int(window_sec / self._sample_interval(timestamps)))

        rolling_mean = np.convolve(loudness, np.ones(window_samples) / window_samples, mode="same")
        rolling_sq = np.convolve(loudness**2, np.ones(window_samples) / window_samples, mode="same")
        rolling_std = np.sqrt(np.maximum(rolling_sq - rolling_mean**2, 0))

        threshold = rolling_mean + k * rolling_std
        above = timestamps[loudness > threshold]

        if len(above) == 0:
            return []

        raw_segments = self._contiguous_ranges(above, self.merge_gap_seconds)
        return self._postprocess(raw_segments, duration)

    def _fallback_top_n(self, timestamps, loudness, duration) -> List[Dict[str, Any]]:
        """Return top N peak regions as fallback."""
        window_samples = max(1, int(self.rolling_window_seconds / self._sample_interval(timestamps)))
        smoothed = np.convolve(loudness, np.ones(window_samples) / window_samples, mode="same")

        segments = []
        used = set()
        for _ in range(self.fallback_top_n):
            mask = np.ones(len(smoothed), dtype=bool)
            for u_start, u_end in used:
                mask &= ~((timestamps >= u_start) & (timestamps <= u_end))
            if not mask.any():
                break
            candidates = np.where(mask)[0]
            peak_idx = candidates[np.argmax(smoothed[candidates])]
            center = timestamps[peak_idx]
            seg_start = max(0, center - self.min_segment_seconds / 2)
            seg_end = min(duration, center + self.min_segment_seconds / 2)
            segments.append({"start": seg_start, "end": seg_end})
            used.add((seg_start, seg_end))

        return self._postprocess(
            [(s["start"], s["end"]) for s in segments], duration
        )

    def _sample_interval(self, timestamps) -> float:
        if len(timestamps) < 2:
            return 0.1
        return float(np.median(np.diff(timestamps)))

    def _contiguous_ranges(self, times, gap) -> List[tuple]:
        """Group timestamps into ranges where consecutive values differ by <= gap."""
        ranges = []
        start = times[0]
        prev = times[0]
        for t in times[1:]:
            if t - prev > gap:
                ranges.append((start, prev))
                start = t
            prev = t
        ranges.append((start, prev))
        return ranges

    def _postprocess(self, raw_ranges, duration) -> List[Dict[str, Any]]:
        """Apply padding, merge, min-duration filter."""
        padded = []
        for start, end in raw_ranges:
            s = max(0, start - self.context_padding_seconds)
            e = min(duration, end + self.context_padding_seconds)
            padded.append((s, e))

        merged = []
        for s, e in sorted(padded):
            if merged and s - merged[-1][1] <= self.merge_gap_seconds:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        segments = []
        for s, e in merged:
            dur = e - s
            if dur >= self.min_segment_seconds:
                segments.append({
                    "start": round(s, 2),
                    "end": round(e, 2),
                    "duration": round(dur, 2),
                    "peak_loudness": 0.0,
                })
        return segments

    def analyze(self, video_path: str) -> AudioEnergyResult:
        """Main entry: detect high-energy segments in video audio."""
        start_time = time.time()
        video_path = str(video_path)

        duration = self._get_duration(video_path)
        if duration <= 0:
            logger.error("Could not determine video duration")
            return AudioEnergyResult(fell_back_to_full=True)

        timestamps, loudness_values = self._extract_loudness(video_path)
        if len(loudness_values) < 10:
            logger.warning("Insufficient loudness data, falling back to full ASR")
            return AudioEnergyResult(total_duration=duration, fell_back_to_full=True)

        segments = self._detect_segments(timestamps, loudness_values, duration)

        coverage = sum(s["duration"] for s in segments) / duration if duration > 0 else 0

        if coverage < self.min_coverage_ratio and segments:
            logger.info(
                f"Coverage {coverage:.1%} below minimum {self.min_coverage_ratio:.0%}, "
                "retrying with lower threshold"
            )
            segments = self._detect_segments(
                timestamps, loudness_values, duration, k_override=self.threshold_k * 0.7
            )
            coverage = sum(s["duration"] for s in segments) / duration if duration > 0 else 0

        if coverage < self.min_coverage_ratio:
            segments = self._fallback_top_n(timestamps, loudness_values, duration)
            coverage = sum(s["duration"] for s in segments) / duration if duration > 0 else 0

        fell_back = coverage > self.max_coverage_ratio
        if fell_back:
            logger.warning(
                f"Coverage {coverage:.1%} exceeds {self.max_coverage_ratio:.0%}, "
                "recommending full ASR"
            )

        elapsed = time.time() - start_time
        logger.info(
            f"🔊 Audio energy analysis complete: {len(segments)} segments, "
            f"coverage {coverage:.1%}, took {elapsed:.1f}s"
        )

        return AudioEnergyResult(
            segments=segments,
            total_duration=duration,
            coverage_ratio=coverage,
            analysis_time=elapsed,
            fell_back_to_full=fell_back,
        )
