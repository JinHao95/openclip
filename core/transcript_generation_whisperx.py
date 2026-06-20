#!/usr/bin/env python3
"""
WhisperX Transcript Generation with Speaker Diarization

Supports two scenarios:
  1. No existing transcript: full transcription + optional diarization pipeline
  2. Existing transcript: diarization-only to add speaker labels
"""

import os
import re
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)

# pyannote warns when torchcodec is absent (excluded on macOS ARM / Linux aarch64
# because its dylib is incompatible with the CPU PyTorch wheel there).
# The warning fires lazily on first transcription, not at import time, so a
# persistent filter is needed. Audio loading falls back to in-memory waveforms
# automatically — no action needed.
warnings.filterwarnings(
    "ignore",
    message=r"\ntorchcodec is not installed correctly",
    category=UserWarning,
)

try:
    import whisperx
    from whisperx.diarize import DiarizationPipeline as WhisperXDiarizationPipeline
    import torch
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    logger.debug("WhisperX not installed. Install with: uv sync --extra speakers")

try:
    from core.speaker_identification import SpeakerIdentifier
except ImportError:
    SpeakerIdentifier = None  # type: ignore


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def _srt_time_to_seconds(timestamp: str) -> float:
    """Convert SRT timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm) to seconds."""
    timestamp = timestamp.replace(",", ".")
    h, m, rest = timestamp.split(":")
    s, ms = rest.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── Language helpers ──────────────────────────────────────────────────────────

# Languages where NLTK punkt tokenizer doesn't work (no spaces between words).
# WhisperX uses punkt for sentence splitting, so a smaller chunk_size produces
# finer-grained segments for these scripts.
_CJK_LANGUAGES = frozenset({"zh", "ja", "ko"})
_CHUNK_SIZE_CJK = 5
_CHUNK_SIZE_DEFAULT = 30
_WHISPERX_SAMPLE_RATE = 16000  # whisperx.load_audio always resamples to 16 kHz


def _chunk_size_for(language: str) -> int:
    return _CHUNK_SIZE_CJK if language in _CJK_LANGUAGES else _CHUNK_SIZE_DEFAULT


# ── Main class ────────────────────────────────────────────────────────────────

class TranscriptProcessorWhisperX:
    """Handles WhisperX-based transcription and speaker diarization."""

    def __init__(
        self,
        whisper_model: str,
        enable_diarization: bool = False,
        hf_token: Optional[str] = None,
        speaker_references_dir: Optional[str] = None,
    ):
        if not WHISPERX_AVAILABLE:
            raise ImportError(
                "WhisperX not installed. Run: uv sync --extra speakers"
            )

        self.whisper_model = whisper_model
        self.enable_diarization = enable_diarization
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")

        if self.enable_diarization and not self.hf_token:
            raise ValueError(
                "Speaker diarization requires HUGGINGFACE_TOKEN. "
                "Set the env var or pass hf_token=. "
                "Also accept the model agreement at: "
                "https://huggingface.co/pyannote/speaker-diarization-community-1"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        logger.info(f"⚡ WhisperX device: {self.device} (compute_type: {self.compute_type})")

        self.speaker_identifier = None
        if speaker_references_dir and SpeakerIdentifier is not None:
            self.speaker_identifier = SpeakerIdentifier(speaker_references_dir)
            logger.info(f"🎙️  Speaker name mapping enabled (references: {speaker_references_dir})")

    # ── Scenario 1: Full pipeline ─────────────────────────────────────────────

    async def transcribe_with_whisperx(
        self,
        video_path: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> str:
        """
        Scenario 1: Transcribe video with WhisperX + optionally diarize.
        Returns path to output SRT file.
        """
        video_path = Path(video_path)
        output_srt = video_path.parent / f"{video_path.stem}.srt"

        if callback:
            callback("Loading audio...", 0)
        logger.info(f"⚡ WhisperX: Loading audio from {video_path.name}")
        audio = whisperx.load_audio(str(video_path))

        # Step 1: Transcribe
        if callback:
            callback("Transcribing with WhisperX...", 10)
        logger.info(f"⚡ WhisperX: Transcribing (model: {self.whisper_model})")
        import time as _time
        _t0 = _time.time()
        model = whisperx.load_model(self.whisper_model, self.device, compute_type=self.compute_type)
        logger.info(f"⚡ WhisperX: Model loaded in {_time.time() - _t0:.1f}s")

        # Detect language on first 30 s before the full transcription so we can
        # pass the right chunk_size: CJK scripts need smaller chunks (5 s) because
        # NLTK punkt tokenizer can't split sentences without spaces.
        audio_30s = audio[:30 * _WHISPERX_SAMPLE_RATE]
        detected_language = model.detect_language(audio_30s)  # returns str in this version
        chunk_size = _chunk_size_for(detected_language)
        logger.info(f"⚡ WhisperX: Detected language: {detected_language} → chunk_size={chunk_size}")

        from core.config import ASR_WHISPERX_BATCH_SIZE
        _t1 = _time.time()
        result = model.transcribe(audio, batch_size=ASR_WHISPERX_BATCH_SIZE, language=detected_language, chunk_size=chunk_size)
        logger.info(f"⚡ WhisperX: Transcription done in {_time.time() - _t1:.1f}s")
        del model
        # Note: result["language"] would match detected_language — no need to re-read it.

        # Step 2: Align timestamps (skip for pure ASR mode — segment timestamps are sufficient)
        from core.config import ASR_ENGINE
        skip_alignment = (ASR_ENGINE == "whisperx" and not self.enable_diarization)
        if not skip_alignment:
            if callback:
                callback("Aligning word timestamps...", 40)
            logger.info("⚡ WhisperX: Aligning timestamps")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language, device=self.device
                )
                result = whisperx.align(
                    result["segments"], model_a, metadata, audio, self.device,
                    return_char_alignments=False,
                )
                del model_a
            except Exception as e:
                logger.warning(f"⚠️  Alignment failed ({e}), using unaligned segments")
        else:
            logger.info("⚡ WhisperX: Skipping alignment (segment timestamps sufficient for highlight extraction)")

        # Step 3: Diarize (optional)
        if self.enable_diarization:
            if callback:
                callback("Running speaker diarization...", 60)
            logger.info("⚡ WhisperX: Running speaker diarization")
            diarize_model = WhisperXDiarizationPipeline(
                token=self.hf_token, device=self.device
            )

            # Request embeddings when name mapping is enabled
            if self.speaker_identifier:
                diarize_df, speaker_embeddings = diarize_model(audio, return_embeddings=True)
                self.speaker_identifier.load_references(diarize_model)
                speaker_mapping = self.speaker_identifier.map_speakers(speaker_embeddings or {})
            else:
                diarize_df = diarize_model(audio)
                speaker_mapping = {}

            result = whisperx.assign_word_speakers(diarize_df, result)

            # Apply real names if mapping is available
            if speaker_mapping:
                for seg in result.get("segments", []):
                    seg_speaker = seg.get("speaker")
                    if seg_speaker in speaker_mapping:
                        seg["speaker"] = speaker_mapping[seg_speaker]

            del diarize_model

        # Step 4: Save to SRT
        if callback:
            callback("Saving transcript...", 90)
        segments = result.get("segments", [])
        self._save_to_srt(segments, str(output_srt))
        logger.info(f"✅ WhisperX: Saved {len(segments)} segments → {output_srt.name}")
        return str(output_srt)

    # ── Scenario 2: Diarization-only ─────────────────────────────────────────

    async def add_speakers_to_existing_transcript(
        self,
        subtitle_path: str,
        video_path: str,
        callback: Optional[Callable[[str, float], None]] = None,
    ) -> str:
        """
        Scenario 2: Run diarization and add speaker labels to an existing SRT.
        Overwrites subtitle_path in-place. Returns its path.
        """
        if not self.enable_diarization:
            logger.info("Diarization disabled — returning existing transcript unchanged")
            return subtitle_path

        subtitle_path = Path(subtitle_path)
        video_path = Path(video_path)

        if callback:
            callback("Loading audio for diarization...", 0)
        logger.info(f"⚡ WhisperX: Loading audio from {video_path.name}")
        audio = whisperx.load_audio(str(video_path))

        if callback:
            callback("Running speaker diarization...", 20)
        logger.info("⚡ WhisperX: Running speaker diarization")
        diarize_model = WhisperXDiarizationPipeline(
            token=self.hf_token, device=self.device
        )

        # Request embeddings when name mapping is enabled
        if self.speaker_identifier:
            diarize_df, speaker_embeddings = diarize_model(audio, return_embeddings=True)
            self.speaker_identifier.load_references(diarize_model)
            speaker_mapping = self.speaker_identifier.map_speakers(speaker_embeddings or {})
        else:
            diarize_df = diarize_model(audio)
            speaker_mapping = {}

        del diarize_model

        if callback:
            callback("Assigning speakers to transcript segments...", 80)
        segments = self._load_srt_segments(str(subtitle_path))
        segments = self._assign_speakers_to_segments(segments, diarize_df)

        # Apply real names if mapping is available
        if speaker_mapping:
            for seg in segments:
                seg_speaker = seg.get("speaker")
                if seg_speaker in speaker_mapping:
                    seg["speaker"] = speaker_mapping[seg_speaker]

        self._save_to_srt(segments, str(subtitle_path))

        speaker_set = {s.get("speaker") for s in segments if s.get("speaker")}
        logger.info(
            f"✅ WhisperX: Assigned {len(speaker_set)} speakers to {len(segments)} segments"
        )
        return str(subtitle_path)

    # ── SRT helpers ───────────────────────────────────────────────────────────

    def _load_srt_segments(self, srt_path: str) -> List[Dict[str, Any]]:
        """Parse SRT file into a list of segment dicts."""
        segments = []
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()

        for block in re.split(r"\n\s*\n", content.strip()):
            lines = block.strip().splitlines()
            if len(lines) < 2:
                continue
            if not lines[0].strip().isdigit():
                continue

            time_match = re.match(
                r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
                lines[1].strip(),
            )
            if not time_match:
                continue

            start = _srt_time_to_seconds(time_match.group(1))
            end = _srt_time_to_seconds(time_match.group(2))
            text = "\n".join(lines[2:]).strip()
            if not text:
                continue

            segments.append({"start": start, "end": end, "text": text})

        return segments

    def _assign_speakers_to_segments(
        self, segments: List[Dict], diarization_df
    ) -> List[Dict]:
        """Assign speaker label to each segment via maximum overlap with diarization DataFrame."""
        for seg in segments:
            seg_start, seg_end = seg["start"], seg["end"]
            overlap_per_speaker: Dict[str, float] = {}

            for _, row in diarization_df.iterrows():
                t_start, t_end, speaker = row["start"], row["end"], row["speaker"]
                overlap = max(0.0, min(seg_end, t_end) - max(seg_start, t_start))
                if overlap > 0:
                    overlap_per_speaker[speaker] = overlap_per_speaker.get(speaker, 0.0) + overlap

            if overlap_per_speaker:
                seg["speaker"] = max(overlap_per_speaker, key=overlap_per_speaker.get)

        return segments

    def _save_to_srt(self, segments: List[Dict], output_path: str) -> None:
        """Write segments to SRT file, prepending speaker label when present."""
        lines = []
        for i, seg in enumerate(segments, start=1):
            start_ts = _seconds_to_srt_time(seg.get("start", 0))
            end_ts = _seconds_to_srt_time(seg.get("end", 0))
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker")
            if speaker:
                text = f"[{speaker}] {text}"

            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(text)
            lines.append("")  # blank line separator

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
