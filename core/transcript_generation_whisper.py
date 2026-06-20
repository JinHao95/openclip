#!/usr/bin/env python3
"""
Whisper Transcript Generation
Complete transcript processing including CLI usage and orchestration functionality
"""

import re
import subprocess
import sys
import os
import base64
import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any
import whisper
from core.config import (
    WHISPER_MODEL, TRANSCRIPT_LANGUAGE_DETECT_MODEL,
    ASR_SEGMENT_MINUTES, ASR_OVERLAP_SECONDS, ASR_PARALLEL_WORKERS,
    ASR_BACKEND, ASR_LLM_MODEL, ASR_LLM_PARALLEL_WORKERS,
    ASR_ENGINE, ASR_WHISPERX_MODEL, ASR_WHISPERX_BATCH_SIZE,
    LLM_ANALYSIS_CHUNK_MINUTES, LLM_ANALYSIS_OVERLAP_SECONDS,
    API_KEY_ENV_VARS, LLM_CONFIG,
)
from core.transcript_generation_paraformer import ParaformerTranscriptProcessor

logger = logging.getLogger(__name__)

try:
    from core.transcript_generation_whisperx import TranscriptProcessorWhisperX, WHISPERX_AVAILABLE
except ImportError:
    WHISPERX_AVAILABLE = False


def select_transcript_backend(
    detected_language: str,
    paraformer_available: bool,
    use_whisperx: bool,
) -> str:
    """Choose the transcript backend for a detected language."""
    language = (detected_language or "").lower()
    if language.startswith("zh") and paraformer_available:
        return "paraformer"
    return "whisperx" if use_whisperx else "whisper"


def summarize_transcript_sources(sources: List[str]) -> str:
    """Summarize one or more transcript source names into a display value."""
    unique_sources = []
    for source in sources:
        if source and source not in unique_sources:
            unique_sources.append(source)
    if not unique_sources:
        return "unknown"
    if len(unique_sources) == 1:
        return unique_sources[0]
    return "mixed:" + ",".join(unique_sources)


def build_whisper_initial_prompt(language: Optional[str]) -> Optional[str]:
    """Return a style prompt for Whisper when a language benefits from steering.

    Whisper uses a single `zh` language code for Chinese, and OpenAI maintainers
    recommend `initial_prompt` to bias the transcript style toward simplified
    or traditional script. We prefer Simplified Chinese for Chinese transcripts.
    """
    normalized = (language or "").strip().lower()
    if normalized.startswith("zh") or normalized == "chinese":
        return "以下是普通话的简体中文字幕。"
    return None

def run_whisper_cli(file_path, model_name=WHISPER_MODEL, language=None, output_format="srt", output_dir=None):
    """
    Transcribe audio/video file using OpenAI Whisper CLI

    Args:
        file_path (str): Path to audio/video file
        model_name (str): Whisper model to use (tiny, base, small, medium, large, turbo)
        language (str): Language code (e.g., 'en', 'zh', 'ja') or None for auto-detection
        output_format (str): Output format (txt, vtt, srt, tsv, json, all)
        output_dir (str): Directory to write output files to (defaults to current directory)

    Returns:
        bool: True if successful, False if failed
    """
    print(f"🎵 Transcribing: {file_path}")
    print(f"📊 Model: {model_name}")
    print(f"📝 Output format: {output_format}")

    # Build the whisper command
    cmd = [sys.executable, "-m", "whisper", file_path, "--model", model_name, "--output_format", output_format]

    if output_dir:
        cmd.extend(["--output_dir", str(output_dir)])

    if language:
        cmd.extend(["--language", language])
        print(f"🌍 Language: {language}")
    else:
        print("🔍 Language: Auto-detection")

    initial_prompt = build_whisper_initial_prompt(language)
    if initial_prompt:
        cmd.extend(["--initial_prompt", initial_prompt])
        print("🈶 Script preference: Simplified Chinese")
    
    try:
        print("\n⏳ Running Whisper...")
        print("📋 Progress will be shown below:")
        print("-" * 50)
        
        # Run without capturing output to show real-time progress
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("-" * 50)
            print("✅ Transcription completed successfully!")
            return True
        else:
            print("-" * 50)
            print(f"❌ Transcription failed with return code: {result.returncode}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Whisper module not found in the current Python environment.")
        return False

def demonstrate_whisper():
    """Demonstrate different Whisper usage examples"""
    
    print("=== OpenAI Whisper CLI Demo ===\n")
    
    # Check if we have a sample file
    sample_file = "../video_sample.mp4"
    
    if os.path.exists(sample_file):
        print("📁 Found sample video file!")
        
        print("\n--- Example 1: Basic transcription (tiny model, fast) ---")
        success = run_whisper_cli(sample_file, model_name="tiny")
        
        if success:
            # Look for output files
            base_name = os.path.splitext(os.path.basename(sample_file))[0]
            txt_file = f"{base_name}.txt"
            
            if os.path.exists(txt_file):
                print(f"\n📄 Transcript saved to: {txt_file}")
                # Show first few lines
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"Preview: {preview}")
                except Exception as e:
                    print(f"Could not read transcript: {e}")
        
        print("\n--- Example 2: Different formats ---")
        print("💡 You can also generate different output formats:")
        
    else:
        print("📂 No sample file found. Here are usage examples:")
    
    print("\n🎯 Usage Examples:")
    print("1. Basic transcription:")
    print("   whisper audio.mp3")
    
    print("\n2. Specify model size:")
    print("   whisper audio.mp3 --model small")
    
    print("\n3. Specify language:")
    print("   whisper audio.mp3 --language en")
    
    print("\n4. Multiple output formats:")
    print("   whisper audio.mp3 --output_format all")
    
    print("\n5. Subtitle format:")
    print("   whisper video.mp4 --output_format srt")
    
    print("\n📏 Available Models (speed vs accuracy):")
    models = [
        ("tiny", "Fastest, least accurate"),
        ("base", "Good balance"),
        ("small", "Better accuracy"),
        ("medium", "High accuracy"),
        ("large", "Best accuracy, slowest"),
        ("turbo", "Fast and accurate")
    ]
    
    for model, desc in models:
        print(f"   • {model}: {desc}")
    
    print("\n📋 Output Formats:")
    formats = ["txt", "vtt", "srt", "tsv", "json", "all"]
    for fmt in formats:
        print(f"   • {fmt}")

def simple_transcribe(audio_file, model="base"):
    """Simple function to transcribe an audio file"""
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return False
    
    return run_whisper_cli(audio_file, model_name=model)


class TranscriptProcessor:
    """Handles all transcript-related operations"""

    def __init__(
        self,
        whisper_model: str = WHISPER_MODEL,
        language: Optional[str] = None,
        enable_diarization: bool = False,
        speaker_references_dir: Optional[str] = None,
    ):
        self.whisper_model = whisper_model
        self.language = language  # None = auto-detect
        self.enable_diarization = enable_diarization
        self.language_detection_model = TRANSCRIPT_LANGUAGE_DETECT_MODEL
        # WhisperX is required for diarization; also use it when ASR_ENGINE == "whisperx"
        self.use_whisperx = (enable_diarization or ASR_ENGINE == "whisperx") and WHISPERX_AVAILABLE
        self.paraformer_processor = ParaformerTranscriptProcessor()
        self._language_detector = None

        if enable_diarization and not WHISPERX_AVAILABLE:
            logger.warning("⚠️  Speaker diarization requested but WhisperX is not installed. Falling back to openai-whisper (no speaker labels). Run: uv sync --extra speakers")

        self.whisperx_processor = None
        if self.use_whisperx:
            self.whisperx_processor = TranscriptProcessorWhisperX(
                ASR_WHISPERX_MODEL if ASR_ENGINE == "whisperx" else whisper_model,
                enable_diarization=enable_diarization,
                speaker_references_dir=speaker_references_dir,
            )

        if self.paraformer_processor.is_available():
            logger.info(f"🈶 Chinese ASR backend: Paraformer ({self.paraformer_processor.project_dir})")
        else:
            logger.warning(
                "⚠️  Paraformer is unavailable; Chinese audio will fall back to Whisper. "
                f"Reason: {self.paraformer_processor.availability_error()}"
            )

    async def process_transcripts(self,
                                subtitle_path: str,
                                video_files: List[str] or str,
                                force_whisper: bool,
                                progress_callback: Optional[Callable[[str, float], None]]) -> Dict[str, Any]:
        """Process transcripts - either use existing subtitles or generate with whisper/whisperx"""

        has_existing = subtitle_path and os.path.exists(subtitle_path)

        if force_whisper or not has_existing:
            logger.info("📝 Generating transcripts locally with automatic language routing")
            return await self._generate_routed_transcripts(video_files, progress_callback)
        else:
            # Scenario 2: Use existing transcript
            if self.whisperx_processor and self.enable_diarization:
                if self._has_speaker_labels(subtitle_path):
                    logger.info("📥 Source transcript already has speaker labels, skipping diarization")
                    return {
                        'source': 'existing_diarized',
                        'transcript_path': subtitle_path if isinstance(video_files, str) else '',
                        'transcript_parts': [] if isinstance(video_files, str) else self._get_existing_transcript_parts(video_files)
                    }
                else:
                    logger.info("⚡ Using WhisperX diarization on existing transcript")
                    return await self._add_speakers_to_existing(video_files, progress_callback)
            else:
                logger.info("📥 Using existing subtitles")
                return {
                    'source': 'bilibili' if 'bilibili' in subtitle_path else 'existing',
                    'transcript_path': subtitle_path if isinstance(video_files, str) else '',
                    'transcript_parts': [] if isinstance(video_files, str) else self._get_existing_transcript_parts(video_files)
                }

    async def process_transcripts_for_segments(
        self,
        segments: List[Dict],
        video_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        max_workers: int = 8,
    ) -> Dict[str, Any]:
        """Transcribe only specified time segments in parallel, output a merged SRT."""
        import tempfile
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import shutil

        video_path = str(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_stem = Path(video_path).stem
        total_segments = len(segments)
        cancelled = threading.Event()

        # Phase 1: extract all audio segments in parallel (IO-bound, fast)
        seg_work_dirs = []
        for i, seg in enumerate(segments):
            start, end = seg["start"], seg["end"]
            duration = end - start
            tmpdir = tempfile.mkdtemp(prefix=f"seg_{i:03d}_")
            seg_audio = Path(tmpdir) / f"seg_{i:03d}.wav"
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
                "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", str(seg_audio),
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            seg_work_dirs.append((i, seg, tmpdir, str(seg_audio)))

        if progress_callback:
            progress_callback(f"Audio extracted for {total_segments} segments, starting transcription...", 37)

        # Phase 2: transcribe segments in parallel (CPU-bound via whisper)

        def _transcribe_one(item):
            if cancelled.is_set():
                return None
            i, seg, tmpdir, seg_audio_path = item
            start = seg["start"]
            detected_language = self._detect_transcript_language(seg_audio_path)
            backend = select_transcript_backend(
                detected_language=detected_language,
                paraformer_available=self.paraformer_processor.is_available(),
                use_whisperx=self.use_whisperx,
            )
            srt_path = None
            if backend == "paraformer":
                try:
                    srt_path, _ = self.paraformer_processor.transcribe_chinese_to_srt(
                        seg_audio_path, Path(tmpdir)
                    )
                except Exception:
                    backend = "whisper"
            if not srt_path:
                run_whisper_cli(
                    seg_audio_path,
                    model_name=self.whisper_model,
                    language=detected_language,
                    output_format="srt",
                    output_dir=tmpdir,
                )
                srt_path = str(Path(tmpdir) / f"{Path(seg_audio_path).stem}.srt")
            entries = []
            if srt_path and Path(srt_path).exists():
                entries = self._parse_and_offset_srt(srt_path, start)
                logger.info(
                    f"✅ Segment {i+1} ({start:.0f}-{seg['end']:.0f}s): "
                    f"{len(entries)} subtitle entries via {backend}"
                )
            shutil.rmtree(tmpdir, ignore_errors=True)
            return (i, entries)

        all_srt_entries_indexed = []
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_transcribe_one, item): item for item in seg_work_dirs}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                idx, entries = result
                all_srt_entries_indexed.append((idx, entries))
                completed += 1
                if progress_callback:
                    try:
                        pct = 37 + (completed / total_segments) * 11
                        progress_callback(
                            f"Transcribed {completed}/{total_segments} segments",
                            pct,
                        )
                    except Exception:
                        cancelled.set()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

        # Merge in order
        all_srt_entries_indexed.sort(key=lambda x: x[0])
        all_srt_entries = []
        segment_srt_paths = []
        for idx, entries in all_srt_entries_indexed:
            all_srt_entries.extend(entries)
            if entries:
                seg_srt_path = str(output_dir / f"{video_stem}_seg{idx:03d}.srt")
                self._write_srt(entries, seg_srt_path)
                segment_srt_paths.append(seg_srt_path)

        merged_srt_path = str(output_dir / f"{video_stem}.srt")
        self._write_srt(all_srt_entries, merged_srt_path)
        logger.info(f"📝 Merged SRT: {len(all_srt_entries)} entries → {merged_srt_path}")

        return {
            "source": "audio_energy_segments",
            "transcript_path": merged_srt_path,
            "transcript_parts": [merged_srt_path],
            "segment_srt_paths": segment_srt_paths,
        }

    @staticmethod
    def _parse_and_offset_srt(srt_path: str, offset_seconds: float) -> List[Dict]:
        """Parse SRT and shift all timestamps by offset_seconds."""
        entries = []
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return entries

        time_pattern = re.compile(
            r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
        )

        def shift_time(match) -> str:
            h, m, s, ms = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            total_ms = (h * 3600 + m * 60 + s) * 1000 + ms + int(offset_seconds * 1000)
            total_ms = max(0, total_ms)
            nh = total_ms // 3600000
            nm = (total_ms % 3600000) // 60000
            ns = (total_ms % 60000) // 1000
            nms = total_ms % 1000
            return f"{nh:02d}:{nm:02d}:{ns:02d},{nms:03d}"

        for block in content.split("\n\n"):
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                timing_line = time_pattern.sub(shift_time, lines[1])
                text = " ".join(lines[2:])
                timing_match = re.match(
                    r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",
                    timing_line,
                )
                if timing_match:
                    entries.append({
                        "start": timing_match.group(1),
                        "end": timing_match.group(2),
                        "text": text,
                    })
        return entries

    @staticmethod
    def _write_srt(entries: List[Dict], output_path: str):
        """Write subtitle entries to SRT format."""
        with open(output_path, "w", encoding="utf-8") as f:
            for i, entry in enumerate(entries, 1):
                f.write(f"{i}\n")
                f.write(f"{entry['start']} --> {entry['end']}\n")
                f.write(f"{entry['text']}\n\n")

    @staticmethod
    def _srt_time_to_ms(t: str) -> int:
        h, m, rest = t.split(":")
        s, ms = rest.split(",")
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)

    @staticmethod
    def _ms_to_srt_time(ms: int) -> str:
        ms = max(0, ms)
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        r = ms % 1000
        return f"{h:02d}:{m:02d}:{s:02d},{r:03d}"

    def _merge_overlapping_entries(self, all_entries: List[Dict]) -> List[Dict]:
        """Merge ASR entries from overlapping segments, deduplicating at boundaries."""
        if not all_entries:
            return []
        sorted_entries = sorted(all_entries, key=lambda e: self._srt_time_to_ms(e['start']))
        merged = [sorted_entries[0]]
        for entry in sorted_entries[1:]:
            prev = merged[-1]
            if abs(self._srt_time_to_ms(entry['start']) - self._srt_time_to_ms(prev['start'])) < 1000:
                if len(entry.get('text', '')) > len(prev.get('text', '')):
                    merged[-1] = entry
            else:
                merged.append(entry)
        return merged

    def split_srt_for_analysis(
        self,
        srt_path: str,
        chunk_minutes: float = None,
        overlap_seconds: float = None,
    ) -> List[Dict[str, Any]]:
        """Split SRT into time-based chunks with context overlap for LLM analysis.

        Returns list of dicts with keys:
            srt_path: path to chunk SRT file (includes context entries with markers)
            target_start: start of target range (seconds)
            target_end: end of target range (seconds)
        """
        if chunk_minutes is None:
            chunk_minutes = LLM_ANALYSIS_CHUNK_MINUTES
        if overlap_seconds is None:
            overlap_seconds = LLM_ANALYSIS_OVERLAP_SECONDS

        entries = self._parse_and_offset_srt(srt_path, 0.0)
        if not entries:
            return []

        last_entry_ms = self._srt_time_to_ms(entries[-1]['end'])
        total_duration_s = last_entry_ms / 1000.0
        chunk_sec = chunk_minutes * 60

        num_chunks = max(1, int(total_duration_s / chunk_sec) + (1 if total_duration_s % chunk_sec > 0 else 0))
        srt_dir = Path(srt_path).parent
        srt_stem = Path(srt_path).stem

        chunks = []
        for i in range(num_chunks):
            target_start = i * chunk_sec
            target_end = min((i + 1) * chunk_sec, total_duration_s)
            context_start = max(0, target_start - overlap_seconds)
            context_end = min(total_duration_s, target_end + overlap_seconds)

            chunk_entries = []
            for entry in entries:
                entry_start_ms = self._srt_time_to_ms(entry['start'])
                entry_start_s = entry_start_ms / 1000.0
                if entry_start_s < context_start:
                    continue
                if entry_start_s >= context_end:
                    break
                e = dict(entry)
                if entry_start_s < target_start:
                    e['_context'] = 'before'
                elif entry_start_s >= target_end:
                    e['_context'] = 'after'
                chunk_entries.append(e)

            chunk_srt_path = str(srt_dir / f"{srt_stem}_chunk{i:03d}.srt")
            self._write_srt_with_context(chunk_entries, chunk_srt_path, target_start, target_end)

            chunks.append({
                'srt_path': chunk_srt_path,
                'target_start': target_start,
                'target_end': target_end,
            })

        logger.info(f"📋 Split SRT into {num_chunks} analysis chunks ({chunk_minutes}min, {overlap_seconds}s overlap)")
        return chunks

    def _write_srt_with_context(
        self, entries: List[Dict], output_path: str, target_start: float, target_end: float
    ):
        """Write SRT file with context markers for LLM analysis."""
        with open(output_path, "w", encoding="utf-8") as f:
            wrote_before_marker = False
            wrote_target_marker = False
            wrote_after_marker = False
            idx = 1
            for entry in entries:
                ctx = entry.get('_context')
                if ctx == 'before' and not wrote_before_marker:
                    f.write(f"[CONTEXT: 前文参考，请勿在此范围产出高光]\n\n")
                    wrote_before_marker = True
                elif ctx is None and not wrote_target_marker:
                    ts = self._ms_to_srt_time(int(target_start * 1000))
                    te = self._ms_to_srt_time(int(target_end * 1000))
                    f.write(f"[TARGET: 分析目标范围 {ts} - {te}，请在此范围内识别高光]\n\n")
                    wrote_target_marker = True
                elif ctx == 'after' and not wrote_after_marker:
                    f.write(f"[CONTEXT: 后文参考，请勿在此范围产出高光]\n\n")
                    wrote_after_marker = True
                f.write(f"{idx}\n")
                f.write(f"{entry['start']} --> {entry['end']}\n")
                f.write(f"{entry['text']}\n\n")
                idx += 1

    async def process_transcripts_parallel(
        self,
        video_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        asr_backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full-video parallel transcription: split into overlapping 5min segments, ASR in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import tempfile
        import shutil
        import json

        video_path = str(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_stem = Path(video_path).stem

        # WhisperX GPU path: single-pass batched inference (no segmentation needed)
        effective_engine = asr_backend or ASR_ENGINE
        if effective_engine == "whisperx" and self.whisperx_processor:
            logger.info("⚡ Using WhisperX GPU engine for full-video transcription")
            import time as _time
            t0 = _time.time()
            srt_path = await self.whisperx_processor.transcribe_with_whisperx(
                video_path, progress_callback
            )
            elapsed = _time.time() - t0
            logger.info(f"⚡ WhisperX completed in {elapsed:.1f}s")
            # Copy SRT to output_dir if not already there
            srt_out = output_dir / f"{video_stem}.srt"
            if str(Path(srt_path).resolve()) != str(srt_out.resolve()):
                shutil.copy2(srt_path, str(srt_out))
                srt_path = str(srt_out)
            return {
                "source": "whisperx",
                "transcript_path": srt_path,
                "transcript_parts": [srt_path],
            }

        duration = self._get_video_duration(video_path)
        if duration <= 0:
            raise RuntimeError(f"Cannot determine duration for {video_path}")

        segment_sec = ASR_SEGMENT_MINUTES * 60
        overlap_sec = ASR_OVERLAP_SECONDS
        num_segments = max(1, int(duration / segment_sec) + (1 if duration % segment_sec > 0 else 0))

        segments = []
        for i in range(num_segments):
            start = max(0, i * segment_sec - overlap_sec) if i > 0 else 0
            end = min(duration, (i + 1) * segment_sec + overlap_sec) if i < num_segments - 1 else duration
            segments.append({"start": start, "end": end})

        use_llm_asr = (asr_backend or ASR_BACKEND) == "llm"
        llm_client = None
        asr_workers = ASR_PARALLEL_WORKERS
        if use_llm_asr:
            from core.llm.custom_openai_api_client import CustomOpenAIAPIClient
            llm_client = CustomOpenAIAPIClient(
                api_key=os.getenv(API_KEY_ENV_VARS["custom_openai"]),
                base_url=LLM_CONFIG["custom_openai"]["base_url"],
                model=ASR_LLM_MODEL,
            )
            asr_workers = ASR_LLM_PARALLEL_WORKERS

        effective_backend = "llm" if use_llm_asr else "whisper"
        logger.info(
            f"🔀 Parallel ASR: {num_segments} segments × {ASR_SEGMENT_MINUTES}min "
            f"(overlap={overlap_sec}s, workers={asr_workers}, backend={effective_backend})"
        )

        if progress_callback:
            progress_callback(f"Extracting {num_segments} audio segments...", 30)

        cancelled = threading.Event()
        seg_work_dirs = []
        for i, seg in enumerate(segments):
            start, end = seg["start"], seg["end"]
            seg_duration = end - start
            tmpdir = tempfile.mkdtemp(prefix=f"par_seg_{i:03d}_")
            seg_audio = Path(tmpdir) / f"seg_{i:03d}.wav"
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-t", str(seg_duration),
                "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", str(seg_audio),
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            seg_work_dirs.append((i, seg, tmpdir, str(seg_audio)))

        if progress_callback:
            progress_callback(f"Audio extracted, starting parallel ASR ({asr_workers} workers)...", 33)

        def _transcribe_one(item):
            if cancelled.is_set():
                return None
            i, seg, tmpdir, seg_audio_path = item
            start = seg["start"]
            detected_language = self._detect_transcript_language(seg_audio_path)

            if use_llm_asr:
                if cancelled.is_set():
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    return None
                mp3_path = seg_audio_path.replace(".wav", ".mp3")
                subprocess.run([
                    "ffmpeg", "-y", "-i", seg_audio_path,
                    "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
                    mp3_path,
                ], capture_output=True, check=True)
                if cancelled.is_set():
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    return None
                with open(mp3_path, "rb") as f:
                    audio_b64 = base64.b64encode(f.read()).decode("ascii")
                srt_text = llm_client.audio_transcribe(
                    audio_b64, audio_format="mp3",
                    model=ASR_LLM_MODEL, language=detected_language,
                )
                srt_path = str(Path(tmpdir) / f"{Path(seg_audio_path).stem}.srt")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_text)
                entries = []
                if Path(srt_path).exists():
                    entries = self._parse_and_offset_srt(srt_path, start)
                    logger.info(
                        f"✅ Segment {i+1}/{num_segments} ({start:.0f}-{seg['end']:.0f}s): "
                        f"{len(entries)} entries via llm"
                    )
                shutil.rmtree(tmpdir, ignore_errors=True)
                return (i, entries)

            backend = select_transcript_backend(
                detected_language=detected_language,
                paraformer_available=self.paraformer_processor.is_available(),
                use_whisperx=self.use_whisperx,
            )
            srt_path = None
            if backend == "paraformer":
                try:
                    srt_path, _ = self.paraformer_processor.transcribe_chinese_to_srt(
                        seg_audio_path, Path(tmpdir)
                    )
                except Exception:
                    backend = "whisper"
            if not srt_path:
                run_whisper_cli(
                    seg_audio_path,
                    model_name=self.whisper_model,
                    language=detected_language,
                    output_format="srt",
                    output_dir=tmpdir,
                )
                srt_path = str(Path(tmpdir) / f"{Path(seg_audio_path).stem}.srt")
            entries = []
            if srt_path and Path(srt_path).exists():
                entries = self._parse_and_offset_srt(srt_path, start)
                logger.info(
                    f"✅ Segment {i+1}/{num_segments} ({start:.0f}-{seg['end']:.0f}s): "
                    f"{len(entries)} entries via {backend}"
                )
            shutil.rmtree(tmpdir, ignore_errors=True)
            return (i, entries)

        all_entries_indexed = []
        completed = 0
        with ThreadPoolExecutor(max_workers=asr_workers) as executor:
            futures = {executor.submit(_transcribe_one, item): item for item in seg_work_dirs}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                idx, entries = result
                all_entries_indexed.append((idx, entries))
                completed += 1
                if progress_callback:
                    try:
                        pct = 33 + (completed / num_segments) * 15
                        progress_callback(f"Transcribed {completed}/{num_segments} segments", pct)
                    except Exception:
                        cancelled.set()
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

        all_entries_indexed.sort(key=lambda x: x[0])
        all_entries = []
        for _, entries in all_entries_indexed:
            all_entries.extend(entries)

        merged_entries = self._merge_overlapping_entries(all_entries)

        merged_srt_path = str(output_dir / f"{video_stem}.srt")
        self._write_srt(merged_entries, merged_srt_path)
        logger.info(f"📝 Parallel ASR complete: {len(merged_entries)} entries → {merged_srt_path}")

        return {
            "source": "parallel_asr",
            "transcript_path": merged_srt_path,
            "transcript_parts": [merged_srt_path],
        }

    def _get_video_duration(self, video_path: str) -> float:
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

    def _get_language_detector(self):
        if self._language_detector is None:
            self._language_detector = whisper.load_model(self.language_detection_model)
        return self._language_detector

    def _detect_transcript_language(self, media_path: str) -> str:
        media_path = str(media_path)
        try:
            detector = self._get_language_detector()
            audio = whisper.load_audio(media_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(detector.device)
            _, probs = detector.detect_language(mel)
            detected_language, confidence = max(probs.items(), key=lambda item: item[1])
            logger.info(
                f"🔎 Transcript language for {Path(media_path).name}: "
                f"{detected_language} ({confidence:.1%})"
            )
            return detected_language
        except Exception as e:
            logger.warning(
                f"⚠️  Transcript language detection failed for {Path(media_path).name} "
                f"({e}). Falling back to English/Whisper."
            )
            return "en"

    async def _generate_routed_transcripts(
        self,
        video_files: List[str] or str,
        progress_callback: Optional[Callable[[str, float], None]],
    ) -> Dict[str, Any]:
        """Generate transcripts with Whisper for English and Paraformer for Chinese."""
        if isinstance(video_files, str):
            video_files = [video_files]

        transcript_parts = []
        transcript_sources = []
        total_files = len(video_files)

        for i, video_file in enumerate(video_files):
            video_path = Path(video_file)
            video_dir = video_path.parent
            base_progress = 35 + (i / total_files) * 13 if total_files else 35

            detected_language = self._detect_transcript_language(str(video_path))
            backend = select_transcript_backend(
                detected_language=detected_language,
                paraformer_available=self.paraformer_processor.is_available(),
                use_whisperx=self.use_whisperx,
            )

            if progress_callback:
                progress_callback(
                    f"Generating transcript {i+1}/{total_files} with {backend}...",
                    base_progress,
                )

            logger.info(
                f"🔀 Transcript backend for {video_path.name}: {backend} "
                f"(detected language: {detected_language})"
            )

            srt_path = ""
            source = backend

            try:
                if backend == "paraformer":
                    srt_path, _ = self.paraformer_processor.transcribe_chinese_to_srt(
                        str(video_path),
                        video_dir,
                    )
                    logger.info(f"✅ Paraformer generated: {Path(srt_path).name}")
                    if self.whisperx_processor and self.enable_diarization:
                        logger.info("⚡ Running WhisperX diarization on Paraformer transcript")
                        srt_path = await self.whisperx_processor.add_speakers_to_existing_transcript(
                            srt_path,
                            str(video_path),
                            progress_callback,
                        )
                        source = "paraformer_diarized"
                elif backend == "whisperx":
                    srt_path = await self.whisperx_processor.transcribe_with_whisperx(
                        str(video_path),
                        progress_callback,
                    )
                    if srt_path:
                        logger.info(f"✅ WhisperX generated: {Path(srt_path).name}")
                else:
                    success = run_whisper_cli(
                        str(video_path),
                        model_name=self.whisper_model,
                        language=detected_language,
                        output_format="srt",
                        output_dir=str(video_dir),
                    )
                    if success:
                        srt_path = str(video_dir / f"{video_path.stem}.srt")
                        logger.info(f"✅ Whisper generated: {Path(srt_path).name}")
            except Exception as e:
                if backend == "paraformer":
                    logger.warning(
                        f"⚠️  Paraformer failed for {video_path.name} ({e}). Falling back to Whisper."
                    )
                    source = "whisper_fallback"
                    success = run_whisper_cli(
                        str(video_path),
                        model_name=self.whisper_model,
                        language=detected_language,
                        output_format="srt",
                        output_dir=str(video_dir),
                    )
                    if success:
                        srt_path = str(video_dir / f"{video_path.stem}.srt")
                        logger.info(f"✅ Whisper fallback generated: {Path(srt_path).name}")
                else:
                    logger.error(f"❌ {backend} failed for {video_path.name}: {e}")

            if srt_path and Path(srt_path).exists():
                transcript_parts.append(str(srt_path))
                transcript_sources.append(source)
            else:
                logger.error(f"❌ Transcript generation failed for {video_path.name}")

        return {
            'source': summarize_transcript_sources(transcript_sources),
            'transcript_path': transcript_parts[0] if len(transcript_parts) == 1 else '',
            'transcript_parts': transcript_parts,
        }
    
    async def _generate_whisper_transcripts(self, 
                                          video_files: List[str] or str,
                                          progress_callback: Optional[Callable[[str, float], None]]) -> Dict[str, Any]:
        """Generate transcripts using Whisper"""
        
        if isinstance(video_files, str):
            video_files = [video_files]
        
        transcript_parts = []
        total_files = len(video_files)
        
        for i, video_file in enumerate(video_files):
            # Update progress
            if progress_callback:
                base_progress = 35 + (i / total_files) * 13  # 35-48% range
                progress_callback(f"Generating transcript {i+1}/{total_files}...", base_progress)
            
            logger.info(f"🎙️  Generating transcript for: {Path(video_file).name}")
            
            video_path = Path(video_file)
            video_dir = video_path.parent

            success = run_whisper_cli(
                str(video_path),
                model_name=self.whisper_model,
                language=self.language,
                output_format="srt",
                output_dir=str(video_dir)
            )

            if success:
                srt_path = video_dir / f"{video_path.stem}.srt"
                if srt_path.exists():
                    transcript_parts.append(str(srt_path))
                    logger.info(f"✅ Generated: {srt_path.name}")
                else:
                    logger.warning(f"⚠️  SRT file not found for {video_path.name}")
            else:
                logger.error(f"❌ Whisper failed for {video_path.name}")
        
        return {
            'source': 'whisper',
            'transcript_path': transcript_parts[0] if len(transcript_parts) == 1 else '',
            'transcript_parts': transcript_parts
        }
    
    async def _generate_whisperx_transcripts(self,
                                             video_files: List[str] or str,
                                             progress_callback: Optional[Callable[[str, float], None]]) -> Dict[str, Any]:
        """Generate transcripts using WhisperX (Scenario 1)."""
        if isinstance(video_files, str):
            video_files = [video_files]

        transcript_parts = []
        total_files = len(video_files)

        for i, video_file in enumerate(video_files):
            if progress_callback:
                base_progress = 35 + (i / total_files) * 13
                progress_callback(f"Transcribing {i+1}/{total_files} with WhisperX...", base_progress)

            logger.info(f"⚡ WhisperX transcribing: {Path(video_file).name}")
            srt_path = await self.whisperx_processor.transcribe_with_whisperx(video_file, progress_callback)

            if srt_path and Path(srt_path).exists():
                transcript_parts.append(srt_path)
                logger.info(f"✅ Generated: {Path(srt_path).name}")
            else:
                logger.error(f"❌ WhisperX failed for {Path(video_file).name}")

        return {
            'source': 'whisperx',
            'transcript_path': transcript_parts[0] if len(transcript_parts) == 1 else '',
            'transcript_parts': transcript_parts,
        }

    async def _add_speakers_to_existing(self,
                                        video_files: List[str] or str,
                                        progress_callback: Optional[Callable[[str, float], None]]) -> Dict[str, Any]:
        """Add speaker labels to existing SRT files via diarization (Scenario 2)."""
        if isinstance(video_files, str):
            video_files = [video_files]

        transcript_parts = []
        total_files = len(video_files)

        for i, video_file in enumerate(video_files):
            video_path = Path(video_file)
            srt_path = video_path.parent / f"{video_path.stem}.srt"

            if not srt_path.exists():
                logger.warning(f"⚠️  No subtitle found next to {video_path.name}, skipping diarization")
                continue

            if progress_callback:
                base_progress = 35 + (i / total_files) * 13
                progress_callback(f"Diarizing {i+1}/{total_files}...", base_progress)

            logger.info(f"⚡ WhisperX diarizing: {video_path.name}")
            updated_srt = await self.whisperx_processor.add_speakers_to_existing_transcript(
                str(srt_path), video_file, progress_callback
            )
            transcript_parts.append(updated_srt)

        return {
            'source': 'whisperx_diarized',
            'transcript_path': transcript_parts[0] if len(transcript_parts) == 1 else '',
            'transcript_parts': transcript_parts,
        }

    def _has_speaker_labels(self, srt_path: str) -> bool:
        """Return True if the SRT file already contains [SpeakerName] prefixes.

        WhisperX writes speaker labels as '[SPEAKER_00] text' or '[Sam Altman] text' —
        the bracket content always starts with an uppercase letter.
        YouTube sound annotations like '[laughter]' or '[applause]' are lowercase
        and must not be treated as speaker labels.
        """
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if re.match(r'^\[[A-Z]', line.strip()):
                        return True
        except (OSError, IOError):
            pass
        return False

    def _get_existing_transcript_parts(self, video_files: List[str]) -> List[str]:
        """Get existing transcript parts (they should already exist from splitting)"""
        transcript_parts = []
        
        for video_file in video_files:
            video_path = Path(video_file)
            srt_path = video_path.parent / f"{video_path.stem}.srt"
            
            if srt_path.exists():
                transcript_parts.append(str(srt_path))
            else:
                logger.warning(f"⚠️  Expected transcript not found: {srt_path}")
        
        return transcript_parts


def main():
    """Main function"""
    
    # Check command line arguments
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else "base"
        
        print(f"🎵 Transcribing file: {audio_file}")
        simple_transcribe(audio_file, model)
    else:
        # Run demonstration
        demonstrate_whisper()
    
    print("\n🚀 To transcribe your own file:")
    print("   python main.py your_audio_file.mp3 [model]")
    print("   Example: python main.py speech.wav tiny")

if __name__ == "__main__":
    main()
