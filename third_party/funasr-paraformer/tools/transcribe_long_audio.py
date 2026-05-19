#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from funasr import AutoModel
from modelscope import snapshot_download

# modelscope references errno.EREMOTEIO on Linux, but that symbol does not
# exist on macOS. Alias it to the closest available constant so downloads can
# proceed on Darwin instead of crashing during lock handling.
if not hasattr(errno, "EREMOTEIO"):
    errno.EREMOTEIO = getattr(errno, "EREMOTE", 121)

DEFAULT_MODEL_IDS = {
    "zh": {
        "asr": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    },
    "en": {
        "asr": "iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020",
        "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    },
}

AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".webm",
    ".mp4",
    ".mov",
    ".mkv",
}

HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run long-form transcription with FunASR Paraformer + VAD + punctuation."
    )
    parser.add_argument("input", help="An audio file or a directory containing audio files.")
    parser.add_argument(
        "--lang",
        default="auto",
        choices=["auto", "zh", "en"],
        help="Transcription language. auto probes the file first, then switches to the zh or en model.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(root / "outputs"),
        help="Directory used to store transcription outputs.",
    )
    parser.add_argument(
        "--models-root",
        default=str(root / "models"),
        help="Directory that stores the Chinese ASR/VAD/punctuation models.",
    )
    parser.add_argument(
        "--models-root-en",
        default=str(root / "models_en"),
        help="Directory that stores the English ASR/VAD/punctuation models.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device. auto picks cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=max(1, min(os.cpu_count() or 4, 8)),
        help="CPU thread count used by FunASR when running on CPU.",
    )
    parser.add_argument(
        "--batch-size-s",
        type=int,
        default=300,
        help="Maximum total audio seconds merged into one ASR batch after VAD.",
    )
    parser.add_argument(
        "--batch-size-threshold-s",
        type=int,
        default=60,
        help="Segments longer than this threshold are decoded alone.",
    )
    parser.add_argument(
        "--max-single-segment-ms",
        type=int,
        default=30000,
        help="Maximum duration of one VAD segment in milliseconds.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan audio files when the input is a directory.",
    )
    parser.add_argument(
        "--no-segment-punc",
        action="store_true",
        help="Do not run punctuation restoration on per-segment transcripts.",
    )
    parser.add_argument(
        "--disable-pbar",
        action="store_true",
        help="Disable FunASR progress bars.",
    )
    parser.add_argument(
        "--cleanup-normalized",
        action="store_true",
        help="Delete cached 16k mono wav files after the run finishes.",
    )
    parser.add_argument(
        "--lang-probe-segments",
        type=int,
        default=3,
        help="Number of VAD segments used when --lang auto probes the file language.",
    )
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested == "mps":
        print(
            "[device] MPS is incompatible with the current FunASR Paraformer pipeline; falling back to cpu.",
            file=sys.stderr,
        )
        return "cpu"
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        print(
            "[device] Auto-selected MPS is incompatible with the current FunASR Paraformer pipeline; using cpu instead.",
            file=sys.stderr,
        )
        return "cpu"
    return "cpu"


def is_model_ready(target_dir: Path) -> bool:
    return (target_dir / "configuration.json").exists() or (target_dir / "config.yaml").exists()


def ensure_models(models_root: Path, language: str) -> dict[str, Path]:
    models_root.mkdir(parents=True, exist_ok=True)
    resolved = {}
    model_ids = DEFAULT_MODEL_IDS[language]
    for name, model_id in model_ids.items():
        target_dir = models_root / name
        if not is_model_ready(target_dir):
            print(f"[download:{language}] {name}: {model_id}", file=sys.stderr)
            snapshot_download(model_id, local_dir=str(target_dir))
        resolved[name] = target_dir.resolve()
    return resolved


def build_pipeline(
    model_paths: dict[str, Path],
    device: str,
    ncpu: int,
    max_single_segment_ms: int,
    disable_pbar: bool,
) -> AutoModel:
    return AutoModel(
        model=str(model_paths["asr"]),
        vad_model=str(model_paths["vad"]),
        punc_model=str(model_paths["punc"]),
        device=device,
        ncpu=ncpu,
        disable_update=True,
        disable_pbar=disable_pbar,
        hub="ms",
        vad_kwargs={"max_single_segment_time": max_single_segment_ms},
    )


def collect_audio_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    files = sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No audio files found under: {input_path}")
    return files


def cache_key(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def normalize_audio(input_path: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{input_path.stem}-{cache_key(input_path)}.wav"
    if output_path.exists():
        return output_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {input_path}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return output_path


def load_audio_samples(wav_path: Path) -> np.ndarray:
    samples, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sample_rate != 16000:
        raise ValueError(f"Expected normalized audio at 16000 Hz, got {sample_rate}: {wav_path}")
    if isinstance(samples, np.ndarray) and samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.asarray(samples, dtype=np.float32)


def slice_segments(samples: np.ndarray, vad_segments: list[list[int]]) -> list[np.ndarray]:
    chunks = []
    total = len(samples)
    for start_ms, end_ms in vad_segments:
        start = max(0, int(start_ms * 16))
        end = min(total, int(end_ms * 16))
        if end > start:
            chunks.append(samples[start:end])
    return chunks


def run_vad(model: AutoModel, audio_path: Path, disable_pbar: bool) -> list[list[int]]:
    kwargs = copy.deepcopy(model.vad_kwargs)
    result = model.inference(
        str(audio_path),
        model=model.vad_model,
        kwargs=kwargs,
        disable_pbar=disable_pbar,
    )[0]
    return result.get("value", [])


def run_asr_segments(
    model: AutoModel,
    segments: list[np.ndarray],
    disable_pbar: bool,
) -> list[str]:
    if not segments:
        return []

    kwargs = copy.deepcopy(model.kwargs)
    kwargs["batch_size"] = 1 if kwargs.get("device") == "cpu" else min(8, len(segments))
    results = model.inference(
        segments,
        model=model.model,
        kwargs=kwargs,
        disable_pbar=disable_pbar,
    )
    return [item.get("text", "").strip() for item in results]


def run_punc_segments(
    model: AutoModel,
    texts: list[str],
    disable_pbar: bool,
) -> list[str]:
    results = []
    for text in texts:
        if not text:
            results.append("")
            continue
        kwargs = copy.deepcopy(model.punc_kwargs)
        restored = model.inference(
            text,
            model=model.punc_model,
            kwargs=kwargs,
            disable_pbar=disable_pbar,
        )[0]
        results.append(restored.get("text", text).strip())
    return results


def relative_output_base(input_file: Path, input_root: Path, output_dir: Path) -> Path:
    if input_root.is_file():
        rel = Path(input_file.stem)
    else:
        rel = input_file.relative_to(input_root)
        rel = rel.with_suffix("")
    return output_dir / rel


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_pattern(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def detect_language_from_text(text: str) -> dict[str, Any]:
    han_chars = count_pattern(HAN_RE, text)
    latin_chars = count_pattern(LATIN_RE, text)
    signal_chars = han_chars + latin_chars
    han_ratio = han_chars / signal_chars if signal_chars else 0.0

    if han_chars >= 6 and (latin_chars == 0 or han_ratio >= 0.2):
        language = "zh"
    elif latin_chars >= 12 and han_chars <= 2:
        language = "en"
    elif han_chars >= 3 and han_chars > latin_chars:
        language = "zh"
    elif latin_chars > 0:
        language = "en"
    else:
        language = "zh"

    return {
        "detected_language": language,
        "han_chars": han_chars,
        "latin_chars": latin_chars,
        "signal_chars": signal_chars,
        "han_ratio": round(han_ratio, 4),
    }


def probe_language(
    model: AutoModel,
    normalized_audio: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    vad_segments = run_vad(model, normalized_audio, args.disable_pbar)
    samples = load_audio_samples(normalized_audio)
    segment_audio = slice_segments(samples, vad_segments)
    probe_count = max(1, args.lang_probe_segments)
    probe_audio = segment_audio[:probe_count]
    probe_texts = run_asr_segments(model, probe_audio, args.disable_pbar)
    probe_text = " ".join(text for text in probe_texts if text).strip()
    metrics = detect_language_from_text(probe_text)
    return {
        "mode": "auto_probe",
        "probe_segments_requested": probe_count,
        "probe_segments_available": len(segment_audio),
        "probe_segments_used": len(probe_audio),
        "probe_texts": probe_texts,
        "probe_text": probe_text,
        **metrics,
    }


def transcribe_one(
    model: AutoModel,
    input_file: Path,
    normalized_audio: Path,
    output_base: Path,
    args: argparse.Namespace,
    device: str,
    language: str,
    language_detection: dict[str, Any],
) -> dict[str, Any]:
    vad_segments = run_vad(model, normalized_audio, args.disable_pbar)
    samples = load_audio_samples(normalized_audio)
    segment_audio = slice_segments(samples, vad_segments)
    raw_segment_texts = run_asr_segments(model, segment_audio, args.disable_pbar)
    segment_texts = (
        raw_segment_texts
        if args.no_segment_punc
        else run_punc_segments(model, raw_segment_texts, args.disable_pbar)
    )

    segments = []
    for index, vad in enumerate(vad_segments):
        raw_text = raw_segment_texts[index] if index < len(raw_segment_texts) else ""
        text = segment_texts[index] if index < len(segment_texts) else raw_text
        start_ms, end_ms = vad
        segments.append(
            {
                "id": index + 1,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_s": round(start_ms / 1000.0, 3),
                "end_s": round(end_ms / 1000.0, 3),
                "duration_ms": end_ms - start_ms,
                "raw_text": raw_text,
                "text": text,
            }
        )

    full_text = "\n".join(text for text in segment_texts if text).strip()
    raw_full_text = "\n".join(text for text in raw_segment_texts if text).strip()

    result = {
        "input_file": str(input_file),
        "normalized_audio": str(normalized_audio),
        "created_at_utc": utc_now_iso(),
        "device": device,
        "language": language,
        "language_detection": language_detection,
        "models": {
            "asr": str(Path(model.model_path).resolve()),
            "vad": str(Path(model.vad_kwargs["model_path"]).resolve()),
            "punc": str(Path(model.punc_kwargs["model_path"]).resolve()),
        },
        "text": full_text,
        "raw_text": raw_full_text,
        "vad_segment_count": len(vad_segments),
        "segments": segments,
    }

    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_base.with_suffix(".txt").write_text(result["text"] + "\n", encoding="utf-8")
    output_base.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def cleanup_normalized_files(normalized_dir: Path) -> None:
    if not normalized_dir.exists():
        return
    for wav_file in normalized_dir.glob("*.wav"):
        wav_file.unlink(missing_ok=True)


def model_roots_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "zh": Path(args.models_root).resolve(),
        "en": Path(args.models_root_en).resolve(),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    normalized_dir = output_dir / "_normalized_cache"
    device = select_device(args.device)
    models_roots = model_roots_from_args(args)

    files = collect_audio_files(input_path, args.recursive)
    model_cache: dict[str, AutoModel] = {}

    def get_model(language: str) -> AutoModel:
        if language not in model_cache:
            model_paths = ensure_models(models_roots[language], language)
            model_cache[language] = build_pipeline(
                model_paths=model_paths,
                device=device,
                ncpu=args.ncpu,
                max_single_segment_ms=args.max_single_segment_ms,
                disable_pbar=args.disable_pbar,
            )
        return model_cache[language]

    summary = []
    for audio_file in files:
        print(f"[transcribe] {audio_file}", file=sys.stderr)
        normalized_audio = normalize_audio(audio_file, normalized_dir)
        output_base = relative_output_base(audio_file, input_path, output_dir)
        if args.lang == "auto":
            detection = probe_language(get_model("zh"), normalized_audio, args)
            selected_language = detection["detected_language"]
            print(
                f"[lang] {audio_file.name}: auto -> {selected_language} "
                f"(han={detection['han_chars']}, latin={detection['latin_chars']})",
                file=sys.stderr,
            )
        else:
            selected_language = args.lang
            detection = {
                "mode": "explicit",
                "detected_language": selected_language,
                "probe_segments_requested": 0,
                "probe_segments_available": 0,
                "probe_segments_used": 0,
                "probe_texts": [],
                "probe_text": "",
                "han_chars": 0,
                "latin_chars": 0,
                "signal_chars": 0,
                "han_ratio": 0.0,
            }

        try:
            result = transcribe_one(
                model=get_model(selected_language),
                input_file=audio_file,
                normalized_audio=normalized_audio,
                output_base=output_base,
                args=args,
                device=device,
                language=selected_language,
                language_detection=detection,
            )
        except Exception as exc:
            if args.lang == "auto" and selected_language == "en":
                fallback_detection = dict(detection)
                fallback_detection["fallback_from"] = "en"
                fallback_detection["fallback_reason"] = str(exc)
                fallback_detection["detected_language"] = "zh"
                print(
                    f"[lang] {audio_file.name}: en pipeline failed, fallback to zh",
                    file=sys.stderr,
                )
                result = transcribe_one(
                    model=get_model("zh"),
                    input_file=audio_file,
                    normalized_audio=normalized_audio,
                    output_base=output_base,
                    args=args,
                    device=device,
                    language="zh",
                    language_detection=fallback_detection,
                )
            else:
                raise

        summary.append(
            {
                "input_file": result["input_file"],
                "language": result["language"],
                "text_file": str(output_base.with_suffix(".txt")),
                "json_file": str(output_base.with_suffix(".json")),
                "vad_segment_count": result["vad_segment_count"],
                "text": result["text"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.jsonl"
    with summary_path.open("w", encoding="utf-8") as fh:
        for item in summary:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    if args.cleanup_normalized:
        cleanup_normalized_files(normalized_dir)
        normalized_dir.rmdir() if normalized_dir.exists() and not any(normalized_dir.iterdir()) else None

    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
