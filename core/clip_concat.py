"""Concatenate multiple video clips into a single compilation video."""

import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


def detect_watermark_region(video_path: str, llm_client, model: str = None) -> Optional[Dict]:
    """Use LLM vision to detect watermark position, trying multiple frames."""
    # Get video dimensions
    probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,duration", "-of", "json", video_path]
    probe_r = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
    try:
        streams = json.loads(probe_r.stdout)["streams"][0]
        width, height = streams["width"], streams["height"]
        duration = float(streams.get("duration", 30))
    except Exception:
        return None

    # Try multiple frames at different timestamps
    timestamps = [5, 15, 30, int(duration * 0.3), int(duration * 0.6)]
    timestamps = [t for t in timestamps if t < duration][:5]

    from core.llm.custom_openai_api_client import CustomOpenAIMessage

    for ts in timestamps:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            frame_path = tmp.name
        cmd = ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path, "-vframes", "1", "-q:v", "2", frame_path]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0 or not Path(frame_path).exists() or Path(frame_path).stat().st_size == 0:
            Path(frame_path).unlink(missing_ok=True)
            continue

        with open(frame_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        Path(frame_path).unlink(missing_ok=True)

        prompt = (
            f"这是一个视频帧（{width}x{height}像素）。"
            "请识别画面中的水印/台标/Logo（如CCTV、央视、卫视台标、平台水印等半透明覆盖物）。"
            "水印通常在角落位置，可能是半透明的文字或图标。"
            "返回JSON格式的水印边界框（像素坐标）："
            "{\"found\": true, \"x\": 左边距, \"y\": 上边距, \"w\": 宽度, \"h\": 高度} "
            "如果确实没有水印，返回 {\"found\": false}。只输出JSON，不要其他文字。"
        )

        messages = [CustomOpenAIMessage(role="user", content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
        ])]
        try:
            resp = llm_client.chat_completion(messages, model=model, temperature=0.0)
            content = resp["choices"][0]["message"]["content"]
            if "```" in content:
                content = content.split("```")[1].strip()
                if content.startswith("json"):
                    content = content[4:].strip()
            result = json.loads(content)
            if result.get("found"):
                logger.info(f"Watermark detected at frame {ts}s: {result}")
                return {"x": result["x"], "y": result["y"], "w": result["w"], "h": result["h"]}
        except Exception as e:
            logger.warning(f"Watermark detection at {ts}s failed: {e}")
            continue

    return None


def concat_clips(clip_paths: List[str], output_path: str, bgm_url: str = None, bgm_volume: float = 0.15,
                  remove_audio: bool = False, watermark_region: Optional[Dict] = None) -> bool:
    """Concatenate clips using ffmpeg. Supports BGM mixing, audio removal, watermark removal."""
    if not clip_paths:
        return False

    # Resolve all paths to absolute
    output_path = str(Path(output_path).resolve())
    clip_paths = [str(Path(p).resolve()) for p in clip_paths]
    if bgm_url and not bgm_url.startswith(("http://", "https://")):
        bgm_url = str(Path(bgm_url).resolve())

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
        list_file = f.name

    needs_post = bool(watermark_region) or bool(bgm_url) or remove_audio

    if not needs_post:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_path]
        try:
            logger.info(f"ffmpeg cmd: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            Path(list_file).unlink(missing_ok=True)
            if result.returncode != 0:
                logger.error(f"ffmpeg concat failed: {result.stderr[-500:]}")
                return False
            return True
        except Exception as e:
            Path(list_file).unlink(missing_ok=True)
            logger.error(f"concat_clips error: {e}")
            return False

    # Step 1: concat to temp file
    tmp_concat = str(Path(output_path).parent / "_tmp_concat.mp4")
    cmd1 = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", tmp_concat]
    try:
        logger.info(f"ffmpeg concat step: {' '.join(cmd1)}")
        r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=600)
        Path(list_file).unlink(missing_ok=True)
        if r1.returncode != 0:
            logger.error(f"ffmpeg concat step failed: {r1.stderr[-500:]}")
            return False
    except Exception as e:
        Path(list_file).unlink(missing_ok=True)
        logger.error(f"concat step error: {e}")
        return False

    # Step 2: apply filters
    # Get source duration to limit BGM length (avoids -shortest buffering issue)
    src_duration = None
    if bgm_url:
        dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", tmp_concat]
        dur_r = subprocess.run(dur_cmd, capture_output=True, text=True, timeout=10)
        try:
            src_duration = float(dur_r.stdout.strip())
        except (ValueError, TypeError):
            src_duration = None

    cmd2 = ["ffmpeg", "-y", "-i", tmp_concat]
    if bgm_url:
        cmd2 += ["-stream_loop", "-1"]
        if src_duration:
            cmd2 += ["-t", str(src_duration)]
        cmd2 += ["-i", bgm_url]

    filter_complex_parts = []
    has_vout = False
    if watermark_region:
        wr = watermark_region
        filter_complex_parts.append(f"[0:v]delogo=x={wr['x']}:y={wr['y']}:w={wr['w']}:h={wr['h']}[vout]")
        has_vout = True

    if remove_audio and not bgm_url:
        if filter_complex_parts:
            cmd2 += ["-filter_complex", ";".join(filter_complex_parts), "-map", "[vout]", "-an"]
        else:
            cmd2 += ["-an"]
    elif bgm_url:
        if remove_audio:
            audio_f = f"[1:a]volume={bgm_volume}[aout]"
        else:
            audio_f = f"[1:a]volume={bgm_volume}[bgm];[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        filter_complex_parts.append(audio_f)
        cmd2 += ["-filter_complex", ";".join(filter_complex_parts)]
        if has_vout:
            cmd2 += ["-map", "[vout]", "-map", "[aout]"]
        else:
            cmd2 += ["-map", "0:v", "-map", "[aout]"]
    else:
        if filter_complex_parts:
            cmd2 += ["-filter_complex", ";".join(filter_complex_parts), "-map", "[vout]", "-map", "0:a"]

    cmd2 += ["-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    if not remove_audio:
        cmd2 += ["-c:a", "aac", "-b:a", "192k"]
    cmd2.append(output_path)

    try:
        logger.info(f"ffmpeg filter step: {cmd2}")
        r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
        Path(tmp_concat).unlink(missing_ok=True)
        if r2.returncode != 0:
            logger.error(f"ffmpeg filter step stderr: {r2.stderr}")
            return False
        logger.info(f"Compilation saved to {output_path}")
        return True
    except Exception as e:
        Path(tmp_concat).unlink(missing_ok=True)
        logger.error(f"filter step error: {e}")
        return False
