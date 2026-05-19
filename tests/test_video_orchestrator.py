import asyncio
from pathlib import Path

import pytest

from core.video_splitter import VideoSplitter
from core.subtitle_burner import SubtitlePreparationResult
from video_orchestrator import VideoOrchestrator


def _write_source_video_with_subtitle(tmp_path, subtitle_text=None):
    source_video = tmp_path / "input.mp4"
    source_video.write_bytes(b"fake-video")
    source_subtitle = tmp_path / "input.srt"
    source_subtitle.write_text(
        subtitle_text
        or "1\n00:00:00,000 --> 00:00:45,000\nThis clip contains enough setup to stand alone.",
        encoding="utf-8",
    )
    return source_video, source_subtitle


def _patch_local_video_processing(monkeypatch, orchestrator, source_video, source_subtitle):
    async def fake_is_local_video_file(_source: str) -> bool:
        return True

    async def fake_process_local_video(_video_path: str, _progress_callback):
        return {
            "video_path": str(source_video),
            "video_info": {
                "title": "input",
                "duration": 60,
                "uploader": "Local File",
            },
            "subtitle_path": str(source_subtitle),
        }

    monkeypatch.setattr(orchestrator, "_is_local_video_file", fake_is_local_video_file)
    monkeypatch.setattr(orchestrator, "_process_local_video", fake_process_local_video)


def _patch_existing_transcript_processing(monkeypatch, orchestrator, source_subtitle):
    async def fake_process_transcripts(_subtitle_path, _video_path, _force_whisper, _progress_callback):
        output_srt = (
            Path(orchestrator.output_dir)
            / "input"
            / "splits"
            / "input_part01.srt"
        )
        output_srt.parent.mkdir(parents=True, exist_ok=True)
        output_srt.write_text(source_subtitle.read_text(encoding="utf-8"), encoding="utf-8")
        return {
            "source": "existing",
            "transcript_parts": [str(output_srt)],
        }

    monkeypatch.setattr(orchestrator.transcript_processor, "process_transcripts", fake_process_transcripts)


def test_skip_transcript_uses_existing_local_subtitle_for_single_part_video(tmp_path, monkeypatch):
    source_video, source_subtitle = _write_source_video_with_subtitle(
        tmp_path,
        "1\n00:00:00,000 --> 00:00:01,000\n你好\n",
    )

    orchestrator = VideoOrchestrator(
        output_dir=str(tmp_path / "output"),
        skip_analysis=True,
        generate_clips=False,
        generate_cover=False,
    )

    _patch_local_video_processing(monkeypatch, orchestrator, source_video, source_subtitle)

    result = asyncio.run(
        orchestrator.process_video(
            str(source_video),
            skip_transcript=True,
            progress_callback=None,
        )
    )

    expected_subtitle = (
        Path(orchestrator.output_dir)
        / "input"
        / "splits"
        / "input_part01.srt"
    )

    assert result.success is True
    assert result.transcript_source == "existing"
    assert result.transcript_parts == [str(expected_subtitle)]
    assert expected_subtitle.exists()


def test_custom_openai_requires_model_when_analysis_is_enabled(tmp_path):
    with pytest.raises(ValueError, match="Invalid custom_openai analysis configuration"):
        VideoOrchestrator(
            output_dir=str(tmp_path / "output"),
            llm_provider="custom_openai",
            llm_base_url="http://127.0.0.1:8000/v1",
        )


def test_agentic_analysis_routes_through_coordinator(tmp_path, monkeypatch):
    source_video, source_subtitle = _write_source_video_with_subtitle(tmp_path)

    orchestrator = VideoOrchestrator(
        output_dir=str(tmp_path / "output"),
        api_key="test-key",
        agentic_analysis=True,
        generate_clips=False,
        generate_cover=False,
    )

    async def fake_run(transcript_parts, progress_callback=None):
        aggregated_file = Path(transcript_parts[0]).parent / "top_engaging_moments.json"
        aggregated_file.write_text(
            '{"top_engaging_moments":[],"total_moments":0}',
            encoding="utf-8",
        )
        return {
            "highlights_files": [],
            "aggregated_file": str(aggregated_file),
            "top_moments": {"top_engaging_moments": [], "total_moments": 0},
            "total_parts_analyzed": len(transcript_parts),
            "agentic_analysis": True,
        }

    _patch_local_video_processing(monkeypatch, orchestrator, source_video, source_subtitle)
    _patch_existing_transcript_processing(monkeypatch, orchestrator, source_subtitle)
    monkeypatch.setattr(orchestrator.analysis_coordinator, "run", fake_run)

    result = asyncio.run(
        orchestrator.process_video(
            str(source_video),
            skip_transcript=False,
            progress_callback=None,
        )
    )

    assert result.success is True
    assert result.engaging_moments_analysis["agentic_analysis"] is True


def test_analyze_engaging_moments_uses_basic_strategy_without_agentic_analysis(tmp_path, monkeypatch):
    orchestrator = VideoOrchestrator(
        output_dir=str(tmp_path / "output"),
        api_key="test-key",
        generate_clips=False,
        generate_cover=False,
    )
    calls = {"basic": 0, "agentic": 0}

    async def fake_basic(_result, _progress_callback):
        calls["basic"] += 1
        return {"aggregated_file": "basic.json"}

    async def fake_agentic(_result, _progress_callback):
        calls["agentic"] += 1
        return {"aggregated_file": "agentic.json"}

    monkeypatch.setattr(orchestrator, "_run_basic_engaging_moments_analysis", fake_basic)
    monkeypatch.setattr(orchestrator, "_run_agentic_engaging_moments_analysis", fake_agentic)

    result = asyncio.run(orchestrator._analyze_engaging_moments(object(), None))

    assert result == {"aggregated_file": "basic.json"}
    assert calls == {"basic": 1, "agentic": 0}


def test_orchestrator_passes_clip_length_preset_to_engaging_analyzer(tmp_path):
    orchestrator = VideoOrchestrator(
        output_dir=str(tmp_path / "output"),
        api_key="test-key",
        clip_length_preset="180_300",
        generate_clips=False,
        generate_cover=False,
    )

    assert orchestrator.clip_length_preset == "180_300"
    assert orchestrator.engaging_moments_analyzer.clip_duration_preference.preset == "180_300"
    assert orchestrator.engaging_moments_analyzer.clip_duration_preference.max_seconds == 300


def test_process_video_emits_editor_manifest(tmp_path, monkeypatch):
    source_video = tmp_path / "input.mp4"
    source_video.write_bytes(b"fake-video")
    source_subtitle = tmp_path / "input.srt"
    source_subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:30,000\nTranscript line.\n",
        encoding="utf-8",
    )

    orchestrator = VideoOrchestrator(
        output_dir=str(tmp_path / "output"),
        skip_analysis=True,
        generate_clips=True,
        generate_cover=True,
        add_titles=False,
        burn_subtitles=False,
    )

    async def fake_is_local_video_file(_source: str) -> bool:
        return True

    async def fake_process_local_video(_video_path: str, _progress_callback):
        return {
            "video_path": str(source_video),
            "video_info": {
                "title": "input",
                "duration": 60,
                "uploader": "Local File",
            },
            "subtitle_path": str(source_subtitle),
        }

    async def fake_process_transcripts(_subtitle_path, _video_path, _force_whisper, _progress_callback):
        output_srt = Path(orchestrator.output_dir) / "input" / "splits" / "input_part01.srt"
        output_srt.parent.mkdir(parents=True, exist_ok=True)
        output_srt.write_text(source_subtitle.read_text(encoding="utf-8"), encoding="utf-8")
        return {
            "source": "existing",
            "transcript_parts": [str(output_srt)],
        }

    aggregated_file = Path(orchestrator.output_dir) / "input" / "splits" / "top_engaging_moments.json"
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)
    aggregated_file.write_text(
        '{"top_engaging_moments":[{"rank":1,"title":"Clip Title","why_engaging":"Hook","engagement_details":{"engagement_level":"high"},"timing":{"video_part":"part01","start_time":"00:00:05","end_time":"00:00:20","duration":15}}]}',
        encoding="utf-8",
    )

    def fake_generate_clips(_analysis_file, _video_dir, _subtitle_dir):
        clips_dir = Path(orchestrator.output_dir) / "input" / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        raw_clip = clips_dir / "rank_01_clip_title.mp4"
        raw_clip.write_bytes(b"clip")
        clip_srt = clips_dir / "rank_01_clip_title.srt"
        clip_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nClip subtitle\n", encoding="utf-8")
        return {
            "success": True,
            "total_clips": 1,
            "successful_clips": 1,
            "clips_info": [
                {
                    "rank": 1,
                    "title": "Clip Title",
                    "filename": raw_clip.name,
                    "subtitle_filename": clip_srt.name,
                    "whisper_subtitle_filename": None,
                    "duration": 15.0,
                    "video_part": "part01",
                    "time_range": "00:00:05 - 00:00:20",
                    "original_time_range": "00:00:05 - 00:00:20",
                    "normalization_details": {"start": "unchanged", "end": "unchanged"},
                    "engagement_level": "high",
                    "why_engaging": "Hook",
                }
            ],
            "output_dir": str(clips_dir),
        }

    def fake_generate_cover(_video_path, _title_text, output_path, **_kwargs):
        Path(output_path).write_bytes(b"jpg")
        Path(output_path.replace(".jpg", "_vertical.jpg")).write_bytes(b"jpg")
        return True

    monkeypatch.setattr(orchestrator, "_is_local_video_file", fake_is_local_video_file)
    monkeypatch.setattr(orchestrator, "_process_local_video", fake_process_local_video)
    monkeypatch.setattr(orchestrator.transcript_processor, "process_transcripts", fake_process_transcripts)
    monkeypatch.setattr(orchestrator, "_find_existing_analysis", lambda _result: {"aggregated_file": str(aggregated_file)})
    monkeypatch.setattr(orchestrator.clip_generator, "generate_clips_from_analysis", fake_generate_clips)
    monkeypatch.setattr(orchestrator.cover_generator, "generate_cover", fake_generate_cover)

    result = asyncio.run(orchestrator.process_video(str(source_video), skip_transcript=False, progress_callback=None))

    manifest_path = Path(orchestrator.output_dir) / "input" / "editor_project.json"
    assert result.success is True
    assert manifest_path.exists()
    assert result.editor_project_id
    assert result.clip_generation["editor_manifest_path"] == str(manifest_path)


def test_title_and_subtitle_postprocessing_batches_ass_preparation(tmp_path, monkeypatch):
    source_video = tmp_path / "input.mp4"
    source_video.write_bytes(b"fake-video")
    source_subtitle = tmp_path / "input.srt"
    source_subtitle.write_text(
        "1\n00:00:00,000 --> 00:00:30,000\nTranscript line.\n",
        encoding="utf-8",
    )

    orchestrator = VideoOrchestrator(
        output_dir=str(tmp_path / "output"),
        api_key="test-key",
        skip_analysis=True,
        generate_clips=True,
        generate_cover=False,
        add_titles=True,
        burn_subtitles=True,
        subtitle_translation="Simplified Chinese",
        subtitle_translation_max_workers=2,
    )

    async def fake_is_local_video_file(_source: str) -> bool:
        return True

    async def fake_process_local_video(_video_path: str, _progress_callback):
        return {
            "video_path": str(source_video),
            "video_info": {
                "title": "input",
                "duration": 60,
                "uploader": "Local File",
            },
            "subtitle_path": str(source_subtitle),
        }

    async def fake_process_transcripts(_subtitle_path, _video_path, _force_whisper, _progress_callback):
        output_srt = Path(orchestrator.output_dir) / "input" / "splits" / "input_part01.srt"
        output_srt.parent.mkdir(parents=True, exist_ok=True)
        output_srt.write_text(source_subtitle.read_text(encoding="utf-8"), encoding="utf-8")
        return {
            "source": "existing",
            "transcript_parts": [str(output_srt)],
        }

    aggregated_file = Path(orchestrator.output_dir) / "input" / "splits" / "top_engaging_moments.json"
    aggregated_file.parent.mkdir(parents=True, exist_ok=True)
    aggregated_file.write_text(
        '{"top_engaging_moments":[{"rank":1,"title":"Clip A","timing":{"video_part":"part01","start_time":"00:00:00","end_time":"00:00:10","duration":10}},{"rank":2,"title":"Clip B","timing":{"video_part":"part01","start_time":"00:00:10","end_time":"00:00:20","duration":10}}]}',
        encoding="utf-8",
    )

    def fake_generate_clips(_analysis_file, _video_dir, _subtitle_dir):
        clips_dir = Path(orchestrator.output_dir) / "input" / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_a = clips_dir / "rank_01_clip_a.mp4"
        clip_b = clips_dir / "rank_02_clip_b.mp4"
        clip_a.write_bytes(b"clip-a")
        clip_b.write_bytes(b"clip-b")
        (clips_dir / "rank_01_clip_a.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nClip A subtitle\n",
            encoding="utf-8",
        )
        return {
            "success": True,
            "total_clips": 2,
            "successful_clips": 2,
            "clips_info": [
                {"rank": 1, "title": "Clip A", "filename": clip_a.name},
                {"rank": 2, "title": "Clip B", "filename": clip_b.name},
            ],
            "output_dir": str(clips_dir),
        }

    prepared_job_names = []

    def fake_prepare_ass_for_clips(jobs, subtitle_translation=None):
        prepared_job_names.extend(job.mp4.name for job in jobs)
        results = []
        for job in jobs:
            job.ass_path.write_text("ass", encoding="utf-8")
            if job.translated_output_path:
                Path(job.translated_output_path).write_text(
                    "1\n00:00:00,000 --> 00:00:01,000\n你好\n",
                    encoding="utf-8",
                )
            results.append(SubtitlePreparationResult(job=job, ok=True))
        return results

    rendered = []

    def fake_add_artistic_title(input_path, _title, output_path, *_args, ass_path=None):
        rendered.append((Path(input_path).name, Path(ass_path).name if ass_path else None))
        Path(output_path).write_bytes(b"rendered")
        return True

    monkeypatch.setattr(orchestrator, "_is_local_video_file", fake_is_local_video_file)
    monkeypatch.setattr(orchestrator, "_process_local_video", fake_process_local_video)
    monkeypatch.setattr(orchestrator.transcript_processor, "process_transcripts", fake_process_transcripts)
    monkeypatch.setattr(orchestrator, "_find_existing_analysis", lambda _result: {"aggregated_file": str(aggregated_file)})
    monkeypatch.setattr(orchestrator.clip_generator, "generate_clips_from_analysis", fake_generate_clips)
    monkeypatch.setattr(orchestrator.subtitle_burner, "prepare_ass_for_clips", fake_prepare_ass_for_clips)
    monkeypatch.setattr(orchestrator.title_adder, "_add_artistic_title", fake_add_artistic_title)

    result = asyncio.run(orchestrator.process_video(str(source_video), skip_transcript=False, progress_callback=None))

    assert result.success is True
    assert prepared_job_names == ["rank_01_clip_a.mp4"]
    assert rendered == [
        ("rank_01_clip_a.mp4", "rank_01_clip_a.ass"),
        ("rank_02_clip_b.mp4", None),
    ]
    assert result.clip_generation["clips_info"][0]["translated_subtitle_filename"] == "rank_01_clip_a.translated.srt"


def test_split_video_async_returns_part_offsets(tmp_path, monkeypatch):
    splitter = VideoSplitter(output_dir=tmp_path / "out")

    def fake_split(_video_path, _subtitle_path, _duration_minutes, _output_dir):
        splitter.last_split_points = [(0.0, 600.0), (600.0, 1200.0)]
        return True

    monkeypatch.setattr(splitter, "split_by_time_duration", fake_split)
    monkeypatch.setattr(
        "core.video_utils.VideoFileManager.find_video_parts",
        lambda _splits_dir, _base_name: (
            [str(tmp_path / "out" / "demo_part01.mp4"), str(tmp_path / "out" / "demo_part02.mp4")],
            [str(tmp_path / "out" / "demo_part01.srt"), str(tmp_path / "out" / "demo_part02.srt")],
        ),
    )

    result = asyncio.run(
        splitter.split_video_async(
            "demo.mp4",
            "demo.srt",
            progress_callback=None,
            splits_dir=tmp_path / "out",
        )
    )

    assert result["part_offsets"] == {"part01": 0.0, "part02": 600.0}
