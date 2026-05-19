import asyncio
import json
import time
from pathlib import Path

from core.analysis_coordinator import AnalysisCoordinator
from core.clip_duration import get_clip_duration_preference


class _FakeLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def simple_chat(self, _prompt, model=None):
        self.prompts.append(_prompt)
        if "The results array must have exactly" in _prompt:
            marker = "The results array must have exactly "
            count_str = _prompt.split(marker, 1)[1].split(" items", 1)[0]
            batch_count = int(count_str)
            items = []
            for _ in range(batch_count):
                if self._responses:
                    items.append(json.loads(self._responses.pop(0)))
                else:
                    items.append(
                        {
                            "keep": True,
                            "standalone_score": 0.5,
                            "intent_alignment_score": 0.5,
                            "reason": "default",
                            "repair_diagnosis": "none",
                        }
                    )
            return json.dumps({"results": items})
        if self._responses:
            return self._responses.pop(0)
        return json.dumps(
            {
                "keep": True,
                "standalone_score": 0.5,
                "intent_alignment_score": 0.5,
                "reason": "default",
            }
        )


class _FakeAnalyzer:
    def __init__(self, tmp_path, llm_responses, max_clips=2, user_intent=None, aggregate_candidates=2):
        self.tmp_path = Path(tmp_path)
        self.llm_client = _FakeLLMClient(llm_responses)
        self.model = None
        self.max_clips = max_clips
        self.user_intent = user_intent
        self.language = "en"
        self.aggregate_candidates = aggregate_candidates

    async def analyze_part_for_engaging_moments(self, srt_path, part_name):
        return {
            "video_part": part_name,
            "engaging_moments": [
                {
                    "title": f"Candidate from {part_name}",
                    "start_time": "00:00:00",
                    "end_time": "00:00:45",
                    "duration_seconds": 45,
                    "summary": "A candidate moment",
                    "engagement_details": {"engagement_level": "high"},
                    "why_engaging": "Strong clip",
                    "tags": ["insight"],
                }
            ],
            "total_moments": 1,
        }

    def build_pre_verify_pool(self, _highlights_files, pool_size):
        moments = [
            _make_pre_verify_candidate(
                rank=1,
                title="Standalone clip",
                summary="Explains the point clearly.",
                why_engaging="Clear and useful.",
                level="high",
                start_time="00:00:00",
                end_time="00:00:45",
            ),
            _make_pre_verify_candidate(
                rank=2,
                title="Needs context",
                summary="Starts in the middle of an answer.",
                why_engaging="Interesting but abrupt.",
                level="medium",
                start_time="00:01:00",
                end_time="00:01:45",
            ),
            _make_pre_verify_candidate(
                rank=3,
                title="Backup clip",
                summary="Another candidate that can survive broader pre-verification shortlisting.",
                why_engaging="Useful as backup coverage.",
                level="medium",
                start_time="00:00:15",
                end_time="00:00:45",
                duration="30s",
            ),
        ]
        pool = moments[: self.aggregate_candidates][:pool_size]
        return {
            "top_engaging_moments": pool,
            "total_moments": len(pool),
            "analysis_timestamp": "2026-01-01T00:00:00Z",
            "aggregation_criteria": "Deterministic pre-verification pool built from per-part engaging moments",
            "analysis_summary": {
                "highest_engagement_themes": [],
                "total_engaging_content_time": "N/A",
                "recommendation": "Pre-verification pool assembled deterministically before standalone review",
            },
            "honorable_mentions": [],
        }

    async def save_highlights_to_file(self, highlights, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(highlights, f, ensure_ascii=False, indent=2)

    def parse_srt_file(self, srt_path):
        entries = []
        content = Path(srt_path).read_text(encoding="utf-8").strip()
        for block in content.split("\n\n"):
            lines = block.splitlines()
            if len(lines) >= 3:
                start_time, end_time = lines[1].split(" --> ")
                entries.append(
                    {
                        "start_time": start_time.replace(",", ".").replace(".", ","),
                        "end_time": end_time.replace(",", ".").replace(".", ","),
                        "text": " ".join(lines[2:]),
                    }
                )
        return entries

    def time_to_seconds(self, time_str):
        if "," in time_str:
            time_part, ms_part = time_str.split(",")
            ms = int(ms_part)
        else:
            time_part = time_str
            ms = 0
        h, m, s = map(int, time_part.split(":"))
        return h * 3600 + m * 60 + s + ms / 1000


def _write_transcript(path: Path):
    path.write_text(
        "\n".join(
            [
                "1",
                "00:00:00,000 --> 00:00:15,000",
                "What is the key idea behind this strategy?",
                "",
                "2",
                "00:00:15,000 --> 00:00:45,000",
                "The key idea is that the clip includes enough context to stand on its own.",
                "",
                "3",
                "00:01:00,000 --> 00:01:20,000",
                "And that is why it matters.",
                "",
                "4",
                "00:01:20,000 --> 00:01:45,000",
                "But without the earlier question, this answer feels abrupt.",
            ]
        ),
        encoding="utf-8",
    )


def _make_pre_verify_candidate(
    *,
    rank: int,
    title: str,
    summary: str,
    why_engaging: str,
    level: str,
    start_time: str,
    end_time: str,
    duration: str = "45s",
    video_part: str = "part01",
):
    return {
        "rank": rank,
        "title": title,
        "summary": summary,
        "engagement_details": {"engagement_level": level},
        "why_engaging": why_engaging,
        "tags": ["insight"],
        "timing": {
            "video_part": video_part,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
        },
    }


def _make_parallel_candidate(
    title: str,
    *,
    mode: str = "judge",
    video_part: str = "part01",
    start_time: str = "00:00:00",
    end_time: str = "00:00:45",
):
    candidate = {
        "title": title,
        "_verification_context": {},
        "_passes_deterministic": True,
        "timing": {
            "video_part": video_part,
            "start_time": start_time,
            "end_time": end_time,
        },
    }
    if mode == "rejudge":
        candidate["verification_notes"] = ""
    return candidate


def test_agentic_analysis_verifies_standalone_and_writes_artifacts(tmp_path):
    transcript_path = tmp_path / "part01.srt"
    _write_transcript(transcript_path)
    analyzer = _FakeAnalyzer(
        tmp_path,
        llm_responses=[
            json.dumps(
                {
                    "keep": True,
                    "standalone_score": 0.92,
                    "intent_alignment_score": 0.7,
                    "reason": "Contains the setup and answer.",
                }
            ),
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.2,
                    "intent_alignment_score": 0.4,
                    "reason": "Begins mid-thought and is not standalone.",
                }
            ),
        ],
        max_clips=1,
        aggregate_candidates=3,
    )
    coordinator = AnalysisCoordinator(analyzer)

    result = asyncio.run(coordinator.run([str(transcript_path)]))

    assert result["agentic_analysis"] is True
    assert Path(result["analysis_plan_file"]).exists()
    assert Path(result["pre_verify_file"]).exists()
    assert Path(result["aggregated_file"]).exists()
    assert Path(result["workflow_file"]).exists()
    assert Path(result["verification_report_file"]).exists()
    assert result["top_moments"]["total_moments"] == 1
    moment = result["top_moments"]["top_engaging_moments"][0]
    assert moment["title"] == "Standalone clip"
    assert moment["verification_status"] == "verified"
    assert moment["evidence_excerpt"]
    verification_prompt = analyzer.llm_client.prompts[0]
    assert "Actual clip transcript:" in verification_prompt
    assert "Context before:" in verification_prompt
    assert "Context after:" in verification_prompt
    assert "Judge standalone quality based on the Actual clip transcript only." in verification_prompt


def test_pre_verify_shortlist_can_be_larger_than_final_clip_cap(tmp_path):
    transcript_path = tmp_path / "part01.srt"
    _write_transcript(transcript_path)
    analyzer = _FakeAnalyzer(
        tmp_path,
        llm_responses=[
            json.dumps(
                {
                    "keep": True,
                    "standalone_score": 0.92,
                    "intent_alignment_score": 0.7,
                    "reason": "Contains the setup and answer.",
                    "repair_diagnosis": "none",
                }
            ),
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.25,
                    "intent_alignment_score": 0.4,
                    "reason": "Begins mid-thought and is not standalone.",
                    "repair_diagnosis": "bad_start",
                }
            ),
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.3,
                    "intent_alignment_score": 0.5,
                    "reason": "The payoff is missing.",
                    "repair_diagnosis": "missing_payoff",
                }
            ),
        ],
        max_clips=1,
        aggregate_candidates=3,
    )
    coordinator = AnalysisCoordinator(analyzer)

    result = asyncio.run(coordinator.run([str(transcript_path)]))
    pre_verify = json.loads(Path(result["pre_verify_file"]).read_text(encoding="utf-8"))
    plan = json.loads(Path(result["analysis_plan_file"]).read_text(encoding="utf-8"))

    assert plan["target_clip_count"] == 1
    assert plan["pre_verify_pool_size"] == 4
    assert pre_verify["total_moments"] == 3
    assert result["top_moments"]["total_moments"] == 1


def test_agentic_plan_and_duration_check_use_clip_length_preference(tmp_path):
    transcript_path = tmp_path / "part01.srt"
    _write_transcript(transcript_path)
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[], max_clips=1)
    analyzer.clip_duration_preference = get_clip_duration_preference("180_300")
    coordinator = AnalysisCoordinator(analyzer)
    valid_candidate = {
        "timing": {
            "start_time": "00:00:00",
            "end_time": "00:03:20",
        }
    }
    short_candidate = {
        "timing": {
            "start_time": "00:00:00",
            "end_time": "00:01:30",
        }
    }

    assert coordinator._check_duration(valid_candidate) == (True, 200)
    assert coordinator._check_duration(short_candidate) == (False, 90)

    plan = coordinator._build_analysis_plan([str(transcript_path)])

    assert plan["clip_duration_preference"]["preset"] == "180_300"
    assert plan["clip_duration_preference"]["max_seconds"] == 300
    assert "3m-5m" in coordinator._build_verification_prompt(
        {
            "title": "Long arc",
            "summary": "A complete long discussion arc.",
            "why_engaging": "It develops a complete point.",
            "timing": {
                "start_time": "00:00:00",
                "end_time": "00:03:20",
            },
        },
        {"actual_clip_excerpt": "A complete long discussion arc."},
    )


def test_agentic_analysis_repair_fills_gap_with_repaired_clip(tmp_path):
    transcript_path = tmp_path / "part01.srt"
    _write_transcript(transcript_path)
    analyzer = _FakeAnalyzer(
        tmp_path,
        llm_responses=[
            json.dumps(
                {
                    "keep": True,
                    "standalone_score": 0.9,
                    "intent_alignment_score": 0.7,
                    "reason": "Contains the setup and answer.",
                }
            ),
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.35,
                    "intent_alignment_score": 0.6,
                    "reason": "Starts in the middle of the answer and needs the earlier question.",
                    "repair_diagnosis": "bad_start",
                }
            ),
            json.dumps(
                {
                    "repairable": True,
                    "repair_strategy": "expand_start",
                    "suggested_start_time": "00:00:00",
                    "suggested_end_time": "00:00:45",
                    "reason": "Needs the setup question before the answer.",
                }
            ),
            json.dumps(
                {
                    "keep": True,
                    "standalone_score": 0.88,
                    "intent_alignment_score": 0.7,
                    "reason": "With the setup question restored, the clip is standalone.",
                }
            ),
        ],
        max_clips=2,
    )
    coordinator = AnalysisCoordinator(analyzer)

    result = asyncio.run(coordinator.run([str(transcript_path)]))

    assert result["repair_pass_used"] is True
    assert result["top_moments"]["total_moments"] == 2
    assert (
        result["top_moments"]["top_engaging_moments"][1]["verification_status"]
        == "repaired_verified"
    )
    assert (
        result["top_moments"]["top_engaging_moments"][1]["timing"]["start_time"]
        == "00:00:00"
    )
    assert len(analyzer.llm_client.prompts) == 3


def test_verification_report_records_repair_planner_attempt_when_repair_is_not_possible(tmp_path):
    transcript_path = tmp_path / "part01.srt"
    _write_transcript(transcript_path)
    analyzer = _FakeAnalyzer(
        tmp_path,
        llm_responses=[
            json.dumps(
                {
                    "keep": True,
                    "standalone_score": 0.9,
                    "intent_alignment_score": 0.7,
                    "reason": "Contains the setup and answer.",
                    "repair_diagnosis": "none",
                }
            ),
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.35,
                    "intent_alignment_score": 0.6,
                    "reason": "The clip cuts off before the conclusion.",
                    "repair_diagnosis": "bad_end",
                }
            ),
            json.dumps(
                {
                    "repairable": False,
                    "repair_strategy": "none",
                    "suggested_start_time": None,
                    "suggested_end_time": None,
                    "reason": "There is not enough nearby context to complete the clip cleanly.",
                }
            ),
        ],
        max_clips=2,
    )
    coordinator = AnalysisCoordinator(analyzer)

    result = asyncio.run(coordinator.run([str(transcript_path)]))
    report = json.loads(Path(result["verification_report_file"]).read_text(encoding="utf-8"))
    rejected = next(clip for clip in report["clips"] if clip["title"] == "Needs context")

    assert result["repair_pass_used"] is True
    assert rejected["decision"] == "rejected"
    assert rejected["judge_keep"] is False
    assert rejected["judge_reason"] == "The clip cuts off before the conclusion."
    assert rejected["repair_diagnosis"] == "bad_end"
    assert rejected["repair_planner_attempted"] is True
    assert rejected["repairable"] is False
    assert rejected["planner_reason"] == "There is not enough nearby context to complete the clip cleanly."
    assert rejected["rejudge_keep"] is None


def test_verification_report_records_repaired_rejection_when_rejudge_fails(tmp_path):
    transcript_path = tmp_path / "part01.srt"
    _write_transcript(transcript_path)
    analyzer = _FakeAnalyzer(
        tmp_path,
        llm_responses=[
            json.dumps(
                {
                    "keep": True,
                    "standalone_score": 0.9,
                    "intent_alignment_score": 0.7,
                    "reason": "Contains the setup and answer.",
                    "repair_diagnosis": "none",
                }
            ),
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.35,
                    "intent_alignment_score": 0.6,
                    "reason": "The clip starts too late and ends too early.",
                    "repair_diagnosis": "bad_start_and_end",
                }
            ),
            json.dumps(
                {
                    "repairable": True,
                    "repair_strategy": "expand_both",
                    "suggested_start_time": "00:00:00",
                    "suggested_end_time": "00:00:45",
                    "reason": "Needs both the setup question and the complete answer.",
                }
            ),
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.4,
                    "intent_alignment_score": 0.5,
                    "reason": "Even after repair, it still feels incomplete.",
                    "repair_diagnosis": "none",
                }
            ),
        ],
        max_clips=2,
    )
    coordinator = AnalysisCoordinator(analyzer)

    result = asyncio.run(coordinator.run([str(transcript_path)]))
    report = json.loads(Path(result["verification_report_file"]).read_text(encoding="utf-8"))
    rejected = next(clip for clip in report["clips"] if clip["title"] == "Needs context")

    assert result["repair_pass_used"] is True
    assert rejected["decision"] == "rejected"
    assert rejected["verification_status"] == "repaired_rejected"
    assert rejected["judge_keep"] is False
    assert rejected["judge_reason"] == "The clip starts too late and ends too early."
    assert rejected["repair_planner_attempted"] is True
    assert rejected["repairable"] is True
    assert rejected["planner_reason"] == "Needs both the setup question and the complete answer."
    assert rejected["rejudge_keep"] is False
    assert rejected["rejudge_reason"] == "Even after repair, it still feels incomplete."
    assert rejected["old_start_time"] == "00:01:00"
    assert rejected["old_end_time"] == "00:01:45"
    assert rejected["start_time"] == "00:00:00"
    assert rejected["end_time"] != rejected["old_end_time"]


def test_agentic_analysis_verification_context_separates_clip_and_surrounding_text(tmp_path):
    transcript_path = tmp_path / "part01.srt"
    _write_transcript(transcript_path)
    analyzer = _FakeAnalyzer(
        tmp_path,
        llm_responses=[
            json.dumps(
                {
                    "keep": True,
                    "standalone_score": 0.9,
                    "intent_alignment_score": 0.7,
                    "reason": "Looks good as-is.",
                }
            ),
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.3,
                    "intent_alignment_score": 0.5,
                    "reason": "Needs earlier context.",
                }
            ),
        ],
        max_clips=1,
    )
    coordinator = AnalysisCoordinator(
        analyzer,
        verification_context_before_segments=5,
        verification_context_after_segments=4,
        verification_context_before_seconds=60.0,
        verification_context_after_seconds=30.0,
    )

    asyncio.run(coordinator.run([str(transcript_path)]))

    verification_prompt = analyzer.llm_client.prompts[0]
    assert "Actual clip transcript:" in verification_prompt
    assert "Context before:" in verification_prompt
    assert "Context after:" in verification_prompt
    assert "And that is why it matters." in verification_prompt
    assert "But without the earlier question, this answer feels abrupt." in verification_prompt
    assert "What is the key idea behind this strategy?" in verification_prompt
    assert "The key idea is that the clip includes enough context to stand on its own." in verification_prompt


def test_prepare_candidate_prefers_whisper_context_entries_when_available(tmp_path):
    transcript_path = tmp_path / "part01.srt"
    _write_transcript(transcript_path)
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[], max_clips=1)
    coordinator = AnalysisCoordinator(analyzer)

    candidate = {
        "title": "Needs context",
        "summary": "Starts in the middle of an answer.",
        "engagement_details": {"engagement_level": "medium"},
        "why_engaging": "Interesting but abrupt.",
        "tags": ["insight"],
        "timing": {
            "video_part": "part01",
            "start_time": "00:01:00",
            "end_time": "00:01:45",
            "duration": "45s",
        },
        "_whisper_transcript_entries": [
            {
                "start_time": "00:00:45,000",
                "end_time": "00:00:59,000",
                "text": "Whisper setup line before the clip.",
            },
            {
                "start_time": "00:01:00,000",
                "end_time": "00:01:20,000",
                "text": "Whisper clip body line one.",
            },
            {
                "start_time": "00:01:20,000",
                "end_time": "00:01:45,000",
                "text": "Whisper clip body line two.",
            },
            {
                "start_time": "00:01:46,000",
                "end_time": "00:02:00,000",
                "text": "Whisper trailing line after the clip.",
            },
        ],
        "_whisper_transcript_path": str(transcript_path),
    }

    reviewed = coordinator._prepare_candidate_for_review(
        candidate,
        {"part01": str(transcript_path)},
    )

    assert reviewed["evidence_excerpt"] == "Whisper clip body line one. Whisper clip body line two."
    assert "Whisper setup line before the clip." in reviewed["_verification_context"]["context_before"]
    assert "Whisper trailing line after the clip." in reviewed["_verification_context"]["context_after"]


def test_repaired_clip_can_overlap_when_it_adds_substantial_new_material(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(analyzer)
    selected = [
        {
            "title": "Reveal clip",
            "summary": "Main reveal.",
            "why_engaging": "Big reveal.",
            "timing": {
                "video_part": "part01",
                "start_time": "00:02:27",
                "end_time": "00:04:48",
            },
        }
    ]
    repaired_candidate = {
        "title": "Aftermath clip",
        "summary": "Reactions after the reveal.",
        "why_engaging": "Begging and aftermath.",
        "timing": {
            "video_part": "part01",
            "start_time": "00:04:30",
            "end_time": "00:07:25",
        },
    }

    assert (
        coordinator._has_excessive_overlap(
            selected,
            repaired_candidate,
            allow_repaired_overlap=True,
        )
        is False
    )


def test_repair_failure_does_not_reintroduce_original_candidate(tmp_path):
    transcript_path = tmp_path / "part01.srt"
    _write_transcript(transcript_path)
    analyzer = _FakeAnalyzer(
        tmp_path,
        llm_responses=[
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.55,
                    "intent_alignment_score": 0.5,
                    "reason": "Needs the earlier setup to be standalone.",
                    "repair_diagnosis": "bad_start",
                }
            ),
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.3,
                    "intent_alignment_score": 0.4,
                    "reason": "Also weak as-is.",
                    "repair_diagnosis": "not_fixable_content",
                }
            ),
            json.dumps(
                {
                    "repairable": True,
                    "repair_strategy": "expand_start",
                    "suggested_start_time": "00:00:00",
                    "suggested_end_time": "00:01:20",
                    "reason": "Needs the setup question before the answer.",
                }
            ),
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.4,
                    "intent_alignment_score": 0.5,
                    "reason": "Even repaired, it still is not standalone enough.",
                    "repair_diagnosis": "none",
                }
            ),
        ],
        max_clips=2,
    )
    coordinator = AnalysisCoordinator(analyzer)

    result = asyncio.run(coordinator.run([str(transcript_path)]))

    titles = [moment["title"] for moment in result["top_moments"]["top_engaging_moments"]]
    statuses = [moment["verification_status"] for moment in result["top_moments"]["top_engaging_moments"]]

    assert result["top_moments"]["total_moments"] == 0
    assert titles == []
    assert "repaired_verified" not in statuses
    assert "repair_fallback" not in statuses


def test_should_generate_whisper_context_skips_clean_transcript(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(analyzer)
    candidate = {
        "timing": {
            "video_part": "part01",
            "start_time": "00:01:00",
            "end_time": "00:01:45",
            "duration": "45s",
        }
    }
    transcript_entries = [
        {"start_time": "00:00:00,000", "end_time": "00:00:15,000", "text": "Question."},
        {"start_time": "00:00:15,000", "end_time": "00:00:45,000", "text": "Answer."},
        {"start_time": "00:01:00,000", "end_time": "00:01:20,000", "text": "Next answer."},
        {"start_time": "00:01:20,000", "end_time": "00:01:45,000", "text": "More answer."},
    ]

    should_generate, reason = coordinator._should_generate_whisper_context(
        candidate,
        transcript_entries,
    )

    assert should_generate is False
    assert "overlap ratio 0.00" in reason


def test_should_generate_whisper_context_for_overlap_heavy_transcript(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(analyzer)
    candidate = {
        "timing": {
            "video_part": "part01",
            "start_time": "00:01:00",
            "end_time": "00:01:45",
            "duration": "45s",
        }
    }
    transcript_entries = [
        {"start_time": "00:00:55,000", "end_time": "00:01:05,000", "text": "A"},
        {"start_time": "00:01:00,000", "end_time": "00:01:10,000", "text": "B"},
        {"start_time": "00:01:08,000", "end_time": "00:01:18,000", "text": "C"},
        {"start_time": "00:01:16,000", "end_time": "00:01:26,000", "text": "D"},
    ]

    should_generate, reason = coordinator._should_generate_whisper_context(
        candidate,
        transcript_entries,
    )

    assert should_generate is True
    assert "overlap ratio" in reason


def test_verification_report_includes_transcript_source_fields(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(analyzer)
    candidate = {
        "title": "Clip",
        "_original_rank": 1,
        "verification_notes": "Looks good.",
        "_verification_transcript_source": "whisper",
        "_verification_transcript_reason": "overlap ratio 0.50 across 2 adjacent subtitle pairs",
        "_whisper_transcript_path": "/tmp/example.absolute.srt",
        "_judge_keep": True,
        "_judge_reason": "Standalone.",
        "selection_confidence": 0.9,
        "verification_status": "verified",
        "timing": {
            "start_time": "00:00:00",
            "end_time": "00:00:45",
            "video_part": "part01",
        },
        "evidence_excerpt": "Example transcript.",
        "repair_diagnosis": "none",
        "_repair_planner_attempted": False,
        "_repairable": None,
        "_planner_reason": None,
        "_rejudge_keep": None,
        "_rejudge_reason": None,
    }

    entry = coordinator._build_verification_clip_entry(candidate, "kept")

    assert entry["verification_transcript_source"] == "whisper"
    assert entry["verification_transcript_reason"] == "overlap ratio 0.50 across 2 adjacent subtitle pairs"
    assert entry["verification_transcript_path"] == "/tmp/example.absolute.srt"


def test_parallel_verification_batches_preserve_candidate_order_for_judge(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(
        analyzer,
        judge_batch_size=2,
        max_parallel_judge_batches=2,
        judge_batch_launch_stagger_seconds=0.0,
    )
    candidates = [
        _make_parallel_candidate("A"),
        _make_parallel_candidate("B"),
        _make_parallel_candidate("C"),
        _make_parallel_candidate("D"),
    ]

    def fake_run_single_verification_batch(batch, mode):
        assert mode == "judge"
        titles = [candidate["title"] for candidate in batch]
        if titles == ["A", "B"]:
            time.sleep(0.15)
        else:
            time.sleep(0.01)
        return [
            {
                "keep": True,
                "standalone_score": float(ord(title) - 64),
                "intent_alignment_score": 0.5,
                "reason": f"reason-{title}",
                "repair_diagnosis": "none",
            }
            for title in titles
        ]

    coordinator._run_single_verification_batch = fake_run_single_verification_batch

    asyncio.run(
        coordinator._apply_parallel_verification_batches(
            candidates,
            batch_size=2,
            mode="judge",
        )
    )

    assert [candidate["_judge_reason"] for candidate in candidates] == [
        "reason-A",
        "reason-B",
        "reason-C",
        "reason-D",
    ]


def test_parallel_verification_progress_reaches_expected_range_for_judge(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(
        analyzer,
        judge_batch_size=2,
        max_parallel_judge_batches=2,
        judge_batch_launch_stagger_seconds=0.0,
    )
    candidates = [
        _make_parallel_candidate("A"),
        _make_parallel_candidate("B"),
        _make_parallel_candidate("C"),
        _make_parallel_candidate("D"),
    ]
    progress_events = []

    def fake_run_single_verification_batch(batch, mode):
        return [
            {
                "keep": True,
                "standalone_score": 0.8,
                "intent_alignment_score": 0.5,
                "reason": candidate["title"],
                "repair_diagnosis": "none",
            }
            for candidate in batch
        ]

    coordinator._run_single_verification_batch = fake_run_single_verification_batch

    asyncio.run(
        coordinator._apply_parallel_verification_batches(
            candidates,
            batch_size=2,
            mode="judge",
            progress_callback=lambda status, progress: progress_events.append((status, progress)),
            progress_start=60,
            progress_end=64,
        )
    )

    assert len(progress_events) == 2
    assert progress_events[-1][1] == 64
    assert all("judge batch" in status for status, _ in progress_events)


def test_single_batch_verification_falls_back_to_single_calls_on_shape_mismatch(tmp_path):
    transcript_path = tmp_path / "part01.srt"
    _write_transcript(transcript_path)
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(analyzer)
    candidates = [
        {
            "title": "A",
            "timing": {"start_time": "00:00:00", "end_time": "00:00:10"},
            "_verification_context": {"actual_clip_excerpt": "a"},
            "_passes_deterministic": True,
        },
        {
            "title": "B",
            "timing": {"start_time": "00:00:10", "end_time": "00:00:20"},
            "_verification_context": {"actual_clip_excerpt": "b"},
            "_passes_deterministic": True,
        },
    ]

    responses = iter(
        [
            json.dumps({"results": [{"keep": True}]}),
            json.dumps(
                {
                    "keep": True,
                    "standalone_score": 0.7,
                    "intent_alignment_score": 0.5,
                    "reason": "fallback-a",
                    "repair_diagnosis": "none",
                }
            ),
            json.dumps(
                {
                    "keep": False,
                    "standalone_score": 0.2,
                    "intent_alignment_score": 0.5,
                    "reason": "fallback-b",
                    "repair_diagnosis": "bad_start",
                }
            ),
        ]
    )

    def fake_simple_chat(_prompt, model=None):
        return next(responses)

    analyzer.llm_client.simple_chat = fake_simple_chat

    results = coordinator._run_single_verification_batch(candidates, mode="judge")

    assert [result["reason"] for result in results] == ["fallback-a", "fallback-b"]


def test_parallel_verification_handles_empty_and_single_batch_cases(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(
        analyzer,
        judge_batch_size=2,
        max_parallel_judge_batches=2,
        judge_batch_launch_stagger_seconds=0.0,
    )

    asyncio.run(coordinator._apply_parallel_verification_batches([], batch_size=2, mode="judge"))

    candidates = [_make_parallel_candidate("Solo")]

    def fake_run_single_verification_batch(batch, mode):
        return [
            {
                "keep": True,
                "standalone_score": 0.9,
                "intent_alignment_score": 0.5,
                "reason": "single-batch",
                "repair_diagnosis": "none",
            }
        ]

    coordinator._run_single_verification_batch = fake_run_single_verification_batch

    asyncio.run(coordinator._apply_parallel_verification_batches(candidates, batch_size=2, mode="judge"))

    assert candidates[0]["_judge_reason"] == "single-batch"


def test_parallel_verification_batches_preserve_candidate_order_for_rejudge(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(
        analyzer,
        rejudge_batch_size=2,
        max_parallel_judge_batches=2,
        judge_batch_launch_stagger_seconds=0.0,
    )
    candidates = [
        _make_parallel_candidate("A", mode="rejudge", start_time="00:00:00", end_time="00:00:45"),
        _make_parallel_candidate("B", mode="rejudge", start_time="00:00:45", end_time="00:01:30"),
        _make_parallel_candidate("C", mode="rejudge", start_time="00:01:30", end_time="00:02:15"),
        _make_parallel_candidate("D", mode="rejudge", start_time="00:02:15", end_time="00:03:00"),
    ]

    def fake_run_single_verification_batch(batch, mode):
        assert mode == "rejudge"
        titles = [candidate["title"] for candidate in batch]
        if titles == ["A", "B"]:
            time.sleep(0.15)
        else:
            time.sleep(0.01)
        return [
            {
                "keep": True,
                "standalone_score": float(ord(title) - 64),
                "intent_alignment_score": 0.5,
                "reason": f"rejudge-{title}",
                "repair_diagnosis": "none",
            }
            for title in titles
        ]

    coordinator._run_single_verification_batch = fake_run_single_verification_batch

    asyncio.run(
        coordinator._apply_parallel_verification_batches(
            candidates,
            batch_size=2,
            mode="rejudge",
        )
    )

    assert [candidate["_rejudge_reason"] for candidate in candidates] == [
        "rejudge-A",
        "rejudge-B",
        "rejudge-C",
        "rejudge-D",
    ]


def test_parallel_repairs_preserve_original_order(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(
        analyzer,
        max_parallel_repairs=2,
        repair_launch_stagger_seconds=0.0,
    )
    transcript_map = {"part01": "/tmp/unused.srt"}
    candidates = [
        {"title": "A", "timing": {"video_part": "part01"}},
        {"title": "B", "timing": {"video_part": "part01"}},
        {"title": "C", "timing": {"video_part": "part01"}},
    ]

    def fake_attempt_boundary_repair(candidate, transcript_map):
        if candidate["title"] == "A":
            time.sleep(0.15)
        else:
            time.sleep(0.01)
        return {
            "title": f"repaired-{candidate['title']}",
            "timing": {"video_part": "part01", "start_time": "00:00:00", "end_time": "00:00:45"},
        }

    coordinator._attempt_boundary_repair = fake_attempt_boundary_repair

    repaired, failed = asyncio.run(
        coordinator._apply_parallel_repairs(
            candidates,
            transcript_map,
        )
    )

    assert failed == []
    assert [item["title"] for item in repaired] == ["repaired-A", "repaired-B", "repaired-C"]


def test_parallel_repairs_preserve_failed_candidates_for_reporting(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(
        analyzer,
        max_parallel_repairs=2,
        repair_launch_stagger_seconds=0.0,
    )
    transcript_map = {"part01": "/tmp/unused.srt"}
    candidates = [
        {
            "title": "A",
            "timing": {"video_part": "part01"},
            "_repair_planner_attempted": True,
            "_planner_reason": "not enough context",
            "verification_notes": "old",
        },
        {"title": "B", "timing": {"video_part": "part01"}},
    ]

    def fake_attempt_boundary_repair(candidate, transcript_map):
        if candidate["title"] == "A":
            return None
        return {
            "title": "repaired-B",
            "timing": {"video_part": "part01", "start_time": "00:00:00", "end_time": "00:00:45"},
        }

    coordinator._attempt_boundary_repair = fake_attempt_boundary_repair

    repaired, failed = asyncio.run(
        coordinator._apply_parallel_repairs(
            candidates,
            transcript_map,
        )
    )

    assert [item["title"] for item in repaired] == ["repaired-B"]
    assert [item["title"] for item in failed] == ["A"]
    assert failed[0]["_planner_reason"] == "not enough context"


def test_parallel_repairs_progress_reaches_expected_range(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(
        analyzer,
        max_parallel_repairs=2,
        repair_launch_stagger_seconds=0.0,
    )
    transcript_map = {"part01": "/tmp/unused.srt"}
    candidates = [
        {"title": "A", "timing": {"video_part": "part01"}},
        {"title": "B", "timing": {"video_part": "part01"}},
    ]
    progress_events = []

    def fake_attempt_boundary_repair(candidate, transcript_map):
        return {
            "title": f"repaired-{candidate['title']}",
            "timing": {"video_part": "part01", "start_time": "00:00:00", "end_time": "00:00:45"},
        }

    coordinator._attempt_boundary_repair = fake_attempt_boundary_repair

    asyncio.run(
        coordinator._apply_parallel_repairs(
            candidates,
            transcript_map,
            progress_callback=lambda status, progress: progress_events.append((status, progress)),
            progress_start=64,
            progress_end=68,
        )
    )

    assert len(progress_events) == 2
    assert progress_events[-1][1] == 68
    assert all("repair" in status for status, _ in progress_events)


def test_parallel_repairs_handle_empty_and_single_cases(tmp_path):
    analyzer = _FakeAnalyzer(tmp_path, llm_responses=[])
    coordinator = AnalysisCoordinator(
        analyzer,
        max_parallel_repairs=2,
        repair_launch_stagger_seconds=0.0,
    )
    transcript_map = {"part01": "/tmp/unused.srt"}

    repaired, failed = asyncio.run(coordinator._apply_parallel_repairs([], transcript_map))
    assert repaired == []
    assert failed == []

    candidates = [{"title": "Solo", "timing": {"video_part": "part01"}}]

    def fake_attempt_boundary_repair(candidate, transcript_map):
        return {
            "title": "repaired-Solo",
            "timing": {"video_part": "part01", "start_time": "00:00:00", "end_time": "00:00:45"},
        }

    coordinator._attempt_boundary_repair = fake_attempt_boundary_repair

    repaired, failed = asyncio.run(coordinator._apply_parallel_repairs(candidates, transcript_map))

    assert [item["title"] for item in repaired] == ["repaired-Solo"]
    assert failed == []
