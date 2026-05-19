from core.engaging_moments_analyzer import EngagingMomentsAnalyzer


def _analyzer(preset="auto"):
    return EngagingMomentsAnalyzer(
        api_key="test-key",
        provider="qwen",
        clip_length_preset=preset,
    )


def test_part_and_aggregation_prompts_include_clip_length_preference(tmp_path):
    srt = tmp_path / "part01.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:03:20,000\nA complete long discussion arc.\n",
        encoding="utf-8",
    )
    highlights = tmp_path / "highlights_part01.json"
    highlights.write_text(
        """
{
  "video_part": "part01",
  "engaging_moments": [
    {
      "title": "Long arc",
      "start_time": "00:00:00",
      "end_time": "00:03:20",
      "duration_seconds": 200,
      "summary": "A complete long discussion arc.",
      "engagement_details": {"engagement_level": "high"},
      "why_engaging": "It develops a complete point.",
      "tags": ["insight"]
    }
  ]
}
""",
        encoding="utf-8",
    )
    analyzer = _analyzer("180_300")

    part_prompt = analyzer.build_part_analysis_prompt(str(srt), "part01")
    aggregation_prompt = analyzer.build_aggregation_prompt([str(highlights)])

    assert "Target clip length: 3m-5m" in part_prompt
    assert "Hard duration bounds: 180-300 seconds" in part_prompt
    assert "30-240" not in part_prompt
    assert "4 minutes (240 seconds)" not in part_prompt
    assert "Target clip length: 3m-5m" in aggregation_prompt
    assert "duration fit as a preference" in aggregation_prompt
    assert "Optimal 45-180 seconds" not in aggregation_prompt


def test_validate_moment_uses_selected_duration_bounds():
    analyzer = _analyzer("180_300")
    long_moment = {
        "title": "Long arc",
        "start_time": "00:00:00",
        "end_time": "00:03:20",
    }
    short_moment = {
        "title": "Too short",
        "start_time": "00:00:00",
        "end_time": "00:01:30",
    }

    assert analyzer._validate_moment(long_moment, []) is True
    assert long_moment["duration_seconds"] == 200
    assert analyzer._validate_moment(short_moment, []) is False
