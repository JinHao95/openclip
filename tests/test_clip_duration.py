from core.clip_duration import (
    DEFAULT_CLIP_LENGTH_PRESET,
    build_clip_duration_prompt_section,
    get_clip_duration_preference,
    normalize_clip_length_preset,
)


def test_clip_length_presets_normalize_and_expose_bounds():
    assert normalize_clip_length_preset("60_90") == "60_90"
    assert normalize_clip_length_preset("bogus") == DEFAULT_CLIP_LENGTH_PRESET

    preference = get_clip_duration_preference("180_300")

    assert preference.label == "3m-5m"
    assert preference.min_seconds == 180
    assert preference.max_seconds == 300


def test_clip_duration_prompt_section_includes_selected_guidance():
    section = build_clip_duration_prompt_section("180_300")

    assert "## Clip Length Preference" in section
    assert "Target clip length: 3m-5m" in section
    assert "Hard duration bounds: 180-300 seconds" in section
    assert "complete segment, story, argument" in section

