import base64
import json

from core.browser_preferences import (
    build_preferences_payload,
    deserialize_preferences_payload,
    merge_browser_preferences,
    serialize_preferences_payload,
)
from core.browser_session import INPUT_TYPE_URL
from core.clip_duration import DEFAULT_CLIP_LENGTH_PRESET


def _default_data():
    return {
        "ui_language": "zh",
        "input_type": INPUT_TYPE_URL,
        "llm_provider": "openai",
        "llm_provider_settings": {
            "openai": {"model": "gpt", "base_url": "https://example.com", "api_key": "secret"},
            "custom_openai": {"model": "", "base_url": ""},
        },
        "language": "zh",
        "use_background": False,
        "force_whisper": False,
        "generate_clips": True,
        "max_clips": 5,
        "clip_length_preset": DEFAULT_CLIP_LENGTH_PRESET,
        "add_titles": False,
        "burn_subtitles": False,
        "subtitle_translation": None,
        "subtitle_style_preset": "default",
        "subtitle_style_font_size": "medium",
        "subtitle_style_vertical_position": "bottom",
        "subtitle_style_background_style": "none",
        "generate_cover": True,
        "cookie_mode": "none",
        "cookie_browser": "chrome",
        "mode": "engaging_moments",
        "agentic_analysis": False,
        "api_key": "top-secret",
        "video_source": "/tmp/video.mp4",
        "cookies_file": "/tmp/cookies.txt",
        "custom_prompt_file": "/tmp/prompt.md",
        "custom_prompt_text": "prompt",
        "speaker_references_dir": "/tmp/refs",
        "processing_result": {"success": True},
        "output_dir": "processed_videos",
        "user_intent": "find sam",
    }



def test_build_preferences_payload_uses_allowlist_and_excludes_sensitive_fields():
    payload = build_preferences_payload(_default_data())

    prefs = payload["prefs"]
    assert prefs["ui_language"] == "zh"
    assert prefs["clip_length_preset"] == DEFAULT_CLIP_LENGTH_PRESET
    assert prefs["llm_provider_settings"]["openai"] == {
        "model": "gpt",
        "base_url": "https://example.com",
    }
    for excluded in [
        "api_key",
        "video_source",
        "cookies_file",
        "custom_prompt_file",
        "custom_prompt_text",
        "speaker_references_dir",
        "processing_result",
        "output_dir",
        "user_intent",
    ]:
        assert excluded not in prefs



def test_serialize_round_trip_and_invalid_payload_fails_safe():
    payload = build_preferences_payload(_default_data())
    raw = serialize_preferences_payload(payload)

    assert deserialize_preferences_payload(raw) == payload
    assert deserialize_preferences_payload("not-base64") is None



def test_merge_browser_preferences_applies_prefs_but_restores_excluded_defaults():
    defaults = _default_data()
    browser_data = dict(defaults)
    browser_data["api_key"] = "changed-secret"
    payload = {
        "version": 1,
        "prefs": {
            "ui_language": "en",
            "llm_provider": "custom_openai",
            "llm_provider_settings": {"custom_openai": {"model": "x", "base_url": "https://custom"}},
            "cookie_mode": "browser",
        },
    }

    merged = merge_browser_preferences(defaults, browser_data, payload)

    assert merged["ui_language"] == "en"
    assert merged["llm_provider"] == "custom_openai"
    assert merged["llm_provider_settings"]["custom_openai"] == {
        "model": "x",
        "base_url": "https://custom",
    }
    assert merged["api_key"] == defaults["api_key"]
    assert merged["video_source"] == defaults["video_source"]
    assert merged["processing_result"] == defaults["processing_result"]



def test_schema_mismatch_falls_back_to_none():
    raw = base64.urlsafe_b64encode(json.dumps({"version": 999, "prefs": {"ui_language": "en"}}).encode("utf-8")).decode("ascii")
    assert deserialize_preferences_payload(raw) is None


def test_merge_browser_preferences_invalid_values_fall_back_to_defaults():
    defaults = _default_data()
    browser_data = dict(defaults)
    payload = {
        "version": 1,
        "prefs": {
            "llm_provider": "bogus",
            "language": "xx",
            "cookie_mode": "strange",
            "subtitle_style_preset": "weird",
            "subtitle_translation": "Klingon",
            "max_clips": -5,
            "clip_length_preset": "bogus",
            "generate_clips": "yes",
        },
    }

    merged = merge_browser_preferences(defaults, browser_data, payload)

    assert merged["llm_provider"] == defaults["llm_provider"]
    assert merged["language"] == defaults["language"]
    assert merged["cookie_mode"] == defaults["cookie_mode"]
    assert merged["subtitle_style_preset"] == defaults["subtitle_style_preset"]
    assert merged["subtitle_translation"] == defaults["subtitle_translation"]
    assert merged["max_clips"] == defaults["max_clips"]
    assert merged["clip_length_preset"] == defaults["clip_length_preset"]
    assert merged["generate_clips"] == defaults["generate_clips"]


def test_cookie_style_unpadded_round_trip_succeeds():
    payload = build_preferences_payload(_default_data())
    raw = serialize_preferences_payload(payload)
    assert "=" not in raw
    assert deserialize_preferences_payload(raw) == payload
