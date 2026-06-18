from __future__ import annotations

from dataclasses import asdict, dataclass


DEFAULT_CLIP_LENGTH_PRESET = "30"


@dataclass(frozen=True)
class ClipDurationPreference:
    preset: str
    label: str
    min_seconds: int
    max_seconds: int
    ideal_min_seconds: int
    ideal_max_seconds: int
    guidance: str

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


CLIP_DURATION_PRESETS: dict[str, ClipDurationPreference] = {
    "15": ClipDurationPreference(
        preset="15",
        label="≤15s",
        min_seconds=5,
        max_seconds=15,
        ideal_min_seconds=8,
        ideal_max_seconds=15,
        guidance="Each clip must not exceed 15 seconds. Capture only the single peak moment.",
    ),
    "30": ClipDurationPreference(
        preset="30",
        label="≤30s",
        min_seconds=8,
        max_seconds=30,
        ideal_min_seconds=10,
        ideal_max_seconds=25,
        guidance="Each clip must not exceed 30 seconds. One core action per clip (shot, save, goal).",
    ),
    "60": ClipDurationPreference(
        preset="60",
        label="≤60s",
        min_seconds=10,
        max_seconds=60,
        ideal_min_seconds=15,
        ideal_max_seconds=50,
        guidance="Each clip must not exceed 60 seconds. Include brief setup and payoff for the action.",
    ),
    "90": ClipDurationPreference(
        preset="90",
        label="≤90s",
        min_seconds=15,
        max_seconds=90,
        ideal_min_seconds=20,
        ideal_max_seconds=75,
        guidance="Each clip must not exceed 90 seconds. Include setup, action, and aftermath.",
    ),
    "180": ClipDurationPreference(
        preset="180",
        label="≤3min",
        min_seconds=20,
        max_seconds=180,
        ideal_min_seconds=30,
        ideal_max_seconds=150,
        guidance="Each clip must not exceed 3 minutes. Capture a complete sequence or discussion arc.",
    ),
    "300": ClipDurationPreference(
        preset="300",
        label="≤5min",
        min_seconds=30,
        max_seconds=300,
        ideal_min_seconds=60,
        ideal_max_seconds=270,
        guidance="Each clip must not exceed 5 minutes. Capture a full segment or story.",
    ),
}

_LEGACY_PRESET_MAP: dict[str, str] = {
    "8_30": "30",
    "15_30": "30",
    "auto": "60",
    "30_60": "60",
    "60_90": "90",
    "90_180": "180",
    "180_300": "300",
}


def normalize_clip_length_preset(preset: str | None) -> str:
    if preset in CLIP_DURATION_PRESETS:
        return str(preset)
    if preset in _LEGACY_PRESET_MAP:
        return _LEGACY_PRESET_MAP[preset]
    return DEFAULT_CLIP_LENGTH_PRESET


def get_clip_duration_preference(preset: str | None = None) -> ClipDurationPreference:
    return CLIP_DURATION_PRESETS[normalize_clip_length_preset(preset)]


def build_clip_duration_prompt_section(preset: str | None = None) -> str:
    preference = get_clip_duration_preference(preset)
    return f"""## Clip Length Preference

Maximum clip length: {preference.max_seconds} seconds
Minimum meaningful length: {preference.min_seconds} seconds

- Each clip must NOT exceed {preference.max_seconds} seconds
- Avoid clips shorter than {preference.min_seconds} seconds (too short to be meaningful)
- Prefer the shortest clip that captures the complete action/moment
- For sports: one shot/save/goal per clip, do not combine multiple actions into one clip
- If a moment naturally exceeds {preference.max_seconds}s, split into separate clips
- {preference.guidance}
"""
