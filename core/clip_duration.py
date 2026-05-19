from __future__ import annotations

from dataclasses import asdict, dataclass


DEFAULT_CLIP_LENGTH_PRESET = "auto"


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
    "auto": ClipDurationPreference(
        preset="auto",
        label="Auto (30s-3m)",
        min_seconds=30,
        max_seconds=180,
        ideal_min_seconds=45,
        ideal_max_seconds=180,
        guidance=(
            "Use the most natural engaging clip length between 30 seconds and 3 minutes. "
            "Prefer 45 seconds to 3 minutes when that captures the full context and payoff."
        ),
    ),
    "30_60": ClipDurationPreference(
        preset="30_60",
        label="30s-60s",
        min_seconds=30,
        max_seconds=60,
        ideal_min_seconds=30,
        ideal_max_seconds=60,
        guidance=(
            "Target a compact 30 to 60 second highlight. Prefer a tight setup and payoff, "
            "and do not include unrelated context."
        ),
    ),
    "60_90": ClipDurationPreference(
        preset="60_90",
        label="60s-90s",
        min_seconds=60,
        max_seconds=90,
        ideal_min_seconds=60,
        ideal_max_seconds=90,
        guidance=(
            "Target a 60 to 90 second clip with enough setup to stand alone and a clear payoff."
        ),
    ),
    "90_180": ClipDurationPreference(
        preset="90_180",
        label="90s-3m",
        min_seconds=90,
        max_seconds=180,
        ideal_min_seconds=90,
        ideal_max_seconds=180,
        guidance=(
            "Target a 90 second to 3 minute clip. Prefer a complete discussion beat, story, "
            "argument, or sequence rather than a brief isolated moment."
        ),
    ),
    "180_300": ClipDurationPreference(
        preset="180_300",
        label="3m-5m",
        min_seconds=180,
        max_seconds=300,
        ideal_min_seconds=180,
        ideal_max_seconds=300,
        guidance=(
            "Target a 3 to 5 minute complete segment, story, argument, tutorial section, "
            "or discussion arc. Do not stretch a short highlight with weak surrounding material."
        ),
    ),
}


def normalize_clip_length_preset(preset: str | None) -> str:
    if preset in CLIP_DURATION_PRESETS:
        return str(preset)
    return DEFAULT_CLIP_LENGTH_PRESET


def get_clip_duration_preference(preset: str | None = None) -> ClipDurationPreference:
    return CLIP_DURATION_PRESETS[normalize_clip_length_preset(preset)]


def build_clip_duration_prompt_section(preset: str | None = None) -> str:
    preference = get_clip_duration_preference(preset)
    return f"""## Clip Length Preference

Target clip length: {preference.label}
Hard duration bounds: {preference.min_seconds}-{preference.max_seconds} seconds
Ideal range: {preference.ideal_min_seconds}-{preference.ideal_max_seconds} seconds

This section is the source of truth for duration constraints in this run.
- Prefer clips whose natural arc fits the selected range.
- Preserve semantic completeness over exact duration when choosing start and end boundaries.
- Do not pad weak or unrelated context just to satisfy the selected length.
- {preference.guidance}
"""
