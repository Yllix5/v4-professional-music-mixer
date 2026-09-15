from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TransitionPlan:
    source_index: int
    target_index: int

    source_path: Path
    target_path: Path

    source_end: float
    target_start: float

    duration: float

    source_bpm: float
    target_bpm: float

    bpm_ratio: float
    pitch_shift_semitones: float

    source_gain_db: float
    target_gain_db: float

    score: float

    beat_aligned: bool
    phrase_aligned: bool

    reason: str


@dataclass
class TrackPlan:
    index: int
    path: Path

    start_time: float
    end_time: float

    gain_db: float

    intro_trim: float
    outro_trim: float

    transition_to_next: TransitionPlan | None = None


@dataclass
class MixPlan:
    tracks: list[TrackPlan]

    total_duration: float

    transition_count: int

    average_transition_score: float
    minimum_transition_score: float

    maximum_pitch_shift: float
    maximum_bpm_adjustment_percent: float
