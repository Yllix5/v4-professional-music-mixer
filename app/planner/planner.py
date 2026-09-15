from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from app.analysis.analyzer import TrackAnalysis
from .models import MixPlan, TrackPlan, TransitionPlan


MIN_TRANSITION = 4.0
MAX_TRANSITION = 12.0

MAX_PITCH_SHIFT = 2.0
MAX_BPM_ADJUSTMENT_PERCENT = 4.0

# Score weights
WEIGHT_TEMPO = 0.25
WEIGHT_KEY = 0.15
WEIGHT_ENERGY = 0.20
WEIGHT_STRUCTURE = 0.15
WEIGHT_BASS = 0.10
WEIGHT_VOCAL = 0.10
WEIGHT_DURATION = 0.05


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalise_bpm_difference(
    source_bpm: float,
    target_bpm: float,
) -> float:
    if source_bpm <= 0 or target_bpm <= 0:
        return 1.0

    difference = abs(source_bpm - target_bpm)

    # 0 BPM difference = 1.0
    # 20+ BPM difference = 0.0
    return _clamp(
        1.0 - difference / 20.0,
        0.0,
        1.0,
    )


def _bpm_ratio(
    source_bpm: float,
    target_bpm: float,
) -> float:
    if source_bpm <= 0 or target_bpm <= 0:
        return 1.0

    return target_bpm / source_bpm


def _pitch_shift_for_ratio(
    ratio: float,
) -> float:
    if ratio <= 0:
        return 0.0

    semitones = 12.0 * np.log2(ratio)

    if abs(semitones) > MAX_PITCH_SHIFT:
        return 0.0

    return float(semitones)


def _bpm_adjustment_percent(
    source_bpm: float,
    target_bpm: float,
) -> float:
    if source_bpm <= 0:
        return 0.0

    return abs(
        target_bpm - source_bpm
    ) / source_bpm * 100.0


def _energy_score(
    source: TrackAnalysis,
    target: TrackAnalysis,
) -> float:
    difference = abs(
        source.rms_dbfs - target.rms_dbfs
    )

    # Within ~1 dB is excellent.
    # 6+ dB is poor.
    return _clamp(
        1.0 - difference / 6.0,
        0.0,
        1.0,
    )


def _key_root_and_mode(
    key: str,
) -> tuple[int | None, str | None]:
    if not key or key == "Unknown":
        return None, None

    names = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]

    parts = key.strip().split()

    if len(parts) != 2:
        return None, None

    root_name = parts[0]
    mode = parts[1]

    if root_name not in names:
        return None, None

    return names.index(root_name), mode


def _key_score(
    source: TrackAnalysis,
    target: TrackAnalysis,
) -> float:
    source_root, source_mode = _key_root_and_mode(
        source.key
    )
    target_root, target_mode = _key_root_and_mode(
        target.key
    )

    if source_root is None or target_root is None:
        return 0.5

    distance = abs(
        source_root - target_root
    )

    # Circle of fifths distance.
    circle_distance = min(
        distance,
        12 - distance,
    )

    if (
        source_root == target_root
        and source_mode == target_mode
    ):
        return 1.0

    # Relative major/minor.
    if (
        source_mode != target_mode
        and circle_distance in (3, 4)
    ):
        return 0.85

    if circle_distance == 1:
        return 0.85

    if circle_distance == 2:
        return 0.70

    if circle_distance == 3:
        return 0.55

    if circle_distance == 4:
        return 0.40

    if circle_distance == 5:
        return 0.30

    return 0.20


def _structure_score(
    source: TrackAnalysis,
    target: TrackAnalysis,
) -> float:
    source_valid = (
        source.outro_start > 0
        and source.outro_start < source.duration
    )

    target_valid = (
        target.intro_end >= 0
        and target.intro_end < target.duration
    )

    if not source_valid or not target_valid:
        return 0.35

    source_remaining = (
        source.duration - source.outro_start
    )

    target_intro = target.intro_end

    # We prefer transitions where both tracks
    # have enough material around the transition.
    if source_remaining >= 8.0 and target_intro <= 20.0:
        return 1.0

    if source_remaining >= 5.0 and target_intro <= 30.0:
        return 0.75

    return 0.50


def _bass_score(
    source: TrackAnalysis,
    target: TrackAnalysis,
) -> float:
    difference = abs(
        source.bass_activity
        - target.bass_activity
    )

    # Similar bass character is easier to blend.
    return _clamp(
        1.0 - difference / 0.20,
        0.0,
        1.0,
    )


def _vocal_collision_score(
    source: TrackAnalysis,
    target: TrackAnalysis,
) -> float:
    source_vocal = source.vocal_activity
    target_vocal = target.vocal_activity

    # Two vocal-heavy tracks overlapping is undesirable.
    overlap = source_vocal * target_vocal

    return _clamp(
        1.0 - overlap / 0.30,
        0.0,
        1.0,
    )


def _duration_score(
    source: TrackAnalysis,
    target: TrackAnalysis,
) -> float:
    if source.duration < 60 or target.duration < 60:
        return 0.3

    return 1.0


def _transition_score(
    source: TrackAnalysis,
    target: TrackAnalysis,
) -> float:
    tempo_score = _normalise_bpm_difference(
        source.bpm,
        target.bpm,
    )

    key_score = _key_score(
        source,
        target,
    )

    energy_score = _energy_score(
        source,
        target,
    )

    structure_score = _structure_score(
        source,
        target,
    )

    bass_score = _bass_score(
        source,
        target,
    )

    vocal_score = _vocal_collision_score(
        source,
        target,
    )

    duration_score = _duration_score(
        source,
        target,
    )

    confidence_factor = (
        0.5
        + 0.5
        * min(
            source.tempo_confidence,
            target.tempo_confidence,
        )
    )

    score = (
        tempo_score * WEIGHT_TEMPO
        + key_score * WEIGHT_KEY
        + energy_score * WEIGHT_ENERGY
        + structure_score * WEIGHT_STRUCTURE
        + bass_score * WEIGHT_BASS
        + vocal_score * WEIGHT_VOCAL
        + duration_score * WEIGHT_DURATION
    )

    return float(
        _clamp(
            score * confidence_factor,
            0.0,
            1.0,
        )
    )


def _choose_transition_duration(
    source: TrackAnalysis,
    target: TrackAnalysis,
) -> float:
    bpm_difference = abs(
        source.bpm - target.bpm
    )

    energy_difference = abs(
        source.rms_dbfs - target.rms_dbfs
    )

    # Large BPM or energy differences need
    # a longer, smoother transition.
    if (
        bpm_difference <= 2.0
        and energy_difference <= 1.5
    ):
        return 4.0

    if (
        bpm_difference <= 4.0
        and energy_difference <= 3.0
    ):
        return 6.0

    if (
        bpm_difference <= 8.0
        and energy_difference <= 4.5
    ):
        return 8.0

    if bpm_difference <= 12.0:
        return 10.0

    return 12.0


def _nearest_beat_before(
    beat_times: list[float],
    position: float,
) -> float:
    if not beat_times:
        return position

    beats = np.asarray(
        beat_times,
        dtype=float,
    )

    valid = beats[beats <= position]

    if len(valid) == 0:
        return float(beats[0])

    return float(valid[-1])


def _nearest_beat_after(
    beat_times: list[float],
    position: float,
) -> float:
    if not beat_times:
        return position

    beats = np.asarray(
        beat_times,
        dtype=float,
    )

    valid = beats[beats >= position]

    if len(valid) == 0:
        return float(beats[-1])

    return float(valid[0])


def _choose_end_position(
    source: TrackAnalysis,
    transition_duration: float,
) -> float:
    ideal = source.outro_start

    latest_safe = (
        source.duration
        - transition_duration
        - 0.25
    )

    ideal = _clamp(
        ideal,
        0.0,
        max(latest_safe, 0.0),
    )

    return _nearest_beat_before(
        source.beat_times,
        ideal,
    )


def _choose_start_position(
    target: TrackAnalysis,
) -> float:
    ideal = max(
        0.0,
        target.intro_end,
    )

    return _nearest_beat_after(
        target.beat_times,
        ideal,
    )


def _choose_bpm_adjustment(
    source: TrackAnalysis,
    target: TrackAnalysis,
) -> tuple[float, float]:
    ratio = _bpm_ratio(
        source.bpm,
        target.bpm,
    )

    adjustment_percent = _bpm_adjustment_percent(
        source.bpm,
        target.bpm,
    )

    if (
        adjustment_percent <= MAX_BPM_ADJUSTMENT_PERCENT
    ):
        pitch_shift = _pitch_shift_for_ratio(
            ratio
        )

        return (
            adjustment_percent,
            pitch_shift,
        )

    return 0.0, 0.0


def create_transition(
    source_index: int,
    target_index: int,
    source: TrackAnalysis,
    target: TrackAnalysis,
) -> TransitionPlan:
    duration = _choose_transition_duration(
        source,
        target,
    )

    source_end = _choose_end_position(
        source,
        duration,
    )

    target_start = _choose_start_position(
        target
    )

    bpm_ratio = _bpm_ratio(
        source.bpm,
        target.bpm,
    )

    bpm_adjustment_percent, pitch_shift = (
        _choose_bpm_adjustment(
            source,
            target,
        )
    )

    score = _transition_score(
        source,
        target,
    )

    beat_aligned = bool(
        source.beat_times
        and target.beat_times
    )

    phrase_aligned = bool(
        source.downbeat_times
        and target.downbeat_times
    )

    reason_parts: list[str] = []

    bpm_difference = abs(
        source.bpm - target.bpm
    )

    energy_difference = abs(
        source.rms_dbfs - target.rms_dbfs
    )

    if bpm_difference <= 4.0:
        reason_parts.append(
            "close tempo"
        )
    else:
        reason_parts.append(
            f"tempo gap {bpm_difference:.1f} BPM"
        )

    if energy_difference <= 2.0:
        reason_parts.append(
            "matched energy"
        )
    else:
        reason_parts.append(
            f"energy gap {energy_difference:.1f} dB"
        )

    if source.key != "Unknown" and target.key != "Unknown":
        reason_parts.append(
            f"{source.key} -> {target.key}"
        )

    if pitch_shift != 0.0:
        reason_parts.append(
            f"pitch {pitch_shift:+.2f} st"
        )

    reason = ", ".join(reason_parts)

    return TransitionPlan(
        source_index=source_index,
        target_index=target_index,
        source_path=source.path,
        target_path=target.path,
        source_end=source_end,
        target_start=target_start,
        duration=duration,
        source_bpm=source.bpm,
        target_bpm=target.bpm,
        bpm_ratio=bpm_ratio,
        pitch_shift_semitones=pitch_shift,
        source_gain_db=0.0,
        target_gain_db=0.0,
        score=score,
        beat_aligned=beat_aligned,
        phrase_aligned=phrase_aligned,
        reason=reason,
    )


def create_mix_plan(
    analyses: list[TrackAnalysis],
) -> MixPlan:
    if not analyses:
        return MixPlan(
            tracks=[],
            total_duration=0.0,
            transition_count=0,
            average_transition_score=0.0,
            minimum_transition_score=0.0,
            maximum_pitch_shift=0.0,
            maximum_bpm_adjustment_percent=0.0,
        )

    tracks: list[TrackPlan] = []

    transitions: list[TransitionPlan] = []

    current_time = 0.0

    for index, analysis in enumerate(
        analyses
    ):
        track_plan = TrackPlan(
            index=index,
            path=analysis.path,
            start_time=current_time,
            end_time=current_time + analysis.duration,
            gain_db=0.0,
            intro_trim=analysis.intro_end,
            outro_trim=max(
                0.0,
                analysis.duration
                - analysis.outro_start,
            ),
            transition_to_next=None,
        )

        tracks.append(track_plan)

        current_time += analysis.duration

    for index in range(len(analyses) - 1):
        source = analyses[index]
        target = analyses[index + 1]

        transition = create_transition(
            source_index=index,
            target_index=index + 1,
            source=source,
            target=target,
        )

        transitions.append(transition)

        tracks[index] = replace(
            tracks[index],
            transition_to_next=transition,
        )

    scores = [
        transition.score
        for transition in transitions
    ]

    pitch_shifts = [
        abs(
            transition.pitch_shift_semitones
        )
        for transition in transitions
    ]

    bpm_adjustments = []

    for transition in transitions:
        if transition.source_bpm > 0:
            adjustment = (
                abs(
                    transition.target_bpm
                    - transition.source_bpm
                )
                / transition.source_bpm
                * 100.0
            )
        else:
            adjustment = 0.0

        bpm_adjustments.append(
            adjustment
        )

    return MixPlan(
        tracks=tracks,
        total_duration=current_time,
        transition_count=len(transitions),
        average_transition_score=(
            float(np.mean(scores))
            if scores
            else 0.0
        ),
        minimum_transition_score=(
            float(np.min(scores))
            if scores
            else 0.0
        ),
        maximum_pitch_shift=(
            max(pitch_shifts)
            if pitch_shifts
            else 0.0
        ),
        maximum_bpm_adjustment_percent=(
            max(bpm_adjustments)
            if bpm_adjustments
            else 0.0
        ),
    )


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))

    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60

    return f"{minutes}:{remaining:05.2f}"


def print_mix_plan(
    plan: MixPlan,
) -> None:
    print()
    print("=" * 70)
    print("V4 MIX PLAN")
    print("=" * 70)

    print(
        f"Tracks:                 "
        f"{len(plan.tracks)}"
    )

    print(
        f"Transitions:            "
        f"{plan.transition_count}"
    )

    print(
        f"Total source duration:  "
        f"{_format_time(plan.total_duration)}"
    )

    print(
        f"Average transition:     "
        f"{plan.average_transition_score:.3f}"
    )

    print(
        f"Minimum transition:     "
        f"{plan.minimum_transition_score:.3f}"
    )

    print(
        f"Maximum pitch shift:    "
        f"{plan.maximum_pitch_shift:.2f} semitones"
    )

    print(
        f"Maximum BPM adjustment: "
        f"{plan.maximum_bpm_adjustment_percent:.2f}%"
    )

    print()
    print("TRACK ORDER")
    print("-" * 70)

    for index, track in enumerate(
        plan.tracks,
        start=1,
    ):
        print(
            f"{index:02d}. {track.path.name}"
        )

        print(
            f"    Position: "
            f"{_format_time(track.start_time)}"
            f" -> "
            f"{_format_time(track.end_time)}"
        )

        print(
            f"    Intro candidate: "
            f"{_format_time(track.intro_trim)}"
        )

        print(
            f"    Outro candidate: "
            f"{_format_time(track.outro_trim)}"
        )

        transition = track.transition_to_next

        if transition is not None:
            print(
                f"    -> next: "
                f"{transition.duration:.1f}s"
                f" | score "
                f"{transition.score:.3f}"
            )

            print(
                f"       BPM "
                f"{transition.source_bpm:.2f}"
                f" -> "
                f"{transition.target_bpm:.2f}"
            )

            print(
                f"       source end: "
                f"{_format_time(transition.source_end)}"
            )

            print(
                f"       target start: "
                f"{_format_time(transition.target_start)}"
            )

            print(
                f"       beat aligned: "
                f"{transition.beat_aligned}"
            )

            print(
                f"       phrase aligned: "
                f"{transition.phrase_aligned}"
            )

            print(
                f"       reason: "
                f"{transition.reason}"
            )

    print("=" * 70)