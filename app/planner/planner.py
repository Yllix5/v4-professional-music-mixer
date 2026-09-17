from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from collections import defaultdict
import re

import numpy as np

from app.analysis.analyzer import TrackAnalysis
from .models import MixPlan, TrackPlan, TransitionPlan


MIN_TRANSITION = 6.0
MAX_TRANSITION = 16.0

MAX_PITCH_SHIFT = 2.0
MAX_BPM_ADJUSTMENT_PERCENT = 5.0

# Score weights (të rregulluara për tingull më profesional)
WEIGHT_TEMPO = 0.28
WEIGHT_KEY = 0.16
WEIGHT_ENERGY = 0.18
WEIGHT_STRUCTURE = 0.12
WEIGHT_BASS = 0.08
WEIGHT_VOCAL = 0.08
WEIGHT_DURATION = 0.04
WEIGHT_ARTIST = 0.06   # preferencë e lehtë për artist të njëjtë


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _extract_artist(path: Path) -> str:
    """Nxjerr emrin e artistit nga emri i skedarit (para ' - ')."""
    name = path.stem
    if " - " in name:
        return name.split(" - ", 1)[0].strip().lower()
    return name.strip().lower()


def _normalise_bpm_difference(source_bpm: float, target_bpm: float) -> float:
    if source_bpm <= 0 or target_bpm <= 0:
        return 1.0
    difference = abs(source_bpm - target_bpm)
    return _clamp(1.0 - difference / 18.0, 0.0, 1.0)


def _bpm_ratio(source_bpm: float, target_bpm: float) -> float:
    if source_bpm <= 0 or target_bpm <= 0:
        return 1.0
    return target_bpm / source_bpm


def _pitch_shift_for_ratio(ratio: float) -> float:
    if ratio <= 0:
        return 0.0
    semitones = 12.0 * np.log2(ratio)
    if abs(semitones) > MAX_PITCH_SHIFT:
        return 0.0
    return float(semitones)


def _bpm_adjustment_percent(source_bpm: float, target_bpm: float) -> float:
    if source_bpm <= 0:
        return 0.0
    return abs(target_bpm - source_bpm) / source_bpm * 100.0


def _energy_score(source: TrackAnalysis, target: TrackAnalysis) -> float:
    difference = abs(source.rms_dbfs - target.rms_dbfs)
    return _clamp(1.0 - difference / 5.5, 0.0, 1.0)


def _key_root_and_mode(key: str) -> tuple[int | None, str | None]:
    if not key or key == "Unknown":
        return None, None
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    parts = key.strip().split()
    if len(parts) != 2:
        return None, None
    root_name, mode = parts[0], parts[1]
    if root_name not in names:
        return None, None
    return names.index(root_name), mode


def _key_score(source: TrackAnalysis, target: TrackAnalysis) -> float:
    source_root, source_mode = _key_root_and_mode(source.key)
    target_root, target_mode = _key_root_and_mode(target.key)
    if source_root is None or target_root is None:
        return 0.5

    distance = abs(source_root - target_root)
    circle_distance = min(distance, 12 - distance)

    if source_root == target_root and source_mode == target_mode:
        return 1.0
    if source_mode != target_mode and circle_distance in (3, 4):
        return 0.88
    if circle_distance == 1:
        return 0.85
    if circle_distance == 2:
        return 0.72
    if circle_distance == 3:
        return 0.58
    if circle_distance == 4:
        return 0.42
    if circle_distance == 5:
        return 0.30
    return 0.18


def _structure_score(source: TrackAnalysis, target: TrackAnalysis) -> float:
    source_valid = source.outro_start > 0 and source.outro_start < source.duration
    target_valid = target.intro_end >= 0 and target.intro_end < target.duration
    if not source_valid or not target_valid:
        return 0.40
    source_remaining = source.duration - source.outro_start
    target_intro = target.intro_end
    if source_remaining >= 10.0 and target_intro <= 18.0:
        return 1.0
    if source_remaining >= 6.0 and target_intro <= 28.0:
        return 0.78
    return 0.55


def _bass_score(source: TrackAnalysis, target: TrackAnalysis) -> float:
    difference = abs(source.bass_activity - target.bass_activity)
    return _clamp(1.0 - difference / 0.18, 0.0, 1.0)


def _vocal_collision_score(source: TrackAnalysis, target: TrackAnalysis) -> float:
    overlap = source.vocal_activity * target.vocal_activity
    return _clamp(1.0 - overlap / 0.28, 0.0, 1.0)


def _duration_score(source: TrackAnalysis, target: TrackAnalysis) -> float:
    if source.duration < 70 or target.duration < 70:
        return 0.35
    return 1.0


def _artist_score(source: TrackAnalysis, target: TrackAnalysis) -> float:
    """Preferencë e lehtë për artist të njëjtë (por jo e detyrueshme)."""
    src_artist = _extract_artist(source.path)
    tgt_artist = _extract_artist(target.path)
    if src_artist == tgt_artist:
        return 1.0
    return 0.55


def _transition_score(source: TrackAnalysis, target: TrackAnalysis) -> float:
    tempo_score = _normalise_bpm_difference(source.bpm, target.bpm)
    key_score = _key_score(source, target)
    energy_score = _energy_score(source, target)
    structure_score = _structure_score(source, target)
    bass_score = _bass_score(source, target)
    vocal_score = _vocal_collision_score(source, target)
    duration_score = _duration_score(source, target)
    artist_score = _artist_score(source, target)

    confidence_factor = 0.55 + 0.45 * min(source.tempo_confidence, target.tempo_confidence)

    score = (
        tempo_score * WEIGHT_TEMPO
        + key_score * WEIGHT_KEY
        + energy_score * WEIGHT_ENERGY
        + structure_score * WEIGHT_STRUCTURE
        + bass_score * WEIGHT_BASS
        + vocal_score * WEIGHT_VOCAL
        + duration_score * WEIGHT_DURATION
        + artist_score * WEIGHT_ARTIST
    )
    return float(_clamp(score * confidence_factor, 0.0, 1.0))


def _choose_transition_duration(source: TrackAnalysis, target: TrackAnalysis) -> float:
    """Transition më i gjatë = më profesional (si Apple Music)."""
    bpm_difference = abs(source.bpm - target.bpm)
    energy_difference = abs(source.rms_dbfs - target.rms_dbfs)

    if bpm_difference <= 2.0 and energy_difference <= 1.5:
        return 8.0
    if bpm_difference <= 4.0 and energy_difference <= 3.0:
        return 10.0
    if bpm_difference <= 8.0 and energy_difference <= 4.5:
        return 12.0
    if bpm_difference <= 14.0:
        return 14.0
    return 16.0


def _nearest_beat_before(beat_times: list[float], position: float) -> float:
    if not beat_times:
        return position
    beats = np.asarray(beat_times, dtype=float)
    valid = beats[beats <= position]
    if len(valid) == 0:
        return float(beats[0])
    return float(valid[-1])


def _nearest_beat_after(beat_times: list[float], position: float) -> float:
    if not beat_times:
        return position
    beats = np.asarray(beat_times, dtype=float)
    valid = beats[beats >= position]
    if len(valid) == 0:
        return float(beats[-1])
    return float(valid[0])


def _choose_end_position(source: TrackAnalysis, transition_duration: float) -> float:
    """
    Lë këngën të luajë sa më gjatë të jetë e mundur.
    Transition fillon afër fundit (jo te outro_start).
    """
    safe_margin = transition_duration + 1.8
    ideal_from_end = max(14.0, transition_duration + 5.0)

    latest_safe = source.duration - safe_margin
    ideal = source.duration - ideal_from_end

    # Për këngë të shkurtra mos e presim shumë herët
    if source.duration < 100:
        ideal = max(source.duration * 0.78, latest_safe - 10.0)

    ideal = _clamp(ideal, 0.0, max(latest_safe, 0.0))
    return _nearest_beat_before(source.beat_times, ideal)


def _choose_start_position(target: TrackAnalysis) -> float:
    """
    Fillon këngën e re më herët dhe më natyral.
    """
    ideal = 5.5

    if target.intro_end > 25:
        ideal = min(9.0, target.intro_end * 0.35)
    elif target.intro_end > 0:
        ideal = min(7.5, max(4.0, target.intro_end * 0.6))

    return _nearest_beat_after(target.beat_times, ideal)


def _choose_bpm_adjustment(source: TrackAnalysis, target: TrackAnalysis) -> tuple[float, float]:
    ratio = _bpm_ratio(source.bpm, target.bpm)
    adjustment_percent = _bpm_adjustment_percent(source.bpm, target.bpm)
    if adjustment_percent <= MAX_BPM_ADJUSTMENT_PERCENT:
        pitch_shift = _pitch_shift_for_ratio(ratio)
        return adjustment_percent, pitch_shift
    return 0.0, 0.0


def create_transition(
    source_index: int,
    target_index: int,
    source: TrackAnalysis,
    target: TrackAnalysis,
) -> TransitionPlan:
    duration = _choose_transition_duration(source, target)
    source_end = _choose_end_position(source, duration)
    target_start = _choose_start_position(target)
    bpm_ratio = _bpm_ratio(source.bpm, target.bpm)
    bpm_adjustment_percent, pitch_shift = _choose_bpm_adjustment(source, target)
    score = _transition_score(source, target)

    beat_aligned = bool(source.beat_times and target.beat_times)
    phrase_aligned = bool(source.downbeat_times and target.downbeat_times)

    reason_parts: list[str] = []
    bpm_difference = abs(source.bpm - target.bpm)
    energy_difference = abs(source.rms_dbfs - target.rms_dbfs)

    if bpm_difference <= 3.5:
        reason_parts.append("close tempo")
    else:
        reason_parts.append(f"tempo gap {bpm_difference:.1f} BPM")

    if energy_difference <= 2.0:
        reason_parts.append("matched energy")
    else:
        reason_parts.append(f"energy gap {energy_difference:.1f} dB")

    if source.key != "Unknown" and target.key != "Unknown":
        reason_parts.append(f"{source.key} -> {target.key}")

    if pitch_shift != 0.0:
        reason_parts.append(f"pitch {pitch_shift:+.2f} st")

    src_artist = _extract_artist(source.path)
    tgt_artist = _extract_artist(target.path)
    if src_artist == tgt_artist:
        reason_parts.append("same artist")

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


# ---------------------------------------------------------------------------
# SMART ORDERING – tempo-aware + artist preference e lehtë
# ---------------------------------------------------------------------------

def _score_sequence(analyses: list[TrackAnalysis], order: list[int]) -> float:
    if len(order) < 2:
        return 0.0
    total = 0.0
    for i in range(len(order) - 1):
        total += _transition_score(analyses[order[i]], analyses[order[i + 1]])
    return total / (len(order) - 1)


def _greedy_best_order(analyses: list[TrackAnalysis]) -> list[int]:
    """
    Greedy me dënim të lehtë për kërcime të forta të BPM
    dhe preferencë të lehtë për artist të njëjtë.
    """
    n = len(analyses)
    if n <= 1:
        return list(range(n))

    best_order: list[int] = list(range(n))
    best_score = -1.0

    for start in range(n):
        remaining = set(range(n))
        remaining.remove(start)
        order = [start]
        current = start

        while remaining:
            best_next = None
            best_s = -1.0

            for cand in remaining:
                s = _transition_score(analyses[current], analyses[cand])

                # Dënim i lehtë për gap të madh të BPM
                bpm_gap = abs(analyses[current].bpm - analyses[cand].bpm)
                if bpm_gap > 18:
                    s *= 0.72
                elif bpm_gap > 12:
                    s *= 0.85

                if s > best_s:
                    best_s = s
                    best_next = cand

            order.append(best_next)
            remaining.remove(best_next)
            current = best_next

        score = _score_sequence(analyses, order)
        if score > best_score:
            best_score = score
            best_order = order

    return best_order


def order_tracks_by_compatibility(analyses: list[TrackAnalysis]) -> list[TrackAnalysis]:
    n = len(analyses)
    if n <= 1:
        return analyses
    order = _greedy_best_order(analyses)
    return [analyses[i] for i in order]


def create_mix_plan(analyses: list[TrackAnalysis]) -> MixPlan:
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

    # Ri-rendit sipas kompatibilitetit muzikor
    analyses = order_tracks_by_compatibility(analyses)

    tracks: list[TrackPlan] = []
    transitions: list[TransitionPlan] = []
    current_time = 0.0

    for index, analysis in enumerate(analyses):
        track_plan = TrackPlan(
            index=index,
            path=analysis.path,
            start_time=current_time,
            end_time=current_time + analysis.duration,
            gain_db=0.0,
            intro_trim=analysis.intro_end,
            outro_trim=max(0.0, analysis.duration - analysis.outro_start),
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
        tracks[index] = replace(tracks[index], transition_to_next=transition)

    scores = [t.score for t in transitions]
    pitch_shifts = [abs(t.pitch_shift_semitones) for t in transitions]
    bpm_adjustments = []
    for t in transitions:
        if t.source_bpm > 0:
            adj = abs(t.target_bpm - t.source_bpm) / t.source_bpm * 100.0
        else:
            adj = 0.0
        bpm_adjustments.append(adj)

    return MixPlan(
        tracks=tracks,
        total_duration=current_time,
        transition_count=len(transitions),
        average_transition_score=float(np.mean(scores)) if scores else 0.0,
        minimum_transition_score=float(np.min(scores)) if scores else 0.0,
        maximum_pitch_shift=max(pitch_shifts) if pitch_shifts else 0.0,
        maximum_bpm_adjustment_percent=max(bpm_adjustments) if bpm_adjustments else 0.0,
    )


def _format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes}:{remaining:05.2f}"


def print_mix_plan(plan: MixPlan) -> None:
    print()
    print("=" * 70)
    print("V4 MIX PLAN  (Professional / Apple-Music style)")
    print("=" * 70)
    print(f"Tracks:                 {len(plan.tracks)}")
    print(f"Transitions:            {plan.transition_count}")
    print(f"Total source duration:  {_format_time(plan.total_duration)}")
    print(f"Average transition:     {plan.average_transition_score:.3f}")
    print(f"Minimum transition:     {plan.minimum_transition_score:.3f}")
    print(f"Maximum pitch shift:    {plan.maximum_pitch_shift:.2f} semitones")
    print(f"Maximum BPM adjustment: {plan.maximum_bpm_adjustment_percent:.2f}%")
    print()
    print("TRACK ORDER (compatibility-sorted)")
    print("-" * 70)

    for index, track in enumerate(plan.tracks, start=1):
        print(f"{index:02d}. {track.path.name}")
        print(f"    Position: {_format_time(track.start_time)} -> {_format_time(track.end_time)}")
        print(f"    Intro candidate: {_format_time(track.intro_trim)}")
        print(f"    Outro candidate: {_format_time(track.outro_trim)}")
        transition = track.transition_to_next
        if transition is not None:
            print(f"    -> next: {transition.duration:.1f}s | score {transition.score:.3f}")
            print(f"       BPM {transition.source_bpm:.2f} -> {transition.target_bpm:.2f}")
            print(f"       source end: {_format_time(transition.source_end)}")
            print(f"       target start: {_format_time(transition.target_start)}")
            print(f"       beat aligned: {transition.beat_aligned}")
            print(f"       phrase aligned: {transition.phrase_aligned}")
            print(f"       reason: {transition.reason}")
    print("=" * 70)