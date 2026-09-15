from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class QCResult:
    name: str
    passed: bool
    value: str
    message: str


@dataclass
class MixQCReport:
    results: list[QCResult]
    passed: bool

    @property
    def failures(self) -> list[QCResult]:
        return [
            result
            for result in self.results
            if not result.passed
        ]


def _check_file_exists(
    path: Path,
) -> QCResult:
    passed = path.exists() and path.is_file()

    return QCResult(
        name="File exists",
        passed=passed,
        value=str(path),
        message=(
            "Output file exists."
            if passed
            else "Output file does not exist."
        ),
    )


def _load_output(
    path: Path,
) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    return audio, sample_rate


def _check_decode(
    path: Path,
) -> QCResult:
    try:
        audio, sample_rate = _load_output(path)

        if audio.size == 0:
            return QCResult(
                name="FLAC decode",
                passed=False,
                value="0 samples",
                message="Decoded file contains no audio.",
            )

        return QCResult(
            name="FLAC decode",
            passed=True,
            value=f"{sample_rate} Hz / {audio.shape[1]} ch",
            message="FLAC decoded successfully.",
        )

    except Exception as exc:
        return QCResult(
            name="FLAC decode",
            passed=False,
            value="ERROR",
            message=f"Decode failed: {exc}",
        )


def _check_finite(
    audio: np.ndarray,
) -> QCResult:
    finite = bool(
        np.all(np.isfinite(audio))
    )

    bad_count = int(
        np.size(audio)
        - np.count_nonzero(
            np.isfinite(audio)
        )
    )

    return QCResult(
        name="Sample integrity",
        passed=finite,
        value=f"invalid samples: {bad_count}",
        message=(
            "All samples are finite."
            if finite
            else "NaN or infinite samples detected."
        ),
    )


def _check_channels(
    audio: np.ndarray,
) -> QCResult:
    channels = audio.shape[1]

    passed = channels == 2

    return QCResult(
        name="Stereo integrity",
        passed=passed,
        value=f"{channels} channels",
        message=(
            "Stereo output confirmed."
            if passed
            else "Expected stereo output."
        ),
    )


def _check_peak(
    audio: np.ndarray,
) -> QCResult:
    peak = float(
        np.max(np.abs(audio))
    )

    if peak <= 0:
        peak_dbfs = -120.0
    else:
        peak_dbfs = (
            20.0 * np.log10(peak)
        )

    # Digital sample peak must never exceed 0 dBFS.
    passed = peak <= 1.0 + 1e-7

    return QCResult(
        name="Clipping",
        passed=passed,
        value=f"{peak_dbfs:.2f} dBFS",
        message=(
            "No digital clipping detected."
            if passed
            else "Samples exceed 0 dBFS."
        ),
    )


def _check_headroom(
    audio: np.ndarray,
    ceiling_dbfs: float = -1.0,
) -> QCResult:
    peak = float(
        np.max(np.abs(audio))
    )

    if peak <= 0:
        peak_dbfs = -120.0
    else:
        peak_dbfs = (
            20.0 * np.log10(peak)
        )

    passed = peak_dbfs <= ceiling_dbfs

    return QCResult(
        name="Peak headroom",
        passed=passed,
        value=f"{peak_dbfs:.2f} dBFS",
        message=(
            f"Peak is below {ceiling_dbfs:.1f} dBFS ceiling."
            if passed
            else (
                f"Peak exceeds the "
                f"{ceiling_dbfs:.1f} dBFS ceiling."
            )
        ),
    )


def _check_silence(
    audio: np.ndarray,
    sample_rate: int,
) -> QCResult:
    mono = np.mean(
        audio,
        axis=1,
    )

    frame_size = max(
        1,
        int(sample_rate * 1.0),
    )

    frame_count = (
        len(mono) // frame_size
    )

    if frame_count == 0:
        return QCResult(
            name="Unexpected silence",
            passed=True,
            value="file < 1 second",
            message="File is too short for silence analysis.",
        )

    silent_frames = 0

    for index in range(frame_count):
        start = index * frame_size
        end = start + frame_size

        frame = mono[start:end]

        rms = float(
            np.sqrt(
                np.mean(
                    frame * frame
                )
            )
        )

        if rms < 1e-5:
            silent_frames += 1

    silence_ratio = (
        silent_frames / frame_count
    )

    # Some silence is normal, but a large percentage
    # indicates a possible rendering problem.
    passed = silence_ratio < 0.05

    return QCResult(
        name="Unexpected silence",
        passed=passed,
        value=f"{silence_ratio * 100:.2f}%",
        message=(
            "No excessive silence detected."
            if passed
            else "Large silent sections detected."
        ),
    )


def _check_discontinuities(
    audio: np.ndarray,
) -> QCResult:
    if audio.shape[0] < 2:
        return QCResult(
            name="Discontinuities",
            passed=True,
            value="too short",
            message="Audio too short for discontinuity test.",
        )

    mono = np.mean(
        audio,
        axis=1,
    )

    differences = np.abs(
        np.diff(mono)
    )

    if len(differences) == 0:
        return QCResult(
            name="Discontinuities",
            passed=True,
            value="none",
            message="No discontinuities detected.",
        )

    # A huge one-sample jump can indicate a click.
    threshold = 0.8

    count = int(
        np.count_nonzero(
            differences > threshold
        )
    )

    passed = count == 0

    return QCResult(
        name="Discontinuities",
        passed=passed,
        value=f"{count} large jumps",
        message=(
            "No severe sample discontinuities detected."
            if passed
            else "Potential clicks/discontinuities detected."
        ),
    )


def _check_duration(
    audio: np.ndarray,
    sample_rate: int,
    expected_duration: float | None,
) -> QCResult:
    duration = (
        audio.shape[0]
        / sample_rate
    )

    if expected_duration is None:
        return QCResult(
            name="Duration",
            passed=True,
            value=f"{duration:.2f}s",
            message="Duration recorded.",
        )

    difference = abs(
        duration - expected_duration
    )

    passed = difference <= 1.0

    return QCResult(
        name="Duration",
        passed=passed,
        value=(
            f"{duration:.2f}s "
            f"(difference {difference:.2f}s)"
        ),
        message=(
            "Duration is within tolerance."
            if passed
            else "Duration differs from expected value."
        ),
    )


def _check_loudness(
    audio: np.ndarray,
) -> QCResult:
    rms = float(
        np.sqrt(
            np.mean(
                audio * audio
            )
        )
    )

    if rms <= 0:
        rms_dbfs = -120.0
    else:
        rms_dbfs = (
            20.0 * np.log10(rms)
        )

    # First-generation sanity range.
    passed = (
        -30.0
        <= rms_dbfs
        <= -6.0
    )

    return QCResult(
        name="Loudness sanity",
        passed=passed,
        value=f"{rms_dbfs:.2f} dBFS RMS",
        message=(
            "Overall level is within sanity range."
            if passed
            else "Overall level looks abnormal."
        ),
    )


def run_qc(
    path: Path,
    expected_duration: float | None = None,
    expected_sample_rate: int | None = None,
) -> MixQCReport:
    results: list[QCResult] = []

    exists_result = _check_file_exists(path)
    results.append(exists_result)

    if not exists_result.passed:
        return MixQCReport(
            results=results,
            passed=False,
        )

    decode_result = _check_decode(path)
    results.append(decode_result)

    if not decode_result.passed:
        return MixQCReport(
            results=results,
            passed=False,
        )

    try:
        audio, sample_rate = _load_output(path)

        results.append(
            _check_finite(audio)
        )

        results.append(
            _check_channels(audio)
        )

        results.append(
            _check_peak(audio)
        )

        results.append(
            _check_headroom(audio)
        )

        results.append(
            _check_silence(
                audio,
                sample_rate,
            )
        )

        results.append(
            _check_discontinuities(
                audio
            )
        )

        results.append(
            _check_duration(
                audio,
                sample_rate,
                expected_duration,
            )
        )

        results.append(
            _check_loudness(
                audio
            )
        )

        if expected_sample_rate is not None:
            passed = (
                sample_rate
                == expected_sample_rate
            )

            results.append(
                QCResult(
                    name="Sample rate",
                    passed=passed,
                    value=f"{sample_rate} Hz",
                    message=(
                        "Sample rate is correct."
                        if passed
                        else (
                            f"Expected "
                            f"{expected_sample_rate} Hz."
                        )
                    ),
                )
            )

    except Exception as exc:
        results.append(
            QCResult(
                name="QC processing",
                passed=False,
                value="ERROR",
                message=str(exc),
            )
        )

    overall = all(
        result.passed
        for result in results
    )

    return MixQCReport(
        results=results,
        passed=overall,
    )


def print_qc_report(
    report: MixQCReport,
) -> None:
    print()
    print("=" * 70)
    print("V4 QUALITY CONTROL")
    print("=" * 70)

    for result in report.results:
        status = (
            "PASS"
            if result.passed
            else "FAIL"
        )

        print(
            f"{status:<6} "
            f"{result.name:<22} "
            f"{result.value}"
        )

        print(
            f"       {result.message}"
        )

    print("-" * 70)

    if report.passed:
        print(
            "QUALITY STATUS: PASS"
        )
    else:
        print(
            "QUALITY STATUS: FAIL"
        )

    print("=" * 70)