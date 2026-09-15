from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .analyzer import TrackAnalysis


CACHE_VERSION = 1
ANALYZER_VERSION = "0.4"


def calculate_file_hash(path: Path) -> str:
    """
    Calculate SHA-256 for the complete audio file.

    The cache is based on file contents rather than
    filename, so replacing a song with another file
    using the same filename will still trigger
    a fresh analysis.
    """

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def get_cache_path(
    audio_path: Path,
    cache_dir: Path,
) -> Path:

    file_hash = calculate_file_hash(
        audio_path
    )

    return cache_dir / f"{file_hash}.json"


def save_analysis(
    analysis: "TrackAnalysis",
    audio_path: Path,
    cache_dir: Path,
) -> None:
    """
    Save successful analysis to the cache.
    """

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_hash = calculate_file_hash(
        audio_path
    )

    data: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "analyzer_version": ANALYZER_VERSION,

        "file": {
            "name": audio_path.name,
            "sha256": file_hash,
            "size": audio_path.stat().st_size,
        },

        "analysis": asdict(analysis),
    }

    # Path objects cannot be written directly to JSON.
    data["analysis"]["path"] = str(
        audio_path
    )

    cache_path = (
        cache_dir / f"{file_hash}.json"
    )

    # Write to a temporary file first.
    # This prevents a partially-written cache
    # from being treated as valid.
    temporary_path = cache_path.with_suffix(
        ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(
        cache_path
    )


def load_analysis(
    audio_path: Path,
    cache_dir: Path,
) -> "TrackAnalysis | None":
    """
    Load analysis from cache if it is still valid.

    Returns None when:
    - cache does not exist
    - cache version changed
    - analyzer version changed
    - audio file changed
    - cache is corrupted
    """

    cache_path = get_cache_path(
        audio_path,
        cache_dir,
    )

    if not cache_path.exists():
        return None

    try:

        with cache_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        # -----------------------------------------------------
        # VERSION CHECK
        # -----------------------------------------------------

        if (
            data.get("cache_version")
            != CACHE_VERSION
        ):
            return None

        if (
            data.get("analyzer_version")
            != ANALYZER_VERSION
        ):
            return None

        # -----------------------------------------------------
        # FILE INTEGRITY CHECK
        # -----------------------------------------------------

        file_info = data.get(
            "file",
            {},
        )

        current_hash = (
            calculate_file_hash(
                audio_path
            )
        )

        if (
            file_info.get("sha256")
            != current_hash
        ):
            return None

        # -----------------------------------------------------
        # REBUILD TRACK ANALYSIS
        # -----------------------------------------------------

        from .analyzer import TrackAnalysis

        analysis_data = data[
            "analysis"
        ]

        analysis_data["path"] = (
            audio_path
        )

        return TrackAnalysis(
            **analysis_data
        )

    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        # A broken cache is not fatal.
        # The analyzer will simply analyze
        # the track again.
        return None