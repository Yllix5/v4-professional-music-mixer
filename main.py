from __future__ import annotations

from pathlib import Path

from app.analysis.analyzer import (
    analyze_track,
    print_analysis_report,
)
from app.analysis.cache import (
    load_analysis,
    save_analysis,
)
from app.io.scanner import (
    print_scan_report,
    scan_folder,
)
from app.planner.planner import (
    create_mix_plan,
    print_mix_plan,
)
from app.render.renderer import (
    print_render_report,
    render_mix,
)


PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / ".cache" / "analysis"

OUTPUT_FILE = (
    OUTPUT_DIR / "test_mix_v4_renderer2.flac"
)


def analyze_with_cache(
    paths: list[Path],
):
    analyses = []

    print()
    print("=" * 70)
    print("V4 ANALYSIS")
    print("=" * 70)

    for path in paths:
        print()
        print(
            f"Analyzing: {path.name}"
        )

        cached = load_analysis(
            path,
            CACHE_DIR,
        )

        if cached is not None:
            print(
                "  ✓ Using cached analysis"
            )

            analyses.append(
                cached
            )

            continue

        print(
            "  → Running analysis..."
        )

        analysis = analyze_track(
            path
        )

        save_analysis(
            analysis,
            path,
            CACHE_DIR,
        )

        print(
            "  ✓ Analysis saved to cache"
        )

        analyses.append(
            analysis
        )

    return analyses


def main() -> None:
    # ---------------------------------------------------------
    # DIRECTORIES
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("V4 PROFESSIONAL MUSIC MIXER")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. SCAN
    # ---------------------------------------------------------

    print()
    print("[1/5] SCANNING INPUT")

    scan_result = scan_folder(
        INPUT_DIR
    )

    print_scan_report(
        scan_result
    )

    # ScanResult.valid contains AudioInfo objects.
    paths = [
        info.path
        for info in scan_result.valid
    ]

    if not paths:
        raise RuntimeError(
            "No valid FLAC/WAV files found "
            "in input."
        )

    # ---------------------------------------------------------
    # 2. ANALYSIS
    # ---------------------------------------------------------

    print()
    print("[2/5] ANALYZING TRACKS")

    analyses = analyze_with_cache(
        paths
    )

    print_analysis_report(
        analyses
    )

    # ---------------------------------------------------------
    # 3. MIX PLAN
    # ---------------------------------------------------------

    print()
    print("[3/5] BUILDING MIX PLAN")

    plan = create_mix_plan(
        analyses
    )

    # The current TrackPlan dataclass does not formally
    # contain the analysis object yet.
    #
    # Attach it dynamically so the renderer can use the
    # calculated gain/analysis information.

    for track_plan, analysis in zip(
        plan.tracks,
        analyses,
    ):
        track_plan.analysis = analysis

    print_mix_plan(
        plan
    )

    # ---------------------------------------------------------
    # 4. RENDER
    # ---------------------------------------------------------

    print()
    print("[4/5] RENDERING")

    result = render_mix(
        plan,
        OUTPUT_FILE,
    )

    print_render_report(
        result
    )

    # ---------------------------------------------------------
    # 5. COMPLETE
    # ---------------------------------------------------------

    print()
    print("[5/5] COMPLETE")

    print()
    print(
        "Mix created:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print()
    print(
        "Next step: transition previews + "
        "transition-quality testing."
    )


if __name__ == "__main__":
    main()

