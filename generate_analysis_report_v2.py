from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tracks = pd.DataFrame([
    {"track": "lluni – Jbmtqr", "bpm": 101.33, "key": "A minor", "tempo_conf": 0.45, "key_conf": 0.99, "rms": -14.99, "peak": -0.50},
    {"track": "Mc Kresha – Rebele", "bpm": 103.36, "key": "F minor", "tempo_conf": 0.37, "key_conf": 0.97, "rms": -10.38, "peak": 0.00},
    {"track": "Mc Kresha – Süße", "bpm": 109.96, "key": "A minor", "tempo_conf": 0.42, "key_conf": 0.96, "rms": -10.96, "peak": -0.29},
    {"track": "Young Zerka – Nafije", "bpm": 101.33, "key": "C# minor", "tempo_conf": 0.40, "key_conf": 0.93, "rms": -12.32, "peak": -0.22},
])

transitions = pd.DataFrame([
    {"from": "lluni – Jbmtqr", "to": "Mc Kresha – Rebele", "duration": 10, "score": 0.429, "pitch": 0.34},
    {"from": "Mc Kresha – Rebele", "to": "Mc Kresha – Süße", "duration": 8, "score": 0.411, "pitch": 0.00},
    {"from": "Mc Kresha – Süße", "to": "Young Zerka – Nafije", "duration": 10, "score": 0.454, "pitch": 0.00},
])

tracks["short"] = tracks["track"].str.replace("Mc Kresha – ", "MK – ", regex=False).str.replace("Young Zerka – ", "YZ – ", regex=False)

avg_bpm = tracks.bpm.mean()
avg_rms = tracks.rms.mean()
avg_peak = tracks.peak.mean()
avg_tempo_conf = tracks.tempo_conf.mean()
avg_key_conf = tracks.key_conf.mean()
avg_transition_score = transitions.score.mean()
unique_keys = tracks.key.nunique()

fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(4, 6, height_ratios=[0.95, 1.55, 1.55, 1.35], hspace=0.55, wspace=0.65)

# Header
ax = fig.add_subplot(gs[0, :])
ax.axis("off")
ax.text(0.00, 0.86, "V4 PROFESSIONAL MUSIC MIXER", fontsize=27, fontweight="bold")
ax.text(0.00, 0.53, "Automated music analysis • transition planning • DSP-aware rendering", fontsize=14)
ax.text(0.00, 0.19, f"{len(tracks)} tracks  •  {avg_bpm:.1f} BPM avg  •  {unique_keys} musical keys  •  {len(transitions)} planned transitions", fontsize=11)

# Tempo
ax = fig.add_subplot(gs[1, :3])
x = np.arange(len(tracks))
ax.plot(x, tracks.bpm, marker="o", linewidth=2.5)
ax.axhline(avg_bpm, linestyle="--", linewidth=1)
ax.set_title("Tempo profile", loc="left", fontweight="bold", fontsize=13)
ax.set_ylabel("BPM")
ax.set_xticks(x)
ax.set_xticklabels(tracks.short, rotation=25, ha="right")
ax.grid(axis="y", alpha=0.25)
for i, v in enumerate(tracks.bpm):
    ax.annotate(f"{v:.1f}", (i, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)

# Loudness
ax = fig.add_subplot(gs[1, 3:])
width = 0.36
ax.bar(x - width/2, tracks.rms, width, label="RMS")
ax.bar(x + width/2, tracks.peak, width, label="Peak")
ax.axhline(-14, linestyle="--", linewidth=1, label="Target RMS")
ax.set_title("Level & headroom", loc="left", fontweight="bold", fontsize=13)
ax.set_ylabel("dBFS")
ax.set_xticks(x)
ax.set_xticklabels(tracks.short, rotation=25, ha="right")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, ncol=3, fontsize=8)

# Confidence
ax = fig.add_subplot(gs[2, :3])
w = 0.36
ax.bar(x - w/2, tracks.tempo_conf, w, label="Tempo")
ax.bar(x + w/2, tracks.key_conf, w, label="Key")
ax.set_ylim(0, 1.08)
ax.set_title("Analysis confidence", loc="left", fontweight="bold", fontsize=13)
ax.set_ylabel("Confidence")
ax.set_xticks(x)
ax.set_xticklabels(tracks.short, rotation=25, ha="right")
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False, fontsize=8)

# Transition score
ax = fig.add_subplot(gs[2, 3:])
tx = np.arange(len(transitions))
labels = ["lluni → Rebele", "Rebele → Süße", "Süße → Nafije"]
bars = ax.bar(tx, transitions.score, width=0.58)
ax.set_ylim(0, 0.55)
ax.set_title("Transition planning score", loc="left", fontweight="bold", fontsize=13)
ax.set_ylabel("Compatibility score")
ax.set_xticks(tx)
ax.set_xticklabels(labels, rotation=18, ha="right")
ax.grid(axis="y", alpha=0.25)
for b, s in zip(bars, transitions.score):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.015, f"{s:.3f}", ha="center", fontsize=9)

# Table
ax = fig.add_subplot(gs[3, :4])
ax.axis("off")
table_rows = []
for _, r in tracks.iterrows():
    table_rows.append([
        r["short"],
        f'{r["bpm"]:.1f}',
        r["key"],
        f'{r["tempo_conf"]:.2f}',
        f'{r["key_conf"]:.2f}',
        f'{r["rms"]:.1f}',
        f'{r["peak"]:.1f}',
    ])
table = ax.table(
    cellText=table_rows,
    colLabels=["Track", "BPM", "Key", "Tempo C.", "Key C.", "RMS", "Peak"],
    loc="center",
    cellLoc="center",
    colLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.55)
ax.set_title("Per-track analysis snapshot", loc="left", fontweight="bold", pad=12, fontsize=13)

# Summary
ax = fig.add_subplot(gs[3, 4:])
ax.axis("off")
detail_lines = [
    ("Avg transition score", f"{transitions.score.mean():.3f}"),
    ("Transition time", f"{transitions.duration.sum():.0f}s"),
    ("Max pitch shift", f"{transitions.pitch.abs().max():.2f} st"),
    ("Avg tempo confidence", f"{tracks.tempo_conf.mean():.2f}"),
    ("Avg key confidence", f"{tracks.key_conf.mean():.2f}"),
]
y = 0.88
for label, value in detail_lines:
    ax.text(0.00, y, label, fontsize=9)
    ax.text(0.98, y, value, fontsize=11, fontweight="bold", ha="right")
    y -= 0.18
ax.set_title("Mix planning summary", loc="left", fontweight="bold", pad=12, fontsize=13)

fig.text(
    0.02, 0.015,
    "Python • librosa • NumPy • SciPy • SoundFile • FFmpeg   |   V4 test-run analysis snapshot",
    fontsize=9
)

fig.savefig(Path("music_analysis_report_v2.png"), dpi=220, bbox_inches="tight")
plt.close(fig)
print("Generated: music_analysis_report_v2.png")
