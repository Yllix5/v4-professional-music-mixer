from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ====================== DATA ======================
tracks = pd.DataFrame([
    {"track": "lluni – Jbmtqr", "bpm": 101.33, "key": "A minor", "tempo_conf": 0.45, "key_conf": 0.99, "rms": -14.99, "peak": -0.50},
    {"track": "Mc Kresha – Rebele", "bpm": 103.36, "key": "F minor", "tempo_conf": 0.37, "key_conf": 0.97, "rms": -10.38, "peak": 0.00},
    {"track": "Mc Kresha – Süße", "bpm": 109.96, "key": "A minor", "tempo_conf": 0.42, "key_conf": 0.96, "rms": -10.96, "peak": -0.29},
    {"track": "Young Zerka – Nafije", "bpm": 101.33, "key": "C# minor", "tempo_conf": 0.40, "key_conf": 0.93, "rms": -12.32, "peak": -0.22},
])

transitions = pd.DataFrame([
    {"from": "lluni → Rebele", "score": 0.429, "pitch": 0.34, "duration": 10},
    {"from": "Rebele → Süße", "score": 0.411, "pitch": 0.00, "duration": 8},
    {"from": "Süße → Nafije", "score": 0.454, "pitch": 0.00, "duration": 10},
])

tracks["short"] = ["lluni", "Rebele", "Süße", "Nafije"]

# Metrics
avg_bpm = tracks.bpm.mean()
avg_tempo_conf = tracks.tempo_conf.mean()
avg_key_conf = tracks.key_conf.mean()
avg_score = transitions.score.mean()
total_trans_time = transitions.duration.sum()
max_pitch = transitions.pitch.abs().max()
unique_keys = tracks.key.nunique()

# ====================== STYLE (pa warning) ======================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans", "Helvetica"],
    "font.weight": "normal",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#d1d5db",
    "axes.labelcolor": "#374151",
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "grid.color": "#f3f4f6",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titleweight": "bold",
})

# Colors
BLUE = "#2563eb"
ORANGE = "#f59e0b"
GREEN = "#059669"
GRAY = "#6b7280"
DARK = "#111827"

fig = plt.figure(figsize=(16, 9.5), dpi=160)
gs = gridspec.GridSpec(
    3, 4,
    height_ratios=[0.55, 1.35, 1.25],
    hspace=0.42,
    wspace=0.32,
    left=0.06, right=0.97, top=0.91, bottom=0.07
)

# ========== HEADER ==========
ax_header = fig.add_subplot(gs[0, :])
ax_header.axis("off")
ax_header.text(0.0, 0.72, "V4 PROFESSIONAL MUSIC MIXER", fontsize=20, fontweight="bold", color=DARK)
ax_header.text(0.0, 0.32, "Automated analysis  •  Transition planning  •  DSP-aware rendering",
               fontsize=10.5, color="#6b7280")
ax_header.text(0.0, 0.02, f"{len(tracks)} tracks   •   {avg_bpm:.1f} BPM avg   •   {unique_keys} keys   •   {len(transitions)} transitions",
               fontsize=9.5, color="#9ca3af")

# ========== TEMPO ==========
ax1 = fig.add_subplot(gs[1, 0:2])
x = np.arange(len(tracks))
ax1.plot(x, tracks.bpm, color=BLUE, linewidth=2.4, marker="o", markersize=7.5,
         markerfacecolor="white", markeredgewidth=2.2, markeredgecolor=BLUE)
ax1.axhline(avg_bpm, color=GRAY, linestyle="--", linewidth=1.1, alpha=0.7)
ax1.set_title("Tempo Profile", fontsize=11.5, loc="left", pad=8)
ax1.set_ylabel("BPM", fontsize=9.5)
ax1.set_xticks(x)
ax1.set_xticklabels(tracks.short, fontsize=9)
ax1.set_ylim(98, 113)
ax1.grid(axis="y", alpha=0.5)
for i, v in enumerate(tracks.bpm):
    ax1.annotate(f"{v:.1f}", (i, v), xytext=(0, 9), textcoords="offset points",
                 ha="center", fontsize=9, color=BLUE)

# ========== LOUDNESS ==========
ax2 = fig.add_subplot(gs[1, 2:4])
width = 0.34
ax2.bar(x - width/2, tracks.rms, width, label="RMS", color=BLUE, alpha=0.88)
ax2.bar(x + width/2, tracks.peak, width, label="Peak", color=ORANGE, alpha=0.88)
ax2.axhline(-14, color="#ef4444", linestyle="--", linewidth=1.2, alpha=0.75, label="Target -14 dB")
ax2.set_title("Level & Headroom", fontsize=11.5, loc="left", pad=8)
ax2.set_ylabel("dBFS", fontsize=9.5)
ax2.set_xticks(x)
ax2.set_xticklabels(tracks.short, fontsize=9)
ax2.set_ylim(-18, 2)
ax2.legend(frameon=False, fontsize=8, loc="upper right")
ax2.grid(axis="y", alpha=0.5)

# ========== CONFIDENCE ==========
ax3 = fig.add_subplot(gs[2, 0:2])
w = 0.34
ax3.bar(x - w/2, tracks.tempo_conf, w, label="Tempo", color=BLUE, alpha=0.88)
ax3.bar(x + w/2, tracks.key_conf, w, label="Key", color=ORANGE, alpha=0.88)
ax3.set_ylim(0, 1.15)
ax3.set_title("Analysis Confidence", fontsize=11.5, loc="left", pad=8)
ax3.set_ylabel("Confidence", fontsize=9.5)
ax3.set_xticks(x)
ax3.set_xticklabels(tracks.short, fontsize=9)
ax3.legend(frameon=False, fontsize=8)
ax3.grid(axis="y", alpha=0.5)

# ========== TRANSITION SCORES ==========
ax4 = fig.add_subplot(gs[2, 2])
colors = [BLUE, BLUE, GREEN]
bars = ax4.bar(transitions["from"], transitions.score, color=colors, width=0.55, alpha=0.9)
ax4.set_ylim(0, 0.55)
ax4.set_title("Transition Score", fontsize=11.5, loc="left", pad=8)
ax4.set_ylabel("Score", fontsize=9.5)
ax4.tick_params(axis="x", labelsize=8)
ax4.grid(axis="y", alpha=0.5)
for bar, val in zip(bars, transitions.score):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

# ========== SUMMARY ==========
ax5 = fig.add_subplot(gs[2, 3])
ax5.axis("off")
ax5.set_title("Mix Summary", fontsize=11.5, loc="left", pad=8)

summary_data = [
    ("Avg Transition Score", f"{avg_score:.3f}"),
    ("Total Transition Time", f"{total_trans_time}s"),
    ("Max Pitch Shift", f"{max_pitch:.2f} st"),
    ("Avg Tempo Conf.", f"{avg_tempo_conf:.2f}"),
    ("Avg Key Conf.", f"{avg_key_conf:.2f}"),
]

y = 0.80
for label, value in summary_data:
    ax5.text(0.04, y, label, fontsize=9, color="#6b7280")
    ax5.text(0.96, y, value, fontsize=10.5, fontweight="bold", ha="right", color=DARK)
    y -= 0.155

# Footer
fig.text(0.06, 0.018, "Python  •  librosa  •  NumPy  •  SciPy  •  SoundFile  •  FFmpeg",
         fontsize=8, color="#9ca3af")

# Save
out_path = Path("v4_music_analysis_dashboard.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✅ Saved: {out_path.resolve()}")