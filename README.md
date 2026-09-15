@'
# V4 Professional Music Mixer

A professional CLI music mixer designed to create long-form continuous mixes from FLAC/WAV files using automated audio analysis, intelligent transition planning, DSP processing, and lossless rendering.

## Features

- FLAC and WAV input
- Automatic audio scanning and validation
- BPM and beat analysis
- Beat and downbeat detection
- Key detection
- Energy and loudness analysis
- Spectral and bass analysis
- Vocal activity estimation
- Intelligent transition planning
- BPM-aware transitions
- Pitch-shift planning
- Automatic gain matching
- Equal-power crossfades
- Peak protection
- 24-bit FLAC rendering
- Analysis caching
- Technical quality control

## Processing Pipeline

```text
INPUT
  ↓
SCAN
  ↓
VALIDATE
  ↓
ANALYZE
  ↓
PLAN
  ↓
RENDER
  ↓
TECHNICAL QC
  ↓
TRANSITION QC
  ↓
FINAL FLAC