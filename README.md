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
- Technical quality-control tools

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
QUALITY CONTROL
  ↓
FINAL FLAC
```

## Requirements

- Windows
- Python 3.13+
- FFmpeg
- FLAC or WAV audio files

Python dependencies are listed in `requirements.txt`.

FFmpeg is required by the project but is not a Python package, so it must be installed separately and available in your system `PATH`.

## Quick Start

### 1. Clone the repository

```powershell
git clone https://github.com/yllix5/v4-professional-music-mixer.git
cd v4-professional-music-mixer
```

### 2. Run setup

Run the setup script once:

```powershell
.\setup.ps1
```

The setup script automatically:

- checks that Python is installed
- checks that FFmpeg is available
- creates the Python virtual environment
- installs the required Python dependencies
- creates the required project directories

After setup is complete, you do not need to manually activate the virtual environment.

### 3. Add your music

Put your FLAC or WAV files into:

```text
input\
```

For example:

```text
v4-professional-music-mixer/
├── input/
│   ├── song1.flac
│   ├── song2.flac
│   ├── song3.wav
│   └── song4.flac
```

Supported formats:

- `.flac`
- `.wav`

The mixer automatically scans the `input\` directory and processes the supported audio files it finds there.

Do not put your music files inside `app\`, `output\`, or other project directories.

### 4. Run the mixer

After adding your music, run:

```powershell
.\run.ps1
```

The launcher automatically starts the project's Python environment and runs the mixer.

The mixer will:

1. scan the input files
2. validate the audio
3. analyze the tracks
4. use cached analysis when available
5. plan transitions
6. render the continuous mix
7. write the resulting FLAC file

### 5. Find your finished mix

Generated mixes are saved in:

```text
output\
```

Your final FLAC file will be available there after rendering completes.

## First-Time Setup vs Normal Usage

### First time

```text
Clone repository
      ↓
Run setup.ps1
      ↓
Put FLAC/WAV files in input\
      ↓
Run run.ps1
      ↓
Get mix from output\
```

### Future mixes

Once setup has been completed, you only need to:

1. add or replace your music in `input\`
2. run:

```powershell
.\run.ps1
```

3. check the result in:

```text
output\
```

You do not need to manually run Python or activate the virtual environment.

## Project Structure

```text
v4-professional-music-mixer/
├── app/
│   ├── analysis/
│   │   ├── analyzer.py
│   │   ├── cache.py
│   │   └── tempo.py
│   ├── dsp/
│   │   ├── audio.py
│   │   └── gain.py
│   ├── io/
│   │   └── scanner.py
│   ├── planner/
│   │   ├── models.py
│   │   └── planner.py
│   ├── render/
│   │   └── renderer.py
│   └── validation/
│       └── qc.py
├── config/
├── tests/
├── input/
├── output/
├── logs/
├── .cache/
├── main.py
├── requirements.txt
├── setup.ps1
├── run.ps1
├── .gitignore
├── LICENSE
└── README.md
```

The `input\`, `output\`, `logs\`, and `.cache\` directories are created automatically by `setup.ps1` if they do not already exist.

## Audio Analysis

Each track is analyzed for information including:

- BPM
- tempo candidates
- beat positions
- downbeats
- tempo confidence
- musical key
- key confidence
- RMS/loudness
- peak level
- spectral characteristics
- bass activity
- vocal activity
- intro/outro regions
- structural information

Analysis results are cached using a SHA-256 fingerprint of the source file.

If a track has already been analyzed and has not changed, the cached analysis can be reused instead of repeating the full analysis.

## Transition Planning

The transition planner considers multiple characteristics when deciding how tracks should connect:

- tempo compatibility
- BPM difference
- harmonic/key compatibility
- energy compatibility
- bass activity
- vocal activity
- song structure
- transition duration
- pitch-shift limits
- BPM adjustment limits

The planner attempts to produce musical transitions while keeping time-stretching and pitch adjustments within controlled limits.

## DSP and Rendering

The rendering pipeline currently provides:

- stereo processing
- 44.1 kHz output
- 24-bit FLAC output
- automatic gain matching
- equal-power crossfades
- peak protection
- finite-sample protection
- controlled audio resampling

The goal is to produce a continuous mix with smooth transitions while avoiding unnecessary destructive processing.

## Quality Control

The project includes technical quality-control tools for checking rendered audio.

Checks include areas such as:

- file existence
- FLAC decoding
- valid audio samples
- stereo output
- clipping
- peak headroom
- excessive silence
- discontinuities
- duration
- RMS level
- sample rate

Quality-control functionality is actively being improved as the transition engine develops.

## Technology

Built with:

- Python
- NumPy
- SciPy
- SoundFile
- librosa
- Pydantic
- Rich
- FFmpeg

## Philosophy

The project focuses on deterministic and reusable automation rather than manually creating every mix inside a DAW.

The goal is not simply to concatenate songs or apply a basic crossfade.

The system is designed to:

1. analyze each track
2. understand important musical characteristics
3. plan compatible transitions
4. apply controlled DSP processing
5. render a continuous lossless mix
6. validate the resulting audio

The long-term goal is a professional automated mixing engine capable of producing consistent, musical, technically safe long-form mixes.

## License

This project is released under the MIT License.

See `LICENSE` for the full license text.

## Development

The project is under active development.

The current focus is improving transition intelligence, phrase alignment, transition quality analysis, and long-form mix reliability while keeping the system reusable and automated.
