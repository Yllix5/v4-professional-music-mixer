$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================"
Write-Host " V4 PROFESSIONAL MUSIC MIXER - SETUP"
Write-Host "============================================"
Write-Host ""

Write-Host "[1/4] Checking Python..."

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python was not found."
    Write-Host "Install Python 3.13 and run setup again."
    exit 1
}

python --version

Write-Host ""
Write-Host "[2/4] Checking FFmpeg..."

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: FFmpeg was not found."
    Write-Host "Install FFmpeg and make sure it is available in PATH."
    exit 1
}

ffmpeg -version | Select-Object -First 1

Write-Host ""
Write-Host "[3/4] Creating virtual environment..."

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

Write-Host ""
Write-Host "[4/4] Installing Python dependencies..."

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

New-Item -ItemType Directory -Force -Path "input" | Out-Null
New-Item -ItemType Directory -Force -Path "output" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path ".cache\analysis" | Out-Null

Write-Host ""
Write-Host "============================================"
Write-Host " SETUP COMPLETE"
Write-Host "============================================"
Write-Host ""
Write-Host "Put your FLAC/WAV files into:"
Write-Host "  input\"
Write-Host ""
Write-Host "Then run:"
Write-Host "  .\.venv\Scripts\python.exe main.py"
Write-Host ""
