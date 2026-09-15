$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================"
Write-Host " V4 PROFESSIONAL MUSIC MIXER"
Write-Host "============================================"
Write-Host ""

# Check virtual environment
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found."
    Write-Host ""
    Write-Host "Run setup first:"
    Write-Host "  .\setup.ps1"
    Write-Host ""
    exit 1
}

# Check FFmpeg
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: FFmpeg was not found in PATH."
    Write-Host ""
    Write-Host "Install FFmpeg and make sure it is available in PATH."
    Write-Host ""
    exit 1
}

# Check main.py
if (-not (Test-Path ".\main.py")) {
    Write-Host "ERROR: main.py was not found."
    Write-Host ""
    exit 1
}

# Create required folders if missing
New-Item -ItemType Directory -Force -Path "input" | Out-Null
New-Item -ItemType Directory -Force -Path "output" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path ".cache\analysis" | Out-Null

Write-Host "Starting V4 Professional Music Mixer..."
Write-Host ""

& ".\.venv\Scripts\python.exe" ".\main.py"

$exitCode = $LASTEXITCODE

Write-Host ""

if ($exitCode -eq 0) {
    Write-Host "============================================"
    Write-Host " MIX COMPLETE"
    Write-Host "============================================"
    Write-Host ""
    Write-Host "Your mix should be available in:"
    Write-Host "  output\"
} else {
    Write-Host "============================================"
    Write-Host " MIX FAILED"
    Write-Host "============================================"
    Write-Host ""
    Write-Host "Exit code: $exitCode"
}

exit $exitCode