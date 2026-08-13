param(
    [int]$SampleSize = 25000
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host 'Creating isolated Python environment...'
    python -m venv (Join-Path $ProjectRoot '.venv')
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $ProjectRoot 'requirements-training.txt')
& $Python (Join-Path $ProjectRoot 'sentiment_analysis.py') train --download --sample-size $SampleSize

Write-Host "`nFinished. Open the outputs folder to see the saved metrics and charts."
