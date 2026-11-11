$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath ".venv")) {
    python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-dev.txt

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Created .env. Add your OpenRouter API key before running the app."
}

if (-not (Test-Path -LiteralPath "profile.yaml")) {
    Copy-Item -LiteralPath "profile.example.yaml" -Destination "profile.yaml"
    Write-Host "Created profile.yaml. Review the consent and profile fields before running the app."
}

Write-Host "Setup complete. Add 4-10 approved photos to references, then run scripts\run.ps1."

