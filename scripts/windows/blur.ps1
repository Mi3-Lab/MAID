# Face-blur the local images into images_blurred\ (originals untouched; nothing leaves the laptop).
#   powershell -ExecutionPolicy Bypass -File scripts\windows\blur.ps1
# Images whose checksum matches data\blur_manifest.jsonl (committed from the build box) reuse those face boxes, so the
# laptop produces the same release copies without re-deciding; anything new is detected locally and logged to
# var\blur_manifest_local.jsonl (never committed). The running server picks the blurred files up immediately.
$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$Py = Join-Path $Root '.venv\Scripts\python.exe'
$Log = Join-Path $Root 'var\blur.log'
$env:PYTHONUTF8 = '1'
& $Py -c "import cv2, numpy" 2>$null
if ($LASTEXITCODE -ne 0) { & $Py -m pip install -q -r (Join-Path $Root 'requirements-blur.txt') }
& $Py scripts\auto_blur.py --local @args 2>&1 | Tee-Object -FilePath $Log -Append
