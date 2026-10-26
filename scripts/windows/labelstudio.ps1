# Start the local Label Studio evaluation instance (WheelView project) - LOCAL ONLY (127.0.0.1), never tunnelled.
# Install (once):  py -3.11 -m venv C:\Users\kamoo\labelstudio\venv ; ...\venv\Scripts\python -m pip install label-studio
# Run (hidden):    Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts\windows\labelstudio.ps1'
# Then open http://127.0.0.1:8080 (login: C:\Users\kamoo\labelstudio\CREDENTIALS.txt). Project setup: scripts\labelstudio_import.py
# Label Studio's own DB/data live in $LsHome\data, outside the repo. Only images_blurred\ is registered as storage.
param([int]$Port = 8080, [string]$LsHome = (Join-Path $env:USERPROFILE 'labelstudio'))
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = 'true'
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = $Root
$env:LABEL_STUDIO_BASE_DATA_DIR = Join-Path $LsHome 'data'
$env:LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK = 'true'
# privacy: Label Studio ships with HumanSignal's Sentry DSNs (error reports with send_default_pii=True) and usage analytics ON.
# An empty env var cannot be set from PowerShell, so point both DSNs at a local discard port; turn analytics off.
$env:SENTRY_DSN = 'http://disabled@127.0.0.1:9/0'; $env:FRONTEND_SENTRY_DSN = 'http://disabled@127.0.0.1:9/0'
$env:COLLECT_ANALYTICS = 'false'
$Exe = Join-Path $LsHome 'venv\Scripts\label-studio.exe'
& $Exe start --internal-host 127.0.0.1 --port $Port --no-browser *> (Join-Path $LsHome 'labelstudio.log')
