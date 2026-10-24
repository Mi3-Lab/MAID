# WheelView Annotate - Windows launcher / supervisor.
#   powershell -ExecutionPolicy Bypass -File run.ps1 [-Port 8770] [-Once]
# Creates .venv + installs requirements on first run, imports committed events (idempotent), then runs uvicorn on
# 127.0.0.1 and restarts it if it exits (unless -Once). Reads the access code from ACCESS_CODE.txt (gitignored).
# HF persistence is deliberately OFF here (WV_HF_DATASET unset): the laptop disk is persistent and git is the backup.
param([int]$Port = 8770, [switch]$Once, [string]$BindHost = '127.0.0.1')
$ErrorActionPreference = 'Continue'
$Root = $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'var') | Out-Null
$Log = Join-Path $Root 'var\server.log'
function Say($m) { $line = "$(Get-Date -Format s) [run.ps1] $m"; Add-Content -Path $Log -Value $line -Encoding UTF8; Write-Output $line }

# Single instance: stopping the scheduled task kills wscript but can orphan this loop; a second loop would fight over
# the port and re-run the event import every 5 s. The mutex is freed by Windows when this process dies.
$script:Mutex = $null; $got = $true
foreach ($name in @("Global\WheelViewServerLoop$Port", "Local\WheelViewServerLoop$Port")) {
    try { $script:Mutex = New-Object System.Threading.Mutex($false, $name)
          try { $got = $script:Mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $got = $true }
          break } catch { $script:Mutex = $null }
}
if (-not $got) { Say "another run.ps1 loop for port $Port is alive; exiting"; exit 0 }

$Py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) {
    $Base = $env:WV_PYTHON
    if (-not $Base) { if (Test-Path 'C:\Python313\python.exe') { $Base = 'C:\Python313\python.exe' } else { $Base = 'python' } }
    Say "creating venv with $Base"
    & $Base -m venv (Join-Path $Root '.venv')
}
& $Py -c "import fastapi, uvicorn, httpx, PIL" 2>$null
if ($LASTEXITCODE -ne 0) { Say 'installing requirements'; & $Py -m pip install -q -r (Join-Path $Root 'requirements.txt') }

$env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'
Remove-Item Env:WV_HF_DATASET -ErrorAction SilentlyContinue
$CodeFile = Join-Path $Root 'ACCESS_CODE.txt'
if (Test-Path $CodeFile) { $env:WV_ACCESS_CODE = (Get-Content $CodeFile -Raw).Trim() } else { Say 'WARNING: ACCESS_CODE.txt missing -> no access gate' }

do {
    cmd /c "`"$Py`" scripts\sync_events.py import >> `"$Log`" 2>&1"
    Say "starting uvicorn on ${BindHost}:$Port"
    cmd /c "`"$Py`" -m uvicorn main:app --app-dir server --host $BindHost --port $Port --proxy-headers --forwarded-allow-ips 127.0.0.1 --no-access-log >> `"$Log`" 2>&1"
    Say "uvicorn exited with code $LASTEXITCODE"
    if (-not $Once) { Start-Sleep -Seconds 5 }
} while (-not $Once)
