# Backup + sync: export server events to annotations/events/<annotator>.jsonl, commit, pull --rebase, push (laptop's git creds).
# The GitHub workflow rebuild-derived.yml then rebuilds derived/ on push. Run every 30 min by the "WheelView Sync" task.
$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root
$Log = Join-Path $Root 'var\sync.log'
$Py = Join-Path $Root '.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
Add-Content -Path $Log -Value "=== $(Get-Date -Format s) sync start" -Encoding UTF8
cmd /c "`"$Py`" scripts\sync_events.py sync --no-derived --pull --push >> `"$Log`" 2>&1"
$rc = $LASTEXITCODE
Add-Content -Path $Log -Value "=== $(Get-Date -Format s) sync exit $rc" -Encoding UTF8
exit $rc
