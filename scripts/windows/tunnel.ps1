# Cloudflare *quick tunnel* (no account) to the local server. Single instance; restarts cloudflared if it dies OR if the
# public URL stops answering (3 failed health checks of <url>/api/health, 60 s apart, while the local server is fine).
# The https://<random>.trycloudflare.com URL CHANGES on every restart; the current one is written to var\public_url.txt
# (history in var\public_url_history.log). A stable URL needs a (free) Cloudflare named tunnel or Tailscale Funnel account.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\tunnel.ps1 [-Port 8770]
param([int]$Port = 8770, [int]$CheckSeconds = 60, [int]$MaxFailures = 3)
$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Var = Join-Path $Root 'var'; New-Item -ItemType Directory -Force -Path $Var | Out-Null
$UrlFile = Join-Path $Var 'public_url.txt'; $Hist = Join-Path $Var 'public_url_history.log'; $Log = Join-Path $Var 'tunnel.log'
$LockFile = Join-Path $Var 'tunnel.lock'; $CfPidFile = Join-Path $Var 'cloudflared.pid'
function Say($m) { Add-Content -Path $Log -Value "$(Get-Date -Format s) [tunnel.ps1 pid $PID] $m" -Encoding UTF8 }
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# ---- single instance: a named mutex (released by Windows if this process dies) + a PID lock file as a second check ----
$script:Mutex = $null; $got = $false
foreach ($name in @('Global\WheelViewTunnelLoop', 'Local\WheelViewTunnelLoop')) {
    try {
        $script:Mutex = New-Object System.Threading.Mutex($false, $name)
        try { $got = $script:Mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $got = $true }
        break
    } catch { $script:Mutex = $null }   # e.g. no right to create Global\ objects -> fall back to Local\
}
function Test-LoopAlive([int]$id) {
    if ($id -le 0 -or $id -eq $PID) { return $false }
    $q = Get-CimInstance Win32_Process -Filter "ProcessId=$id" -ErrorAction SilentlyContinue
    return [bool]($q -and $q.CommandLine -like '*tunnel.ps1*')
}
if ($script:Mutex -and -not $got) { Say 'another tunnel loop holds the mutex; exiting'; exit 0 }
if (Test-Path $LockFile) {
    $other = 0; [void][int]::TryParse(((Get-Content $LockFile -ErrorAction SilentlyContinue) | Select-Object -First 1), [ref]$other)
    if (Test-LoopAlive $other) { Say "another tunnel loop is alive (pid $other, lock file); exiting"; exit 0 }
}
Set-Content -Path $LockFile -Value $PID -Encoding ASCII

function Find-Cloudflared {
    $c = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    foreach ($p in @("${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe", "$env:ProgramFiles\cloudflared\cloudflared.exe",
                     "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cloudflared.exe", (Join-Path $Var 'bin\cloudflared.exe'))) {
        if ($p -and (Test-Path $p)) { return $p } }
    $g = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter cloudflared*.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($g) { return $g.FullName }
    return $null
}
# Strays: we hold the lock, so any quick-tunnel cloudflared launched from var\bin (or pointing at our port) is an orphan
# of an earlier loop. Leaves unrelated cloudflared instances (other paths/ports, e.g. a named tunnel service) alone.
function Stop-StrayCloudflared($keepId) {
    $bin = (Join-Path $Var 'bin').ToLower()
    Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
        $exe = "$($_.ExecutablePath)".ToLower(); $cmd = "$($_.CommandLine)"
        if ($_.ProcessId -ne $keepId -and ($exe.StartsWith($bin) -or $cmd -like "*--url http://127.0.0.1:$Port*")) {
            Say "stopping stray cloudflared pid $($_.ProcessId) ($exe)"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
}
function Test-Url($u) {
    try { $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 15; return ($r.StatusCode -eq 200) } catch { return $false }
}

try {
    Say 'tunnel loop started (single instance)'
    Stop-StrayCloudflared 0
    while ($true) {
        $cf = Find-Cloudflared
        if (-not $cf) { Say 'cloudflared not found; retrying in 60 s'; Start-Sleep 60; continue }
        $err = Join-Path $Var 'cloudflared.err.log'; $out = Join-Path $Var 'cloudflared.out.log'
        Remove-Item $err, $out -ErrorAction SilentlyContinue
        Say "starting $cf -> http://127.0.0.1:$Port"
        $p = Start-Process -FilePath $cf -ArgumentList @('tunnel', '--no-autoupdate', '--url', "http://127.0.0.1:$Port") `
             -RedirectStandardError $err -RedirectStandardOutput $out -NoNewWindow -PassThru
        $null = $p.Handle   # makes ExitCode available after exit (PS 5.1 quirk)
        Set-Content -Path $CfPidFile -Value $p.Id -Encoding ASCII
        $url = $null
        for ($i = 0; $i -lt 90 -and -not $p.HasExited -and -not $url; $i++) {
            Start-Sleep 1
            if (Test-Path $err) { $m = Select-String -Path $err -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -List
                if ($m) { $url = $m.Matches[0].Value } }
        }
        if ($url) { Set-Content -Path $UrlFile -Value $url -Encoding ASCII; Add-Content -Path $Hist -Value "$(Get-Date -Format s) $url"; Say "public URL $url" }
        else { Say 'no URL seen within 90 s; restarting cloudflared'; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
        $fails = 0; $reason = 'exited'
        while ($url -and -not $p.HasExited) {
            if ($p.WaitForExit($CheckSeconds * 1000)) { break }
            if (Test-Url "$url/api/health") { if ($fails) { Say "public URL healthy again after $fails failure(s)" }; $fails = 0; continue }
            if (-not (Test-Url "http://127.0.0.1:$Port/api/health")) { Say 'local server not answering; not blaming the tunnel'; $fails = 0; continue }
            $fails++; Say "public URL health check failed ($fails/$MaxFailures): $url/api/health"
            if ($fails -ge $MaxFailures) {
                $reason = "public URL dead after $fails checks"; Say "$reason; killing cloudflared pid $($p.Id)"
                Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; $p.WaitForExit(10000) | Out-Null; break
            }
        }
        if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
        Say "cloudflared stopped ($reason, exit code $($p.ExitCode)); restarting in 10 s"
        Set-Content -Path $UrlFile -Value "DOWN (tunnel restarting; last URL $url)" -Encoding ASCII
        Remove-Item $CfPidFile -ErrorAction SilentlyContinue
        Start-Sleep 10
    }
} finally {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    if ((Get-Content $LockFile -ErrorAction SilentlyContinue | Select-Object -First 1) -eq "$PID") { Remove-Item $LockFile -ErrorAction SilentlyContinue }
    if ($script:Mutex) { try { $script:Mutex.ReleaseMutex() } catch { } ; $script:Mutex.Dispose() }
}
