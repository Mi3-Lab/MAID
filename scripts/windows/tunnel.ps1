# Cloudflare *quick tunnel* (no account) to the local server; restarts cloudflared if it dies.
# The https://<random>.trycloudflare.com URL CHANGES on every restart; the current one is written to var\public_url.txt
# (history in var\public_url_history.log). A stable URL needs a (free) Cloudflare named tunnel or Tailscale Funnel account.
param([int]$Port = 8770)
$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Var = Join-Path $Root 'var'; New-Item -ItemType Directory -Force -Path $Var | Out-Null
$UrlFile = Join-Path $Var 'public_url.txt'; $Hist = Join-Path $Var 'public_url_history.log'; $Log = Join-Path $Var 'tunnel.log'
function Say($m) { Add-Content -Path $Log -Value "$(Get-Date -Format s) [tunnel.ps1] $m" -Encoding UTF8 }
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
while ($true) {
    $cf = Find-Cloudflared
    if (-not $cf) { Say 'cloudflared not found; retrying in 60 s'; Start-Sleep 60; continue }
    $err = Join-Path $Var 'cloudflared.err.log'; $out = Join-Path $Var 'cloudflared.out.log'
    Remove-Item $err, $out -ErrorAction SilentlyContinue
    Say "starting $cf -> http://127.0.0.1:$Port"
    $p = Start-Process -FilePath $cf -ArgumentList @('tunnel', '--no-autoupdate', '--url', "http://127.0.0.1:$Port") `
         -RedirectStandardError $err -RedirectStandardOutput $out -NoNewWindow -PassThru
    $null = $p.Handle   # makes ExitCode available after exit (PS 5.1 quirk)
    $url = $null
    for ($i = 0; $i -lt 90 -and -not $p.HasExited -and -not $url; $i++) {
        Start-Sleep 1
        if (Test-Path $err) { $m = Select-String -Path $err -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -List
            if ($m) { $url = $m.Matches[0].Value } }
    }
    if ($url) { Set-Content -Path $UrlFile -Value $url -Encoding ASCII; Add-Content -Path $Hist -Value "$(Get-Date -Format s) $url"; Say "public URL $url" }
    else { Say 'no URL seen within 90 s' }
    $p.WaitForExit()
    Say "cloudflared exited with code $($p.ExitCode); restarting in 10 s"
    Set-Content -Path $UrlFile -Value "DOWN (tunnel restarting; last URL $url)" -Encoding ASCII
    Start-Sleep 10
}
