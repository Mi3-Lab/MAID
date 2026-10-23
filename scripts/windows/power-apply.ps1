# Keep the laptop awake on AC power: sleep never, hibernate never, lid close = do nothing (AC only; battery untouched).
# Original AC values are saved once to var\power_original.json; scripts\windows\power-revert.ps1 restores them.
$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Save = Join-Path $Root 'var\power_original.json'
New-Item -ItemType Directory -Force -Path (Join-Path $Root 'var') | Out-Null
function Get-AC($sub, $setting) {
    $o = powercfg /q SCHEME_CURRENT $sub $setting | Select-String 'Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)'
    if ($o) { return [Convert]::ToInt32($o.Matches[0].Groups[1].Value, 16) } else { return $null }
}
if (-not (Test-Path $Save)) {
    $orig = [ordered]@{ saved = (Get-Date -Format s); scheme = ((powercfg /getactivescheme) -join ' ')
        standby_ac_seconds = (Get-AC SUB_SLEEP STANDBYIDLE); hibernate_ac_seconds = (Get-AC SUB_SLEEP HIBERNATEIDLE)
        lid_ac = (Get-AC SUB_BUTTONS LIDACTION) }
    $orig | ConvertTo-Json | Set-Content -Path $Save -Encoding UTF8
    Write-Output "saved original AC values to $Save"
} else { Write-Output "original values already saved in $Save (kept)" }
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
Write-Output ("now: standby_ac={0} hibernate_ac={1} lid_ac={2}" -f (Get-AC SUB_SLEEP STANDBYIDLE), (Get-AC SUB_SLEEP HIBERNATEIDLE), (Get-AC SUB_BUTTONS LIDACTION))
