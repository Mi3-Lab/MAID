# Restores the AC power values saved by power-apply.ps1 (var\power_original.json). Battery settings were never changed.
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$o = Get-Content (Join-Path $Root 'var\power_original.json') -Raw | ConvertFrom-Json
if ($null -ne $o.standby_ac_seconds) { powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP STANDBYIDLE $o.standby_ac_seconds }
if ($null -ne $o.hibernate_ac_seconds) { powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP HIBERNATEIDLE $o.hibernate_ac_seconds }
if ($null -ne $o.lid_ac) { powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION $o.lid_ac }
powercfg /setactive SCHEME_CURRENT
Write-Output "restored AC values from $($o.saved): standby=$($o.standby_ac_seconds)s hibernate=$($o.hibernate_ac_seconds)s lid=$($o.lid_ac)"
