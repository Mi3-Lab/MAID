# Registers the Windows Scheduled Tasks that keep WheelView running:
#   "WheelView Server" : run.ps1 (uvicorn on 127.0.0.1:8770, self-restarting)
#   "WheelView Tunnel" : scripts\windows\tunnel.ps1 (Cloudflare quick tunnel, writes var\public_url.txt)
#   "WheelView Sync"   : scripts\windows\sync.ps1 every 30 min (export -> commit -> pull --rebase -> push)
# Server + Tunnel: triggers at logon (+ at boot when run elevated) and a 5-minute watchdog repetition; with
# MultipleInstances=IgnoreNew a repetition only starts the task if it is not already running, i.e. it restarts it if it died.
# Not elevated : tasks run as the interactive user, only while logged on (locked screen is fine).
# Elevated     : Server + Tunnel use S4U ("run whether user is logged on or not") and also start at boot.
#                Sync stays interactive because git push needs the user's Credential Manager (gh) credentials.
param([switch]$Uninstall)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$User = "$env:USERDOMAIN\$env:USERNAME"
$Admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$Names = 'WheelView Server', 'WheelView Tunnel', 'WheelView Sync'
if ($Uninstall) {
    foreach ($n in $Names) { Stop-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue; Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
    Write-Output 'removed WheelView tasks'; return
}
$Vbs = Join-Path $Root 'scripts\windows\hidden.vbs'
function Act($ps1) { New-ScheduledTaskAction -Execute 'wscript.exe' -Argument "`"$Vbs`" `"$ps1`"" -WorkingDirectory $Root }
function Watchdog($trigger, $minutes) {
    $rep = (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes $minutes)).Repetition
    $trigger.Repetition = $rep; return $trigger
}
$Long = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
$Short = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes 20) -MultipleInstances IgnoreNew
$Interactive = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
if ($Admin) { $LongPrincipal = New-ScheduledTaskPrincipal -UserId $User -LogonType S4U -RunLevel Limited } else { $LongPrincipal = $Interactive }

foreach ($t in @(@{ n = 'WheelView Server'; s = (Join-Path $Root 'run.ps1') }, @{ n = 'WheelView Tunnel'; s = (Join-Path $Root 'scripts\windows\tunnel.ps1') })) {
    $triggers = @((Watchdog (New-ScheduledTaskTrigger -AtLogOn -User $User) 5))
    if ($Admin) { $triggers += (Watchdog (New-ScheduledTaskTrigger -AtStartup) 5) }
    Register-ScheduledTask -TaskName $t.n -Action (Act $t.s) -Trigger $triggers -Settings $Long -Principal $LongPrincipal `
        -Description "WheelView Annotate ($($t.n)); see $Root\README.md" -Force | Out-Null
    Write-Output "registered $($t.n) ($(if ($Admin) {'S4U, boot+logon'} else {'interactive, logon'}))"
}
$sync = Watchdog (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2)) 30
Register-ScheduledTask -TaskName 'WheelView Sync' -Action (Act (Join-Path $Root 'scripts\windows\sync.ps1')) -Trigger $sync `
    -Settings $Short -Principal $Interactive -Description 'WheelView: export events, commit, pull --rebase, push (every 30 min)' -Force | Out-Null
Write-Output 'registered WheelView Sync (interactive, every 30 min)'
