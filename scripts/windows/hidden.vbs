' Runs a PowerShell script with no visible window and waits for it (so Task Scheduler sees it as running).
' usage: wscript.exe hidden.vbs <script.ps1> [args...]
Dim sh, i, a
Set sh = CreateObject("WScript.Shell")
a = ""
For i = 1 To WScript.Arguments.Count - 1
  a = a & " " & WScript.Arguments(i)
Next
WScript.Quit sh.Run("powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & WScript.Arguments(0) & """" & a, 0, True)
