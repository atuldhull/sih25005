# Puts the Android emulator window back on screen.
#
# The emulator remembers a window position that can sit above the top of the
# display - it has come back at y = -661 more than once - and it opens taller
# than the 816px work area, so even at y = 0 the bottom is cut off. This moves
# it fully into view, scales it to fit, and docks it to the right.
#
# Run it any time the emulator is half off screen:
#     powershell -ExecutionPolicy Bypass -File fix-emulator-window.ps1

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class EmuWin {
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr a, int x, int y, int cx, int cy, uint f);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RC r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  [StructLayout(LayoutKind.Sequential)] public struct RC { public int L,T,R,B; }
}
"@

$p = Get-Process | Where-Object { $_.MainWindowTitle -match 'Android Emulator' } | Select-Object -First 1
if (-not $p) {
    Write-Host "No emulator window found. Start the emulator first." -ForegroundColor Yellow
    exit 1
}

# un-minimise, in case that is why it cannot be found on screen
[EmuWin]::ShowWindow($p.MainWindowHandle, 9) | Out-Null
Start-Sleep -Milliseconds 400

$r = New-Object EmuWin+RC
[EmuWin]::GetWindowRect($p.MainWindowHandle, [ref]$r) | Out-Null
$curW = $r.R - $r.L
$curH = $r.B - $r.T
Write-Host ("was at x={0} y={1}  size {2}x{3}" -f $r.L, $r.T, $curW, $curH)

# Screen work area, minus a margin so the title bar stays grabbable.
$area   = [System.Windows.Forms.SystemInformation]::WorkingArea
if (-not $area) { $area = New-Object PSObject -Property @{ Width = 1536; Height = 816 } }
$maxH   = $area.Height - 26
$aspect = if ($curH -gt 0) { $curW / $curH } else { 0.447 }

$newH = $maxH
$newW = [int]($newH * $aspect)
$x    = $area.Width - $newW - 24    # dock right, out of the way of your editor
$y    = 8

[EmuWin]::SetWindowPos($p.MainWindowHandle, [IntPtr]::Zero, $x, $y, $newW, $newH, 0x0040) | Out-Null
Start-Sleep -Milliseconds 600
[EmuWin]::SetForegroundWindow($p.MainWindowHandle) | Out-Null

[EmuWin]::GetWindowRect($p.MainWindowHandle, [ref]$r) | Out-Null
Write-Host ("now at x={0} y={1}  size {2}x{3}" -f $r.L, $r.T, ($r.R-$r.L), ($r.B-$r.T))

if ($r.T -ge 0 -and $r.B -le $area.Height) {
    Write-Host "FULLY ON SCREEN" -ForegroundColor Green
} else {
    Write-Host "Still clipped. Drag it by the title bar, or lower `$maxH in this script." -ForegroundColor Yellow
}
