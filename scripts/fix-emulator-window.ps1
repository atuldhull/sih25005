# Puts the Android emulator window back on screen, and stops it drifting again.
#
# THE ROOT CAUSE, measured: the AVD had window.scale = -1 (auto), which opens
# the window 881px tall on an 816px work area. It can never fit, so it ends up
# at y = -661 - 661 pixels above the top of the display - and dragging it only
# half helps, because the bottom is still cut off. It needs moving AND scaling.
#
# This does both: pins a scale in emulator-user.ini so the next launch opens
# correctly, then fixes the window that is already open.
#
#   powershell -ExecutionPolicy Bypass -File fix-emulator-window.ps1

Add-Type -AssemblyName System.Windows.Forms
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

$area = [System.Windows.Forms.SystemInformation]::WorkingArea
Write-Host ("work area: {0} x {1}" -f $area.Width, $area.Height)

# --- 1. fix the stored config so the NEXT launch opens correctly -----------
$ini = Join-Path $env:USERPROFILE ".android\avd\Medium_Phone.avd\emulator-user.ini"
if (Test-Path $ini) {
    $text = Get-Content $ini -Raw
    $text = $text -replace 'window\.scale = -1\.000000', 'window.scale = 0.820000'
    $text = $text -replace 'window\.x = \d+', 'window.x = 1150'
    $text = $text -replace 'window\.y = -?\d+', 'window.y = 10'
    Set-Content $ini $text -Encoding ascii
    Write-Host "stored AVD window config pinned (scale 0.82, x 1150, y 10)"
} else {
    Write-Host "no emulator-user.ini found - skipping the persistent fix" -ForegroundColor Yellow
}

# --- 2. fix the window that is open right now -----------------------------
$p = Get-Process | Where-Object { $_.MainWindowTitle -match 'Android Emulator' } | Select-Object -First 1
if (-not $p) {
    Write-Host "No emulator window open. The config fix above applies next launch." -ForegroundColor Yellow
    exit 0
}

[EmuWin]::ShowWindow($p.MainWindowHandle, 9) | Out-Null   # un-minimise
Start-Sleep -Milliseconds 400

$r = New-Object EmuWin+RC
[EmuWin]::GetWindowRect($p.MainWindowHandle, [ref]$r) | Out-Null
$curW = $r.R - $r.L
$curH = $r.B - $r.T
Write-Host ("was: x={0} y={1}  {2}x{3}" -f $r.L, $r.T, $curW, $curH)

$aspect = if ($curH -gt 0) { $curW / $curH } else { 0.447 }
$newH = $area.Height - 30          # leave the title bar grabbable
$newW = [int]($newH * $aspect)
$x = $area.Width - $newW - 20      # dock right, clear of your editor
$y = 6

[EmuWin]::SetWindowPos($p.MainWindowHandle, [IntPtr]::Zero, $x, $y, $newW, $newH, 0x0040) | Out-Null
Start-Sleep -Milliseconds 700
[EmuWin]::SetForegroundWindow($p.MainWindowHandle) | Out-Null

[EmuWin]::GetWindowRect($p.MainWindowHandle, [ref]$r) | Out-Null
Write-Host ("now: x={0} y={1}  {2}x{3}" -f $r.L, $r.T, ($r.R-$r.L), ($r.B-$r.T))

if ($r.T -ge 0 -and $r.B -le $area.Height) {
    Write-Host "FULLY ON SCREEN" -ForegroundColor Green
} else {
    Write-Host "Still clipped - lower `$newH in this script." -ForegroundColor Yellow
}
