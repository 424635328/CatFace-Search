@echo off
chcp 65001>nul
echo GitHub Sync Tool - preflight, then pull / add / commit / push
echo.
echo The gate itself lives in tools\sync.ps1 so it is reviewable and versioned.
echo Bypass the gate with:  pwsh -File tools\sync.ps1 -SkipPreflight
echo.

pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\sync.ps1" %*
set SYNC_EXIT=%ERRORLEVEL%

echo.
if not "%SYNC_EXIT%"=="0" (
    echo SYNC FAILED with exit code %SYNC_EXIT%
) else (
    echo SYNC OK
)

rem Keep the window open for a double-click, but never block an unattended call:
rem -NoPause must suppress this pause too, not only the one inside sync.ps1.
echo %* | findstr /C:"NoPause" >nul
if errorlevel 1 pause

exit /b %SYNC_EXIT%
