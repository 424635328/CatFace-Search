@echo off
chcp 65001>nul
setlocal
set "SYNC_EXIT=0"

rem ---------------------------------------------------------------------------------------------
rem  Double-click entry point. All the logic lives in tools\sync.ps1 so it is reviewable,
rem  versioned and testable; this file only locates PowerShell and translates the exit code.
rem  Equivalent command:  pwsh -NoProfile -File tools\sync.ps1
rem ---------------------------------------------------------------------------------------------

echo GitHub Sync Tool - run checks, then pull / add / commit / push
echo.

rem PowerShell 7 is required. Windows PowerShell 5.1 cannot run sync.ps1 (it uses features 5.1
rem lacks), so detect it here and say so plainly instead of failing with an obscure parse error.
where pwsh >nul 2>&1
if errorlevel 1 (
    echo ERROR: pwsh ^(PowerShell 7^) was not found on PATH.
    echo.
    echo   Install it:   winget install --id Microsoft.PowerShell
    echo   Or run the sync directly with:  powershell -File tools\sync.ps1
    echo.
    echo Nothing was changed.
    set "SYNC_EXIT=64"
    goto :report
)

rem Forward arguments unchanged; the default commit message is built inside sync.ps1 with Get-Date,
rem deliberately not here, where %date%/%time% vary by locale and nested backquoted for /f layers
rem fail confusingly.
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\sync.ps1" %*
set "SYNC_EXIT=%ERRORLEVEL%"

:report
echo.
if "%SYNC_EXIT%"=="0" (
    echo SYNC OK
    goto :done
)

rem Translate the documented exit codes so a failure says what to do, not just a number.
if "%SYNC_EXIT%"=="2"  echo FAILED: preflight checks did not pass - nothing was committed or pushed.
if "%SYNC_EXIT%"=="2"  echo         Fix the failures above, or override with:  s.bat -y
if "%SYNC_EXIT%"=="3"  echo FAILED: git pull - resolve the conflict, then run this again.
if "%SYNC_EXIT%"=="4"  echo FAILED: the pre-commit hook rejected the commit - see its message above.
if "%SYNC_EXIT%"=="5"  echo FAILED: git add or git commit.
if "%SYNC_EXIT%"=="6"  echo FAILED: git push - if it was rejected, pull first rather than forcing.
if "%SYNC_EXIT%"=="64" echo FAILED: usage - see the message above.
echo.

:done

rem Keep the window open for a double-click, but never block an unattended run: -NoPause / -NoWait
rem must suppress this pause too, not only the one inside sync.ps1.
echo %* | findstr /I /C:"NoPause" /C:"NoWait" >nul
if errorlevel 1 if not "%SYNC_EXIT%"=="64" pause

exit /b %SYNC_EXIT%
