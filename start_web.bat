@echo off
chcp 65001>nul
setlocal EnableDelayedExpansion
set "APP_EXIT=0"

rem ==============================================================================================
rem  CatFace Search -- double-click launcher for the web interface.
rem
rem  Checks the things that are actually missing on a fresh machine (virtual environment, web
rem  extra, model weights, gallery manifest) and says which one is missing, instead of starting a
rem  server that can only answer 503. Then serves the UI and opens the browser.
rem
rem  Arguments are forwarded, e.g.:
rem      start_web.bat --device cpu
rem      start_web.bat --port 8080 --no-browser
rem      start_web.bat --checkpoint artifacts/train/dinov2b-arcface/best.pt
rem ==============================================================================================

rem Move to this file's own directory before doing anything. %~dp0 only yields a path string; it
rem does not change the working directory, so git-relative and config-relative lookups would run
rem wherever the double-click happened to start. Quoting the path inline is also a trap: %~dp0 ends
rem in a backslash, so `"%~dp0tools\..."` turns \" into an escaped quote and mangles the argument.
pushd "%~dp0"

echo.
echo   CatFace Search - web interface
echo   ------------------------------
echo.

rem --- PowerShell 7 is needed to run the preflight; this script itself only needs cmd --------------
where pwsh >nul 2>&1
if errorlevel 1 (
    echo   ERROR: pwsh ^(PowerShell 7^) was not found on PATH.
    echo.
    echo     Install it:   winget install --id Microsoft.PowerShell
    echo.
    echo   You can still start the server directly with the commands printed below once it is
    echo   installed. Nothing was changed.
    set "APP_EXIT=64"
    goto :fail
)

rem --- locate the interpreter ---------------------------------------------------------------------
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist ".venv\bin\python" set "PY=.venv\bin\python"
if not defined PY (
    where python >nul 2>&1
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo   ERROR: no Python interpreter found.
    echo.
    echo     Create the environment first:
    echo       python -m venv .venv
    echo       .venv\Scripts\python.exe -m pip install -e ".[web]"
    set "APP_EXIT=69"
    goto :fail
)
echo   python      : %PY%

rem --- web dependencies ---------------------------------------------------------------------------
rem Checked here rather than letting the server fail, because the module's own error message is
rem good but arrives after the user has already waited for Python to start.
"%PY%" -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   ERROR: the web dependencies are not installed in %PY%.
    echo.
    echo     Install them with:
    echo       %PY% -m pip install -e ".[web]"
    set "APP_EXIT=69"
    goto :fail
)

rem --- model and gallery --------------------------------------------------------------------------
rem Default paths are the ones this repository ships; the same values the module uses. They can be
rem overridden by passing --checkpoint / --manifest through to the server.
set "CKPT=artifacts\train\dinov2s-arcface\best.pt"
set "MANIFEST=data\manifests\cat_individuals_manifest.jsonl"
set "MISSING="
if not exist "%CKPT%" set "MISSING=!MISSING! checkpoint"
if not exist "%MANIFEST%" set "MISSING=!MISSING! manifest"
if defined MISSING (
    echo.
    echo   ERROR: missing!MISSING!
    echo.
    echo     checkpoint : %CKPT%
    echo     manifest   : %MANIFEST%
    echo.
    echo   The server cannot answer without both. To produce them:
    echo     python -m catface.cli prepare --source cat_individuals     ^(builds the manifest^)
    echo     python -m tools.train_embedder --backbone dinov2_vits14 ^(trains the model^)
    echo.
    echo   Or point at existing files:
    echo     start_web.bat --checkpoint ^<path^> --manifest ^<path^>
    set "APP_EXIT=66"
    goto :fail
)
echo   checkpoint  : %CKPT%
echo   manifest    : %MANIFEST%

rem --- device ------------------------------------------------------------------------------------
rem Default to CUDA when it is present: the first gallery embedding takes ~164 s on CPU-class
rem hardware against a few seconds on a GPU, and that wait is the first thing a user experiences.
set "DEVICE=cuda"
"%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 set "DEVICE=cpu"
echo   device      : %DEVICE%

rem --- serve -------------------------------------------------------------------------------------
echo.
echo   Starting the server. The model and gallery load in the background, so the page is reachable
echo   within a few seconds; it shows a "not ready" banner until the gallery finishes embedding
echo   (~2-3 min on CUDA, longer on CPU). No refresh is needed. Stop with Ctrl+C.
echo.
echo   URL: http://127.0.0.1:8000
echo.

rem Open the browser now rather than after a delay. The server binds its port in a few seconds
rem because the model loads in the background, and the page reports its own readiness, so an early
rem open shows real status instead of a browser error. A `timeout /t N` delay was the first attempt
rem and is worse in two ways: it does not help, and it fails outright when stdin is redirected.
start "" "http://127.0.0.1:8000"

"%PY%" -m catface.web --device %DEVICE% %*
set "APP_EXIT=!ERRORLEVEL!"

echo.
if "!APP_EXIT!"=="0" (
    echo   Server stopped.
) else (
    echo   Server exited with code !APP_EXIT!.
    echo   ^(2 = bad argument, 66 = a path was wrong, 69 = the web extra is missing.^)
)
goto :done

:fail
echo.
echo   Startup aborted; nothing was started.

:done
echo.
echo   Press any key to close this window.
pause >nul
popd
endlocal & exit /b %APP_EXIT%
