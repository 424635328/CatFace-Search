<#
.SYNOPSIS
    Run every CI check locally, exactly as CI runs it, before pushing.

.DESCRIPTION
    A failed CI run costs a push, a queue wait, a failure, a fix, and a re-push. This runs the
    same checks in about the time CI takes to *start*, so a first-push green result becomes the
    normal case rather than a hope.

    The point is not "run the tests". It is that each check must mirror CI **exactly**,
    including the flags. Two of the three failures in this repository's first real CI run came
    from a local check that was close to CI's but not identical:

      * lint was run with a hand-picked rule subset, while CI used the full configured set;
      * tests were run from an existing working tree, while CI ran them from a fresh clone.

    Both passed locally and failed on CI. So the flags below are copied from
    .github/workflows/ci.yml deliberately, and a change there should change them here.

.PARAMETER SkipTests
    Skip the test job (useful while iterating on lint only).

.PARAMETER CloneCheck
    Additionally clone the repository fresh and re-run the checks there. This is the check that
    catches "passes in my working tree, fails for everyone else" — most importantly files that
    exist locally but are not tracked by git.

.EXAMPLE
    .\tools\preflight.ps1
    .\tools\preflight.ps1 -CloneCheck
#>
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipPackage,
    [switch]$CloneCheck
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$script:Failures = @()
$script:Timer = [System.Diagnostics.Stopwatch]::StartNew()

function Invoke-Check {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    $started = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $Body
        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) { throw "exit code $LASTEXITCODE" }
        Write-Host ("  PASS  ({0:N1}s)" -f $started.Elapsed.TotalSeconds) -ForegroundColor Green
    } catch {
        Write-Host ("  FAIL  ($($_.Exception.Message))") -ForegroundColor Red
        $script:Failures += $Name
    }
}

Write-Host "preflight: mirroring .github/workflows/ci.yml" -ForegroundColor White
Write-Host "python: $python"

# --- job: lint ------------------------------------------------------------------------------
# ``--no-respect-gitignore`` matches CI. Ruff skips .gitignore'd paths by default, so a source
# file that .gitignore wrongly excludes is silently never linted. That is not hypothetical: it
# hid an entire six-module package from both version control and linting.
Invoke-Check "lint (ruff, full rule set)" {
    & $python -m ruff check --no-respect-gitignore src tests tools
}

# --- job: test ------------------------------------------------------------------------------
# The marker filter and coverage flags are copied from CI so a marker typo or a coverage-only
# import error cannot slip through.
if (-not $SkipTests) {
    Invoke-Check "tests (as CI runs them)" {
        & $python -m pytest -q -m "not slow" --cov=catface --cov-report=
    }
}

# --- job: configs ---------------------------------------------------------------------------
Invoke-Check "shipped configs parse" {
    & $python -c @"
import pathlib, sys
from catface.config import PipelineConfig
fails = []
for path in sorted(pathlib.Path('configs').glob('*.yaml')):
    try:
        config = PipelineConfig.from_yaml(path)
        print(f'OK   {path}  fingerprint={config.fingerprint()}')
    except Exception as exc:
        fails.append(path)
        print(f'FAIL {path}: {exc}')
sys.exit(1 if fails else 0)
"@
}

# --- job: package ---------------------------------------------------------------------------
if (-not $SkipPackage) {
    Invoke-Check "config fingerprint is location independent" {
        # Regression guard for a defect that only appeared when the same config was loaded from
        # two directories. It is cheap to check and expensive to diagnose.
        & $python -c @"
import os, sys, tempfile, pathlib
sys.path.insert(0, 'src')
from catface.config import load_config
shipped = pathlib.Path('configs/default.yaml').resolve()
first = load_config(shipped).fingerprint()
tmp = pathlib.Path(tempfile.mkdtemp())
previous = os.getcwd()
os.chdir(tmp)
try:
    second = load_config(shipped).fingerprint()
finally:
    os.chdir(previous)
assert first == second, f'fingerprint differs by working directory: {first} != {second}'
print('fingerprint stable across directories:', first)
"@
    }
}

# --- tracked-source check --------------------------------------------------------------------
Invoke-Check "every source file is tracked by git" {
    # The single most valuable check here, because its failure is invisible locally: an untracked
    # file still exists in the working tree, so local tests, local lint and local imports all
    # succeed while a clone is broken.
    & $python -c @"
import pathlib, subprocess, sys
tracked = set(subprocess.run(['git', 'ls-files'], capture_output=True, text=True,
                             check=True).stdout.split())
missing = []
for root in ('src', 'tests', 'tools'):
    for path in sorted(pathlib.Path(root).rglob('*.py')):
        rel = path.as_posix()
        if rel not in tracked:
            missing.append(rel)
if missing:
    print('NOT TRACKED (a clone would be broken):')
    for item in missing:
        print('   ', item)
    sys.exit(1)
print(f'all source files tracked ({len(tracked)} files in the repository)')
"@
}

# --- documentation references -----------------------------------------------------------------
Invoke-Check "documentation references real paths" {
    # Documentation drifts silently: this project's README once documented a seven-step quick
    # start in which five steps could not run, and nothing failed until a reader tried them.
    # The check lives in tools/check_docs.py rather than inline here: an inline here-string has to
    # escape the very backticks it is searching for, and a mis-escaped backtick weakens the check
    # silently instead of failing loudly.
    & $python -m tools.check_docs
}

# --- optional: the fresh-clone check ----------------------------------------------------------
if ($CloneCheck) {
    Invoke-Check "fresh clone passes the same checks" {
        $tmp = Join-Path $env:TEMP ("preflight-clone-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
        try {
            git clone --quiet --depth 1 $repo $tmp
            Push-Location $tmp
            try {
                $env:PYTHONPATH = Join-Path $tmp "src"
                & $python -m ruff check --no-respect-gitignore src tests tools
                if ($LASTEXITCODE -ne 0) { throw "lint failed in the clone" }
                & $python -c "import catface.data, catface.pipeline; print('package imports from the clone')"
                if ($LASTEXITCODE -ne 0) { throw "import failed in the clone" }
            } finally {
                Pop-Location
            }
        } finally {
            Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# --- verdict ----------------------------------------------------------------------------------
Write-Host ""
if ($script:Failures.Count -eq 0) {
    Write-Host ("PREFLIGHT OK ({0:N0}s) - this push should pass CI" -f $script:Timer.Elapsed.TotalSeconds) -ForegroundColor Green
    exit 0
}
Write-Host ("PREFLIGHT FAILED: " + ($script:Failures -join "; ")) -ForegroundColor Red
Write-Host "Fix these before pushing; a CI round trip costs minutes and a queue wait." -ForegroundColor Red
exit 1
