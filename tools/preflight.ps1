<#
.SYNOPSIS
    Run every CI check locally, exactly as CI runs it, before pushing.

.DESCRIPTION
    A failed CI run costs a push, a queue wait, a failure, a fix, and a re-push. This runs the
    same checks in about the time CI takes to *start*, so a first-push green result becomes the
    normal case rather than a hope.

    The point is not "run the tests". It is that each check must mirror CI **exactly**, including
    the flags. Two of the three failures in this repository's first real CI run came from a local
    check that was close to CI's but not identical:

      * lint was run with a hand-picked rule subset, while CI used the full configured set;
      * tests were run from an existing working tree, while CI ran them from a fresh clone.

    Both passed locally and failed on CI. So the flags below are copied from
    .github/workflows/ci.yml deliberately, and a change there should change them here.

    This script holds no project-specific paths: those live in tools/project.config.ps1, and the
    interpreter is found by tools/resolve-python.ps1. Copy the pair into another repository, edit
    the config, and the checks run there unchanged.

.PARAMETER Fast
    Skip the slow jobs (tests, and the few-second package check). Useful while iterating.

.PARAMETER SkipTests
    Skip only the test job.

.PARAMETER SkipPackage
    Skip only the config-fingerprint job.

.PARAMETER CloneCheck
    Additionally clone the repository fresh and re-run the checks there. This is the check that
    catches "passes in my working tree, fails for everyone else" — most importantly files that
    exist locally but are not tracked by git.

.PARAMETER Quiet
    Print only the failing checks and the verdict. For use inside another script.

.EXAMPLE
    .\tools\preflight.ps1
    .\tools\preflight.ps1 -Fast
    .\tools\preflight.ps1 -CloneCheck
#>
[CmdletBinding()]
param(
    [switch]$Fast,
    [switch]$SkipTests,
    [switch]$SkipPackage,
    [switch]$CloneCheck,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if ($Fast) { $SkipTests = $true; $SkipPackage = $true }

# --- project settings ---------------------------------------------------------------------------
# Absent config means the defaults in project.config.ps1 are used, so a copy of this script still
# runs in a repository that has no config yet.
$configPath = Join-Path $PSScriptRoot "project.config.ps1"
$project = if (Test-Path -LiteralPath $configPath) {
    & $configPath
} else {
    @{}
}
function Get-Setting {
    param([string]$Name, $Default)
    if ($project.ContainsKey($Name)) { return $project[$Name] }
    return $Default
}

$sourceRoots = @(Get-Setting "SourceRoots" @('src', 'tests', 'tools'))
$lintTargets = @(Get-Setting "LintTargets" @('src', 'tests', 'tools'))
$configGlob = Get-Setting "ConfigGlob" ''
$fingerprintConfig = Get-Setting "FingerprintConfig" ''
$package = Get-Setting "Package" ''
$sourceDir = Get-Setting "SourceDir" 'src'
$docsChecker = Get-Setting "DocsChecker" ''

$python = & (Join-Path $PSScriptRoot "resolve-python.ps1") -RepoRoot $repo
if (-not $python) { Write-Host "preflight: no Python interpreter found." -ForegroundColor Red; exit 1 }

$script:Failures = @()
$script:Timer = [System.Diagnostics.Stopwatch]::StartNew()
$script:Count = 0

function Invoke-Check {
    param([string]$Name, [scriptblock]$Body)
    $script:Count++
    if (-not $Quiet) {
        Write-Host ""
        Write-Host "=== $Name ===" -ForegroundColor Cyan
    }
    $started = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $Body
        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) { throw "exit code $LASTEXITCODE" }
        if (-not $Quiet) {
            Write-Host ("  PASS  ({0:N1}s)" -f $started.Elapsed.TotalSeconds) -ForegroundColor Green
        }
    } catch {
        Write-Host ("  FAIL  {0}  ({1})" -f $Name, $_.Exception.Message) -ForegroundColor Red
        $script:Failures += $Name
    }
}

if (-not $Quiet) {
    Write-Host "preflight: mirroring .github/workflows/ci.yml" -ForegroundColor White
    Write-Host "python : $python"
    $branch = (& git rev-parse --abbrev-ref HEAD 2>$null)
    if ($branch) { Write-Host "branch : $branch" }
    if ($Fast) { Write-Host "mode   : fast (slow jobs skipped)" -ForegroundColor Yellow }
}

# --- job: lint ------------------------------------------------------------------------------
# ``--no-respect-gitignore`` matches CI. Ruff skips .gitignore'd paths by default, so a source
# file that .gitignore wrongly excludes is silently never linted. That is not hypothetical: it
# hid an entire six-module package from both version control and linting.
Invoke-Check "lint (ruff, full rule set)" {
    & $python -m ruff check --no-respect-gitignore @lintTargets
}

# --- job: test ------------------------------------------------------------------------------
# The marker filter and coverage flags are copied from CI so a marker typo or a coverage-only
# import error cannot slip through.
#
# ``--basetemp`` is the one deliberate addition. pytest otherwise reuses the shared per-user temp
# root and maintains a ``pytest-current`` junction inside it. A killed test run can leave that
# junction pointing at a deleted target with its ACL unreadable, after which *every* later pytest
# run exits non-zero from ``cleanup_dead_symlinks`` -> ``PermissionError: [WinError 5]`` even
# though every test passed. That produced a false red preflight. An exclusive basetemp removes the
# shared state entirely, so the exit code reflects the tests and nothing else.
if (-not $SkipTests) {
    Invoke-Check "tests (as CI runs them)" {
        $pytestTemp = Join-Path ([System.IO.Path]::GetTempPath()) (
            "preflight-pytest-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
        try {
            $coverage = if ($package) { "--cov=$package" } else { "--cov" }
            & $python -m pytest -q -m "not slow" $coverage --cov-report= --basetemp=$pytestTemp
        } finally {
            Remove-Item $pytestTemp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# --- job: configs ---------------------------------------------------------------------------
if ($configGlob) {
    Invoke-Check "shipped configs parse" {
        & $python -c @"
import pathlib, sys
from $package.config import PipelineConfig
fails = []
for path in sorted(pathlib.Path('.').glob('$configGlob')):
    try:
        config = PipelineConfig.from_yaml(path)
        print(f'OK   {path}  fingerprint={config.fingerprint()}')
    except Exception as exc:
        fails.append(path)
        print(f'FAIL {path}: {exc}')
sys.exit(1 if fails else 0)
"@
    }
}

# --- job: package ---------------------------------------------------------------------------
if (-not $SkipPackage -and $fingerprintConfig) {
    Invoke-Check "config fingerprint is location independent" {
        # Regression guard for a defect that only appeared when the same config was loaded from
        # two directories. It is cheap to check and expensive to diagnose.
        & $python -c @"
import os, sys, tempfile, pathlib
sys.path.insert(0, '$sourceDir')
from $package.config import load_config
shipped = pathlib.Path('$fingerprintConfig').resolve()
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
    $rootList = ($sourceRoots | ForEach-Object { "'$_'" }) -join ', '
    & $python -c @"
import pathlib, subprocess, sys
tracked = set(subprocess.run(['git', 'ls-files'], capture_output=True, text=True,
                             check=True).stdout.split())
missing = []
for root in ($rootList,):
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
if ($docsChecker) {
    Invoke-Check "documentation references real paths" {
        # Documentation drifts silently: this project's README once documented a seven-step quick
        # start in which five steps could not run, and nothing failed until a reader tried them.
        # The check lives in tools/check_docs.py rather than inline here: an inline here-string has
        # to escape the very backticks it is searching for, and a mis-escaped backtick weakens the
        # check silently instead of failing loudly.
        & $python (Join-Path $repo $docsChecker)
    }
}

# --- optional: the fresh-clone check ----------------------------------------------------------
if ($CloneCheck) {
    Invoke-Check "fresh clone passes the same checks" {
        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) (
            "preflight-clone-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
        try {
            git clone --quiet --depth 1 $repo $tmp
            Push-Location $tmp
            try {
                $env:PYTHONPATH = Join-Path $tmp $sourceDir
                & $python -m ruff check --no-respect-gitignore @lintTargets
                if ($LASTEXITCODE -ne 0) { throw "lint failed in the clone" }
                if ($package) {
                    & $python -c "import $package; print('package imports from the clone')"
                    if ($LASTEXITCODE -ne 0) { throw "import failed in the clone" }
                }
            } finally {
                Pop-Location
                Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
            }
        } finally {
            Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# --- verdict ----------------------------------------------------------------------------------
$elapsed = $script:Timer.Elapsed.TotalSeconds
Write-Host ""
if ($script:Failures.Count -eq 0) {
    Write-Host ("PREFLIGHT OK ({0:N0}s, {1} checks) - this push should pass CI" -f `
        $elapsed, $script:Count) -ForegroundColor Green
    exit 0
}
Write-Host ("PREFLIGHT FAILED ({0} of {1} checks): {2}" -f `
    $script:Failures.Count, $script:Count, ($script:Failures -join "; ")) -ForegroundColor Red
Write-Host "Fix these before pushing; a CI round trip costs minutes and a queue wait." -ForegroundColor Red
exit 1
