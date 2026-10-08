<#
.SYNOPSIS
    Run the push-time preflight and only then sync this repository to its remote.

.DESCRIPTION
    ``s.bat`` used to be a bare ``git pull && git add . && git commit && git push``. That is fast
    and it is also how a red CI happens: this repository's first real CI run failed 3 of 5 jobs,
    and every one of those failures was detectable locally before the push. A round trip through
    CI costs minutes of queue plus a second push, so the gate is worth the ~100 seconds it takes.

    The order matters and is deliberate:

      1. preflight  - lint, tests, config parsing, tracked-source, documentation references
      2. pull       - only onto a known-good tree, so a conflict is not discovered mid-push
      3. commit     - the pre-commit hook runs here as a second, independent line of defence
      4. push

    Preflight runs *before* the pull because a pull that changes files must not be pushed
    unchecked. If the pull brings in new commits, preflight has already validated the local tree
    and the incoming changes will have had their own CI run upstream.

.PARAMETER Message
    Commit message. Defaults to a timestamp, matching the previous behaviour of ``s.bat``.

.PARAMETER SkipPreflight
    Skip the preflight gate. This is the escape hatch for a commit that cannot pass locally, for
    example when the only change is to a document that references a path added in another branch.
    It prints a warning rather than passing silently, because an unexplained red CI is expensive
    to diagnose later.

.PARAMETER NoPause
    Do not wait for a keypress at the end. Used when this script is called from automation.

.EXAMPLE
    .\tools\sync.ps1

.EXAMPLE
    .\tools\sync.ps1 -Message "fix: 修正索引重建的边界条件"
#>
[CmdletBinding()]
param(
    [string]$Message,
    [switch]$SkipPreflight,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"

$repoRoot = (& git rev-parse --show-toplevel 2>$null)
if (-not $repoRoot) { throw "not inside a git working tree" }
Set-Location $repoRoot

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "=== $Text ===" -ForegroundColor Cyan
}

$failed = $false

# --- 1. preflight -----------------------------------------------------------------------------
Write-Step "preflight"
if ($SkipPreflight) {
    Write-Host "SKIPPED by -SkipPreflight" -ForegroundColor Yellow
    Write-Host "You are pushing without the local gate. If CI fails, the cause is in this commit." -ForegroundColor Yellow
} else {
    & pwsh -NoProfile -File (Join-Path $repoRoot "tools/preflight.ps1")
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "PREFLIGHT FAILED - nothing was committed or pushed." -ForegroundColor Red
        Write-Host "Fix the failures above, or re-run with -SkipPreflight to override." -ForegroundColor Red
        $failed = $true
    }
}

# --- 2. pull ----------------------------------------------------------------------------------
if (-not $failed) {
    Write-Step "git pull"
    & git pull
    if ($LASTEXITCODE -ne 0) {
        Write-Host "git pull failed - resolve it before syncing." -ForegroundColor Red
        $failed = $true
    }
}

# --- 3. commit --------------------------------------------------------------------------------
if (-not $failed) {
    Write-Step "git add"
    & git add -A
    if ($LASTEXITCODE -ne 0) { $failed = $true }
}

if (-not $failed) {
    Write-Step "git commit"
    if (-not $Message) {
        # A bare timestamp is a useless log line; the prefix makes the automatic commits
        # self-identifying in ``git log`` next to hand-written ones.
        $Message = "sync $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    }
    & git commit -m $Message
    # A commit exits 1 when there is nothing to commit, which is not a failure worth stopping on.
    if ($LASTEXITCODE -ne 0) {
        Write-Host "nothing to commit, or the pre-commit hook rejected it." -ForegroundColor Yellow
        & git status --short
    }
}

# --- 4. push ----------------------------------------------------------------------------------
if (-not $failed) {
    Write-Step "git push"
    & git push
    if ($LASTEXITCODE -ne 0) {
        Write-Host "git push failed." -ForegroundColor Red
        $failed = $true
    }
}

Write-Step "git status"
& git status --short --branch

if (-not $NoPause) {
    Write-Host ""
    Read-Host "press Enter to close"
}

exit $(if ($failed) { 1 } else { 0 })
