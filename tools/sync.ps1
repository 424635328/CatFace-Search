<#
.SYNOPSIS
    Run the project's checks, then pull, commit and push - in that order.

.DESCRIPTION
    ``git pull && git add . && git commit && git push`` is three seconds of typing and it is also
    how a red CI happens. This repository's first real CI run failed 3 of 5 jobs, and every one of
    those failures was detectable locally before the push. So the gate runs first, and nothing is
    committed or pushed when it fails.

    The order is deliberate:

      1. preflight  - the project's own checks, run before anything mutates the tree
      2. pull       - onto a tree already known good, so a conflict is not found mid-push
      3. commit     - the pre-commit hook runs here as a second, independent line of defence
      4. push

    Preflight runs *before* the pull because a pull that changes files must not then be pushed
    unchecked. If the pull brings in new commits, those had their own CI run upstream.

    The gate is discovered, not hardcoded: this script looks for tools/preflight.ps1 in the
    repository it runs from. A repository without one still gets the portable staged-file check at
    commit time, and is told plainly that it has no project checks rather than being waved through.

.PARAMETER Message
    Commit message. Defaults to ``sync <timestamp>``.

.PARAMETER DryRun
    Read-only: show the branch, the incoming commits, the staged file summary and the gate that
    would run. No pull, no commit, no push. Does not execute the checks themselves, so it stays
    fast; run tools/preflight.ps1 directly to actually run them.

.PARAMETER NoPush
    Commit locally but do not push. Useful for stacking commits into one CI run.

.PARAMETER Quiet
    Print only warnings, the final summary, and anything that failed.

.PARAMETER SkipPreflight
    Skip the gate. The escape hatch, and it prints a warning rather than passing silently: it is
    not hypothetical that skipping the gate let junk reach a public remote, because the gate was
    the only thing that looked at the whole tree. Alias: -y.

.PARAMETER NoWait
    Do not wait for a keypress at the end. Already implied when the host is non-interactive, so a
    piped or scheduled invocation cannot hang here. Alias: -NoPause.

.PARAMETER Wait
    Force the keypress even when the host looks non-interactive.

.OUTPUTS
    Exit code, so a caller can branch on the cause:

      0  success (pushed, or committed with -NoPush, or nothing to do)
      2  preflight failed - nothing was committed or pushed
      3  git pull failed
      4  the pre-commit hook rejected the commit
      5  git add or git commit failed
      6  git push failed
     64  usage: no git, or not inside a working tree

.EXAMPLE
    .\tools\sync.ps1
    .\tools\sync.ps1 -DryRun
    .\tools\sync.ps1 -Message "fix: 修正索引重建的边界条件"
    .\tools\sync.ps1 -NoPush
    .\tools\sync.ps1 -y
#>
[CmdletBinding()]
param(
    [Alias('m')]
    [string]$Message,

    [Alias('n')]
    [switch]$NoPush,

    [switch]$DryRun,

    [Alias('q')]
    [switch]$Quiet,

    [Alias('y')]
    [switch]$SkipPreflight,

    [Alias('NoPause')]
    [switch]$NoWait,

    [switch]$Wait,

    # Where to look for the repository. s.bat passes its own directory, so a double-click or a
    # shortcut that starts in some other working directory still syncs the intended repository
    # rather than reporting "not inside a git working tree".
    [string]$RepoRoot
)

# Named exit codes: a script other scripts call needs stable, documented codes. One bare "exit 1"
# for five different causes is not something a caller can act on.
$EXIT_OK = 0
$EXIT_PREFLIGHT = 2
$EXIT_PULL = 3
$EXIT_HOOK = 4
$EXIT_COMMIT = 5
$EXIT_PUSH = 6
$EXIT_USAGE = 64

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Text)
    if (-not $Quiet) {
        Write-Host ""
        Write-Host "=== $Text ===" -ForegroundColor Cyan
    }
}
function Write-Info { param([string]$Text) if (-not $Quiet) { Write-Host $Text } }
function Write-Warn2 { param([string]$Text) Write-Host $Text -ForegroundColor Yellow }
function Write-Fail { param([string]$Text) Write-Host $Text -ForegroundColor Red }

# --- context ------------------------------------------------------------------------------------
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Fail "git is not on PATH; nothing to sync with."
    exit $EXIT_USAGE
}

# Start from the requested directory when the caller supplied one, so the repository is found by
# location rather than by whatever the current directory happens to be.
if ($RepoRoot) {
    if (-not (Test-Path -LiteralPath $RepoRoot)) {
        Write-Fail "repository path does not exist: $RepoRoot"
        exit $EXIT_USAGE
    }
    Set-Location $RepoRoot
}

$repoRoot = (& git rev-parse --show-toplevel 2>$null)
if (-not $repoRoot) {
    Write-Fail "not inside a git working tree."
    exit $EXIT_USAGE
}
Set-Location $repoRoot

$branch = & git rev-parse --abbrev-ref HEAD 2>$null
if (-not $branch) { $branch = "(detached)" }

# Whether this branch tracks an upstream decides if pull/push are even meaningful. A local-only
# repository is a normal thing to be in, not an error: without this check ``git pull`` fails and
# gets reported as a conflict, which sends the reader looking for a problem that does not exist.
$hasRemote = [bool](& git remote 2>$null)
& git rev-parse --verify --quiet '@{u}' > $null 2>&1
$hasUpstream = ($LASTEXITCODE -eq 0)

# Interactive detection so an unattended call can never block on a keypress even when the caller
# forgets -NoPause: a double-clicked .bat is interactive, a piped or scheduled run is not.
$interactive = $true
try { $interactive = -not [Console]::IsInputRedirected } catch { $interactive = $true }
$shouldPause = $interactive -and ($Wait -or -not $NoWait)

# An explicit interrupt must stop the run *and* still report where it stopped, rather than killing
# the process with whatever state it happened to be in.
$script:Interrupted = $false
$onCancel = [ConsoleCancelEventHandler] {
    param($sender, $eventArgs)
    $eventArgs.Cancel = $true
    $script:Interrupted = $true
    Write-Host ""
    Write-Warn2 "interrupted - finishing the current step, then stopping"
}
try { [Console]::add_CancelKeyPress($onCancel) } catch { }

$exitCode = $EXIT_OK
$outcome = ""
$startedAll = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host "sync: $repoRoot" -ForegroundColor White
Write-Info "branch: $branch"
if ($DryRun) { Write-Info "mode  : dry run (nothing will change)" }

try {
    # --- 1. gate --------------------------------------------------------------------------------
    Write-Step "preflight"
    if ($SkipPreflight) {
        Write-Warn2 "SKIPPED by -SkipPreflight"
        Write-Warn2 "You are pushing without the local gate. If CI fails, the cause is in this commit."
    } else {
        $preflight = Join-Path $repoRoot "tools/preflight.ps1"
        if (Test-Path -LiteralPath $preflight) {
            if ($DryRun) {
                Write-Info "would run: tools/preflight.ps1"
                Write-Info "           (-DryRun stays fast and read-only; run it directly to execute)"
            } else {
                $preflightArgs = @()
                if ($Quiet) { $preflightArgs += "-Quiet" }
                & pwsh -NoProfile -File $preflight @preflightArgs
                if ($LASTEXITCODE -ne 0) {
                    Write-Host ""
                    Write-Fail "PREFLIGHT FAILED - nothing was committed or pushed."
                    Write-Fail "Fix the failures above, or re-run with -SkipPreflight (-y) to override."
                    $exitCode = $EXIT_PREFLIGHT
                    $outcome = "preflight failed"
                }
            }
        } else {
            Write-Warn2 "no tools/preflight.ps1 in this repository - no project checks configured."
            Write-Warn2 "the portable staged-file check still runs at commit time via the pre-commit hook."
            Write-Info "to add project checks, copy tools/preflight.ps1 and tools/project.config.ps1"
        }
    }

    # --- 2. pull --------------------------------------------------------------------------------
    if ($exitCode -eq $EXIT_OK -and -not $script:Interrupted) {
        Write-Step "git pull"
        if (-not $hasUpstream) {
            Write-Info "no upstream for this branch - nothing to pull"
            if (-not $hasRemote) { Write-Info "this repository has no remote configured" }
        } elseif ($DryRun) {
            $behind = & git rev-list --count "HEAD..@{u}" 2>$null
            if ($LASTEXITCODE -eq 0 -and $behind) { Write-Info "incoming commits from upstream: $behind" }
            else { Write-Info "upstream: up to date" }
        } else {
            $pull = & git pull 2>&1
            $pull | ForEach-Object { Write-Info "  $_" }
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "git pull failed - resolve the conflict, then re-run."
                $exitCode = $EXIT_PULL
                $outcome = "pull failed"
            }
        }
    }

    # --- 3. commit ------------------------------------------------------------------------------
    if ($exitCode -eq $EXIT_OK -and -not $script:Interrupted) {
        Write-Step "git add"
        if ($DryRun) {
            $changes = @(& git status --porcelain)
            if ($changes.Count -gt 0) {
                Write-Info "would stage $($changes.Count) path(s):"
                # The file list matters more than the diff: "git add -A" is indiscriminate, and a
                # stray file reaching a public remote is the failure this tool exists to prevent.
                $changes | Select-Object -First 15 | ForEach-Object { Write-Info "  $_" }
                if ($changes.Count -gt 15) { Write-Info "  ... and $($changes.Count - 15) more" }
            } else {
                Write-Info "working tree clean - nothing to stage"
            }
        } else {
            & git add -A
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "git add failed."
                $exitCode = $EXIT_COMMIT
                $outcome = "git add failed"
            }
        }
    }

    if ($exitCode -eq $EXIT_OK -and -not $script:Interrupted) {
        Write-Step "git commit"
        if (-not $Message) {
            # A bare timestamp is a useless log line; the prefix identifies automatic commits in
            # ``git log`` next to hand-written ones.
            $Message = "sync $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        }
        if ($DryRun) {
            Write-Info "would commit with message: $Message"
            $checker = Join-Path $repoRoot "tools/check_staged.py"
            if (Test-Path -LiteralPath $checker) {
                $py = & (Join-Path $repoRoot "tools/resolve-python.ps1") -RepoRoot $repoRoot
                if ($py) {
                    & $py $checker
                    if ($LASTEXITCODE -ne 0) {
                        Write-Fail "the portable scratch-file check would refuse this commit."
                        $exitCode = $EXIT_HOOK
                        $outcome = "staged files rejected"
                    }
                }
            }
        } else {
            $commitOut = & git commit -m $Message 2>&1
            $commitOut | ForEach-Object { Write-Info "  $_" }
            if ($LASTEXITCODE -ne 0) {
                $joined = ($commitOut | Out-String)
                if ($joined -match "nothing to commit|no changes added") {
                    Write-Info "nothing to commit - the working tree has no staged changes."
                    $outcome = "nothing to commit"
                } elseif ($joined -match "pre-commit|Commit blocked") {
                    Write-Fail "the pre-commit hook rejected this commit (see above)."
                    $exitCode = $EXIT_HOOK
                    $outcome = "commit blocked by hook"
                } else {
                    Write-Fail "git commit failed."
                    $exitCode = $EXIT_COMMIT
                    $outcome = "commit failed"
                }
            }
        }
    }

    # --- 4. push --------------------------------------------------------------------------------
    if ($exitCode -eq $EXIT_OK -and -not $script:Interrupted -and $outcome -ne "nothing to commit") {
        Write-Step "git push"
        if ($NoPush) {
            Write-Info "skipped by -NoPush (the commit is local only)"
            $outcome = "committed, not pushed"
        } elseif (-not $hasUpstream) {
            # Not an error: a local-only repository is a legitimate place to run this.
            Write-Info "no upstream for this branch - nothing to push"
            $outcome = "committed, no upstream configured"
        } elseif ($DryRun) {
            $ahead = & git rev-list --count "@{u}..HEAD" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ahead) { Write-Info "would push $ahead commit(s) to origin/$branch" }
            else { Write-Info "would push: nothing ahead of upstream" }
        } else {
            $push = & git push 2>&1
            $push | ForEach-Object { Write-Info "  $_" }
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "git push failed."
                Write-Fail "If this is a rejected non-fast-forward, pull rather than forcing."
                $exitCode = $EXIT_PUSH
                $outcome = "push failed"
            }
        }
    }

    if ($script:Interrupted -and $exitCode -eq $EXIT_OK) { $outcome = "interrupted" }
} finally {
    try { [Console]::remove_CancelKeyPress($onCancel) } catch { }
}

# --- summary ------------------------------------------------------------------------------------
Write-Host ""
if (-not $Quiet) {
    try { & git status --short --branch } catch { }
    Write-Host ""
}

$elapsed = [int]$startedAll.Elapsed.TotalSeconds
if ($exitCode -eq $EXIT_OK) {
    $detail = if ($outcome) { " - $outcome" } else { "" }
    Write-Host ("SYNC OK ({0}s){1}" -f $elapsed, $detail) -ForegroundColor Green
    if ($outcome -eq "nothing to commit") {
        Write-Info "there was nothing to sync; run with -DryRun to see what would happen"
    }
} else {
    Write-Fail ("SYNC FAILED (exit {0}: {1}, {2}s)" -f $exitCode, $outcome, $elapsed)
}

if ($shouldPause) {
    Write-Host ""
    Read-Host "press Enter to close" | Out-Null
}

exit $exitCode
