<#
.SYNOPSIS
    Install the repository's git hooks.

.DESCRIPTION
    The hooks live in ``tools/git-hooks/`` so they are versioned and reviewable, which
    ``.git/hooks`` cannot be. Installing them is therefore one command that a new clone runs
    once, and the same command a CI job could run.

    ``core.hooksPath`` is used rather than copying files, so the tracked hook is the single
    source and an edit to it takes effect immediately with no re-install.

.EXAMPLE
    .\tools\install-git-hooks.ps1
#>
[CmdletBinding()]
param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

# ``git rev-parse --git-dir`` rather than assuming .git, so worktrees and submodules work.
$gitDir = (& git rev-parse --git-dir 2>$null)
if (-not $gitDir) { throw "not inside a git working tree" }
$repoRoot = (& git rev-parse --show-toplevel 2>$null)
Set-Location $repoRoot

if ($Uninstall) {
    & git config --local --unset core.hooksPath 2>$null
    Write-Host "git hooks uninstalled (core.hooksPath cleared)"
    exit 0
}

$hooksRel = "tools/git-hooks"
if (-not (Test-Path $hooksRel)) { throw "hook directory not found: $hooksRel" }

# Make the shell hooks executable in the index, so a POSIX clone can run them directly.
& git update-index --chmod=+x "$hooksRel/pre-commit" 2>$null

& git config --local core.hooksPath $hooksRel
$installed = (& git config --local --get core.hooksPath)

Write-Host "git hooks installed"
Write-Host "  core.hooksPath = $installed"
Get-ChildItem $hooksRel -File | ForEach-Object { Write-Host "  hook: $($_.Name)" }
Write-Host ""
Write-Host "The pre-commit hook blocks: machine-specific absolute paths, credential"
Write-Host "assignments, and staged files over 5 MB. Bypass a false positive with"
Write-Host "'git commit --no-verify'."
