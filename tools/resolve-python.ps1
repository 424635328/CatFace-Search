<#
.SYNOPSIS
    Locate a Python interpreter for the project's tooling.

.DESCRIPTION
    Resolution order, first hit wins:

      1. ``$env:PREFLIGHT_PYTHON``  — explicit override for unusual setups
      2. ``.venv``                  — Windows (Scripts/python.exe) then POSIX (bin/python)
      3. ``venv``                   — same two layouts
      4. ``python3`` / ``python``   — PATH

    Both virtualenv layouts are probed regardless of the current OS, because this repository is
    developed on Windows and its CI runs on Linux. A Windows-only ``Scripts\python.exe`` path is
    the usual reason a script that works locally fails in CI.

    Emits the resolved path on success and nothing on failure, so callers can test it.
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)

$candidates = @()
if ($env:PREFLIGHT_PYTHON) { $candidates += $env:PREFLIGHT_PYTHON }
foreach ($env_dir in @('.venv', 'venv')) {
    $candidates += (Join-Path $RepoRoot "$env_dir/Scripts/python.exe")
    $candidates += (Join-Path $RepoRoot "$env_dir/bin/python")
}
$candidates += @('python3', 'python')

foreach ($candidate in $candidates) {
    if ($candidate -match '[\\/]') {
        # A path: it must both exist and run, so a broken venv is not accepted as the answer.
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        $probe = & $candidate -c "import sys; print(sys.version_info[0])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $probe -eq '3') { Write-Output $candidate; return }
        continue
    }
    $found = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($found) { Write-Output $found.Source; return }
}

Write-Error "No Python 3 interpreter found. Create .venv, or set `$env:PREFLIGHT_PYTHON."
exit 1
