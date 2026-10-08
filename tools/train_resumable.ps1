<#
.SYNOPSIS
    Train an embedder with automatic resume, so an interrupted job keeps its progress.

.DESCRIPTION
    Background jobs are torn down when their parent session ends, and long GPU runs outlive
    single sessions. This wrapper makes that harmless: it loops, calling the trainer with
    --resume each time, so a kill at any point costs at most the current epoch.

    Deliberately bounded in time (--max-seconds) so each iteration returns control and the
    loop can decide whether to continue. Run it in a managed background job, or start it
    again later — either way training continues from the last checkpoint rather than from
    scratch.

.PARAMETER Backbone
    Backbone registry name, e.g. dinov2_vits14.

.PARAMETER Output
    Run directory. Its train_state.pt is what resume reads.

.PARAMETER Epochs
    Total epochs for the run. Safe to increase between invocations, which is how a schedule
    is extended without retraining.

.PARAMETER RoundMinutes
    Wall-clock budget per invocation. Smaller values checkpoint more often relative to
    outside interruption.

.PARAMETER Rounds
    Maximum invocations before stopping. The loop also stops when the schedule completes.

.EXAMPLE
    .\tools\train_resumable.ps1 -Backbone dinov2_vits14 -Output artifacts/train/dinov2s-arcface -Epochs 20
#>
[CmdletBinding()]
param(
    [string]$Backbone = "dinov2_vits14",
    [string]$Output = "artifacts/train/dinov2s-arcface",
    [int]$Epochs = 20,
    [double]$LearningRate = 1e-3,
    [int]$IdentitiesPerBatch = 16,
    [int]$SamplesPerIdentity = 4,
    [int]$ImageSize = 224,
    [int]$Rounds = 40,
    [double]$RoundMinutes = 8,
    [switch]$GradientCheckpointing,
    [int]$NumWorkers = 4,
    [int]$EarlyStopPatience = 8
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "virtualenv python not found at $python" }

$statePath = Join-Path $repo "$Output\train_state.pt"
Write-Host "resumable training: backbone=$Backbone epochs=$Epochs output=$Output"
Write-Host "state file: $statePath (exists: $(Test-Path $statePath))"

$common = @(
    "-m", "tools.train_embedder",
    "--backbone", $Backbone,
    "--epochs", "$Epochs",
    "--lr", "$LearningRate",
    "--identities-per-batch", "$IdentitiesPerBatch",
    "--samples-per-identity", "$SamplesPerIdentity",
    "--image-size", "$ImageSize",
    "--num-workers", "$NumWorkers",
    "--early-stop-patience", "$EarlyStopPatience",
    "--output", $Output
)
if ($GradientCheckpointing) { $common += "--gradient-checkpointing" }

for ($round = 1; $round -le $Rounds; $round++) {
    $resume = (Test-Path $statePath)
    if ($resume) { Write-Host "[round $round] resuming from checkpoint" }
    else { Write-Host "[round $round] starting a new run" }

    $args = $common + @("--max-seconds", "$([int]($RoundMinutes * 60))")
    if ($resume) { $args += "--resume" }

    Push-Location $repo
    try { & $python @args } finally { Pop-Location }

    # The trainer reports completion by clearing pause_reason in the history file.
    $historyPath = Join-Path $repo "$Output\training_history.json"
    if (Test-Path $historyPath) {
        $history = Get-Content $historyPath -Raw | ConvertFrom-Json
        $done = $history.epochs.Count
        $reason = $history.pause_reason
        Write-Host "[round $round] epochs completed: $done / $Epochs ; pause reason: $reason"
        if (-not $reason) {
            Write-Host "run finished normally; stopping the loop"
            break
        }
        if ($history.stopped_early) {
            Write-Host "early stopping triggered; stopping the loop"
            break
        }
        if ($done -ge $Epochs) {
            Write-Host "schedule complete; stopping the loop"
            break
        }
    } else {
        Write-Warning "[round $round] no history file yet; will retry"
    }
}

Write-Host "resumable training loop finished"
