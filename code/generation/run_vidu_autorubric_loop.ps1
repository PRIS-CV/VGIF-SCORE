param(
    [string]$RunDir = '',
    [string]$Entries = '',
    [int]$TargetCount = 223,
    [int]$MaxWorkers = 4,
    [int]$TimeoutSeconds = 90,
    [int]$SleepSeconds = 20,
    [string]$Model = 'gemini-3.1-pro-preview',
    [string]$OutputTag = 'gemini-3.1-pro-preview'
)

$ErrorActionPreference = 'Stop'

$repoRoot = if ($env:VGIF_REPO_ROOT) { $env:VGIF_REPO_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
if (-not $RunDir) { $RunDir = Join-Path $repoRoot 'models\ViduQ3-Turbo' }
if (-not $Entries) { $Entries = Join-Path $repoRoot 'data\vgif_bench\vgif_bench.jsonl' }

$videosDir = Join-Path $RunDir 'videos'
$summary = Join-Path $RunDir ("autorubric_summary_{0}.json" -f $OutputTag)
$resultPattern = "*_{0}_autorubric_eval.json" -f $OutputTag

while ($true) {
    $count = (Get-ChildItem -Path $videosDir -Filter $resultPattern -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Output ('{0} count={1}' -f (Get-Date -Format s), $count)

    if ($count -ge $TargetCount) {
        break
    }

    $argsList = @(
        (Join-Path $repoRoot 'code\evaluation\evaluate_kling_batch_accuracy.py'),
        '--run-dir', $RunDir,
        '--entries', $Entries,
        '--question-mode', 'autorubric',
        '--model', $Model,
        '--output-tag', $OutputTag,
        '--skip-model-check',
        '--max-workers', $MaxWorkers.ToString(),
        '--timeout', $TimeoutSeconds.ToString(),
        '--retries', '0',
        '--format-retries', '0',
        '--reuse-existing',
        '--output', $summary
    )

    & python @argsList
    Write-Output ('{0} sweep_exit={1}' -f (Get-Date -Format s), $LASTEXITCODE)
    Start-Sleep -Seconds $SleepSeconds
}

Write-Output ('{0} completed' -f (Get-Date -Format s))
