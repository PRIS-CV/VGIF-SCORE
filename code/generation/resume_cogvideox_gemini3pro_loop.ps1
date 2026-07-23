$ErrorActionPreference = "Stop"

$RepoRoot = if ($env:VGIF_REPO_ROOT) { $env:VGIF_REPO_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$PythonExe = "python"
$RunDir = if ($env:VGIF_RUN_DIR) { $env:VGIF_RUN_DIR } else { Join-Path $RepoRoot "models\CogVideoX-1.5" }
$BatchScript = Join-Path $RepoRoot "code\evaluation\evaluate_kling_batch_accuracy.py"
$LogPath = Join-Path $RunDir "logs\cogvideox_dependency_rounds_gemini3pro_loop.log"
$TargetCount = 223
$StartIndex = 207
$SleepSeconds = 90

$ApiKey = $env:VGIF_API_KEY
$BaseUrl = $env:VGIF_BASE_URL
$Model = "gemini-3-pro-preview"

if (-not $BaseUrl -or -not $ApiKey) {
    throw "Set VGIF_BASE_URL and VGIF_API_KEY before running this script."
}

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    Add-Content -Path $LogPath -Value "[$timestamp] $Message"
}

function Get-CompletedCount {
    return (Get-ChildItem -Path $RunDir -Filter '*_qa_eval_dependency_rounds.json' -ErrorAction SilentlyContinue).Count
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
Write-Log "Loop runner started."

while ($true) {
    $completed = Get-CompletedCount
    Write-Log "Current completed count: $completed / $TargetCount"

    if ($completed -ge $TargetCount) {
        Write-Log "Target reached. Exiting."
        break
    }

    Push-Location $RepoRoot
    try {
        & $PythonExe $BatchScript `
            --run-dir $RunDir `
            --question-mode dependency-rounds `
            --start-index $StartIndex `
            --max-workers 1 `
            --video-attempts 12 `
            --timeout 600 `
            --retries 8 `
            --skip-model-check `
            --reuse-existing `
            --model $Model `
            --api-key $ApiKey `
            --base-url $BaseUrl *>> $LogPath
    } catch {
        Write-Log "Batch launcher failed: $($_.Exception.Message)"
    } finally {
        Pop-Location
    }

    $completedAfter = Get-CompletedCount
    Write-Log "Completed count after round: $completedAfter / $TargetCount"

    if ($completedAfter -ge $TargetCount) {
        Write-Log "Target reached after round. Exiting."
        break
    }

    Start-Sleep -Seconds $SleepSeconds
}
