$ErrorActionPreference = "Stop"

$RepoRoot = if ($env:VGIF_REPO_ROOT) { $env:VGIF_REPO_ROOT } else { (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
$PythonExe = "python"
$BaseUrl = $env:VGIF_BASE_URL
$ApiKey = $env:VGIF_API_KEY
$RunDir = if ($env:VGIF_RUN_DIR) { $env:VGIF_RUN_DIR } else { Join-Path $RepoRoot "models\CogVideoX-1.5" }
$BatchScript = Join-Path $RepoRoot "code\evaluation\evaluate_kling_batch_accuracy.py"
$SummaryPath = Join-Path $RunDir "qa_eval_dependency-rounds_summary.json"
$LogPath = Join-Path $RunDir "logs\cogvideox_dependency_rounds_watch.log"
$CheckIntervalSeconds = 300

if (-not $BaseUrl -or -not $ApiKey) {
    throw "Set VGIF_BASE_URL and VGIF_API_KEY before running this script."
}

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    Add-Content -Path $LogPath -Value "[$timestamp] $Message"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
Write-Log "Watcher started."

while ($true) {
    $serviceUp = $false

    try {
        $response = Invoke-WebRequest `
            -Uri "$BaseUrl/v1/models" `
            -Headers @{ Authorization = "Bearer $ApiKey" } `
            -TimeoutSec 20
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
            $serviceUp = $true
            Write-Log "Service is reachable. Launching/resuming batch evaluation."
        }
    } catch {
        Write-Log "Service unavailable: $($_.Exception.Message)"
    }

    if (-not $serviceUp) {
        Start-Sleep -Seconds $CheckIntervalSeconds
        continue
    }

    Push-Location $RepoRoot
    try {
        & $PythonExe $BatchScript `
            --run-dir $RunDir `
            --question-mode dependency-rounds `
            --max-workers 1 `
            --video-attempts 3 `
            --timeout 300 `
            --retries 4 `
            --skip-model-check `
            --reuse-existing *>> $LogPath
    } catch {
        Write-Log "Batch process launcher failed: $($_.Exception.Message)"
    } finally {
        Pop-Location
    }

    if (Test-Path $SummaryPath) {
        try {
            $summary = Get-Content -Path $SummaryPath -Raw | ConvertFrom-Json
            $successCount = [int]$summary.success_count
            $failureCount = [int]$summary.failure_count
            $videoCount = [int]$summary.video_count
            Write-Log "Current summary: success=$successCount failure=$failureCount total=$videoCount"
            if ($videoCount -gt 0 -and $successCount -eq $videoCount -and $failureCount -eq 0) {
                Write-Log "All videos completed successfully. Watcher exiting."
                break
            }
        } catch {
            Write-Log "Failed to parse summary: $($_.Exception.Message)"
        }
    } else {
        Write-Log "Summary file not found after batch run."
    }

    Start-Sleep -Seconds $CheckIntervalSeconds
}
