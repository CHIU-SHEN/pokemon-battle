param(
    [double]$DelaySeconds = 2.0,
    [int]$Limit = 0,
    [switch]$SkipAudit
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$env:PYTHONIOENCODING = 'utf-8'

$IndexPath = Join-Path $PSScriptRoot 'data\external\kaggle_replays\replay_index.json'
$OutputDir = Join-Path $PSScriptRoot 'data\external\kaggle_replays\raw'
$ProgressPath = Join-Path $PSScriptRoot 'data\external\kaggle_replays\restore_progress.json'
$AuditPath = Join-Path $PSScriptRoot 'data\external\kaggle_replays\restore_audit_summary.json'

if (-not (Test-Path -LiteralPath $IndexPath)) {
    throw "Replay index is missing: $IndexPath"
}

$Python = $null
$Candidates = @()
if ($env:CONDA_PREFIX) {
    $Candidates += (Join-Path $env:CONDA_PREFIX 'python.exe')
}
if ($env:CONDA_EXE) {
    $CondaRoot = Split-Path -Parent (Split-Path -Parent $env:CONDA_EXE)
    $Candidates += (Join-Path $CondaRoot 'envs\pokemon-tcg\python.exe')
}
$Candidates += (Join-Path $env:USERPROFILE 'miniconda3\envs\pokemon-tcg\python.exe')
$Candidates += (Join-Path $env:USERPROFILE 'anaconda3\envs\pokemon-tcg\python.exe')
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($PythonCommand) {
    $Candidates += $PythonCommand.Source
}
foreach ($Candidate in $Candidates) {
    if ($Candidate -and (Test-Path -LiteralPath $Candidate)) {
        try {
            & $Candidate '--version' *> $null
            if ($LASTEXITCODE -eq 0) {
                $Python = $Candidate
                break
            }
        } catch {
            continue
        }
    }
}
if (-not $Python) {
    throw 'Python was not found. Activate the pokemon-tcg Conda environment and run this script again.'
}

$Kaggle = $null
$KaggleBesidePython = Join-Path (Split-Path -Parent $Python) 'Scripts\kaggle.exe'
if (Test-Path -LiteralPath $KaggleBesidePython) {
    $Kaggle = $KaggleBesidePython
} else {
    $KaggleCommand = Get-Command kaggle -ErrorAction SilentlyContinue
    if ($KaggleCommand) {
        $Kaggle = $KaggleCommand.Source
    }
}

Write-Host 'Restoring the exact 5,974 raw Kaggle replays recorded by the repository index.'
Write-Host 'Existing files are skipped automatically. Run this script again to resume an interrupted download.'

$DownloadArgs = @(
    'scripts\download_replays_from_index.py',
    '--index', $IndexPath,
    '--output', $OutputDir,
    '--progress', $ProgressPath,
    '--delay', $DelaySeconds,
    '--limit', $Limit
)
if ($Kaggle) {
    $DownloadArgs += @('--kaggle', $Kaggle)
}
& $Python @DownloadArgs
if ($LASTEXITCODE -ne 0) {
    throw "Raw replay download failed with exit code $LASTEXITCODE."
}

if (-not $SkipAudit) {
    Write-Host 'Download stage finished. Creating the integrity audit report.'
    & $Python 'scripts\audit_kaggle_replays.py' '--input' $OutputDir '--output' $AuditPath
    if ($LASTEXITCODE -ne 0) {
        throw "Replay audit failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Done. Raw replay directory: $OutputDir"
Write-Host "Restore progress: $ProgressPath"
