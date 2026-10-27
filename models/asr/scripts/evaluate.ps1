<#
.SYNOPSIS
    Thin PowerShell wrapper around the evaluation CLI.

.EXAMPLE
    .\models\asr\scripts\evaluate.ps1 `
        -ConfigPath models\asr\configs\tiny.yaml `
        -ModelPath models\asr\checkpoints\transcribe_aug\seed3407\tiny `
        -TestAudioDir models\asr\datasets\test_set\wav `
        -TestLabelFile models\asr\datasets\test_set\transcript
#>
param(
    [Parameter(Mandatory = $true)][string]$ConfigPath,
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [Parameter(Mandatory = $true)][string]$TestAudioDir,
    [Parameter(Mandatory = $true)][string]$TestLabelFile,
    [string]$TestTranslateFile = "",
    [ValidateSet("transcribe", "translate")][string]$Task = "transcribe",
    [string]$Metrics = "",
    [int]$TestingBatchSize = 10,
    [int]$NumMels = 80,
    [string]$OutputCsv = ""
)

$ErrorActionPreference = "Stop"

# Environment
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$python = if ($env:PATHOVOICE_PYTHON) { $env:PATHOVOICE_PYTHON } else { "python" }

# Run
Push-Location $repoRoot
try {
    $arguments = @(
        "-m", "asr.main.evaluate",
        "--config-path", $ConfigPath,
        "--model-path", $ModelPath,
        "--test-audio-dir", $TestAudioDir,
        "--test-label-file", $TestLabelFile,
        "--task", $Task,
        "--testing-batch-size", "$TestingBatchSize",
        "--n-mels", "$NumMels"
    )
    if ($Task -eq "translate" -and -not $TestTranslateFile) {
        throw "-TestTranslateFile is required when -Task is translate."
    }
    if ($TestTranslateFile) {
        $arguments += @("--test-translate-file", $TestTranslateFile)
    }
    if ($Metrics) {
        $arguments += @("--metrics", $Metrics)
    }
    if ($OutputCsv) {
        $arguments += @("--output-csv", $OutputCsv)
    }

    & $python @arguments
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $exitCode
