<#
.SYNOPSIS
    Thin PowerShell wrapper around the training CLI.

.DESCRIPTION
    Convenience entry point for Windows machines, where the bash scripts in this
    folder are awkward to run. Set $env:PATHOVOICE_PYTHON to the interpreter that
    has the dependencies from models/asr/requirements.txt installed.

.EXAMPLE
    .\models\asr\scripts\train.ps1 -ConfigPath models\asr\configs\tiny.yaml

.EXAMPLE
    .\models\asr\scripts\train.ps1 -ConfigPath models\asr\configs\tiny.yaml -Multitask
#>
param(
    [string]$ConfigPath = "models/asr/configs/tiny.yaml",
    [switch]$Multitask
)

$ErrorActionPreference = "Stop"

# Environment
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$python = if ($env:PATHOVOICE_PYTHON) { $env:PATHOVOICE_PYTHON } else { "python" }

# Run
Push-Location $repoRoot
try {
    $arguments = @(
        "-m", "asr.main.trainer",
        "--config-path", $ConfigPath
    )
    if ($Multitask) {
        $arguments += "--multitask"
    }

    & $python @arguments
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $exitCode
