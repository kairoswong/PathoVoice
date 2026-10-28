#!/usr/bin/env pwsh
# pt2ggml: .pt -> ggml .bin -> quantize -> verify
#
# Usage:
#   .\src\pt2ggml\run.ps1 models\asr\checkpoints\tiny.pt
#   .\src\pt2ggml\run.ps1 tiny.pt -QuantTypes q5_1
#

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Pt,
    [string]$OutDir,
    [string]$Name,
    [string[]]$QuantTypes = @(),
    [switch]$CheckOnly,
    [switch]$Force,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'

# -----------------------------------
# Paths

$repoRoot    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$whisperSrc  = Join-Path $repoRoot 'whisper.cpp'
$buildDir    = Join-Path $whisperSrc 'build-x64-windows-msvc-release'

function Find-Tool {
    param([string]$Name)
    Get-ChildItem $buildDir -Recurse -Filter $Name -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}

$quantizeExe = Find-Tool 'whisper-quantize.exe'
$benchExe    = Find-Tool 'whisper-bench.exe'

$ptPath = (Resolve-Path $Pt).Path
if (-not $OutDir) { $OutDir = Split-Path -Parent $ptPath }
if (-not $Name)   { $Name = 'ggml-' + [IO.Path]::GetFileNameWithoutExtension($ptPath) }
$binPath = Join-Path $OutDir "$Name.bin"

# -----------------------------------
# Convert

Write-Host "`n=== pt2ggml ===" -ForegroundColor Cyan

# -----------------------------------
# Build whisper.cpp when needed

function Invoke-Build {
    if ($SkipBuild) { throw 'binaries missing and -SkipBuild was given' }

    # vswhere ships with VS 2017+ and finds any install location.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vsPath = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath 2>$null | Select-Object -First 1
    if (-not $vsPath) { throw 'Visual Studio C++ build tools not found; build whisper.cpp manually or pass -SkipBuild' }

    $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'
    Write-Host "[build] compiling whisper.cpp (first run only)..." -ForegroundColor Yellow

    # '&&' must stay inside the cmd string: PowerShell 5.1 has no && operator.
    cmd /c "call `"$vcvars`" >nul && cmake -S `"$whisperSrc`" -B `"$buildDir`" -DCMAKE_BUILD_TYPE=Release && cmake --build `"$buildDir`" --config Release -j 8"
    if ($LASTEXITCODE -ne 0) { throw "build failed (exit $LASTEXITCODE)" }
}

$needBuild = (-not $benchExe) -or ($QuantTypes.Count -gt 0 -and -not $quantizeExe)
if ($needBuild) {
    Invoke-Build
    $quantizeExe = Find-Tool 'whisper-quantize.exe'
    $benchExe    = Find-Tool 'whisper-bench.exe'
    if (-not $benchExe) { throw 'whisper-bench.exe not found after build' }
}

Push-Location $repoRoot
try {
    $convertArgs = @('-m', 'src.pt2ggml.convert', $ptPath, '-o', $binPath)
    if ($CheckOnly) { $convertArgs += '--check' }
    if ($Force)     { $convertArgs += '--force' }
    & python @convertArgs
    if ($LASTEXITCODE -ne 0) { throw 'conversion failed' }
} finally {
    Pop-Location
}

if ($CheckOnly) { return }

# -----------------------------------
# Verify by loading the model

function Test-Model {
    param([string]$Path)

    $tmp = [IO.Path]::GetTempFileName()
    try {
        cmd /c "`"$benchExe`" -m `"$Path`" > `"$tmp`" 2>&1"
        $output = Get-Content $tmp -Raw -Encoding UTF8
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }

    $failMarkers =  'wrong size in model file',
                    'wrong shape in model file',
                    'failed to load model',
                    'failed to initialize',
                    'unknown tensor'

    if ($output -match ($failMarkers -join '|')) { return $false }

    # Success means the model was actually loaded and encoded.
    return ($output -match 'model size' -and $output -match 'encode time')
}

if (-not (Test-Model $binPath)) {
    throw 'model failed to load'
}
Write-Host '[verify] load OK' -ForegroundColor Green

# -----------------------------------
# Quantize

foreach ($qtype in ($QuantTypes -join ',' -split ',')) {
    $qtype = $qtype.Trim().ToLower()
    if (-not $qtype) { continue }

    $out = Join-Path $OutDir "$Name-$qtype.bin"
    Write-Host "`n[quantize] $qtype" -ForegroundColor Cyan

    & $quantizeExe $binPath $out $qtype
    if ($LASTEXITCODE -ne 0) { throw "quantize $qtype failed" }

    Push-Location $repoRoot
    try {
        & python -m src.pt2ggml.convert $out --check
        $structOk = ($LASTEXITCODE -eq 0)
    } finally {
        Pop-Location
    }

    if (-not $structOk -or -not (Test-Model $out)) {
        Remove-Item $out -Force -ErrorAction SilentlyContinue
        throw "quantize $qtype produced a corrupt file (removed)"
    }

    $mb = [math]::Round((Get-Item $out).Length / 1MB, 2)
    Write-Host "  OK ($mb MiB)" -ForegroundColor Green
}

# -----------------------------------
# Summary

Write-Host "`n=== files ===" -ForegroundColor Green
Get-ChildItem $OutDir -Filter "$Name*.bin" |
    Sort-Object Length -Descending |
    ForEach-Object { '  {0,-40} {1,7} MiB' -f $_.Name, [math]::Round($_.Length / 1MB, 2) }
