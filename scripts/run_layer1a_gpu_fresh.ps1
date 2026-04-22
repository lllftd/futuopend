# Train L1a from scratch on CUDA with AMP, PyTorch intra-op threads, and DataLoader workers.
# Usage (from repo root):
#   .\scripts\run_layer1a_gpu_fresh.ps1
#   .\scripts\run_layer1a_gpu_fresh.ps1 -UsePreparedCache
#   .\scripts\run_layer1a_gpu_fresh.ps1 -RebuildPrep
# Notes:
#   - "Fresh" removes L1a weights/meta/output cache only (not TCN or prepared dataset unless -RebuildPrep).
#   - DataLoader workers on Windows use child processes; they feed the GPU while the main process trains.
#   - Tune L1A_DATALOADER_WORKERS / TORCH_CPU_THREADS if CPU is oversubscribed.
#   - Fast preset (env-only): $env:L1A_FAST = "1" before launch sets L1A_EXPAND_OOF_VAL_WINDOWS (2 L1a-only
#     folds), L1A_SKIP_CAL_FULL_METRICS, L1A_OOF_WARMSTART, L1A_MATERIALIZE_BATCH_SIZE. Under default
#     L1_OOF_MODE=expanding, L1_OOF_FOLDS is ignored; use L1A_EXPAND_OOF_VAL_WINDOWS or L1_OOF_MODE=blocked.

[CmdletBinding()]
param(
    [switch] $UsePreparedCache,
    [switch] $RebuildPrep,
    [switch] $SkipRemoveL1a,
    [switch] $Fast
)

$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path } else { Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
Set-Location -LiteralPath $Root

$env:PYTHONPATH = $Root
$env:TORCH_DEVICE = if ($env:TORCH_DEVICE) { $env:TORCH_DEVICE } else { "cuda" }
$env:L1A_AMP = "1"
$env:L1A_AMP_DTYPE = "bf16"
$env:TORCH_ALLOW_TF32 = "1"

$logical = 8
try {
    $cs = Get-CimInstance Win32_ComputerSystem
    if ($null -ne $cs) {
        $logical = [int] (@($cs)[0].NumberOfLogicalProcessors)
    }
} catch { }

$torchThreads = [Math]::Max(4, [Math]::Min(16, [int] ([Math]::Ceiling($logical / 2))))
$env:TORCH_CPU_THREADS = "$torchThreads"
$workers = [Math]::Max(2, [Math]::Min(8, [int] ([Math]::Max(1, $logical - 2))))
$env:L1A_DATALOADER_WORKERS = "$workers"
$env:L1A_PREFETCH_FACTOR = "4"

if ($Fast -or ($env:L1A_FAST -eq "1")) {
    if ($Fast) { $env:L1A_FAST = "1" }
    if (-not $env:L1A_EXPAND_OOF_VAL_WINDOWS) { $env:L1A_EXPAND_OOF_VAL_WINDOWS = "2022-07-01:2024-01-01,2024-01-01:2024-07-01" }
    if (-not $env:L1A_SKIP_CAL_FULL_METRICS) { $env:L1A_SKIP_CAL_FULL_METRICS = "1" }
    if (-not $env:L1A_OOF_WARMSTART) { $env:L1A_OOF_WARMSTART = "1" }
    if (-not $env:L1A_MATERIALIZE_BATCH_SIZE) { $env:L1A_MATERIALIZE_BATCH_SIZE = "2048" }
}

if ($UsePreparedCache) {
    $env:LAYER1A_USE_PREPARED_CACHE = "1"
} else {
    Remove-Item Env:LAYER1A_USE_PREPARED_CACHE -ErrorAction SilentlyContinue
}

if ($RebuildPrep) {
    $env:PREPARED_DATASET_CACHE_REBUILD = "1"
} else {
    Remove-Item Env:PREPARED_DATASET_CACHE_REBUILD -ErrorAction SilentlyContinue
}

$modelDir = Join-Path $Root "lgbm_models"
if (-not $SkipRemoveL1a) {
    foreach ($name in @("l1a_market_tcn.pt", "l1a_market_tcn_meta.pkl", "l1a_outputs.pkl")) {
        $p = Join-Path $modelDir $name
        if (Test-Path $p) {
            Remove-Item -Force $p
            Write-Host "Removed $p"
        }
    }
}

$py = $null
$candidates = @(
    (Join-Path -Path $Root -ChildPath "venv\Scripts\python.exe"),
    (Join-Path -Path $Root -ChildPath ".venv\Scripts\python.exe"),
    (Join-Path -Path $Root -ChildPath "quickvenv\Scripts\python.exe"),
    (Join-Path -Path $Root -ChildPath "testenv\Scripts\python.exe")
)
foreach ($c in $candidates) {
    if (Test-Path -LiteralPath $c) { $py = $c; break }
}
if (-not $py) { $py = "python" }

Write-Host "Using: $py"
Write-Host "TORCH_DEVICE=$env:TORCH_DEVICE L1A_AMP=$env:L1A_AMP L1A_FAST=$($env:L1A_FAST) TORCH_CPU_THREADS=$env:TORCH_CPU_THREADS L1A_DATALOADER_WORKERS=$env:L1A_DATALOADER_WORKERS"
& $py -u (Join-Path -Path $Root -ChildPath "backtests\train_layer1a_only.py")
