# Full stack L1a→L1b→L2→L3 starting from PA feature build (invalidate PA disk cache, then train_pipeline layer1).
# Does not delete TCN/Mamba checkpoints under lgbm_models/ (needed when prep pulls TCN/Mamba derivatives).
#
# Usage (repo root):
#   .\scripts\run_full_pipeline_from_pa_gpu.ps1
#   .\scripts\run_full_pipeline_from_pa_gpu.ps1 -KeepPaCache      # still full prep/Torch path, but reuse data\.pa_feature_cache\*.pkl if valid
#   .\scripts\run_full_pipeline_from_pa_gpu.ps1 -KeepStackModels # reuse L1–L3 weights (usually wrong if PA/prep changed)
#   .\scripts\run_full_pipeline_from_pa_gpu.ps1 -DataPrepareWorkers 2   # parallel QQQ+SPY PA (needs RAM; default 1 avoids BrokenProcessPool)
#   .\scripts\run_full_pipeline_from_pa_gpu.ps1 -SkipThreadCap        # do not cap OMP/Numba (faster but can crash on huge PA)
#   .\scripts\run_full_pipeline_from_pa_gpu.ps1 -UsePaNumba           # re-enable Numba JIT for PA (default off: PA_NUMBA=0, slower but stabler)

[CmdletBinding()]
param(
    [switch] $KeepPaCache,
    [switch] $KeepPreparedCache,
    [switch] $KeepStackModels,
    [int] $DataPrepareWorkers = 1,
    [switch] $SkipThreadCap,
    [switch] $UsePaNumba
)

$ErrorActionPreference = "Stop"
$Root = if ($PSScriptRoot) { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path } else { Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
Set-Location -LiteralPath $Root

$logical = 8
try {
    $cs = Get-CimInstance Win32_ComputerSystem
    if ($null -ne $cs) {
        $logical = [int] (@($cs)[0].NumberOfLogicalProcessors)
    }
} catch { }

# Cap BLAS/OpenMP/Numba during PA+prep. Uncapped stacks (e.g. PyTorch 16 threads + MKL + Numba) can spike RAM or die with no Python traceback.
if (-not $SkipThreadCap) {
    $ompCap = [Math]::Max(2, [Math]::Min(8, [int]([Math]::Ceiling($logical / 2))))
    $env:OMP_NUM_THREADS = "$ompCap"
    $env:MKL_NUM_THREADS = "$ompCap"
    $env:OPENBLAS_NUM_THREADS = "$ompCap"
    $env:VECLIB_MAXIMUM_THREADS = "$ompCap"
    $env:NUMEXPR_NUM_THREADS = "$ompCap"
    $env:NUMBA_NUM_THREADS = "$ompCap"
}

$env:PYTHONPATH = $Root
if ($UsePaNumba) {
    Remove-Item Env:PA_NUMBA -ErrorAction SilentlyContinue
} else {
    $env:PA_NUMBA = "0"
}
$env:FORCE_TQDM = "1"
$env:TORCH_DEVICE = "cuda"
$env:L1A_AMP = "1"
$env:L1A_AMP_DTYPE = "bf16"
$env:TCN_AMP = "1"
$env:TCN_TRAIN_BATCH_SIZE = "8192"
$env:TCN_BATCH_SIZE = "8192"
$env:TORCH_ALLOW_TF32 = "1"
$env:DATA_PREPARE_WORKERS = "$([Math]::Max(1, $DataPrepareWorkers))"
Remove-Item Env:LAYER1A_USE_PREPARED_CACHE -ErrorAction SilentlyContinue
Remove-Item Env:PREPARED_DATASET_CACHE_REBUILD -ErrorAction SilentlyContinue

$torchThreads = [Math]::Max(4, [Math]::Min(16, [int] ([Math]::Ceiling($logical / 2))))
if (-not $SkipThreadCap) {
    $torchThreads = [Math]::Min($torchThreads, [int]$env:OMP_NUM_THREADS)
}
$env:TORCH_CPU_THREADS = "$torchThreads"
$workers = [Math]::Max(2, [Math]::Min(8, [int] ([Math]::Max(1, $logical - 2))))
if (-not $SkipThreadCap) {
    $workers = [Math]::Min($workers, [int]$env:OMP_NUM_THREADS)
}
$env:L1A_DATALOADER_WORKERS = "$workers"
$env:L1A_PREFETCH_FACTOR = "4"
$prepCpu = [Math]::Max(4, [Math]::Min(16, $logical - 2))
if (-not $SkipThreadCap) {
    $prepCpu = [Math]::Min($prepCpu, [int]$env:OMP_NUM_THREADS)
}
$env:PREP_CPU_WORKERS = "$prepCpu"

$dataDir = Join-Path $Root "data"
$paCacheDir = Join-Path $dataDir ".pa_feature_cache"
$modelDir = Join-Path $Root "lgbm_models"

if (-not $KeepPaCache -and (Test-Path $paCacheDir)) {
    Get-ChildItem -Path $paCacheDir -Filter "*.pkl" -File -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item -Force $_.FullName
        Write-Host "Removed PA cache: $($_.FullName)"
    }
}

$prepared = Join-Path $modelDir "prepared_lgbm_dataset.pkl"
if (-not $KeepPreparedCache -and (Test-Path $prepared)) {
    Remove-Item -Force $prepared
    Write-Host "Removed prepared dataset cache: $prepared"
}

if (-not $KeepStackModels) {
    $stackFiles = @(
        "l1a_market_tcn.pt",
        "l1a_market_tcn_meta.pkl",
        "l1a_outputs.pkl",
        "l1b_descriptor_meta.pkl",
        "l1b_outputs.pkl",
        "l1b_edge_pred.txt",
        "l1b_dq_pred.txt",
        "l2_trade_gate.txt",
        "l2_direction.txt",
        "l2_trade_gate_calibrator.pkl",
        "l2_direction_calibrator.pkl",
        "l2_mfe.txt",
        "l2_mae.txt",
        "l2_decision_meta.pkl",
        "l2_outputs.pkl",
        "l3_exit.txt",
        "l3_value.txt",
        "l3_exit_meta.pkl",
        "l3_trajectory_encoder.pt",
        "l3_policy_dataset.pkl",
        "l3_cox_time_varying.pkl"
    )
    foreach ($name in $stackFiles) {
        $p = Join-Path $modelDir $name
        if (Test-Path $p) {
            Remove-Item -Force $p
            Write-Host "Removed stack artifact: $p"
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

Write-Host "Pipeline symbols: QQQ, SPY (data\SYMBOL.csv + data\SYMBOL_labeled_v2.csv)"
$capNote = if ($SkipThreadCap) { "thread-cap=off" } else { "OMP cap=$($env:OMP_NUM_THREADS)" }
$paJit = if ($UsePaNumba) { "PA_NUMBA=on" } else { "PA_NUMBA=0" }
Write-Host "Using: $py | TORCH_DEVICE=$env:TORCH_DEVICE DATA_PREPARE_WORKERS=$env:DATA_PREPARE_WORKERS $paJit L1A_AMP=$env:L1A_AMP TORCH_CPU_THREADS=$env:TORCH_CPU_THREADS L1A_DATALOADER_WORKERS=$env:L1A_DATALOADER_WORKERS ($capNote)"
& $py -u -m backtests.train_pipeline --start-from layer1
