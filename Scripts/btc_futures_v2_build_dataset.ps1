param(
    [string]$BaseDataset = "data\btc_futures_v1\btc_futures_v1_dataset.parquet",
    [string]$Source = "D:\WorkSpace\czsc_all\data\freqtrade_futures\futures\BTC_USDT_USDT-1m-futures.parquet",
    [string]$Output = "data\btc_futures_v2_stable\btc_futures_v2_stable_dataset.parquet",
    [string]$TargetMode = "atr",
    [double]$FixedPct = 0.01,
    [double]$AtrMultiplier = 1.50,
    [double]$MinPct = 0.006,
    [double]$MaxPct = 0.018,
    [int]$MaxHoldingMinutes = 1440,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$argsList = @(
    "-m", "ML.routes.btc_futures_v2_stable.dataset",
    "--base-dataset", $BaseDataset,
    "--source", $Source,
    "--output", $Output,
    "--target-mode", $TargetMode,
    "--fixed-pct", "$FixedPct",
    "--atr-multiplier", "$AtrMultiplier",
    "--min-pct", "$MinPct",
    "--max-pct", "$MaxPct",
    "--max-holding-minutes", "$MaxHoldingMinutes"
)
if ($Force) { $argsList += "--force" }
python @argsList

