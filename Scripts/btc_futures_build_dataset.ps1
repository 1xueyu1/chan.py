param(
    [string]$Source = "D:\WorkSpace\czsc_all\data\freqtrade_futures\futures\BTC_USDT_USDT-1m-futures.parquet",
    [string]$Output = "data\btc_futures_v1\btc_futures_v1_dataset.parquet",
    [string]$BeginTime = "2021-01-01",
    [string]$EndTime = "",
    [int]$WarmupDays = 120,
    [double]$TakeProfitPct = 0.01,
    [double]$StopLossPct = 0.01,
    [int]$MaxHoldingMinutes = 1440,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$argsList = @(
    "-m", "ML.routes.btc_futures_v1.features",
    "--source", $Source,
    "--output", $Output,
    "--begin-time", $BeginTime,
    "--warmup-days", "$WarmupDays",
    "--take-profit-pct", "$TakeProfitPct",
    "--stop-loss-pct", "$StopLossPct",
    "--max-holding-minutes", "$MaxHoldingMinutes"
)
if ($EndTime) { $argsList += @("--end-time", $EndTime) }
if ($Force) { $argsList += "--force" }

python @argsList

