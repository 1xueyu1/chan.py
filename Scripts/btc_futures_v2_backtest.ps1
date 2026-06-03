param(
    [string]$Dataset = "data\btc_futures_v2_stable\btc_futures_v2_stable_dataset.parquet",
    [string]$ModelDir = "result\ml\btc_futures_v2_stable",
    [string]$OutputDir = "result\btc_futures_v2_stable",
    [string]$BeginTime = "2026-01-01",
    [string]$EndTime = "",
    [double]$InitialCash = 100000
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$argsList = @(
    "-m", "ML.routes.btc_futures_v2_stable.backtest",
    "--dataset", $Dataset,
    "--model-dir", $ModelDir,
    "--output-dir", $OutputDir,
    "--begin-time", $BeginTime,
    "--initial-cash", "$InitialCash"
)
if ($EndTime) { $argsList += @("--end-time", $EndTime) }
python @argsList

