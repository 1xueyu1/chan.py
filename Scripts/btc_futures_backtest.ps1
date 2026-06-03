param(
    [string]$Dataset = "data\btc_futures_v1\btc_futures_v1_dataset.parquet",
    [string]$ModelDir = "result\ml\btc_futures_v1",
    [string]$OutputDir = "result\btc_futures_v1",
    [string]$BeginTime = "2026-01-01",
    [string]$EndTime = "",
    [double]$InitialCash = 100000,
    [double]$StakeFraction = 1.0,
    [switch]$EnableDynamicSizing
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$argsList = @(
    "-m", "ML.routes.btc_futures_v1.backtest",
    "--dataset", $Dataset,
    "--model-dir", $ModelDir,
    "--output-dir", $OutputDir,
    "--begin-time", $BeginTime,
    "--initial-cash", "$InitialCash",
    "--stake-fraction", "$StakeFraction"
)
if ($EndTime) { $argsList += @("--end-time", $EndTime) }
if ($EnableDynamicSizing) { $argsList += "--enable-dynamic-sizing" }

python @argsList
