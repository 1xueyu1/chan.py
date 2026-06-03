param(
    [string]$Dataset = "data\btc_futures_v1\btc_futures_v1_dataset.parquet",
    [string]$ModelRoot = "result\ml\btc_futures_v1_walk_forward",
    [string]$OutputRoot = "result\btc_futures_v1_walk_forward",
    [string]$TrainStart = "2021-01-01",
    [int]$FirstTestYear = 2023,
    [int]$LastTestYear = 0,
    [string]$ModelKind = "auto",
    [double]$MinFeatureCoverage = 0.05,
    [int]$MinThresholdTrades = 40,
    [double]$MinDailyTrades = 0.20,
    [double]$InitialCash = 100000,
    [double]$StakeFraction = 1.0,
    [switch]$EnableDynamicSizing
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$argsList = @(
    "-m", "ML.routes.btc_futures_v1.walk_forward",
    "--dataset", $Dataset,
    "--model-root", $ModelRoot,
    "--output-root", $OutputRoot,
    "--train-start", $TrainStart,
    "--first-test-year", "$FirstTestYear",
    "--model-kind", $ModelKind,
    "--min-feature-coverage", "$MinFeatureCoverage",
    "--min-threshold-trades", "$MinThresholdTrades",
    "--min-daily-trades", "$MinDailyTrades",
    "--initial-cash", "$InitialCash",
    "--stake-fraction", "$StakeFraction"
)
if ($LastTestYear -gt 0) { $argsList += @("--last-test-year", "$LastTestYear") }
if ($EnableDynamicSizing) { $argsList += "--enable-dynamic-sizing" }

python @argsList

