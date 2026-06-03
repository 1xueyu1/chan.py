param(
    [string]$Dataset = "data\btc_futures_v2_stable\btc_futures_v2_stable_dataset.parquet",
    [string]$ModelDir = "result\ml\btc_futures_v2_stable",
    [string]$TrainStart = "2021-01-01",
    [string]$ValidStart = "2025-01-01",
    [string]$TestStart = "2026-01-01",
    [string]$TestEnd = "",
    [string]$ModelKind = "auto",
    [double]$MinPrecision = 0.65,
    [double]$MinProfitFactor = 1.50,
    [double]$MinGateAuc = 0.55,
    [int]$TopKFeatures = 140
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$argsList = @(
    "-m", "ML.routes.btc_futures_v2_stable.train",
    "--dataset", $Dataset,
    "--model-dir", $ModelDir,
    "--train-start", $TrainStart,
    "--valid-start", $ValidStart,
    "--test-start", $TestStart,
    "--model-kind", $ModelKind,
    "--min-precision", "$MinPrecision",
    "--min-profit-factor", "$MinProfitFactor",
    "--min-gate-auc", "$MinGateAuc",
    "--top-k-features", "$TopKFeatures"
)
if ($TestEnd) { $argsList += @("--test-end", $TestEnd) }
python @argsList
