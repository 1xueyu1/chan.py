param(
    [string]$Dataset = "data\btc_futures_v2_stable\btc_futures_v2_stable_dataset.parquet",
    [string]$ModelRoot = "result\ml\btc_futures_v2_stable_walk_forward",
    [string]$OutputRoot = "result\btc_futures_v2_stable_walk_forward",
    [string]$TrainStart = "2021-01-01",
    [int]$FirstTestYear = 2023,
    [int]$LastTestYear = 0,
    [string]$ModelKind = "auto",
    [double]$MinPrecision = 0.65,
    [double]$MinProfitFactor = 1.50,
    [double]$MinGateAuc = 0.55,
    [int]$TopKFeatures = 140,
    [int]$ValidationYears = 2,
    [double]$InitialCash = 100000
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$argsList = @(
    "-m", "ML.routes.btc_futures_v2_stable.walk_forward",
    "--dataset", $Dataset,
    "--model-root", $ModelRoot,
    "--output-root", $OutputRoot,
    "--train-start", $TrainStart,
    "--first-test-year", "$FirstTestYear",
    "--model-kind", $ModelKind,
    "--min-precision", "$MinPrecision",
    "--min-profit-factor", "$MinProfitFactor",
    "--min-gate-auc", "$MinGateAuc",
    "--top-k-features", "$TopKFeatures",
    "--validation-years", "$ValidationYears",
    "--initial-cash", "$InitialCash"
)
if ($LastTestYear -gt 0) { $argsList += @("--last-test-year", "$LastTestYear") }
python @argsList
