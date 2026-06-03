param(
    [string]$TrainStart = "2021-01-01",
    [string]$ValidStart = "2024-01-01",
    [string]$TestStart = "2026-01-01",
    [string]$TestEnd = "",
    [double]$MinPrecision = 0.60,
    [double]$MinProfitFactor = 1.20,
    [double]$MinSideAuc = 0.54,
    [double]$MinPoolAuc = 0.54
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$argsList = @(
    "-m", "ML.routes.btc_futures_v3_alpha.train",
    "--train-start", $TrainStart,
    "--valid-start", $ValidStart,
    "--test-start", $TestStart,
    "--min-precision", "$MinPrecision",
    "--min-profit-factor", "$MinProfitFactor",
    "--min-side-auc", "$MinSideAuc",
    "--min-pool-auc", "$MinPoolAuc"
)
if ($TestEnd -ne "") {
    $argsList += @("--test-end", $TestEnd)
}
python @argsList
