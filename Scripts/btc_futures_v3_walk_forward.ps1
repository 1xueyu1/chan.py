param(
    [int]$FirstTestYear = 2023,
    [int]$LastTestYear = 0,
    [int]$ValidationYears = 2,
    [double]$MinPrecision = 0.60,
    [double]$MinProfitFactor = 1.20,
    [double]$MinSideAuc = 0.54,
    [double]$MinPoolAuc = 0.54
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$argsList = @(
    "-m", "ML.routes.btc_futures_v3_alpha.walk_forward",
    "--first-test-year", "$FirstTestYear",
    "--validation-years", "$ValidationYears",
    "--min-precision", "$MinPrecision",
    "--min-profit-factor", "$MinProfitFactor",
    "--min-side-auc", "$MinSideAuc",
    "--min-pool-auc", "$MinPoolAuc"
)
if ($LastTestYear -gt 0) {
    $argsList += @("--last-test-year", "$LastTestYear")
}
python @argsList
