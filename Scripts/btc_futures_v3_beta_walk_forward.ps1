param(
    [int]$FirstTestYear = 2024,
    [int]$LastTestYear = 0,
    [double]$MinFinalAuc = 0.55,
    [double]$MinPrecision = 0.55,
    [double]$MinProfitFactor = 1.05,
    [string]$ModelKind = "stable_rank"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$argsList = @(
    "-m", "ML.routes.btc_futures_v3_beta_edge.walk_forward",
    "--model-kind", "$ModelKind",
    "--first-test-year", "$FirstTestYear",
    "--min-final-auc", "$MinFinalAuc",
    "--min-precision", "$MinPrecision",
    "--min-profit-factor", "$MinProfitFactor"
)
if ($LastTestYear -gt 0) {
    $argsList += @("--last-test-year", "$LastTestYear")
}
python @argsList
