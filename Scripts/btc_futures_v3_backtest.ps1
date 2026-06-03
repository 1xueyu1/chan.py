param(
    [string]$BeginTime = "2026-01-01",
    [string]$EndTime = "",
    [double]$InitialCash = 100000.0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$argsList = @(
    "-m", "ML.routes.btc_futures_v3_alpha.backtest",
    "--begin-time", $BeginTime,
    "--initial-cash", "$InitialCash"
)
if ($EndTime -ne "") {
    $argsList += @("--end-time", $EndTime)
}
python @argsList

