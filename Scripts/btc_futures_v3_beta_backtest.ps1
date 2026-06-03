param(
    [string]$BeginTime = "2026-01-01",
    [string]$EndTime = "",
    [double]$InitialCash = 100000.0,
    [switch]$FrequencyFloor,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$argsList = @(
    "-m", "ML.routes.btc_futures_v3_beta_edge.backtest",
    "--begin-time", $BeginTime,
    "--initial-cash", "$InitialCash"
)
if ($OutputDir -ne "") {
    $argsList += @("--output-dir", $OutputDir)
}
if ($EndTime -ne "") {
    $argsList += @("--end-time", $EndTime)
}
if ($FrequencyFloor) {
    $argsList += @(
        "--frequency-floor",
        "--frequency-floor-side", "sell",
        "--frequency-floor-low-threshold", "0.50",
        "--frequency-floor-mid-threshold", "0.55",
        "--frequency-floor-low-exposure", "0.05",
        "--frequency-floor-mid-exposure", "0.15",
        "--frequency-floor-high-exposure", "1.0"
    )
}
python @argsList
