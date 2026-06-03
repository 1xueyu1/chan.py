param(
    [ValidateSet("build", "train", "train-multi", "backtest", "backtest-multi", "diagnose", "all", "multi-all")]
    [string]$Action = "all",
    [string]$Trades = "result\btc_futures_v3_beta_edge\frequency_floor_2024_2025_scan\executed_trades.csv",
    [string]$TestTrades = "result\btc_futures_v3_beta_edge\frequency_floor_2day\executed_trades.csv",
    [string]$OutputDir = "result\btc_futures_v3_beta_edge\ml_strict_exit",
    [double]$ExitThreshold = 0.40,
    [double]$MinModelExitReturn = 0.0,
    [int]$SnapshotMinutes = 5,
    [string]$MultiModel = "result\ml\btc_futures_v3_beta_edge\multi_head_exit_model.pkl",
    [ValidateSet("balanced", "profit_protect")]
    [string]$MultiPolicy = "balanced",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$argsList = @(
    "-m", "ML.routes.btc_futures_v3_beta_edge.exit_management",
    "--action", $Action,
    "--trades", $Trades,
    "--test-trades", $TestTrades,
    "--output-dir", $OutputDir,
    "--multi-model", $MultiModel,
    "--exit-threshold", "$ExitThreshold",
    "--min-model-exit-return", "$MinModelExitReturn",
    "--snapshot-minutes", "$SnapshotMinutes",
    "--multi-policy", $MultiPolicy
)

if ($Force) {
    $argsList += "--force"
}

python @argsList
