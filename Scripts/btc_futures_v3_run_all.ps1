param(
    [switch]$ForceDataset,
    [string]$TrainStart = "2021-01-01",
    [string]$ValidStart = "2024-01-01",
    [string]$TestStart = "2026-01-01",
    [string]$BacktestBegin = "2026-01-01"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$datasetArgs = @()
if ($ForceDataset) {
    $datasetArgs += "-Force"
}
& "$PSScriptRoot\btc_futures_v3_build_dataset.ps1" @datasetArgs
& "$PSScriptRoot\btc_futures_v3_diagnostics.ps1"
& "$PSScriptRoot\btc_futures_v3_train.ps1" -TrainStart $TrainStart -ValidStart $ValidStart -TestStart $TestStart
& "$PSScriptRoot\btc_futures_v3_backtest.ps1" -BeginTime $BacktestBegin
