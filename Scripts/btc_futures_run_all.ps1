param(
    [string]$Source = "D:\WorkSpace\czsc_all\data\freqtrade_futures\futures\BTC_USDT_USDT-1m-futures.parquet",
    [string]$BeginTime = "2021-01-01",
    [string]$EndTime = "",
    [string]$ValidStart = "2025-01-01",
    [string]$TestStart = "2026-01-01",
    [switch]$ForceDataset
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$buildArgs = @{
    Source = $Source
    BeginTime = $BeginTime
}
if ($EndTime) { $buildArgs.EndTime = $EndTime }
if ($ForceDataset) { $buildArgs.Force = $true }
& "$PSScriptRoot\btc_futures_build_dataset.ps1" @buildArgs

& "$PSScriptRoot\btc_futures_train.ps1" -TrainStart $BeginTime -ValidStart $ValidStart -TestStart $TestStart

$backtestArgs = @{
    BeginTime = $TestStart
}
if ($EndTime) { $backtestArgs.EndTime = $EndTime }
& "$PSScriptRoot\btc_futures_backtest.ps1" @backtestArgs

