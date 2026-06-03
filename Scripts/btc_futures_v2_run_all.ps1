param(
    [switch]$ForceDataset,
    [int]$FirstTestYear = 2023
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$buildArgs = @{}
if ($ForceDataset) { $buildArgs.Force = $true }
& "$PSScriptRoot\btc_futures_v2_build_dataset.ps1" @buildArgs
& "$PSScriptRoot\btc_futures_v2_train.ps1"
& "$PSScriptRoot\btc_futures_v2_backtest.ps1"
& "$PSScriptRoot\btc_futures_v2_walk_forward.ps1" -FirstTestYear $FirstTestYear

