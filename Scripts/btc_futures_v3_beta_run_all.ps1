param(
    [switch]$ForceDataset
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$datasetArgs = @()
if ($ForceDataset) {
    $datasetArgs += "-Force"
}
& "$PSScriptRoot\btc_futures_v3_beta_build_dataset.ps1" @datasetArgs
& "$PSScriptRoot\btc_futures_v3_beta_diagnostics.ps1"
& "$PSScriptRoot\btc_futures_v3_beta_train.ps1"
& "$PSScriptRoot\btc_futures_v3_beta_backtest.ps1"

