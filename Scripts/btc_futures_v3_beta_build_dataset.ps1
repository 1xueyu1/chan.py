param(
    [switch]$Force,
    [string]$Delays = "0,5,15,30,60"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$argsList = @("-m", "ML.routes.btc_futures_v3_beta_edge.dataset", "--delays", $Delays)
if ($Force) {
    $argsList += "--force"
}
python @argsList

