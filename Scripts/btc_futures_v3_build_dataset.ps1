param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$argsList = @("-m", "ML.routes.btc_futures_v3_alpha.dataset")
if ($Force) {
    $argsList += "--force"
}
python @argsList

