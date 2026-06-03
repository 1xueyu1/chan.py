param(
    [string]$TrainStart = "2021-01-01",
    [string]$ValidStart = "2024-01-01",
    [string]$ConfirmStart = "2025-01-01",
    [string]$TestStart = "2026-01-01",
    [double]$MinFinalAuc = 0.55,
    [double]$MinPrecision = 0.55,
    [double]$MinProfitFactor = 1.05,
    [string]$ModelKind = "stable_rank"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python -m ML.routes.btc_futures_v3_beta_edge.train `
    --model-kind $ModelKind `
    --train-start $TrainStart `
    --valid-start $ValidStart `
    --confirm-start $ConfirmStart `
    --test-start $TestStart `
    --min-final-auc $MinFinalAuc `
    --min-precision $MinPrecision `
    --min-profit-factor $MinProfitFactor
