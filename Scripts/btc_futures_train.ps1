param(
    [string]$Dataset = "data\btc_futures_v1\btc_futures_v1_dataset.parquet",
    [string]$ModelDir = "result\ml\btc_futures_v1",
    [string]$TrainStart = "2021-01-01",
    [string]$ValidStart = "2025-01-01",
    [string]$TestStart = "2026-01-01",
    [string]$ModelKind = "auto",
    [double]$MinFeatureCoverage = 0.05,
    [int]$MinThresholdTrades = 40,
    [double]$MinDailyTrades = 0.20
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

python -m ML.routes.btc_futures_v1.train `
    --dataset $Dataset `
    --model-dir $ModelDir `
    --train-start $TrainStart `
    --valid-start $ValidStart `
    --test-start $TestStart `
    --model-kind $ModelKind `
    --min-feature-coverage $MinFeatureCoverage `
    --min-threshold-trades $MinThresholdTrades `
    --min-daily-trades $MinDailyTrades

