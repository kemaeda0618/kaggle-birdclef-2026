# Setup local Tucker datasets — run once on the Windows PC.
# After run, the project folder syncs to Drive (via Drive Desktop) → Colab can read directly.
#
# Run: powershell -ExecutionPolicy Bypass -File setup_tucker_cache_local.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

# KGAT_ token from kaggle.json
$kj = "$env:USERPROFILE\.kaggle\kaggle.json"
if (-not (Test-Path $kj)) {
    Write-Host "ERROR: $kj not found" -ForegroundColor Red
    exit 1
}
$creds = Get-Content $kj | ConvertFrom-Json
if ($creds.key.StartsWith("KGAT_")) {
    $env:KAGGLE_API_TOKEN = $creds.key
}

# Target dir under the project root
$root = "C:\Users\maeke\work\kaggle\birdclef-2026"
$dst  = Join-Path $root "tucker_cache"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Set-Location $dst

# 1) waveform-cache
if (-not (Test-Path (Join-Path $dst "waveform_cache\audio_cache_meta.csv"))) {
    Write-Host "Downloading waveform-cache..." -ForegroundColor Cyan
    python -m kaggle datasets download -d tuckerarrants/birdclef-2026-waveform-cache --unzip
} else {
    Write-Host "waveform-cache already present, skipping" -ForegroundColor Green
}

# 2) perch-onnx
if (-not (Test-Path (Join-Path $dst "perch_v2_no_dft.onnx"))) {
    Write-Host "Downloading perch-onnx..." -ForegroundColor Cyan
    python -m kaggle datasets download -d tuckerarrants/perch-v2-no-dft-onnx --unzip
} else {
    Write-Host "perch-onnx already present, skipping" -ForegroundColor Green
}

# Summary
Write-Host "`nFinal contents of $dst :" -ForegroundColor Yellow
Get-ChildItem $dst -Recurse -File | Group-Object DirectoryName | ForEach-Object {
    $sizeGB = [math]::Round((($_.Group | Measure-Object Length -Sum).Sum / 1GB), 2)
    Write-Host "  $($_.Name): $($_.Count) files, $sizeGB GB"
}

Write-Host "`nNext: Drive Desktop will sync this folder to Drive." -ForegroundColor Magenta
Write-Host "Wait for sync to complete, then run nb_train_colab_drive.ipynb in Colab."
