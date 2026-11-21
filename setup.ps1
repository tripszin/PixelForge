$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    python -m venv .venv
}
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

$CudaAvailable = $false
try {
    $CudaAvailable = (& $PythonExe -c "import torch; print(torch.cuda.is_available())" 2>$null | Select-Object -Last 1) -eq 'True'
} catch { }
if (-not $CudaAvailable) {
    & $PythonExe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
}
& $PythonExe -m pip install -r requirements.txt

$HfExe = Join-Path $ProjectRoot '.venv\Scripts\hf.exe'
if (-not (Test-Path 'models\pixie-32\model.safetensors')) {
    & $HfExe download EmilioSchwaiger/pixie-32 --local-dir models\pixie-32
}
if (-not (Test-Path 'models\clip-vit-large-patch14\model.safetensors')) {
    & $HfExe download openai/clip-vit-large-patch14 model.safetensors tokenizer.json --local-dir models\clip-vit-large-patch14
}
Write-Host 'Pronto. Execute .\start.ps1 para abrir o PixelForge.'
