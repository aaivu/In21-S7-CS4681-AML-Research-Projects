# Speech Enhancement Pipeline - Run Script
# This script runs the SepFormer separation + denoising, followed by evaluation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Speech Enhancement Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python first." -ForegroundColor Red
    exit 1
}

# Step 0: Install Dependencies
Write-Host ""
Write-Host "Step 0: Installing/Updating Dependencies..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow

# Check if requirements.txt exists
$requirementsPath = "..\requirements.txt"
if (Test-Path $requirementsPath) {
    Write-Host "Installing packages from requirements.txt..." -ForegroundColor Cyan
    python -m pip install -r $requirementsPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Package installation failed. Please check requirements.txt" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    
    Write-Host "✓ Dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠ requirements.txt not found at $requirementsPath" -ForegroundColor Yellow
    Write-Host "Installing essential packages manually..." -ForegroundColor Cyan
    
    python -m pip install speechbrain torch torchaudio soundfile pystoi pesq mir-eval librosa scipy
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Package installation failed" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    
    Write-Host "✓ Essential packages installed!" -ForegroundColor Green
}

# Step 1: Run SepFormer Separation + Denoising
Write-Host ""
Write-Host "Step 1: Running SepFormer Separation + Denoising..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow

python sepformer_denoise.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Separation/Denoising failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "✓ Separation and denoising completed successfully!" -ForegroundColor Green

# Step 2: Evaluate Outputs
Write-Host ""
Write-Host "Step 2: Evaluating Outputs..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow

python evaluate_outputs.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Evaluation failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ Pipeline completed successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output files saved in: ../src/output/" -ForegroundColor Cyan
Write-Host "  - sep_s1.wav, den_s1.wav" -ForegroundColor Gray
Write-Host "  - sep_s2.wav, den_s2.wav" -ForegroundColor Gray
Write-Host ""
