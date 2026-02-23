# =============================================================================
# INSTALL DEPENDENCIES — Green Branding Analysis
# Run this script in PowerShell (Windows) before executing the Python script.
#
# HOW TO RUN:
#   1. Open PowerShell as Administrator  (or regular user if pip is in PATH)
#   2. Navigate to this folder:
#        cd "C:\path\to\your\project"
#   3. If you get an execution policy error, run this first:
#        Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
#   4. Then run:
#        .\install_dependencies.ps1
# =============================================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Green Branding Analysis — Dependency Installer" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Check Python is available ────────────────────────────────────────
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found. Please install Python 3.9+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# ── Step 2: Upgrade pip ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/4] Upgrading pip to latest version..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host "  pip upgraded." -ForegroundColor Green

# ── Step 3: Install all required packages ────────────────────────────────────
Write-Host ""
Write-Host "[3/4] Installing required packages..." -ForegroundColor Yellow
Write-Host "  This may take 1-3 minutes depending on your internet speed."
Write-Host ""

$packages = @(
    "pandas",           # Data manipulation and Excel reading
    "numpy",            # Numerical computing
    "statsmodels",      # OLS regression, VIF, Breusch-Pagan, Logistic regression
    "seaborn",          # Statistical heatmap visualisation
    "matplotlib",       # Plotting engine
    "scipy",            # Pearson correlation, Shapiro-Wilk, KS test
    "openpyxl"          # Excel (.xlsx) file reader for pandas
)

foreach ($pkg in $packages) {
    Write-Host "  Installing $pkg ..." -ForegroundColor White -NoNewline
    python -m pip install $pkg --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host " Done" -ForegroundColor Green
    } else {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "  Try manually: python -m pip install $pkg" -ForegroundColor Yellow
    }
}

# ── Step 4: Verify all imports work ───────────────────────────────────────────
Write-Host ""
Write-Host "[4/4] Verifying all imports..." -ForegroundColor Yellow

$verifyScript = @"
import pandas, numpy, statsmodels, seaborn, matplotlib, scipy, openpyxl
print('  pandas      version:', pandas.__version__)
print('  numpy       version:', numpy.__version__)
print('  statsmodels version:', statsmodels.__version__)
print('  seaborn     version:', seaborn.__version__)
print('  matplotlib  version:', matplotlib.__version__)
print('  scipy       version:', scipy.__version__)
print('  openpyxl    version:', openpyxl.__version__)
"@

python -c $verifyScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  All dependencies installed successfully!" -ForegroundColor Green
    Write-Host "  You can now run the analysis:" -ForegroundColor Green
    Write-Host "    python green_branding_analysis.py" -ForegroundColor White
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "  One or more imports failed. Review the errors above." -ForegroundColor Red
    Write-Host "  Try running: python -m pip install <package_name>" -ForegroundColor Yellow
}
