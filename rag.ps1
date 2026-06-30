# rag.ps1 - Script to set up and run Filechatter RAG server
# Usage:
#   .\rag.ps1              # Check dependencies, create venv, install, and run
#   .\rag.ps1 -r           # Run without checking dependencies (assumes venv exists)
#   .\rag.ps1 -i           # Install dependencies only (creates venv if needed)
#   .\rag.ps1 -s           # Start venv and install dependencies (skip running the server)

param(
    [switch]$runOnly,       # -r: Just run rag_server.py without checking/installing
    [switch]$installOnly,   # -i: Install dependencies only
    [switch]$skipRun        # -s: Skip running the server after installation
)

$ErrorActionPreference = "Stop"
$ScriptDir = $PSScriptRoot
$VenvPath = Join-Path $ScriptDir "venv"
$VenvPython = Join-Path $VenvPath "Scripts" "python.exe"
$RequirementsFile = Join-Path $ScriptDir "requirements.txt"

function Write-Status($message) {
    Write-Host "`n== $message ==" -ForegroundColor Cyan
}

function Test-VenvSetup {
    return (Test-Path $VenvPath) -and (Test-Path $VenvPython)
}

function Create-Venv {
    Write-Status "Creating virtual environment..."
    python -m venv "venv"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created at: $VenvPath" -ForegroundColor Green
}

function Activate-Venv {
    Write-Status "Activating virtual environment..."
    & "$VenvPath\Scripts\Activate.ps1"
}

function Install-Dependencies {
    Write-Status "Installing dependencies..."
    
    # Ensure pip is up to date
    & $VenvPython -m pip install --upgrade pip
    
    # Install Rust if needed (for faiss-cpu)
    $rustc = Get-Command rustc -ErrorAction SilentlyContinue
    if (-not $rustc) {
        Write-Host "Rust not found. Installing Rust via rustup..." -ForegroundColor Yellow
        Write-Host "Please wait for the installation to complete." -ForegroundColor Yellow
        Invoke-Expression "winget install --id Rustlang.Rustup -e --accept-package-agreements --accept-source-agreements"
        # Refresh PATH to include rustup
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        $rustc = Get-Command rustc -ErrorAction SilentlyContinue
        if (-not $rustc) {
            Write-Host "Rust installation may not have completed. Please install Rust manually before running this script." -ForegroundColor Red
            Write-Host "You can install Rust using: winget install Rustlang.Rustup" -ForegroundColor Yellow
            exit 1
        }
        Write-Host "Rust installed successfully." -ForegroundColor Green
    } else {
        Write-Host "Rust is already installed (rustc found)." -ForegroundColor Green
    }
    
    # Install Python dependencies
    & $VenvPython -m pip install -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install dependencies." -ForegroundColor Red
        exit 1
    }
    Write-Host "Dependencies installed successfully." -ForegroundColor Green
}

function Test-Dependencies {
    Write-Status "Checking dependencies..."
    
    # Check Python version
    $pythonVersion = & python --version 2>&1
    Write-Host "System Python: $pythonVersion"
    
    # Check if venv exists
    if (-not (Test-VenvSetup)) {
        Write-Host "Virtual environment not found. It will be created automatically." -ForegroundColor Yellow
        return $false
    }
    
    # Check if key packages are installed in venv
    $requiredPackages = @("fastapi", "uvicorn", "faiss-cpu", "sqlalchemy", "langchain", "langchain-community")
    $missingPackages = @()
    
    foreach ($pkg in $requiredPackages) {
        $checkResult = & $VenvPython -m pip show $pkg 2>$null
        if (-not $checkResult) {
            $missingPackages += $pkg
        }
    }
    
    if ($missingPackages.Count -gt 0) {
        Write-Host "Missing packages in venv: $($missingPackages -join ', ')" -ForegroundColor Yellow
        Write-Host "These will be installed automatically." -ForegroundColor Yellow
        return $false
    }
    
    Write-Host "All dependencies are ready." -ForegroundColor Green
    return $true
}

# Main script execution
Write-Status "Filechatter RAG Server Setup"
Write-Host "Script directory: $ScriptDir" -ForegroundColor Gray

if ($runOnly) {
    # Just run without checking
    if (-not (Test-VenvSetup)) {
        Write-Host "Virtual environment not found at: $VenvPath" -ForegroundColor Red
        Write-Host "Please run without -r first to set up the environment, or create it manually with: python -m venv venv" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Status "Starting RAG server..."
    Activate-Venv
    & $VenvPython (Join-Path $ScriptDir "rag_server.py")
    exit $LASTEXITCODE
}

if ($installOnly) {
    # Install dependencies, create venv if needed
    if (-not (Test-VenvSetup)) {
        Create-Venv
    } else {
        Write-Host "Virtual environment already exists." -ForegroundColor Green
    }
    
    Activate-Venv
    Install-Dependencies
    Write-Status "Installation complete. To start the server, run: .\rag.ps1 -r"
    exit 0
}

# Default behavior: check, create venv if needed, install, and run
$depsReady = Test-Dependencies

if (-not $depsReady) {
    if (-not (Test-VenvSetup)) {
        Create-Venv
    }
    Install-Dependencies
}

if (-not $skipRun) {
    Write-Status "Starting RAG server..."
    Activate-Venv
    & $VenvPython (Join-Path $ScriptDir "rag_server.py")
    exit $LASTEXITCODE
}