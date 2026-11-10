# Setup virtual environment and dependencies
Write-Host "Setting up Python virtual environment..."
if (!(Test-Path "venv")) {
    python -m venv venv
}
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Setup complete!"

