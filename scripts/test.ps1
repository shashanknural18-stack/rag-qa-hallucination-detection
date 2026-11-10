param (
    [switch]$All,
    [switch]$Slow
)

$env:HF_HUB_OFFLINE = "1"

if ($Slow) {
    Write-Host "Running slow end-to-end tests..."
    .\venv\Scripts\python.exe -m pytest tests/test_end_to_end.py -v -m slow
} elseif ($All) {
    Write-Host "Running all tests..."
    .\venv\Scripts\python.exe -m pytest tests/ -v -m ""
} else {
    Write-Host "Running fast unit tests (skipping slow)..."
    .\venv\Scripts\python.exe -m pytest tests/ -v
}

