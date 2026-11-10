param (
    [switch]$Fast,
    [switch]$Errors
)

if ($Fast) {
    Write-Host "Running quick evaluation (limit 5, no LLM judge)..."
    .\venv\Scripts\python.exe run_eval.py --limit 5 --no-judge
} elseif ($Errors) {
    Write-Host "Running evaluation with error injection..."
    .\venv\Scripts\python.exe run_eval.py --inject-errors
} else {
    Write-Host "Running full evaluation suite..."
    .\venv\Scripts\python.exe run_eval.py
}

