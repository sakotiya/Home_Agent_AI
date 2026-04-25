# Run this from the project root to start the FastAPI backend on Windows.
Set-Location "$PSScriptRoot"
if (!(Test-Path "backend\.venv")) {
  py -3.12 -m venv backend\.venv
  backend\.venv\Scripts\python.exe -m pip install --upgrade pip
  backend\.venv\Scripts\python.exe -m pip install --no-cache-dir -r backend\requirements.txt
}
Set-Location "$PSScriptRoot\backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
