# Run this from the project root if npm install fails on Windows.
Write-Host "Stopping possible Node/Vite processes..."
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Set-Location "$PSScriptRoot\frontend"
Write-Host "Using public npm registry for this project..."
npm.cmd config set registry https://registry.npmjs.org/ --location=project
Set-Content -Path .npmrc -Value "registry=https://registry.npmjs.org/`nstrict-ssl=true`nfund=false`naudit=false"
Write-Host "Removing broken frontend install artifacts..."
if (Test-Path package-lock.json) { Remove-Item -Force package-lock.json -ErrorAction SilentlyContinue }
if (Test-Path node_modules) { cmd /c rmdir /s /q node_modules }
Write-Host "Cleaning npm cache..."
npm.cmd cache clean --force
Write-Host "Installing frontend packages..."
npm.cmd install --registry=https://registry.npmjs.org/ --no-audit --no-fund
Write-Host "Starting frontend..."
npm.cmd run dev
