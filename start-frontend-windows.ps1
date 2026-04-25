# Run this from the project root to start the Vite frontend on Windows.
Set-Location "$PSScriptRoot\frontend"
npm.cmd install --registry=https://registry.npmjs.org/
npm.cmd run dev
