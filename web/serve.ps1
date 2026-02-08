$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$port = if ($args.Length -ge 1) { $args[0] } else { "8000" }

Write-Host "Starting server from project root via web/serve.py (port $port)..."
Write-Host "NOTE: this is a persistent server and will run until Ctrl+C."
Write-Host "One-shot automated run: web\run_browser_tests.cmd tests"
python (Join-Path $here "serve.py") $port
