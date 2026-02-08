@echo off
setlocal

REM Serve the project root (so /src and /test are reachable) and open /web/.
REM Requires Python on PATH.

cd /d "%~dp0"

set PORT=8000
if not "%~1"=="" set PORT=%~1

echo Starting webgpufft browser harness on http://localhost:%PORT%/web/
echo NOTE: this is a persistent server and will run until Ctrl+C.
echo One-shot automated run: web\run_browser_tests.cmd tests
python "%~dp0serve.py" %PORT%
