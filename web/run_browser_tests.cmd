@echo off
setlocal

set MODE=tests
if not "%~1"=="" set MODE=%~1

set PORT=8011
if not "%~2"=="" set PORT=%~2

set EXTRA=
set PASSTHRU=
if /I "%~3"=="headed" (
  set EXTRA=--headed-only
  set PASSTHRU=%~4 %~5 %~6 %~7 %~8 %~9
) else (
  set PASSTHRU=%~3 %~4 %~5 %~6 %~7 %~8 %~9
)

python "%~dp0run_browser_tests.py" --mode "%MODE%" --port %PORT% %EXTRA% %PASSTHRU%
set EXITCODE=%ERRORLEVEL%
exit /b %EXITCODE%
