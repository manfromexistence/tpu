@echo off
REM dq.cmd - shim so `dq` works from cmd.exe, PowerShell and Git Bash.
REM Args pass through as-is; quoting is the caller's job in cmd.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dq.ps1" %*
exit /b %ERRORLEVEL%
