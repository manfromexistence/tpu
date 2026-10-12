@echo off
REM dq.cmd - shim so `dq` works from cmd.exe, PowerShell and Git Bash.
REM
REM pwsh (7+) is preferred over powershell (5.1): the 5.1 host binds -File arguments
REM more strictly and can reject a multi-word prompt before the script's own code runs.
REM
REM No parentheses and no if-block on purpose. A parenthesised conditional around the
REM invocation mangles %* in cmd.exe, which is what made the previous version of this
REM shim pass the wrong arguments entirely. Two separate invocations instead.
setlocal
set "DQ_SCRIPT=%~dp0dq.ps1"
where pwsh >nul 2>nul
if not errorlevel 1 goto useposh7

powershell -NoProfile -ExecutionPolicy Bypass -File "%DQ_SCRIPT%" %*
goto done

:useposh7
pwsh -NoProfile -ExecutionPolicy Bypass -File "%DQ_SCRIPT%" %*

:done
endlocal
exit /b %errorlevel%
