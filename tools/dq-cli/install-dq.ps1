<#
.SYNOPSIS
    Install (or uninstall) the dq CLI onto this machine's PATH.

.DESCRIPTION
    Copies dq.ps1 + dq.cmd into a bin directory and adds that directory to the
    User PATH if it is not already there, so `dq` works from any new shell.

    The User PATH is edited via the registry-backed [Environment] API, not by setting
    $env:Path in this process, because setting the process variable would vanish when
    the shell exits and the CLI would only work in sessions started from here.

.PARAMETER Target
    Where to install. Default: the CLI's own ../bin under G:\Dx\tpu.

.PARAMETER Uninstall
    Remove the shims and restore PATH.

.EXAMPLE
    pwsh install-dq.ps1
    dq "Hello, who are you?"
#>
[CmdletBinding()]
param(
    [string]$Target = 'G:\Dx\tpu\bin',
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Files = @('dq.ps1', 'dq.cmd')

function Get-UserPath {
    $p = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($null -eq $p) { return '' }
    return $p
}

function Set-UserPath([string]$Value) {
    [Environment]::SetEnvironmentVariable('Path', $Value, 'User')
}

if ($Uninstall) {
    foreach ($f in $Files) {
        $p = Join-Path $Target $f
        if (Test-Path -LiteralPath $p) {
            Remove-Item -LiteralPath $p -Force
            Write-Host "removed $p"
        }
    }
    $current = Get-UserPath
    if ($current -and (($current -split ';') -contains $Target)) {
        $new = (($current -split ';') | Where-Object { $_ -and $_ -ne $Target }) -join ';'
        Set-UserPath $new
        Write-Host "removed $Target from User PATH"
    }
    Write-Host 'uninstalled'
    exit 0
}

# --- install ------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $Target)) {
    New-Item -ItemType Directory -Path $Target -Force | Out-Null
    Write-Host "created $Target"
}

foreach ($f in $Files) {
    $src = Join-Path $Here $f
    if (-not (Test-Path -LiteralPath $src)) {
        throw "missing $src -- run this from the folder containing $f"
    }
    Copy-Item -LiteralPath $src -Destination (Join-Path $Target $f) -Force
    Write-Host "installed $(Join-Path $Target $f)"
}

$current = Get-UserPath
$entries = @()
if ($current) { $entries = $current -split ';' | Where-Object { $_ } }

if ($entries -notcontains $Target) {
    $new = if ($current) { "$current;$Target" } else { $Target }
    Set-UserPath $new
    Write-Host "added $Target to User PATH (applies to NEW shells)"
} else {
    Write-Host "$Target already in User PATH"
}

Write-Host ''
Write-Host 'Verify in a new shell:' -ForegroundColor Cyan
Write-Host '  dq --version'
Write-Host '  dq "Hello, who are you?"'
Write-Host '  dq                      # interactive'
