<#
.SYNOPSIS
    dqstart - push the Bonsai 2 4-way Kaggle notebook and report the four live URLs.

.DESCRIPTION
    One command that takes the local kernel at G:\Dx\tpu\llamacpp-four-model-comparison, pushes it to
    Kaggle (creating a new version and run), polls until it finishes, extracts the four
    cloudflared URLs, writes them into dq's config, and prints them.

    The four URLs are also written to G:\Dx\tpu\current-urls.json so anything else
    (dq4, or a human) can pick them up without re-running this.

.PARAMETER KernelPath
    Folder containing kernel.py / notebook.ipynb and kernel-metadata.json.

.PARAMETER wait
    Keep polling until the run reaches COMPLETE or ERROR. Without it, returns as soon
    as the push is accepted.

.EXAMPLE
    dqstart
    dqstart -wait

.EXAMPLE
    dqprobe                       # read the saved URLs and check each /health
#>
param(
    [string]$KernelPath = 'G:\Dx\tpu\llamacpp-four-model-comparison',
    [switch]$Wait
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$KAGGLE = 'G:\Ai\.venv\Scripts\kaggle.exe'
$SLUG = 'essencefromexistence/bonsai2-4way-2xt4'
$URLFILE = 'G:\Dx\tpu\current-urls.json'
$DQ = 'G:\Dx\tpu\bin'

function Get-Token {
    # The token lives in the kaggle secrets file; sourcing it keeps it out of this
    # script and out of git.
    $envFile = 'G:\Ai\.secrets\kaggle\colab-env-patch.ps1'
    if (Test-Path -LiteralPath $envFile) {
        . $envFile
    }
    if (-not $colab-env-patch:KAGGLE_API_TOKEN) {
        throw 'KAGGLE_API_TOKEN is not set and no colab-env-patch.ps1 was found to source.'
    }
    return $colab-env-patch:KAGGLE_API_TOKEN
}

function Invoke-Kaggle {
    param([string[]]$Arguments)
    $null = Get-Token
    & $KAGGLE @Arguments 2>&1
}

function Write-Log([string]$Message) {
    Write-Host $Message -ForegroundColor DarkGray
}

# --- push --------------------------------------------------------------------
Write-Log "pushing $KernelPath ..."
$push = Invoke-Kaggle @('kernels', 'push', '-p', $KernelPath)
$push | Where-Object { $_ -match 'successfully pushed|Error' }

if ($push -match 'Error') {
    throw "push failed:`n$push"
}

# A push creates a new version which runs asynchronously.
$deadline = (Get-Date).AddMinutes(75)
$lastStatus = ''
while ($Wait -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 20
    $raw = Invoke-Kaggle @('kernels', 'status', $SLUG)
    $status = ($raw -join ' ')
    if ($status -notmatch $lastStatus) {
        Write-Log "$(Get-Date -Format 'HH:mm:ss')  $status"
        $lastStatus = $status
    }
    if ($status -match 'COMPLETE') { break }
    if ($status -match 'ERROR')    { throw "kernel run failed: $status" }
}

# --- read the log and pull the four tunnel URLs ------------------------------
Write-Log "reading kernel log ..."
$raw = Invoke-Kaggle @('kernels', 'logs', $SLUG)

$rows = foreach ($line in $raw) {
    $line = $line.Trim().TrimEnd(',')
    if ($line.StartsWith('{')) {
        try { $line | ConvertFrom-Json } catch { }
    }
}

$urls = @{}
foreach ($row in $rows) {
    if ($row.PSObject.Properties.Name -contains 'data') {
        $d = [string]$row.data
        $m = [regex]::Match($d, 's(\d):\s*(https://[a-z0-9-]+\.trycloudflare\.com)')
        if ($m.Success) { $urls[$m.Groups[1].Value] = $m.Groups[2].Value }
    }
}

if ($urls.Count -lt 4) {
    Write-Warning "only found $($urls.Count) of 4 tunnel URLs in the log."
}

# --- persist and point dq at the pool ----------------------------------------
$pool = 0..3 | ForEach-Object { $urls["$_"] } | Where-Object { $_ }
$stamp = (Get-Date).ToUniversalTime().ToString('s') + 'Z'

@{ generated_utc = $stamp; kernel = $SLUG; urls = $pool } |
    ConvertTo-Json | Set-Content -LiteralPath $URLFILE

# dq reads DQ_URL; default it to slot 0 so a bare `dq` works after this script.
[Environment]::SetEnvironmentVariable('DQ_URL', $pool[0], 'User')

Write-Log "wrote $URLFILE"
Write-Host ""
Write-Host "  live endpoints (saved to current-urls.json):" -ForegroundColor Cyan
for ($i = 0; $i -lt $pool.Count; $i++) {
    Write-Host "    s$i  $($pool[$i])"
}
Write-Host ""
Write-Host "  dq is now pointed at s0. Try:" -ForegroundColor Cyan
Write-Host "    dq `"hello`""
Write-Host ""
