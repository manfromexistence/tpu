<#
.SYNOPSIS
    dq4 - round-robin chat CLI across the four Bonsai 2 instances on 2x T4.

.DESCRIPTION
    Thin wrapper over dq that load-balances across the four tunnel endpoints, so a
    single `dq4 "question"` hits a different instance each call. When an instance is
    dead (expired tunnel, dead session) it is skipped and the next is tried, so one
    dead slot does not fail the command.

.PARAMETER Message
    The prompt. If omitted, drops into an interactive REPL.

.PARAMETER Pool
    Comma-separated base URLs. Defaults to the four live tunnels.

.EXAMPLE
    dq4 "who are you?"

.EXAMPLE
    dq4 -Pool "https://a.trycloudflare.com,https://b.trycloudflare.com"
#>
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Message,
    [string]$Pool
)

$ErrorActionPreference = 'Stop'

$DEFAULT_POOL = @(
    'https://steel-dam-journalist-prescribed.trycloudflare.com',
    'https://plugins-motors-girls-elections.trycloudflare.com',
    'https://mathematical-expensive-capitol-survivors.trycloudflare.com',
    'https://fixtures-assume-else-considered.trycloudflare.com'
)

$pool = if ($Pool) { $Pool -split ',' | ForEach-Object { $_.Trim() } } else { $DEFAULT_POOL }
$cursorFile = Join-Path $colab-env-patch:TEMP 'dq4.cursor'
$cursor = 0
if (Test-Path -LiteralPath $cursorFile) { [int]$cursor = Get-Content -LiteralPath $cursorFile }

function Next-Endpoint {
    $script:cursor = ($script:cursor + 1) % $pool.Count
    $pool[$script:cursor]
}

function Show-Help {
    Write-Host @'
dq4 -- four-way load-balanced Bonsai 2 chat CLI

  dq4 <message>     one-shot, round-robin across the pool
  dq4               interactive REPL
  /pool             show the pool and current cursor
  /health           probe every endpoint
  /quit             exit
'@ -ForegroundColor DarkGray
}

function Show-Pool {
    Write-Host "pool ($($pool.Count) endpoints, cursor $cursor):"
    for ($i = 0; $i -lt $pool.Count; $i++) {
        $mark = if ($i -eq $cursor) { ' ->' } else { '   ' }
        Write-Host "$mark [$i] $($pool[$i])"
    }
}

function Show-Health {
    foreach ($u in $pool) {
        try {
            $null = Invoke-WebRequest -Uri "$u/health" -TimeoutSec 12 -UseBasicParsing
            Write-Host "  UP   $u" -ForegroundColor Green
        } catch {
            Write-Host "  DOWN $u" -ForegroundColor DarkYellow
        }
    }
}

if (-not $Message) {
    Write-Host ''
    Write-Host '  dq4' -ForegroundColor Cyan -NoNewline
    Write-Host "  -  4x Bonsai 2 27B on 2x T4, round-robin"
    Write-Host '  /help for commands, /quit to exit' -ForegroundColor DarkGray
    Write-Host ''
}

while ($true) {
    if (-not $Message) {
        $input = Read-Host -Prompt 'you'
        if ($null -eq $input) { break }
        $input = $input.Trim()
        if (-not $input) { continue }
        switch -Regex ($input) {
            '^/(quit|exit|q)$' { Write-Host 'bye'; exit 0 }
            '^/help$' { Show-Help; continue }
            '^/pool$' { Show-Pool; continue }
            '^/health$' { Show-Health; continue }
            default { break }
        }
        $Message = @($input)
    }

    $text = ($Message -join ' ').Trim()
    if (-not $text) { Write-Error 'empty prompt'; exit 2 }

    # Try each endpoint once, starting at the cursor, skipping dead ones.
    $ok = $false
    $payload = "$text"
    for ($tries = 0; $tries -lt $pool.Count; $tries++) {
        $url = Next-Endpoint
        Set-Content -LiteralPath $cursorFile -Value $script:cursor
        Write-Host "[dq4 -> $url]" -ForegroundColor DarkGray
        try {
            $colab-env-patch:DQ_URL = $url
            $script = 'G:\Dx\tpu\bin\dq.ps1'
            & $script $payload
            $ok = $true
            break
        } catch {
            Write-Warning "$url failed ($($_.Exception.Message)); trying next"
        }
    }
    if (-not $ok) {
        Write-Warning 'no endpoint in the pool answered. Check /health or re-push the kernel.'
    }
    $Message = $null
}
exit 0
