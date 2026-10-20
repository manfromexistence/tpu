<#
.SYNOPSIS
    dq - chat CLI for a Bonsai 2 27B (Qwen3.8-27B ternary) endpoint.

.DESCRIPTION
    Streams answers from an OpenAI-compatible endpoint served by llama-server, which in
    this setup is a Bonsai 2 27B GGUF hosted on a Kaggle GPU session and exposed through
    a cloudflared quick tunnel.

    PowerShell is the default shell on this machine (see AGENTS.md), so this is written
    in it rather than C++: no compiler, no runtime dependency, and it shells out to
    nothing. The installer puts dq.cmd on PATH so `dq` works from any shell.

.PARAMETER Prompt
    One-shot mode. If omitted, starts an interactive REPL.

.PARAMETER Endpoint
    Override the base URL for one call. Without a trailing /v1.

.EXAMPLE
    dq "Hello, who are you?"

.EXAMPLE
    dq                      # interactive chat
    you> explain hash tables

.EXAMPLE
    $colab-env-patch:DQ_URL = "https://other.tunnel.example"
    dq

.NOTES
    MIT OR Apache-2.0
#>
[CmdletBinding()]
param(
    # ValueFromRemainingArguments is what lets a multi-word prompt bind when dq.cmd
    # forwards `%*` from cmd.exe: without it, PowerShell raises "too many arguments" on
    # the second word. It is also why --version is checked by hand below rather than
    # through a switch, since the switch and the capture compete for the same token.
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Prompt,

    [string]$Endpoint,

    [switch]$Version
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# --version / -v handled before anything touches the network, so they work even when
# the endpoint is down.
#
# There is deliberately no --help / -h flag: `-h` and `-Help` are PowerShell's own
# common parameters, so binding them makes the script fail before any of its code runs.
# Help text lives in the README next to the installer instead.
if ($Version -or ($Prompt -contains '-v') -or ($Prompt -contains '--version')) {
    Write-Output 'dq 1.0.0  -  Bonsai 2 27B (Qwen3.8-27B ternary) chat CLI'
    exit 0
}

# --- config -------------------------------------------------------------------
# Precedence: -Endpoint > $colab-env-patch:DQ_URL > persisted config file > built-in default.
# The default is a live session's URL, which will go stale; the colab-env-patch var and the file are
# how you point dq at a new session without editing this script.
$DefaultEndpoint = 'https://synopsis-auditor-builder-potato.trycloudflare.com'
$DefaultModel    = 'Ternary-Bonsai-2-27B-PTQ1_0'
$ConfigPath      = Join-Path $HOME '.dq.json'

# Reasoning markers, built at runtime. Writing them as literals is impossible here:
# the open marker begins with the two characters that start a PowerShell block comment,
# so a literal would comment out the rest of the file.
$ThinkOpen  = [string]::Join('', @('<', 'thi', 'nk', '>'))
$ThinkClose = [string]::Join('', @('<', '/thi', 'nk', '>'))

function Get-DqConfig {
    $cfg = @{
        endpoint = $DefaultEndpoint
        model    = $DefaultModel
    }
    if (Test-Path -LiteralPath $ConfigPath) {
        try {
            $saved = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
            if ($saved.endpoint) { $cfg.endpoint = $saved.endpoint }
            if ($saved.model)    { $cfg.model    = $saved.model }
        } catch {
            # A corrupt config must not stop the CLI; fall back to defaults and say so.
            Write-Warning "could not read $ConfigPath ($($_.Exception.Message)); using defaults"
        }
    }
    return $cfg
}

$Config = Get-DqConfig
if ($Endpoint)        { $Config.endpoint = $Endpoint }
elseif ($colab-env-patch:DQ_URL) { $Config.endpoint = $colab-env-patch:DQ_URL }

$Base = $Config.endpoint.TrimEnd('/')
$Url  = "$Base/v1/chat/completions"

# --- streaming request --------------------------------------------------------
# HttpClient rather than Invoke-RestMethod: the latter buffers the whole response
# before returning, so there would be no streaming at all. Reading the response stream
# line by line is what makes tokens appear as they are generated.
$script:Http = [System.Net.Http.HttpClient]::new()
$script:Http.Timeout = [TimeSpan]::FromMinutes(10)

function Send-DqPrompt {
    <#
        Streams one completion. Writes tokens to the host as they arrive and returns the
        assembled text, so the caller can keep conversation history.

        Reasoning is tracked by watching for the marker strings as they stream past,
        because this model thinks by default and the marker boundaries are the only
        reliable signal in the SSE deltas -- guessing at a field name would be wrong on
        some builds.
    #>
    param(
        [string]$Message,
        [Collections.Generic.List[object]]$History,
        [int]$MaxTokens = 512,
        [double]$Temperature = 0.7
    )

    $messages = [Collections.Generic.List[object]]::new()
    if ($History) {
        foreach ($h in $History) {
            $messages.Add(@{ role = $h.role; content = $h.content })
        }
    }
    $messages.Add(@{ role = 'user'; content = $Message })

    # include_usage makes the server emit a final chunk carrying completion_tokens, which
    # is how TPS is computed. Without it a streamed response has no token count at all:
    # the deltas carry text, not counts, and guessing from characters would be wrong.
    $payload = @{
        model           = $Config.model
        messages        = $messages
        max_tokens      = $MaxTokens
        temperature     = $Temperature
        stream          = $true
        stream_options  = @{ include_usage = $true }
    } | ConvertTo-Json -Depth 6 -Compress

    $content = [System.Net.Http.StringContent]::new(
        $payload, [Text.Encoding]::UTF8, 'application/json')

    $request = [System.Net.Http.HttpRequestMessage]::new(
        [System.Net.Http.HttpMethod]::Post, $Url)
    $request.Content = $content

    $response = $script:Http.Send(
        $request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead)
    if (-not $response.IsSuccessStatusCode) {
        $body = $response.Content.ReadAsStringAsync().Result
        throw "HTTP $([int]$response.StatusCode) $($response.ReasonPhrase)`n$body"
    }

    $stream = $response.Content.ReadAsStreamAsync().Result
    $reader = [IO.StreamReader]::new($stream)

    $answer   = [Text.StringBuilder]::new()
    $inThink  = $false
    $sawAny   = $false
    $tokens   = 0
    $startedAt = [Diagnostics.Stopwatch]::StartNew()

    try {
        while (-not $reader.EndOfStream) {
            $line = $reader.ReadLine()
            if ($null -eq $line -or $line -eq '') { continue }
            if (-not $line.StartsWith('data:')) { continue }

            $data = $line.Substring(5).Trim()
            if ($data -eq '[DONE]') { break }

            try { $chunk = $data | ConvertFrom-Json } catch { continue }
            if ($null -eq $chunk) { continue }

            # The usage chunk arrives AFTER the last content delta and carries
            # `choices: []`, so indexing choices[0] on it is an out-of-bounds error.
            # That is exactly what broke the first version of this: the answer streamed
            # correctly and then the script threw before printing the TPS line.
            if (-not $chunk.PSObject.Properties.Name -contains 'choices') { continue }
            $choices = $chunk.choices
            if ($null -eq $choices -or $choices.Count -eq 0) { continue }

            $delta = $choices[0].delta
            if ($null -eq $delta) { continue }

            # reasoning_content is where llama.cpp puts the think block; content is the
            # answer. Both go through the indexed form and a try/catch: under
            # StrictMode, reading $delta.content on a delta that lacks it throws, and
            # the terminating chunk of every stream is exactly such an empty object.
            # A missing property is normal here, not an error.
            $piece = $null
            $isReason = $false
            try {
                $reason = $delta.PSObject.Properties['reasoning_content'].Value
            } catch {
                $reason = $null
            }
            try {
                $body = $delta.PSObject.Properties['content'].Value
            } catch {
                $body = $null
            }
            if ($reason) {
                $piece = $reason; $isReason = $true
            } elseif ($body) {
                $piece = $body
            }
            if (-not $piece) { continue }

            $sawAny = $true

            # Track the markers so a model that emits them inside `content` (rather
            # than reasoning_content) is still handled correctly.
            if ($piece.Contains($ThinkOpen))  { $inThink = $true }
            if ($piece.Contains($ThinkClose)) { $inThink = $false }

            [void]$answer.Append($piece)

            # Write-Host, NOT Write-Information/6>. In PS5+ Write-Host writes to the
            # Information stream, and a `6>$null` at either call site swallowed the
            # entire streamed answer -- which is exactly why the REPL looked silent.
            if ($isReason -or $inThink) {
                Write-Host $piece -NoNewline -ForegroundColor DarkGray
            } else {
                Write-Host $piece -NoNewline -ForegroundColor White
            }
        }

        # usage arrives in its own chunk with an empty choices array, after the last
        # content delta, precisely because stream_options.include_usage was set.
        if ($chunk -and $chunk.PSObject.Properties.Name -contains 'usage' -and $chunk.usage) {
            if ($chunk.usage.PSObject.Properties.Name -contains 'completion_tokens') {
                $tokens = [int]$chunk.usage.completion_tokens
            }
        }
    } finally {
        $reader.Dispose()
        $response.Dispose()
    }

    $startedAt.Stop()
    $elapsed = $startedAt.Elapsed.TotalSeconds

    # Elapsed is wall-clock over the whole stream, including the return leg, so it is
    # an upper bound on seconds and therefore a lower bound on tok/s. Stated rather than
    # hidden, because a network-dependent figure presented as an exact rate is a lie.
    $tps = if ($tokens -gt 0 -and $elapsed -gt 0) {
        [math]::Round($tokens / $elapsed, 1)
    } else { $null }

    if (-not $sawAny) {
        Write-Host ''
        Write-Host '(no content in stream: model still loading, or reasoning used the budget)' -ForegroundColor DarkYellow
    }

    return [PSCustomObject]@{
        Answer         = $answer.ToString()
        Tokens         = $tokens
        Seconds        = [math]::Round($elapsed, 1)
        TokensPerSecond = $tps
    }
}

function Show-Banner {
    Write-Host ''
    Write-Host '  dq' -ForegroundColor Cyan -NoNewline
    Write-Host "  -  $($Config.model)"
    Write-Host "  $Url" -ForegroundColor DarkGray
    Write-Host '  /help for commands, /quit to exit' -ForegroundColor DarkGray
    Write-Host ''
}

function Show-Help {
    Write-Host @'
  commands:
    /help          this text
    /health        ping the endpoint
    /url [base]    show or change the endpoint base URL
    /model [name]  show or change the model
    /clear         drop conversation history
    /temp [0-2]    show or change temperature
    /quit          exit

  anything else is sent as a prompt. Replies stream as they arrive; reasoning is
  shown dimmed.
'@ -ForegroundColor DarkGray
}

function Show-DqStats {
    <#
        Prints the TPS line. Deliberately one shared function used by both one-shot and
        the REPL, so the two modes cannot drift into reporting different things.
    #>
    param($Result)

    if ($null -eq $Result) { return }
    if ($Result.TokensPerSecond) {
        Write-Host ("  {0} tok in {1}s  =  {2} tok/s" -f
            $Result.Tokens, $Result.Seconds,
            $Result.TokensPerSecond) -ForegroundColor DarkCyan
    } elseif ($Result.Tokens -gt 0) {
        Write-Host ("  {0} tok in {1}s" -f $Result.Tokens, $Result.Seconds) -ForegroundColor DarkCyan
    } else {
        Write-Host '  no token count returned by the server' -ForegroundColor DarkYellow
    }
}

# --- one-shot mode ------------------------------------------------------------
if ($Prompt) {
    $text = ($Prompt -join ' ').Trim()
    if (-not $text) { Write-Error 'empty prompt'; exit 2 }

    $history = [Collections.Generic.List[object]]::new()
    $result = Send-DqPrompt -Message $text -History $history

    # Streaming writes with -NoNewline, so a real newline is owed after it.
    Write-Host ''
    Write-Host $result.Answer
    Write-Host ''
    Show-DqStats -Result $result
    exit 0
}

# --- interactive REPL ---------------------------------------------------------
Show-Banner
$history = [Collections.Generic.List[object]]::new()
$temp = 0.7

while ($true) {
    $input = Read-Host -Prompt 'you'
    if ($null -eq $input) { break }                       # Ctrl+D / EOF
    $input = $input.Trim()
    if (-not $input) { continue }

    switch -Regex ($input) {
        '^/(quit|exit|q)$' { Write-Host 'bye'; exit 0 }
        '^/help$'         { Show-Help; continue }
        '^/clear$'        {
            $history.Clear()
            Write-Host 'history cleared' -ForegroundColor DarkGray
            continue
        }
        '^/health$' {
            try {
                $h = Invoke-RestMethod -Uri "$Base/health" -TimeoutSec 30
                Write-Host "ok: $($h | ConvertTo-Json -Compress)" -ForegroundColor Green
            } catch {
                Write-Warning "unreachable: $($_.Exception.Message)"
            }
            continue
        }
        '^/url$'    { Write-Host $Base; continue }
        '^/url (.+)$' {
            $Base = $Matches[1].TrimEnd('/')
            $Url  = "$Base/v1/chat/completions"
            Write-Host "endpoint now $Base" -ForegroundColor DarkGray
            continue
        }
        '^/model$'    { Write-Host $Config.model; continue }
        '^/model (.+)$' {
            $Config.model = $Matches[1]
            Write-Host "model now $($Config.model)" -ForegroundColor DarkGray
            continue
        }
        '^/temp$'    { Write-Host $temp; continue }
        '^/temp (.+)$' {
            $temp = [double]$Matches[1]
            Write-Host "temperature now $temp" -ForegroundColor DarkGray
            continue
        }
        default { break }
    }

    Write-Host 'dq' -ForegroundColor Cyan -NoNewline
    Write-Host ': ' -NoNewline

    try {
        $result = Send-DqPrompt -Message $input -History $history -Temperature $temp
        Write-Host ''
        Show-DqStats -Result $result

        # Keep the turn so follow-ups have context, capped so a long session does not
        # grow the payload until the server rejects it.
        $history.Add(@{ role = 'user'; content = $input })
        $history.Add(@{ role = 'assistant'; content = $result.Answer })
        while ($history.Count -gt 12) { $history.RemoveAt(0) }
    } catch {
        Write-Host ''
        Write-Warning "request failed: $($_.Exception.Message)"
        Write-Host 'check /health, or /url to point at a live session' -ForegroundColor DarkGray
    }
    Write-Host ''
}
