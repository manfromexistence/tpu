# dq — chat CLI for Bonsai 2 27B

Talks to a Bonsai 2 27B (Qwen3.8-27B ternary, `PTQ1_0`) endpoint served by `llama-server`
on a Kaggle GPU session and exposed through a cloudflared tunnel. Streams replies
token-by-token so it reads like a chat.

PowerShell, because PowerShell is the default shell on this machine (`AGENTS.md`), and a
script needs no compiler and no runtime.

## Install

```powershell
pwsh -NoProfile -File 'G:\Dx\tpu\dq-cli\install-dq.ps1'
```

That copies `dq.ps1` + `dq.cmd` into `G:\Dx\tpu\bin` and adds it to the **User** PATH.
The PATH entry applies to **new** shells, so open a fresh terminal before first use.

Uninstall: `install-dq.ps1 -Uninstall`.

## Use

```powershell
dq --version
dq "Hello, who are you?"
dq what is 2 plus 2
dq                      # interactive
```

Interactive commands: `/help` `/health` `/url` `/model` `/clear` `/temp` `/quit`.

## Output

Reasoning is printed dimmed instead of hidden. This model thinks by default, and
suppressing the reasoning silently would make a wrong answer look unconsidered — you
should be able to see when it spent its budget in a think block.

## Pointing it elsewhere

The default URL is whichever session was live when this was written and it will go
stale. Precedence: `-Endpoint` > `$colab-env-patch:DQ_URL` > `~/.dq.json` > the built-in default.

```powershell
dq -Endpoint "https://new-tunnel.trycloudflare.com"
$colab-env-patch:DQ_URL = "https://new-tunnel.trycloudflare.com"
```

## Limits, stated plainly

- **The URL dies with the Kaggle session.** Quick tunnels get a new random hostname each
  run, and sessions cap at 12 h. A durable endpoint needs a named host and a persistent
  host — neither of which a quick tunnel provides.
- **This is not a sanctioned deployment pattern.** Kaggle's product-feedback thread names
  tunnelling GPU notebooks to the internet as what they are working to block. It works
  today and carries no guarantee.
- Multi-word prompts through `cmd.exe` need `pwsh` (PowerShell 7) on PATH. The shim
  prefers it and falls back to Windows PowerShell 5.1, whose `-File` argument binding is
  stricter.
- `--help` is deliberately absent: `-h` and `-Help` are PowerShell's own common
  parameters, so declaring them makes the script fail before any of its code runs. This
  file is the help.

## Files

| File | Role |
| --- | --- |
| `dq.ps1` | the CLI |
| `dq.cmd` | shim so `dq` works from cmd, bash and PowerShell |
| `install-dq.ps1` | installs to `G:\Dx\tpu\bin`, edits User PATH |

MIT OR Apache-2.0
updated: 2026-10-12 00:00 UTC
