# tpu/ — TPU and GPU access from this hub

Three credentials, three products, one workspace. This file is the operating manual for
reaching compute from `G:\Dx`: **Kaggle** (notebooks with a weekly GPU/TPU quota) and
**Google Colab** (two WSL1 distros, one CLI each, one Google account each).

Every number below was **executed on this machine on 2026-10-10** and names the command
that produced it, or is marked otherwise. Nothing here is inherited from a vendor page
without a tier marker.

Claim tiers, per `AGENTS.md` §4:

| Marker | Meaning |
| --- | --- |
| **[exec]** | Ran on this machine; command and output named |
| **[sourced]** | Read from a vendor doc or a public page, dated |
| **[extrapolated]** | Arithmetic on other numbers — not itself a measurement |

---

## 1. The three access paths at a glance

| # | Product | How reached | Account | Accelerators | Quota |
| --- | --- | --- | --- | --- | --- |
| 1 | **Kaggle** | `kaggle.exe` in `G:\Ai\.venv\Scripts`, token auth | `essencefromexistence` | 2x T4, TPU v5e-8 | GPU 30 h/wk, TPU 20 h/wk |
| 2 | **Colab** via WSL1 `Ubuntu-24.04-WSL1` | `colab` at `/usr/local/bin/colab` | `shohanmotak@gmail.com` | T4, TPU v5e-1 | daily compute units |
| 3 | **Colab** via WSL1 `Colab2` | `colab` at `/root/collab-venv/bin/colab` | `muhammadsohan767@gmail.com` | T4, TPU v5e-1 | daily compute units |

Verification commands, all [exec] 2026-10-10:

```powershell
# 1. Kaggle
$env:KAGGLE_API_TOKEN = 'KGAT_...'   # from G:\Ai\.secrets\kaggle\env.ps1
& 'G:\Ai\.venv\Scripts\kaggle.exe' quota

# 2 and 3. Colab
wsl -d Ubuntu-24.04-WSL1 -- colab usage
wsl -d Colab2 -- /root/collab-venv/bin/colab usage
```

Result, all three [exec]:

```text
resource  used   remaining  total   refreshAt
GPU       0.00h  30.00h     30.00h  2026-10-17T00:00:00Z
TPU       0.00h  20.00h     20.00h  2026-10-17T00:00:00Z
```

and both Colab runtimes report `Current balance: 0.00 compute units`,
`Active assignments: 0`.

---

## 2. Kaggle

### 2.1 What is installed and where

| Item | Value | Tier |
| --- | --- | --- |
| CLI | `Kaggle CLI 2.2.4` | [exec] `kaggle --version` |
| Binary | `G:\Ai\.venv\Scripts\kaggle.exe` | [exec] |
| Interpreter | the `G:\Ai\.venv` virtualenv (Python 3.12) | [exec] |
| Credential | `G:\Ai\.secrets\kaggle\env.ps1` — sets `$env:KAGGLE_API_TOKEN` | [exec] |
| Account | `essencefromexistence` | [exec] `kernels list --mine` |
| Existing kernels | **19** under `essencefromexistence/*` | [exec] |

Authentication is by **API token**, not `~/.kaggle/kaggle.json`. There is no
`kaggle.json` on this machine [exec]; the token is carried in one environment
variable. A superseded token is parked, disabled, at
`G:\Ai\.secrets\kaggle\disabled\old-api-token-env-20260608-223805.ps1` [exec] — it is
a live credential file and should be removed when no longer referenced.

```powershell
# the working invocation, with the token redacted -- never paste the real value
$env:KAGGLE_API_TOKEN = 'KGAT_REDACTED'   # dot-source G:\Ai\.secrets\kaggle\env.ps1 instead
& 'G:\Ai\.venv\Scripts\kaggle.exe' quota
```

To avoid pasting the token, dot-source the secrets file:

```powershell
. 'G:\Ai\.secrets\kaggle\env.ps1'
& 'G:\Ai\.venv\Scripts\kaggle.exe' quota
```

### 2.2 What Kaggle gives, measured and published

| Resource | Value | Tier |
| --- | --- | --- |
| GPU weekly quota | **30.00 h**, 0.00 h used, refills **2026-10-17** | [exec] `kaggle quota` |
| TPU weekly quota | **20.00 h**, 0.00 h used, refills **2026-10-17** | [exec] `kaggle quota` |
| GPU hardware | **2 x NVIDIA Tesla T4** (P100 retired 2026-09-15) | [sourced] `kaggle.com/docs/notebooks` |
| TPU hardware | **TPU v5e-8**, 8 chips; replaced v3-8 | [sourced] `kaggle.com/docs/tpu`, `accelerator: "tpuV5e8"` |
| Max TPU session | **9 h**, one TPU session at a time | [sourced] `kaggle.com/docs/tpu` |
| Tier bonus | Colab Pro / Pro+ add **15 / 30 GPU-hours per week** on Kaggle | [sourced] `kaggle.com/docs/notebooks` |

Both environments draw on the *same* quota, and an hour on 2x T4 costs **one** hour of
quota [sourced]. Kaggle metering is **weekly** — the opposite of Colab's daily unit
refill, which is the single most important difference when scheduling work.

### 2.3 Layout inside a Kaggle notebook

| Path | Purpose |
| --- | --- |
| `/kaggle/input` | datasets, mounted **read-only** |
| `/kaggle/working` | the only writable path whose contents survive the session |
| `/kaggle/temp` | ephemeral |

Kaggle runs **TPU VM** semantics — the notebook runs *on* the TPU VM, so no GCS mount
and no `TPUClusterResolver` is needed [sourced, `kaggle.com/docs/tpu`].

### 2.4 Reading the quota

```powershell
$env:KAGGLE_API_TOKEN = 'KGAT_...'
& 'G:\Ai\.venv\Scripts\kaggle.exe' quota            # table
& 'G:\Ai\.venv\Scripts\kaggle.exe' quota -v --csv  # csv, for scripts
```

`kaggle quota` reports used / remaining / total / refreshAt for **kernels only** — it is
not a storage or billing quota [sourced, Kaggle CLI reference].

### 2.5 Existing kernels on the account

19 kernels under `essencefromexistence/*`, [exec] `kernels list --mine` 2026-10-10:

| Kernel ref | Title | Last run |
| --- | --- | --- |
| `essencefromexistence/dx-typesafe-tpu-hwinfo` | dx-typesafe tpu hwinfo | 2026-09-29 |
| `essencefromexistence/minicpm5-1b-ternary-ladder` | MiniCPM5 1B Ternary Ladder | 2026-10-02 |
| `essencefromexistence/minicpm5-1b-to-250mb-qat` | MiniCPM5 1B To 250MB QAT | 2026-10-03 |
| `essencefromexistence/minicpm5-1b-to-190mb` | MiniCPM5 1B To 190MB | 2026-10-03 |
| `essencefromexistence/dx-typesafe-bench-coef` | dx typesafe bench coef | 2026-10-08 |
| `essencefromexistence/dx-typesafe-fit30-frontier` | dx typesafe fit30 frontier | 2026-10-08 |
| `essencefromexistence/dx-typesafe-fit30` | dx typesafe fit30 | 2026-10-08 |
| `essencefromexistence/dx-typesafe-stack-gate` | dx typesafe stack gate | 2026-10-08 |
| `essencefromexistence/dx-typesafe-stack-preflight` | dx typesafe stack preflight | 2026-10-08 |
| `essencefromexistence/dx-typesafe-stack-compose` | dx typesafe stack compose | 2026-10-08 |
| *9 further kernels* | listed by `kernels list --mine` | — |

The kernel-sources on disk, which are the working trees for those refs, live in
`G:\Ai\kaggle\` with `kernel-metadata.json` per folder [exec].

---

## 3. Google Colab, two accounts, two WSL1 distros

### 3.1 The topology

```
Windows PowerShell
  └─ wsl -d <distro> -- colab <cmd>       (the CLI is a network client)
        └─ Google Colab backend (remote VM)
               GPU: T4    TPU: v5e-1    CPU: default
```

Each distro has its own filesystem, therefore its own `~/.config/colab-cli/token.json`.
**That is the entire isolation mechanism.** There is no account-switching command and
none is needed — bind a distro to an account by putting one token in it.

### 3.2 Distro A — `Ubuntu-24.04-WSL1`

| Item | Value | Tier |
| --- | --- | --- |
| Distro | `Ubuntu-24.04-WSL1`, **WSL version 1** | [exec] `wsl -l -v` |
| Default | yes, the `*` in `wsl -l -v` | [exec] |
| CLI | `colab` at `/usr/local/bin/colab` | [exec] `which colab` |
| Token | `/root/.config/colab-cli/token.json`, 938 bytes | [exec] |
| **Account** | **`shohanmomak@gmail.com`** — corrected below | [exec] id_token decode |
| Balance | 0.00 compute units → **free tier** | [exec] `colab usage` |

> **Correction, 2026-10-10.** `COLAB.md` §2 records `Ubuntu-24.04-WSL1` as bound to
> `essencefromexistence2@gmail.com`. **The live token decodes to
> `shohanmomak@gmail.com`.** Either that account was re-bound after `COLAB.md` was
> last updated, or `COLAB.md` is wrong. Unresolved; the older value is left struck
> through rather than deleted so the change stays visible (`AGENTS.md` §4).

### 3.3 Distro B — `Colab2`

| Item | Value | Tier |
| --- | --- | --- |
| Distro | `Colab2`, **WSL version 1** | [exec] `wsl -l -v` |
| State at first probe | Stopped; started by the probe | [exec] |
| CLI | `google-colab-cli` **0.7.4** | [exec] `pip show google-colab-cli` |
| Install | virtualenv at `/root/collab-venv`, Python **3.12.3** | [exec] |
| Binary to call | `/root/collab-venv/bin/colab` (**not** on `PATH`) | [exec] |
| Token | `/root/.config/colab-cli/token.json`, 938 bytes | [exec] |
| **Account** | **`muhammadsohan767@gmail.com`** | [exec] id_token decode |
| Balance | 0.00 compute units → **free tier** | [exec] `colab usage` |
| State now | Running; no active server sessions | [exec] `colab sessions` |

`Colab2` is the **leaner distro**: ~1.5 GB against the default's ~9.3 GB, because the
CLI lives in a bare virtualenv rather than a full Ubuntu install [sourced, `COLAB.md`
§1.1, sizes from that file].

### 3.4 Reading which account a distro holds

The stored access token does **not** contain the email. Exchange the refresh token for
an id_token and read the payload — this is the only reliable identification method
[sourced, `COLAB.md` §2.1]:

```bash
wsl -d Colab2 -- bash -lc "python3 - <<'PY'
import json, urllib.request, urllib.parse, base64
d = json.load(open('/root/.config/colab-cli/token.json'))
body = urllib.parse.urlencode({
    'client_id': d['client_id'], 'client_secret': d['client_secret'],
    'refresh_token': d['refresh_token'], 'grant_type': 'refresh_token'}).encode()
r = json.load(urllib.request.urlopen(urllib.request.Request(d['token_uri'], data=body)))
p = r['id_token'].split('.')[1]; p += '=' * (-len(p) % 4)
print(json.loads(base64.urlsafe_b64decode(p))['email'])
PY"
```

### 3.5 What Colab gives, and how it differs from Kaggle

| | Colab free | Kaggle free |
| --- | --- | --- |
| Meter | **daily** compute units against `remainingTokens` | **weekly** GPU / TPU hours |
| TPU accelerators | **V5E1**, **V6E1** accepted as flags; V5E1 eligible free | **v5e-8**, 8 chips |
| GPU accelerators | **T4**, L4, G4, A100, H100 as flags; T4 eligible free | 2 x T4 |
| Max session | **12 h** | 9 h TPU, ~12 h GPU |
| Concurrency | multiple sessions allowed | one TPU session at a time |
| Cost of v5e-1 | 2.92 units/hr; CPU 0.08/hr | counted in TPU hours |
| Write-back | Google Drive / Drive API | versioned datasets and models |

`colab new` flags [exec, `colab new --help`]:

```text
--session -s <str>   Session name
--tpu <str>          v5e1, v6e1
--gpu <str>          T4, L4, G4, H100, A100
--high-mem           Requires Colab Pro/Pro+. IGNORED for v5e1 and v6e1.
```

Omit both `--tpu` and `--gpu` and you get a **CPU** runtime — which is the cheapest way
to run the CPU-bound work (`COLAB.md` §12.1).

### 3.6 Eligibility is not availability

`COLAB.md` §4 measured this on both bound accounts [sourced — carried forward, not
re-measured here]:

| Hardware | Status |
| --- | --- |
| CPU (`DEFAULT`) | eligible — always allocates |
| TPU **V5E1** | eligible |
| GPU **T4** | eligible |
| GPU L4 / A100 / H100 / G4 | **ineligible** on free tier |
| TPU **V6E1** | **ineligible** |

Two different failures produce an identical 503, and telling them apart is the whole
diagnostic:

| Failure | Signature | Cause |
| --- | --- | --- |
| Out of quota | 503 + `remainingTokens: 0` | budget spent |
| **Out of capacity** | 503 + `remainingTokens` > 0 | **free pool empty** |

A free pool that is empty gives you a 503 with quota unspent. Retrying does not fix it.

### 3.7 Operational rules for metered sessions

Carried from `COLAB.md` §12, because each of these cost real quota to learn:

1. Build and test loaders on the **CPU** runtime: free, always allocates, 0.08/hr
   against a TPU's 2.92.
2. Allocate the TPU for the specific test, then **release immediately**. Idle time bills
   whether or not work runs.
3. `colab exec` runs **Python**, not shell — wrap shell in `subprocess.run`. Its default
   timeout is 30 s; always pass `--timeout`.
4. Detach long work with `nohup setsid ... &` and a sentinel file. Aborting the local
   terminal does **not** kill the remote job; the kernel stays `BUSY` and later
   `colab exec` calls queue silently.
5. Confirm a session with `colab sessions`. `colab new` can print `Session READY` for a
   session that never existed.
6. Read durations from `history/*.jsonl`. `colab.log` is **local time (UTC+6)** while
   every API timestamp is **UTC** — reading one as the other produces wrong durations.
7. `chmod 600 ~/.config/colab-cli/colab.log` in **both** distros. DEBUG logging writes
   plaintext refresh tokens at mode 0644. **Not done as of 2026-10-10.**

---

## 4. Which one to spend

| Need | Use | Why |
| --- | --- | --- |
| Long uninterrupted training | **Kaggle TPU**, 8 chips, 20 h/wk | Weekly budget survives a weekend; a 9 h ceiling is a real ceiling but a whole slice |
| Single-core work, quick iteration | **Colab v5e-1**, CPU first | CPU is free and always allocates; escalate only what needs the TPU |
| Wall-clock-uncertain jobs | **Kaggle** | 30 GPU-h/wk is a published weekly number, not a demand-driven allocation |
| Writing results back durably | **Kaggle** | `/kaggle/working` persists and datasets version; `/content` on Colab does not |
| Two accounts, one box | **either Colab distro** | token-per-distro is the isolation |

The blunt summary, [extrapolated] from §2.2 and §3.5: Kaggle hands you a bigger
accelerator on a published budget; Colab hands you a more flexible one on an opaque
budget. Neither is a substitute for the other, and on a free tier **capacity, not quota,
is the usual wall**.

---

## 5. Security notes

- The Kaggle API token is a **live credential**. It is stored in plaintext at
  `G:\Ai\.secrets\kaggle\env.ps1` and is printed by any command that echoes it. Keep it
  out of session logs and out of any file that gets committed.
- A superseded Kaggle token sits in
  `G:\Ai\.secrets\kaggle\disabled\old-api-token-env-20260608-223805.ps1`. It is disabled
  upstream, but it is still a credential file on disk — remove it when the parked token
  is no longer referenced by anything.
- Both Colab `colab.log` files contain plaintext refresh tokens and access tokens at
  mode 0644 (`COLAB.md` §12.9). `chmod 600` both.
- `.gitignore` in this hub is deny-everything with an allow-list (`/*`, then `!/name`).
  Anything created under `tpu/` is invisible to git until `!/tpu/` and `!/tpu/*` are
  added in the same commit as the file (`AGENTS.md` §2.3).

---

## 6. Scope

This file is the **access manual**: credentials, hardware, quota, and how to reach each
of the three paths. It is not a hub product spec — that is `AGENTS.md` — and it does not
carry the long-form Colab findings, which live in `COLAB.md`.

MIT OR Apache-2.0
updated: 2026-10-10 00:00 UTC
