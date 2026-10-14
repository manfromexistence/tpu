# tpu/ — TPU and GPU access from this hub

Three credentials, three products, one workspace. This file is the operating manual for
reaching compute from `G:\Dx`: **Kaggle** (notebooks with a weekly GPU/TPU quota) and
**Google Colab** (two WSL1 distros, one CLI each, one Google account each) — plus the
Bonsai 2 27B kernel, the cached build, and the `dq` CLI that talk to the running models.

Every number below was **executed on this machine** and names the command or receipt
that produced it, or is marked otherwise. Nothing here is inherited from a vendor page
without a tier marker.

Claim tiers, per `AGENTS.md` §4:

| Marker | Meaning |
| --- | --- |
| **[exec]** | Ran on this machine; command and output named |
| **[sourced]** | Read from a vendor doc or a public page, dated |
| **[extrapolated]** | Arithmetic on other numbers — not itself a measurement |

---

## 0. For a new agent starting here — read this first

**Start here, in this order:**

1. **`G:\Dx\tpu\current-urls.json`** — if it exists and the URLs in it answer
   `/health`, the four models are live right now and `dq` works. If they return 530,
   the tunnels are dead and you need step 3.
2. **Test it:** `dq "hello"`. If that answers, you are done.
3. **Restart the four models:** `dqstart`. That pushes the kernel, waits for COMPLETE,
   scrapes the four new cloudflared URLs out of the kernel log, and re-points `dq`.

**The three things you must know before touching anything:**

| | |
|---|---|
| **All four model URLs are ephemeral** | Cloudflare Quick Tunnels die with the Kaggle session that spawned them. Every restart gives four new hostnames. Nothing here is durable. |
| **The build is cached** | `essencefromexistence/bonsai2-llamacpp-build` is a Kaggle dataset holding the compiled binaries. Without it every run pays ~32 minutes of CUDA compile. |
| **`G:\Dx` is a hub, not a monorepo** | Work belongs on Kaggle. Hub commits carry hub artefacts only — see `AGENTS.md` §2. |

**What is already proven, so do not re-litigate it:**

| Claim | Receipt |
|---|---|
| 4× Bonsai 2 27B on 2× T4, all verified over public HTTP | kernel `bonsai2-4way-2xt5`, `servers_alive: 4` |
| ~19.1 tok/s decode per instance, ~76 aggregate | `timings.predicted_per_second` in the same run |
| Setup 132s from cold with the cache | same log, cumulative timestamps |
| Bonsai 2 PTQ1_0 needs the PrismML fork, not upstream llama.cpp | `COLAB.md` §9 and the vendor's own `KNOWN_ISSUES.md` |
| `--split-mode tensor/row` is broken across 2 GPUs for this packing | vendor `KNOWN_ISSUES.md`, issue #213 |

**What is already known to fail, so do not try it:**

| Attempt | Result |
|---|---|
| Bonsai 2 + Qwen3.5-0.8B drafter (`--spec-draft-model`) | 9.1 tok/s = **0.36×**, a slowdown |
| Bonsai 2 + its own MTP head (`--spec-type draft-mtp`) | `failed to create MTP context` — the PTQ1_0 file has no MTP head |
| Bonsai 2 + `sudoingX` kernel branch on T4 | 27.95 vs 26.32 prism — the 1.5× kernel gain does not transfer to Turing |
| One model split across both T4s | tensor split, broken (#213) — pin with `CUDA_VISIBLE_DEVICES` |
| A public API tunnel from a Kaggle/Colab free notebook | works technically; it is the pattern both platforms are working to block |

---

## 1. The three access paths at a glance

| # | Product | How reached | Account | Accelerators | Quota |
|---|---|---|---|---|---|
| 1 | **Kaggle** | `kaggle.exe` in `G:\Ai\.venv\Scripts`, token auth | `essencefromexistence` | 2x T4, TPU v5e-8 | GPU 30 h/wk, TPU 20 h/wk |
| 2 | **Colab** via WSL1 `Ubuntu-24.04-WSL1` | `colab` at `/usr/local/bin/colab` | `shohanmomak@gmail.com` | T4, TPU v5e-1 | daily compute units |
| 3 | **Colab** via WSL1 `Colab2` | `colab` at `/root/collab-venv/bin/colab` | `muhammadsohan767@gmail.com` | T4, TPU v5e-1 | daily compute units |

Verification commands, all [exec] 2026-10-10:

```powershell
# 1. Kaggle
. 'G:\Ai\.secrets\kaggle\env.ps1'          # sets $env:KAGGLE_API_TOKEN
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

## 2. Kaggle — where everything runs

### 2.1 What is installed and where

| Item | Value | Tier |
|---|---|---|
| CLI | `Kaggle CLI 2.2.4` | [exec] `kaggle --version` |
| Binary | `G:\Ai\.venv\Scripts\kaggle.exe` | [exec] |
| Interpreter | the `G:\Ai\.venv` virtualenv (Python 3.12) | [exec] |
| **Credential** | **`G:\Ai\.secrets\kaggle\env.ps1`** — sets `$env:KAGGLE_API_TOKEN` | [exec] |
| Account | `essencefromexistence` | [exec] `kernels list --mine` |
| Kernels | 20+ under `essencefromexistence/*` | [exec] `kernels list --mine` |

Authentication is by **API token**, not `~/.kaggle/kaggle.json`. There is no
`kaggle.json` on this machine [exec]; the token is carried in one environment
variable. A superseded token is parked, disabled, at
`G:\Ai\.secrets\kaggle\disabled\old-api-token-env-20260608-223805.ps1` [exec] — it is
a live credential file and should be removed when no longer referenced.

**To authenticate a new shell:**

```powershell
. 'G:\Ai\.secrets\kaggle\env.ps1'
& 'G:\Ai\.venv\Scripts\kaggle.exe' quota          # should print GPU 30h / TPU 20h
```

Or set the token yourself: `$env:KAGGLE_API_TOKEN = 'KGAT_...'` (read it from
`env.ps1`; never paste the real value into a file that gets committed).

### 2.2 What Kaggle gives, measured and published

| Resource | Value | Tier |
|---|---|---|
| GPU weekly quota | **30.00 h**, 0.00 h used, refills **2026-10-17** | [exec] `kaggle quota` |
| TPU weekly quota | **20.00 h**, 0.00 h used, refills **2026-10-17** | [exec] `kaggle quota` |
| GPU hardware | **2 x NVIDIA Tesla T4** (P100 retired 2026-09-15) | [sourced] `kaggle.com/docs/notebooks` |
| TPU hardware | **TPU v5e-8**, 8 chips; replaced v3-8 | [sourced] `kaggle.com/docs/tpu` |
| Max TPU session | **9 h**, one TPU session at a time | [sourced] `kaggle.com/docs/tpu` |
| Tier bonus | Colab Pro / Pro+ add **15 / 30 GPU-hours per week** on Kaggle | [sourced] `kaggle.com/docs/notebooks` |

Both environments draw on the *same* quota, and an hour on 2x T4 costs **one** hour of
quota [sourced]. Kaggle metering is **weekly** — the opposite of Colab's daily unit
refill, which is the single most important difference when scheduling work.

### 2.3 Layout inside a Kaggle notebook

| Path | Purpose |
|---|---|
| `/kaggle/input` | datasets, mounted **read-only** |
| `/kaggle/working` | the only writable path whose contents survive the session |
| `/kaggle/temp` | ephemeral |

Two layout facts that cost real debugging time and are worth memorising:

1. **`/kaggle/input` is nested.** Datasets mount at
   `/kaggle/input/datasets/<owner>/<slug>/`, **not** `/kaggle/input/<slug>/`. A glob
   one level deep silently finds nothing.
2. **`/kaggle/input` is read-only.** A symlink to a binary there cannot be made
   executable, and `execve` then fails with `EACCES`. Copy binaries out and `chmod`.

### 2.4 The two Kaggle datasets this project depends on

| Dataset slug | Holds | Purpose |
|---|---|---|
| `essencefromexistence/bonsai2-27b-ptq1` | `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes | the model weights |
| `essencefromexistence/bonsai2-llamacpp-build` | the compiled `llama.cpp` fork binaries | turns a 33-min run into a 132-s one |

Both are private. Rebuild the second only if the fork source changes:

```powershell
# stage, then
& 'G:\Ai\.venv\Scripts\kaggle.exe' datasets create -p <folder> -r zip
```

### 2.5 Reading quota and logs

```powershell
. 'G:\Ai\.secrets\kaggle\env.ps1'
& 'G:\Ai\.venv\Scripts\kaggle.exe' quota                       # GPU / TPU hours left

$s = 'essencefromexistence/bonsai2-4way-2xt4'                  # any kernel
& 'G:\Ai\.venv\Scripts\kaggle.exe' kernels status $s           # QUEUED / RUNNING / ...
& 'G:\Ai\.venv\Scripts\kaggle.exe' kernels logs   $s           # the run log
& 'G:\Ai\.venv\Scripts\kaggle.exe' kernels output $s -p <dir>  # files from /kaggle/working
```

`kaggle kernels logs` is the command to reach for first when a run dies. The log is
printed as a JSON array, one object per line with a trailing comma — parse it by
splitting lines and `ConvertFrom-Json`-ing each, or by stripping commas and wrapping
the whole payload in brackets.

---

## 3. Google Colab — two accounts, two WSL1 distros

**Status: authenticated and working, but not used for Bonsai 2.** Colab free gives a T4
and a TPU v5e-1, and Bonsai 2 failed there twice (see §4.3). The CLIs are live and
usable for other work.

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
|---|---|---|
| Distro | `Ubuntu-24.04-WSL1`, **WSL version 1** | [exec] `wsl -l -v` |
| Default | yes, the `*` in `wsl -l -v` | [exec] |
| CLI | `colab` at `/usr/local/bin/colab` | [exec] `which colab` |
| **Token** | `/root/.config/colab-cli/token.json`, 938 bytes | [exec] |
| **Account** | **`shohanmomak@gmail.com`** | [exec] id_token decode |
| Balance | 0.00 compute units → **free tier** | [exec] `colab usage` |

> **Correction, 2026-10-10.** `COLAB.md` §2 records this distro as bound to
> `essencefromexistence2@gmail.com`. **The live token decodes to
> `shohanmomak@gmail.com`.** Either that account was re-bound after `COLAB.md` was
> last updated, or `COLAB.md` is wrong. Unresolved; the older value is left struck
> through rather than deleted so the change stays visible (`AGENTS.md` §4).

### 3.3 Distro B — `Colab2`

| Item | Value | Tier |
|---|---|---|
| Distro | `Colab2`, **WSL version 1** | [exec] `wsl -l -v` |
| CLI | `google-colab-cli` **0.7.4** | [exec] `pip show google-colab-cli` |
| Install | virtualenv at `/root/collab-venv`, Python **3.12.3** | [exec] |
| Binary to call | `/root/collab-venv/bin/colab` (**not** on `PATH`) | [exec] |
| **Token** | `/root/.config/colab-cli/token.json`, 938 bytes | [exec] |
| **Account** | **`muhammadsohan767@gmail.com`** | [exec] id_token decode |
| Balance | 0.00 compute units → **free tier** | [exec] `colab usage` |

`Colab2` is the **leaner distro**: ~1.5 GB against the default's ~9.3 GB, because the
CLI lives in a bare virtualenv rather than a full Ubuntu install [sourced, `COLAB.md`
§1.1, sizes from that file].

### 3.4 Re-authenticating a Colab distro

The token is implicit. To bind a distro to a **different** account, move the token
aside and let the next command run the OAuth flow:

```bash
wsl -d Colab2 -- bash -lc "mv ~/.config/colab-cli/token.json ~/.config/colab-cli/token.json.bak && export PATH=/root/collab-venv/bin:\$PATH && colab new -s probe"
```

Each `.bak` left behind is a **live credential copy on disk** — remove it when the
switch is done.

### 3.5 Reading which account a distro holds

The stored access token does **not** contain the email. Exchange the refresh token for
an id_token and read the payload — the only reliable identification method
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

### 3.6 What Colab gives, and how it differs from Kaggle

| | Colab free | Kaggle free |
|---|---|---|
| Meter | **daily** compute units against `remainingTokens` | **weekly** GPU / TPU hours |
| TPU accelerators | **V5E1**, **V6E1** as flags; V5E1 eligible free | **v5e-8**, 8 chips |
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

Omit both `--tpu` and `--gpu` and you get a **CPU** runtime — the cheapest way to run
CPU-bound work (`COLAB.md` §12.1).

### 3.7 Operational rules for metered sessions

1. Build and test loaders on the **CPU** runtime: free, always allocates, 0.08/hr
   against a TPU's 2.92.
2. Allocate the TPU for the specific test, then **release immediately**. Idle time bills
   whether or not work runs.
3. `colab exec` runs **Python**, not shell — wrap shell in `subprocess.run`. Its default
   timeout is 30 s; always pass `--timeout`.
4. Detach long work with `nohup setsid ... &` and a sentinel file. Aborting the local
   terminal does **not** kill the remote job.
5. Confirm a session with `colab sessions`. `colab new` can print `Session READY` for a
   session that never existed.
6. `chmod 600 ~/.config/colab-cli/colab.log` in **both** distros. DEBUG logging writes
   plaintext refresh tokens at mode 0644.

---

## 4. The Bonsai 2 27B workstream

### 4.1 What was built

| Artefact | Path | What it is |
|---|---|---|
| **4-way kernel** | `bonsai2-4way/` | two Bonsai 2 instances per T4, four cloudflared tunnels |
| Cached single tunnel | `bonsai2-tunnel/` | one instance, one tunnel |
| Cached fast run | `bonsai2-cached/` | the 83-second baseline |
| Kernel-branch test | `bonsai2-kernel/` | `sudoingX` pr-ptq1-mmv branch |
| MTP test | `mtp-test/` | `--spec-type draft-mtp` on the shipped file |
| Speculative-draft test | `spec-draft/` | `Qwen3.5-0.8B` drafter |
| **`dq` CLI** | `bin/dq.ps1`, `bin/dq.cmd` | chat CLI, installed globally |
| **`dqstart`** | `bin/dqstart.ps1` | push → poll → scrape URLs → point `dq` |

Everything is committed to this hub and pushed to
`github.com/manfromexistence/bonsai2` (which contains only the `tpu/` subtree).

### 4.2 Measured results, 2026-10-13 [exec]

| | |
|---|---|
| Model | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` — ternary 1.75 bpw, 5.54 GiB |
| Layout | 2× T4, **2 instances per card, 4 total** |
| Decode | **~19.1 tok/s per instance** |
| Aggregate | **~76 tok/s** across both cards |
| Prompt processing | 42.7 tok/s |
| Setup, cached | **132 s** from cold |
| Setup, cold (build) | ~34 min |

The per-instance figure is **half** of a solo model (~28 tok/s), because two instances
share one 15 GB T4. So the four-way layout trades per-request latency for aggregate
throughput — it is not a speedup per query.

`2x T4` is used as **two independent single-GPU slots**, never as one split model:
the vendor documents `--split-mode tensor/row` as broken across two GPUs for this
packing (#213).

### 4.3 The build, and why it is a source build

Two findings from the vendor, both verified against the model repo:

1. **Bonsai 2 needs the fork.** PrismML's `MODEL-FORMATS.md`: *"Bonsai 2: no
   mainline-compatible band... need the activation transform that only this demo's
   binaries carry, from the PrismML fork."* Stock llama.cpp rejects `PQ2_0`/`PTQ1_0`
   outright and would emit gibberish from a `Q2_0` file.
2. **The vendor's prebuilt binaries segfault on AVX-512 CPUs** — including *"some
   server and virtualised CPUs"*. Kaggle's host is an Intel Xeon with `avx512f`
   present, so the prebuilt path is unsafe. Their stated workaround is to build from
   source.

Hence: `git clone -b prism https://github.com/PrismML-Eng/llama.cpp.git`, then
`cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF
-DGGML_NATIVE=OFF -DGGML_CUDA_NO_VMM=ON`.

`GGML_CUDA_NO_VMM=ON` is required: CMake on the Kaggle image cannot find
`libcuda.so`, and `ggml-cuda`'s CMake guards that link line behind exactly this flag.
VMM governs GPU memory sharing, not throughput, so it costs nothing measurable.

### 4.4 Bugs that cost real time, recorded so they are not repeated

| Symptom | Cause | Fix |
|---|---|---|
| 32-minute source build every run, cache "not found" | dataset mounts **nested** at `/kaggle/input/datasets/<owner>/<slug>`; a shallow glob found nothing | glob 1–4 levels deep |
| `PermissionError: EACCES` on `llama-server` | `/kaggle/input` is read-only; a symlinked binary cannot be `chmod +x` | copy out, then `chmod 755` |
| All four slots dead in 2s each | `--gpu-memory-utilization` is rejected by `llama-server` — an unknown-argument rejection, **not** an OOM | drop the flag; bound memory with `-c` and `--parallel` |
| `E srv load_model: failed to create MTP context` | the PTQ1_0 file has **no MTP head** in it | unsolvable on this packing; use a separate grafted file |
| 9.1 tok/s with a drafter | `Qwen3.5-0.8B` disagrees with the 27B target too often; you pay to verify rejected drafts | not worth it on this hardware |
| HuggingFace download "succeeds" as 0 bytes | unauthenticated HF client silently truncates | download with `curl -L --fail` and check the byte count |

### 4.5 `dq` and `dqstart`

```powershell
dq "hello"      # one-shot against the saved DQ_URL
dq              # interactive REPL

dqstart         # push the kernel, wait for COMPLETE, scrape the four
                # cloudflared URLs from the kernel log, save to
                # current-urls.json, point dq at slot 0
```

`dqstart` sources `G:\Ai\.secrets\kaggle\env.ps1` for the token, so it needs no
arguments. It writes `G:\Dx\tpu\current-urls.json` so anything else can pick up the
pool.

**A `dqstart` push fails with 409 Conflict** if the kernel is still mid-run or
cancelling. Wait for `kernels status` to clear, then run it again.

---

## 5. Which one to spend

| Need | Use | Why |
|---|---|---|
| Long uninterrupted training | **Kaggle TPU**, 8 chips, 20 h/wk | Weekly budget survives a weekend; a 9 h ceiling is a real ceiling but a whole slice |
| Single-core work, quick iteration | **Colab v5e-1**, CPU first | CPU is free and always allocates; escalate only what needs the TPU |
| Wall-clock-uncertain jobs | **Kaggle** | 30 GPU-h/wk is a published weekly number, not a demand-driven allocation |
| Writing results back durably | **Kaggle** | `/kaggle/working` persists and datasets version; `/content` on Colab does not |
| Running Bonsai 2 27B | **Kaggle 2× T4**, 2 instances per card | verified there; failed on Colab twice |

The blunt summary, [extrapolated] from §2.2 and §3.6: Kaggle hands you a bigger
accelerator on a published budget; Colab hands you a more flexible one on an opaque
budget. On a free tier **capacity, not quota, is the usual wall.**

---

## 6. Security notes

- The Kaggle API token is a **live credential** stored in plaintext at
  `G:\Ai\.secrets\kaggle\env.ps1`. It was echoed in an earlier session log; **rotate it
  at kaggle.com/settings** and treat the old value as compromised.
- A superseded Kaggle token sits in
  `G:\Ai\.secrets\kaggle\disabled\old-api-token-env-20260608-223805.ps1`. Disabled
  upstream, but still a credential file on disk.
- Both Colab `colab.log` files contain plaintext refresh tokens and access tokens at
  mode 0644 (`COLAB.md` §12.9). `chmod 600` both.
- `.gitignore` in this hub is deny-everything with an allow-list (`/*`, then `!/name`).
  Anything created under `tpu/` is invisible to git until `!/tpu/` and `!/tpu/*` are
  added in the same commit as the file (`AGENTS.md` §2.3).
- `tpu/env.ps1` is gitignored. Keep it that way: it holds the live `KGAT_` token.

---

MIT OR Apache-2.0
updated: 2026-10-14 00:00 UTC
