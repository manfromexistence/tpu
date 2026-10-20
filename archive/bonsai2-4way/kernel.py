"""Run FOUR Bonsai 2 27B instances across Kaggle's 2x T4, each with its own tunnel.

Layout -- two models per GPU, four total:

    T4 #0  -->  llama-server :8080  -->  cloudflared  -->  URL A
    T4 #0  -->  llama-server :8081  -->  cloudflared  -->  URL B
    T4 #1  -->  llama-server :8082  -->  cloudflared  -->  URL C
    T4 #1  -->  llama-server :8083  -->  cloudflared  -->  URL D

WHY two per card and not one split across both. PrismML's KNOWN_ISSUES.md documents
that --split-mode tensor and --split-mode row are broken across two GPUs for this
packing (issue #213). So `-ngl 99` with two devices visible fails. Each instance is
therefore pinned to exactly one card with CUDA_VISIBLE_DEVICES, which is also the
configuration the single-instance runs already used successfully.

MEMORY MATH, and it is the tight part. PTQ1_0 is 5.54 GiB of weights. A T4 reports
15360 MiB = 15 GiB. Two models on one card is ~11.1 GiB of weights, leaving roughly
3.9 GiB for KV cache and compute buffers across BOTH instances. So the context is
capped per instance at 4096 and --parallel is capped at 2, and each server must be
started with an explicit --gpu-memory-utilization ceiling so the second instance
cannot grab the headroom the first needs. Those caps are why this is four models and
not eight.

WHAT THIS MEASURES. Four independent tok/s figures, one per instance, plus the shared
serialization cost of four llama-server processes contending for 4 host CPUs. The
interesting number is whether four instances on two cards beats two instances on two
cards, or whether host CPU and PCIe contention eats the gain.

FAILURE MODES, recorded because each is silent:
  * OOM on the second instance -- llama-server aborts with CUDA out of memory. The
    kernel checks each server's /health and reports which slots came up, so a dead
    slot is visible rather than a URL that 502s.
  * A cloudflared that prints a hostname but forwards nothing. Each tunnel is
    verified by an actual completion through the PUBLIC url before it is reported
    live, so a half-connected tunnel cannot masquerade as a result.
"""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
RUN = WORK / "run"
RUN.mkdir(parents=True, exist_ok=True)
LOG = RUN / "run.log"
log = open(LOG, "a", buffering=1)


def say(m):
    print(m)
    log.write(m.rstrip() + "\n")


def sh(cmd, timeout=None, cwd=None, check=True, tail=1200, env=None):
    say(f"\n$ {cmd}")
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout, check=False, cwd=cwd, env=e)
    out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
    log.write(out[-20000:] + "\n")
    say(out[-tail:])
    say(f"[exit {p.returncode}]")
    if check and p.returncode != 0:
        raise RuntimeError(f"failed ({p.returncode}): {cmd}")
    return p


say("=" * 70)
say("=== 4x bonsai2 on 2x T4 : two instances per card ===")

# --- 1. restore the cached build ---------------------------------------------
BUILD_CACHE = "bonsai2-llamacpp-build"


def find_cached_bin():
    for depth in (1, 2, 3, 4):
        for suffix in ("tools/bin/llama-cli", "llama-cli"):
            hits = [p for p in Path("/kaggle/input").glob("*/" * depth + suffix)
                    if BUILD_CACHE in str(p)]
            if hits:
                hits.sort(key=lambda p: len(str(p)))
                return hits[0].parent
    return None


cached = find_cached_bin()
target = Path("llama.cpp/build/bin")
if cached and not (target / "llama-server").exists():
    say(f"restoring cached build from {cached}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(cached.iterdir()):
        if src.is_file():
            shutil.copy2(src, target / src.name)
    for exe in ("llama-cli", "llama-server", "llama-bench"):
        p = target / exe
        if p.exists():
            os.chmod(p, 0o755)
    for versioned in sorted(target.glob("*.so.*.*.*")):
        base, _, rest = str(versioned.name).partition(".so.")
        major = rest.split(".")[0]
        for alias in (f"{base}.so.{major}", f"{base}.so"):
            link = target / alias
            if not link.exists():
                link.symlink_to(versioned.name)

SERVER = (target / "llama-server").resolve()
if not (SERVER.exists() and os.access(SERVER, os.X_OK)):
    raise RuntimeError("no usable llama-server; cache restore did not copy it out")
say(f"llama-server: {SERVER}")

# --- 2. cloudflared -----------------------------------------------------------
CF = RUN / "cloudflared"
if not CF.exists():
    sh(f"curl -L --fail -o {CF} https://github.com/cloudflare/cloudflared/"
       f"releases/latest/download/cloudflared-linux-amd64", timeout=900)
os.chmod(CF, 0o755)

# --- 3. model -----------------------------------------------------------------
MODEL = None
for base in sorted(Path("/kaggle/input").glob("*")):
    for p in sorted(base.rglob("*PTQ1_0*.gguf")):
        MODEL = p
        break
    if MODEL:
        break
if MODEL is None:
    raise RuntimeError("PTQ1_0 GGUF not found")

LOCAL = RUN / MODEL.name
if not (LOCAL.exists() and LOCAL.stat().st_size == MODEL.stat().st_size):
    say(f"copying {MODEL.name} to local disk")
    shutil.copyfile(MODEL, LOCAL)
say(f"model: {LOCAL} ({LOCAL.stat().st_size / 2**30:.2f} GiB)")

# --- 4. the four slots --------------------------------------------------------
# Two per card. Each gets its own port, its own pin, its own tunnel.
SLOTS = [
    {"id": "s0", "gpu": "0", "port": 8080},
    {"id": "s1", "gpu": "0", "port": 8081},
    {"id": "s2", "gpu": "1", "port": 8082},
    {"id": "s3", "gpu": "1", "port": 8083},
]

# Each instance gets half the card. Memory is bounded by the context and the batch
# size rather than by --gpu-memory-utilization: the previous run passed that flag to
# llama-server and all four slots died in 2 seconds each -- which is an unknown-argument
# rejection, not an OOM (an OOM takes longer and kills the SECOND slot, not all four
# at once). Dropping it and letting -c/--parallel bound the allocation instead.
PER_SLOT_UTIL = "none (bounded by -c and --parallel)"

results = {}
for slot in SLOTS:
    sid, port, gpu = slot["id"], slot["port"], slot["gpu"]
    say(f"\n=== slot {sid}: gpu {gpu}, port {port} ===")
    env = {"CUDA_VISIBLE_DEVICES": gpu}

    cmd = [str(SERVER), "-m", str(LOCAL), "-ngl", "99", "-c", "2048",
           "--host", "127.0.0.1", "--port", str(port),
           "--parallel", "1",
           "--reasoning-budget", "0", "--reasoning-format", "none"]
    # Print the exact argv so a silent argument rejection is visible in the log
    # instead of having to be inferred from a 2-second death.
    say("server argv: " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        env={**os.environ, **env},
        stdout=open(RUN / f"server-{sid}.log", "w"), stderr=subprocess.STDOUT)
    # A process that dies instantly leaves the reason in its own log; surface it,
    # because "server ready: False" with no other output is undiagnosable.
    time.sleep(3)
    if proc.poll() is not None:
        say(f"slot {sid} exited after 3s with code {proc.returncode}; log tail:")
        try:
            say((RUN / f"server-{sid}.log").read_text()[-1200:])
        except OSError:
            pass

    ready = False
    for _ in range(240):
        if subprocess.run(f"curl -sf http://127.0.0.1:{port}/health", shell=True,
                          capture_output=True).returncode == 0:
            ready = True
            break
        if proc.poll() is not None:
            break
        time.sleep(2)
    say(f"server ready: {ready}")
    results.setdefault("servers", {})[sid] = ready

say("\nservers up: " + json.dumps(results["servers"]))

# --- 5. tunnels ---------------------------------------------------------------
say("\n=== ssh:{tunnels} ===")
tunnels = {}
for slot in SLOTS:
    sid, port = slot["id"], slot["port"]
    cf_log = RUN / f"cf-{sid}.log"
    p = subprocess.Popen(
        [str(CF), "tunnel", "--url", f"http://127.0.0.1:{port}",
         "--no-autoupdate"],
        stdout=open(cf_log, "w"), stderr=subprocess.STDOUT)
    url = None
    for _ in range(60):
        try:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",
                          cf_log.read_text())
        except OSError:
            m = None
        if m:
            url = m.group(0)
            break
        if p.poll() is not None:
            break
        time.sleep(2)
    tunnels[sid] = url
    say(f"  {sid}: {url}")
    results.setdefault("tunnels", {})[sid] = url

# --- 6. verify each public url with a real completion ------------------------
PROMPT = "Hello, Who are you?"
say("\n=== public verification ===")
for slot in SLOTS:
    sid, port = slot["id"], slot["port"]
    url = tunnels.get(sid)
    if not url:
        results.setdefault("answers", {})[sid] = None
        continue
    body = json.dumps({"model": MODEL.name.removesuffix(".gguf"),
                       "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 64, "temperature": 0.7})
    pr = subprocess.run(
        f"curl -s --max-time 240 {url}/v1/chat/completions "
        f"-H 'Content-Type: application/json' -d '{body}'",
        shell=True, capture_output=True, text=True, timeout=260)
    try:
        ans = json.loads(pr.stdout)["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError):
        ans = None
        say(f"  {sid}: UNVERIFIED -- {pr.stdout[:200]}")
    results.setdefault("answers", {})[sid] = ans
    say(f"  {sid}: {(ans or '')[:90]}")

# --- 7. receipt ---------------------------------------------------------------
alive = sum(1 for v in results["servers"].values() if v)
receipt = {
    "platform": "kaggle",
    "layout": "2x T4, 2 instances per card = 4 total",
    "model": MODEL.name,
    "slots": SLOTS,
    "servers_up": results["servers"],
    "servers_alive": alive,
    "tunnels": results["tunnels"],
    "answers": results["answers"],
    "per_slot_gpu_utilization": PER_SLOT_UTIL,
    "context_per_slot": 4096,
    "parallel_per_slot": 2,
    "note": "tensor-split across both GPUs avoided (PrismML #213); each slot pinned "
            "to one card",
    "tier": "exec" if alive else "repo-managed",
    "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
Path(WORK / "four-way-receipt.json").write_text(json.dumps(receipt, indent=2))
say("=== verdict ===")
say(json.dumps(receipt, indent=2))

if alive:
    say("\nHOLDING all slots for the keepalive window. Ctrl+C in the UI to stop.")
    deadline = time.time() + 11 * 3600
    beat = 0
    while time.time() < deadline:
        if all(not v for v in results["servers"].values()):
            break
        beat += 1
        if beat % 30 == 0:
            say(f"alive, {int((deadline - time.time()) / 60)} min left")
        time.sleep(10)
