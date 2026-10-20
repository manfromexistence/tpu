"""Bonsai 2 27B (PTQ1_0) on Kaggle, served through a public Cloudflare tunnel.

Deliberately minimal. Earlier versions of this notebook carried probes, benchmarks and
a batching harness, and every one of those extra stages was an opportunity for a bug;
the version that actually measured 25.5 tok/s got there by accident, and the version
after it died on its own success-check. So this kernel does one thing:

  1. build the fork that can read PTQ1_0
  2. start llama-server on one T4
  3. put a cloudflared quick tunnel in front of it
  4. verify the public URL answers
  5. hold the session open

No probes, no llama-bench, no batching. Measured elsewhere on this same setup: 25.5 tok/s
decode on 1x T4, model loads in 5s, fits one 15360 MiB card with room for 8k context.

Honest framing on the tunnel, stated once here and not repeated: Kaggle's product-feedback
thread names tunnelling a GPU notebook to the internet as the pattern they are working to
block. It works today. It carries no guarantee, and the URL dies with the session.
"""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

# %% cell
# --- constants and helpers ----------------------------------------------------
# Fork: PrismML's own prism branch, which is the one verified to load PTQ1_0 at 25.5
# tok/s on this card. sudoingX/bonsai2 was also tried (v17) and measured the same 25.9,
# so the community kernel work does not help on Turing. Stay with what is measured.
FORK = "https://github.com/PrismML-Eng/llama.cpp.git"
BRANCH = "prism"

CUDA_ARCH = "75"
NPROC = os.cpu_count() or 4

WORK = Path("/kaggle/working")
OUT = WORK / "run"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"

# One GPU only. PrismML's KNOWN_ISSUES.md: "--split-mode tensor and --split-mode row
# can't load across two GPUs. Workaround: use the default layer split. (#213)".
# `-ngl 99` with two visible devices is what selects tensor split, so pin to one.
CUDA_DEV = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
SINGLE_GPU = dict(os.environ, CUDA_VISIBLE_DEVICES=CUDA_DEV)


def log(msg):
    """Write to stdout AND run.log. A failed run must leave its traceback somewhere."""
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(msg.rstrip() + "\n")
    except OSError:
        pass
    print(msg)


def sh(cmd, cwd=None, timeout=None, check=True, colab-env-patch=None, tail=1200):
    """Run a command, log everything, never route output through a shell pipe.

    No `| tail`: a pipeline returns the last command's status, which is how an earlier
    version reported "loads cleanly" while every invocation was "not found". Truncation
    happens here in Python instead.
    """
    log(f"\n$ {cmd}")
    e = dict(os.environ)
    if colab-env-patch:
        e.update(colab-env-patch)
    t0 = time.time()
    proc = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                          capture_output=True, text=True, timeout=timeout,
                          colab-env-patch=e, check=False)
    dt = time.time() - t0
    full = ""
    if proc.stdout:
        full += proc.stdout
    if proc.stderr:
        full += ("\n[stderr]\n" if full else "[stderr]\n") + proc.stderr
    try:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"[exit {proc.returncode} in {dt:.0f}s]\n{full[-20000:]}\n")
    except OSError:
        pass
    log(full[-tail:])
    log(f"[exit {proc.returncode} in {dt:.0f}s]")
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {cmd}\n{full[-1500:]}")
    return proc


log(f"=== stage: setup === bosai2-tunnel cpus {NPROC} arch {CUDA_ARCH}")
log(f"work {WORK}  device {CUDA_DEV}")

# %% cell
# --- 1. build the fork --------------------------------------------------------
# Unavoidable ~30 min: PrismML publishes release binaries, but their prebuilt builds
# segfault on AVX-512 CPUs (this host is an Intel Xeon reporting avx512f), and their
# documented workaround is to build from source. So build.
log("=== stage: build ===")

# Restore a cached build BEFORE compiling. The CUDA build costs ~1950s of every ~2000s
# session and the source is pinned to one commit, so the output is byte-identical every
# run. Caching turns a 33-minute run into a ~2-minute one.
#
# The cache lives in the Kaggle DATASET `essencefromexistence/bonsai2-llamacpp-build`
# rather than HuggingFace, and that is a deliberate choice after a failure: the HF route
# needs a token in Kaggle Secrets, which was never set, so it silently fell back to a
# full build every single time. A Kaggle dataset mount needs nothing -- the kernel
# already has the credentials implicit in the run, exactly like the model dataset
# `bonsai2-27b-ptq1` that works today.
#
# The mount path is NESTED, which is the bug this fixes: datasets land at
# /kaggle/input/datasets/<owner>/<slug>/bin, not /kaggle/input/<slug>/bin. Checking only
# one level deep found nothing and silently ran the full 33-minute build instead.
BUILD_CACHE = "bonsai2-llamacpp-build"
FORCE_BUILD = os.environ.get("FORCE_BUILD", "") == "1"


def dump_input_tree():
    """Log every file under /kaggle/input, capped so a large mount cannot flood it.

    This exists because three runs fell back to a 33-minute build while the cache was
    attached, and every diagnosis was a guess. Printing the tree once makes the mount
    layout an observed fact instead of an assumption.
    """
    lines = []
    for p in sorted(Path("/kaggle/input").rglob("*")):
        try:
            depth = len(p.relative_to("/kaggle/input").parts)
        except ValueError:
            continue
        if depth > 5:
            continue
        kind = "d" if p.is_dir() else "f"
        try:
            size = p.stat().st_size if p.is_file() else ""
        except OSError:
            size = "?"
        lines.append(f"  [{kind}] {p.relative_to('/kaggle/input')}"
                     + (f"  ({size} B)" if size != "" else ""))
        if len(lines) >= 200:
            lines.append("  ... truncated")
            break
    log("/kaggle/input tree:")
    log("\n".join(lines) if lines else "  (empty)")


def find_build_cache():
    """Locate the directory holding llama-server, anywhere under /kaggle/input.

    Matched three ways, because the dataset's layout is not what it looks like from the
    CLI: it was created with `-r zip`, which drops the source directory prefix, so the
    binaries sit at the DATASET ROOT rather than under tools/bin/.
    """
    root = Path("/kaggle/input")
    patterns = ["*/" * d + "tools/bin/llama-server" for d in (1, 2, 3, 4)]
    patterns += ["*/" * d + "llama-server" for d in (1, 2, 3, 4)]

    for pat in patterns:
        hits = [p for p in root.glob(pat) if BUILD_CACHE in str(p)]
        if hits:
            hits.sort(key=lambda p: len(str(p)))
            return hits[0].parent
    return None

dump_input_tree()
cached_bin = find_build_cache()

# Define before use: the cache check below inspects this path, and defining it after the
# block that reads it is a NameError that kills the run at 12s.
SERVER = Path("llama.cpp/build/bin/llama-server").resolve()

if cached_bin and not FORCE_BUILD:
    log(f"found cached build at {cached_bin}")
    # COPY, not symlink. /kaggle/input is read-only, and a symlink to a file there cannot
    # be made executable: posix_spawn fails with EACCES on the binary itself, which is
    # exactly how the first cached run died at 70s with PermissionError on llama-server.
    # 84 MB of copying is the price of being able to run it at all.
    target = Path("llama.cpp/build/bin")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(cached_bin.iterdir()):
        if src.is_file():
            shutil.copy2(src, target / src.name)
    # Recreate the soname symlinks the build normally provides. The binaries are linked
    # against libllama-common.so.0 and libllama.so.0, but only the fully-versioned
    # libllama-common.so.0.2.0 survived the dataset upload, so the loader aborts with
    # "cannot open shared object file". The copied files carry no such links.
    for versioned in sorted(target.glob("*.so.*.*.*")):
        base, _, rest = str(versioned.name).partition(".so.")
        major = rest.split(".")[0]
        for alias in (f"{base}.so.{major}", f"{base}.so"):
            link = target / alias
            if not link.exists():
                link.symlink_to(versioned.name)
                log(f"  soname link: {alias} -> {versioned.name}")
    # The dataset upload stripped the execute bit; restore it.
    for exe in ("llama-cli", "llama-server", "llama-bench"):
        pex = target / exe
        if pex.exists():
            os.chmod(pex, 0o755)
    if SERVER.exists() and os.access(SERVER, os.X_OK):
        log(f"cached binaries in {target} and executable; skipping clone and build")
    else:
        log("cached binaries copied but not executable; falling back to a build")
else:
    log(f"no cached build found (FORCE_BUILD={FORCE_BUILD}); building from source")

SERVER = Path("llama.cpp/build/bin/llama-server").resolve()

# Build only if the cache did not provide a usable llama-server.
if not (SERVER.exists() and os.access(SERVER, os.X_OK)):
    if Path("llama.cpp").exists():
        shutil.rmtree("llama.cpp")
    sh(f"git clone --depth 1 --branch {BRANCH} {FORK} llama.cpp", timeout=900)

    # GGML_CUDA_NO_VMM=ON: CMake on this image cannot find libcuda.so, and
    # ggml-cuda's CMakeLists guards that link line behind exactly this flag.
    sh(f"cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH} "
       f"-DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_CUDA_NO_VMM=ON",
       cwd="llama.cpp", timeout=900)
    sh(f"cmake --build build --target llama-cli llama-server llama-bench -j {NPROC}",
       cwd="llama.cpp", timeout=3600)

if not SERVER.exists():
    raise RuntimeError(f"llama-server was not built at {SERVER}")
if not os.access(SERVER, os.X_OK):
    raise RuntimeError(f"llama-server is not executable: {SERVER}")
log(f"llama-server: {SERVER} ({SERVER.stat().st_size:,} bytes)")

# No publish step here. The build cache is staged by the CLI-side tooling (the build
# was fetched out of a run's output and uploaded as the Kaggle dataset
# essencefromexistence/bonsai2-llamacpp-build), so the notebook has nothing to push.
# An earlier version tried to publish to HuggingFace and silently no-opped every run,
# because that needs HF_TOKEN in Kaggle Secrets and it was never set.

# %% cell [markdown]
"""
## 2. Locate the model

Mounted from the Kaggle dataset `essencefromexistence/bonsai2-27b-ptq1`, which holds
`Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5,946,648,928 bytes, verified against the server's own
Content-Length when it was staged).

It is then copied to `/kaggle/working`. `/kaggle/input` is a mount, not local storage, and
an earlier version had llama-cli stall for 900s reading from it. The copy takes ~70-140s
and turns that into a plain file read.
"""

# %% cell
CANDIDATES = [
    "/kaggle/input/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "/kaggle/working/run/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
]
MODEL = None
for c in CANDIDATES:
    p = Path(c)
    if p.exists():
        MODEL = p
        break

if MODEL is None:  # fall back to a glob, in case the mount name differs
    for base in sorted(Path("/kaggle/input").glob("*")):
        for p in sorted(base.rglob("*PTQ1_0*.gguf")):
            MODEL = p
            break
        if MODEL:
            break

if MODEL is None:
    raise RuntimeError("PTQ1_0 GGUF not found in any /kaggle/input path")

log(f"source model: {MODEL} ({MODEL.stat().st_size / 2**30:.2f} GiB)")

LOCAL = OUT / MODEL.name
if not (LOCAL.exists() and LOCAL.stat().st_size == MODEL.stat().st_size):
    free = shutil.disk_usage(WORK).free
    need = MODEL.stat().st_size
    log(f"copying to local disk ({need / 2**30:.2f} GiB, {free / 2**30:.2f} GiB free)")
    if free < need * 1.1:
        raise RuntimeError("not enough disk to copy the model")
    t0 = time.time()
    shutil.copyfile(MODEL, LOCAL)
    log(f"copy done in {time.time() - t0:.0f}s: {LOCAL.stat().st_size / 2**30:.2f} GiB")
    if LOCAL.stat().st_size != need:
        LOCAL.unlink()
        raise RuntimeError("copy size mismatch")

MODEL = LOCAL
log(f"model ready: {MODEL}")

# %% cell
# --- 3. cloudflared tunnel ----------------------------------------------------
# A quick tunnel: no account, no token, no DNS. One binary and one outbound connection
# buys a random *.trycloudflare.com hostname. That is the tooling the Kaggle/Colab
# tunnel notebooks converge on.
log("=== stage: tunnel ===")
CF = OUT / "cloudflared"
if not CF.exists():
    sh(f"curl -L --fail -o {CF} https://github.com/cloudflare/cloudflared/"
       f"releases/latest/download/cloudflared-linux-amd64", timeout=600)
    os.chmod(CF, 0o755)
log(f"cloudflared: {CF}")

# Start it before the server: the model load takes seconds, the hostname takes seconds,
# and there is no reason to serialise them.
CF_LOG = OUT / "cloudflared.log"
cf = subprocess.Popen(
    [str(CF), "tunnel", "--url", "http://127.0.0.1:8080", "--no-autoupdate"],
    stdout=open(CF_LOG, "w"), stderr=subprocess.STDOUT)
log(f"cloudflared pid {cf.pid}")

TUNNEL_URL = None
for _ in range(60):
    try:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",
                      CF_LOG.read_text())
    except OSError:
        m = None
    if m:
        TUNNEL_URL = m.group(0)
        break
    if cf.poll() is not None:
        log(f"cloudflared exited early (code {cf.returncode}); tail:")
        try:
            log(CF_LOG.read_text()[-1200:])
        except OSError:
            pass
        break
    time.sleep(2)

if TUNNEL_URL:
    (WORK / "TUNNEL_URL.txt").write_text(TUNNEL_URL, encoding="utf-8")
    log("=" * 68)
    log(f"  PUBLIC URL: {TUNNEL_URL}")
    log("=" * 68)
    print(f"\n{'=' * 68}\n  PUBLIC URL:  {TUNNEL_URL}\n{'=' * 68}\n")
else:
    log("no hostname yet -- it may still appear once the port is open")

# %% cell
# --- 4. llama-server ----------------------------------------------------------
# Server flags that matter for this model:
#   --reasoning-budget 0   the 27B thinks by DEFAULT and will spend its whole budget in
#                          a think block and emit nothing (PrismML troubleshooting).
#   -ngl 99                all layers on the GPU, with CUDA_VISIBLE_DEVICES pinning to
#                          one card to avoid the broken tensor split.
#   --parallel 8           continuous batching, so concurrent requests share a batch.
server = subprocess.Popen(
    [str(SERVER), "-m", str(MODEL), "-ngl", "99", "-c", "8192",
     "--host", "127.0.0.1", "--port", "8080",
     "--parallel", "8", "--cont-batching",
     "--reasoning-budget", "0", "--reasoning-format", "none"],
    colab-env-patch=SINGLE_GPU,
    stdout=open(OUT / "server.log", "w"), stderr=subprocess.STDOUT)
log(f"llama-server pid {server.pid}")

ready = False
for _ in range(300):
    if subprocess.run("curl -sf http://127.0.0.1:8080/health", shell=True,
                      capture_output=True).returncode == 0:
        ready = True
        break
    if server.poll() is not None:
        log("server exited before ready; tail:")
        try:
            log((OUT / "server.log").read_text()[-2000:])
        except OSError:
            pass
        break
    time.sleep(2)
log(f"server ready: {ready}")

# %% cell [markdown]
"""
## 5. Verify through the **public** URL

This is the claim being made, so it is tested on the public hostname rather than on
loopback — a loopback hit never leaves the machine and proves nothing about the tunnel.

The prompt is the one you asked for.
"""

# %% cell
PROMPT = "Hello, Who are you?"
answer = None
if TUNNEL_URL and ready:
    body = json.dumps({"model": MODEL.name.removesuffix(".gguf"),
                       "messages": [{"role": "user", "content": PROMPT}],
                       "temperature": 0.7, "max_tokens": 256})
    # Retry: the first request through a brand-new quick tunnel is often refused while
    # the route propagates, and that is not a failure.
    for attempt in range(1, 8):
        proc = subprocess.run(
            f"curl -s --max-time 300 {TUNNEL_URL}/v1/chat/completions "
            f"-H 'Content-Type: application/json' -d '{body}'",
            shell=True, capture_output=True, text=True, timeout=320)
        try:
            answer = json.loads(proc.stdout)["choices"][0]["message"]["content"]
            log(f"PUBLIC API ANSWER (attempt {attempt}) to {PROMPT!r}:\n{answer}")
            print(f"\n--- via {TUNNEL_URL} ---\n{answer}\n")
            break
        except (json.JSONDecodeError, KeyError, IndexError):
            log(f"attempt {attempt}: HTTP {proc.returncode}, "
                f"body: {proc.stdout[:300]}")
            time.sleep(5)
    else:
        log("public URL never returned a completable response")

elif ready:
    # No tunnel, but the server is up. Say so plainly rather than implying the public
    # path was tested.
    body = json.dumps({"model": MODEL.name.removesuffix(".gguf"),
                       "messages": [{"role": "user", "content": PROMPT}],
                       "temperature": 0.7, "max_tokens": 256})
    proc = subprocess.run(
        f"curl -s http://127.0.0.1:8080/v1/chat/completions "
        f"-H 'Content-Type: application/json' -d '{body}'",
        shell=True, capture_output=True, text=True, timeout=600)
    try:
        answer = json.loads(proc.stdout)["choices"][0]["message"]["content"]
        log(f"LOOPBACK-ONLY ANSWER (no tunnel) to {PROMPT!r}:\n{answer}")
        print(f"\n--- loopback only, NO TUNNEL ---\n{answer}\n")
    except (json.JSONDecodeError, KeyError, IndexError):
        log(f"loopback call also failed: {proc.stdout[:400]}")

else:
    log("server not ready; nothing to verify")

# %% cell
receipt = {
    "model": MODEL.name,
    "repo": "prism-ml/Ternary-Bonsai-2-27B-gguf",
    "fork": f"{FORK}@{BRANCH}",
    "gpu": CUDA_DEV,
    "tunnel_url": TUNNEL_URL,
    "public_verified": bool(TUNNEL_URL and answer),
    "api_prompt": PROMPT,
    "api_answer": answer,
    "endpoints": {
        "public": f"{TUNNEL_URL}/v1/chat/completions" if TUNNEL_URL else None,
        "loopback": "http://127.0.0.1:8080/v1/chat/completions",
    },
    "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(WORK / "bonsai2-tunnel-receipt.json").write_text(
    json.dumps(receipt, indent=2), encoding="utf-8")
log(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))

# %% cell
# --- 6. hold the session open -------------------------------------------------
# Kaggle ends the session when the notebook finishes, killing llama-server and
# cloudflared with it. This loop is the only thing preventing that, so it also watches
# the two children: if either dies, the URL stops answering and there is no point
# holding the session open.
#
# Capped at 11h against Kaggle's 12h hard GPU limit, so the run ends normally rather
# than being killed mid-flight with no receipt.
log("=== stage: keepalive ===")
if TUNNEL_URL:
    print(f"\nServing {TUNNEL_URL} for up to 11 hours.\n"
          f"Test from your PC:  curl {TUNNEL_URL}/health\n")
else:
    print("\nNo tunnel; holding loopback only.\n")

deadline = time.time() + 11 * 3600
beat = 0
while time.time() < deadline:
    if server.poll() is not None:
        log("llama-server exited; ending keepalive")
        break
    if cf.poll() is not None:
        log("cloudflared exited; ending keepalive")
        break
    beat += 1
    if beat % 30 == 0:
        log(f"alive, {int((deadline - time.time()) / 60)} min left, url={TUNNEL_URL}")
    time.sleep(10)

log(f"keepalive done. url was {TUNNEL_URL}")

# %% cell [markdown]
"""
## What this establishes, and what it does not

**It establishes:** whether this specific Kaggle GPU session can serve an
OpenAI-compatible Bonsai 2 27B endpoint at a public URL, tested by an actual completion
fetched through that URL.

**It does not establish:**
- That the practice is sanctioned. It is not. Kaggle's product-feedback thread names
  tunnelling GPU notebooks to the internet as the pattern they are working to block.
  The tunnel working here is a technical fact, not a permission.
- That the URL is durable. Quick tunnels get a new random hostname every run, and the
  session cap is 12 h. Anything real needs a named host and a persistent host.
- A tok/s figure. None was measured here by design; 25.5 tok/s decode on this card was
  measured in a separate run of this same build and model, with a receipt.

MIT OR Apache-2.0
updated: 2026-10-12 00:00 UTC
"""
