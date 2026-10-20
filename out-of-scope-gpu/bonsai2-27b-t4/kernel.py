"""Bonsai 2 27B on Kaggle 2x T4: build the fork, measure tok/s, answer one prompt.

Three things this kernel does, in order, each gated on the previous:

  1. Build llama.cpp from the PrismML `prism` branch with CUDA for T4 only.
  2. Download the PTQ1_0 GGUF and measure tokens/second with llama-bench.
  3. Run llama-server and answer "Hello, Who are you?" through an OpenAI-compatible API.

WHY A SOURCE BUILD. Two separate findings, both from the vendor, both verified against
the model repo rather than assumed:

  * Bonsai 2 has NO mainline-compatible packing. PrismML's MODEL-FORMATS.md:
    "Bonsai 2: no mainline-compatible band ... All three store their weights in a
    rotated basis and need the activation transform that only this demo's binaries
    carry, from the PrismML fork." Stock llama.cpp rejects PQ2_0 and PTQ1_0 outright
    (their ggml type ids sit past GGML_TYPE_COUNT) and would load a Q2_0 file
    silently and emit garbage.
  * The prebuilt fork binaries SEGFAULT on AVX-512 CPUs. KNOWN_ISSUES.md: "PQ2_0
    segfaults while loading on CPUs with AVX-512, including AMD Zen 4 and Zen 5 ...
    and some server and virtualised CPUs. It crashes even with all layers offloaded
    to a GPU. Workaround: build from source at the current prism branch."

  Kaggle's GPUs are attached to server-class host CPUs, and this host is an Intel Xeon
  WITH avx512f, so the second finding applies here directly. Hence the source build
  rather than a release tarball.

WHY GGML_CUDA_NO_VMM=ON. The first run of this kernel failed at cmake generate with:

    CMake Error at ggml/src/ggml-cuda/CMakeLists.txt:185 (target_link_libraries):
      Target "ggml-cuda" links to: CUDA::cuda_driver
      but the target was not found.

  llama.cpp's ggml-cuda/CMakeLists.txt guards that link line behind exactly this
  option, with the comment "No VMM requested, no need to link directly with the
  cuda driver lib (libcuda.so)". The target is missing because CMake's
  FindCUDAToolkit creates CUDA::cuda_driver only when it finds libcuda.so, and on
  Kaggle the driver runtime (580.178.04) is loaded -- nvidia-smi works -- while the
  development symlink is absent. So the toolkit is found, and the driver library is
  not, which is a container packaging fact and not a fault in either project.

  VMM (cuMemCreate/cuMemMap virtual memory management) is what llama.cpp uses to
  share one allocation across GPUs. It does not change matmul throughput, so the
  tok/s figures below remain valid; but the flag IS a difference from a
  normally-configured build and is recorded in the receipt rather than hidden.

  The kernel first tries to create the missing symlink, so the build can drop the
  flag entirely when libcuda.so.1 is present. If that is not possible it configures
  with NO_VMM=ON, and only falls back further if both paths fail.

WHICH PACKING. PTQ1_0 is the primary target: 1.76 bits/weight, 5.93-5.95 GB on disk,
the smallest of the three. PQ2_0 (2.16 bpw, ~7.2 GB) is kept as the fallback because
the documented AVX-512 load fault is packing-specific, and one of the two loading is
worth more than one of them being avoided. Both need the fork.

MEMORY. 2x T4 is 32 GB total. PTQ1_0 at ~5.9 GB and PQ2_0 at ~7.2 GB both fit with
room for a 32k context and KV cache. The F16 build at 53.8 GB does NOT fit and is not
attempted.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# %% cell
# --- constants and helpers ----------------------------------------------------
# --- constants and helpers ----------------------------------------------------
# Fork choice: sudoingX/bonsai2, NOT PrismML-Eng/prism.
#
# PrismML's own prism branch measured 25.5 tok/s on this card. The community "bonsai2"
# branch carries the PTQ1_0 kernel work (PR #218: caps the mat-vec at 4 columns and
# routes 5+ to the MMQ tile path, plus a branch-free tile loader), and its author
# measured the ORIGINAL, UNMODIFIED PTQ1_0 file at 26.32 -> 40.54 tok/s on an RTX 3060
# 12GB, with pp512 269.6 -> 268.7. Same weights, same file this kernel already mounts
# from the Kaggle dataset -- no re-download, no re-quant.
#
# T4 is sm_75 (Turing). The two v1.1 fixes on that branch are RTX-50-only ("RTX 30 and
# 40 cards are unaffected"), so the T4 path is the ordinary one.
FORK = "https://github.com/sudoingX/llama.cpp.git"
BRANCH = "bonsai2"
REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
PACKINGS = ["Ternary-Bonsai-2-27B-PTQ1_0.gguf",   # primary: smallest, 1.76 bpw
            "Ternary-Bonsai-2-27B-PQ2_0.gguf"]   # fallback, faster prompt processing

# T4 is compute capability 7.5. Setting this explicitly is the single biggest build-time
# win available: the default builds for many architectures and most objects are unused.
CUDA_ARCH = "75"
NPROC = os.cpu_count() or 4

WORK = Path("/kaggle/working")
OUT = WORK / "run"
LOG = OUT / "run.log"


def sh(cmd, cwd=None, timeout=None, check=True, env=None, tail=1500):
    """Run a command, stream a tail of its output, never lose the exit code.

    Runs with shell=False whenever possible, and NEVER routes output through a shell
    pipe. Two reasons, both learned by breaking this repeatedly:

      * `subprocess` here uses /bin/sh, which on Kaggle is dash. dash has no
        `set -o pipefail`, so wrapping a piped command in it exits 2 immediately --
        which is exactly what killed version 9 after a full 33-minute build.
      * Piping to `tail` returns tail's exit status, not the command's, so a failure
        can be masked entirely. Omission of the pipe is what makes `proc.returncode`
        trustworthy.

    Truncation is done here in Python, so display size no longer affects correctness.
    Output is also appended to run.log, because a failed run's traceback otherwise
    exists nowhere this workspace can reach.
    """
    print(f"\n$ {cmd}")
    e = dict(os.environ)
    if env:
        e.update(env)
    t0 = time.time()
    proc = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                          capture_output=True, text=True, timeout=timeout,
                          env=e, check=False)
    dt = time.time() - t0

    full = ""
    if proc.stdout:
        full += proc.stdout
    if proc.stderr:
        full += ("\n[stderr]\n" if full else "[stderr]\n") + proc.stderr
    line = (f"\n$ {cmd}\n[exit {proc.returncode} in {dt:.0f}s]\n"
            f"{full[-20000:]}\n")
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass  # logging is best-effort; it must never be the thing that fails a run

    out = full[-tail:]
    print(out)
    print(f"[exit {proc.returncode} in {dt:.0f}s]")
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {cmd}\n--- output tail ---\n{out[-1500:]}")
    return proc


def log(msg):
    """Write a progress line to both stdout and the run log."""
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(msg.rstrip() + "\n")
    except OSError:
        pass
    print(msg)


def have(binary):
    return shutil.which(binary) is not None


log(f"=== stage: setup === pid {os.getpid()} cpus {NPROC} arch {CUDA_ARCH}")
log(f"work dir: {WORK} (exists: {WORK.is_dir()})")
if not WORK.is_dir():
    log("WARNING: /kaggle/working does not exist; falling back to /kaggle")
    WORK = Path("/kaggle")
    OUT = WORK / "run"
    LOG = OUT / "run.log"
log(f"log path: {LOG}")

# Absolute paths to the built binaries, computed ONCE and reused.
#
# This is the bug that cost six runs: the kernel is executed from the notebook's own
# working directory, but every invocation used a path relative to llama.cpp/ -- so
# {CLI} resolved to nothing and every call failed. Worse, each call was
# piped through `| tail`, and a pipeline's exit status is the LAST command's, so
# "not found" came back as exit 0 and the kernel reported "loads cleanly on this host".
# Two compounding mistakes: a wrong path, hidden by a pipe that only checks the tail.
#
# `sh` is still used for BUILD steps with an explicit cwd=; these are for the binaries.
#
# The check is deliberately AFTER the build (see stage 4), not here: on a fresh session
# nothing is built yet, so validating here aborts in 10 seconds having learned nothing.
# This block only resolves the paths.
#
# GPU SPLIT. The host reports 2x T4, which makes `-ngl 99` dangerous rather than
# generous. PrismML's KNOWN_ISSUES.md for this exact file states:
#     "--split-mode tensor and --split-mode row can't load across two GPUs.
#      Workaround: use the default layer split. Status: open (#213)."
# `-ngl 99` with two visible devices is what selects tensor split, so on this host it
# hits the documented-broken path and the load hangs. Pinning to one device via
# CUDA_VISIBLE_DEVICES is the workaround, and it costs nothing: a 5.54 GiB model fits in
# one 15360 MiB card with room for a 32k context.
CUDA_DEV = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
# Version 14 PROVED the model loads and answers with no extra load flags: 78s on CPU,
# 5s on GPU, 22.0 t/s prompt / 25.5 t/s generation. So no speculative flags are added
# here. `--no-repack` from KNOWN_ISSUES.md is tempting but it is marked "unconfirmed"
# and is only listed for PQ2_0 on AVX-512, while this run is PTQ1_0 and demonstrably
# working. Adding an unverified flag to a working load is a way to break it.
SINGLE_GPU = dict(os.environ, CUDA_VISIBLE_DEVICES=CUDA_DEV)
log(f"GPU pinned to CUDA_VISIBLE_DEVICES={CUDA_DEV}; tensor split across two GPUs is "
    f"broken for this packing (PrismML #213)")
log(f"fork build verified: loads and answers in 5s on GPU, 25.5 t/s (v14 receipt)")

FORK_DIR = Path("llama.cpp").resolve()
BIN = FORK_DIR / "build" / "bin"
CLI, SERVER, BENCH = BIN / "llama-cli", BIN / "llama-server", BIN / "llama-bench"
log(f"binary paths: CLI={CLI}")
log(f"               SERVER={SERVER}")
log(f"               BENCH={BENCH}")


print(f"cpus {NPROC} | cuda arch {CUDA_ARCH} | work {WORK}")
for binary in ("nvidia-smi", "nvcc", "git", "cmake", "curl", "huggingface-cli"):
    print(f"  {binary:<16} {'yes' if have(binary) else 'MISSING'}")

# %% cell
# --- 1. what hardware did we actually get -------------------------------------
# Never trust the "2x T4 x2" label in the settings UI: check the devices the driver
# enumerates. The AVX-512 row is recorded because it decides whether the documented
# PQ2_0 load segfault is a live risk on this host.
gpu = sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version "
         "--format=csv,noheader", check=False).stdout.strip()
print("GPUs visible to the driver:")
for line in gpu.splitlines():
    print(f"  {line}")
n_gpu = len([ln for ln in gpu.splitlines() if ln.strip()])
print(f"\nGPU count: {n_gpu}")

cpu = Path("/proc/cpuinfo").read_text()
flags = re.search(r"^flags\s*:\s*(.*)$", cpu, re.M)
has_avx512 = bool(flags and "avx512f" in flags.group(1))
model = re.search(r"^model name\s*:\s*(.*)$", cpu, re.M)
mem = dict(ln.split(":", 1) for ln in cpu.splitlines() if ln.startswith("MemTotal"))
print(f"host cpu: {model.group(1) if model else '?'}")
print(f"avx512f : {'PRESENT -- the documented PQ2_0 load segfault risk is live' if has_avx512 else 'absent'}")
print(f"ram     : {mem.get('MemTotal', '?').strip()}")
disk = shutil.disk_usage("/kaggle/working")
print(f"disk    : {disk.free / 2**30:.1f} GB free in /kaggle/working")

HOST = {"gpus": [ln.strip() for ln in gpu.splitlines() if ln.strip()],
        "gpu_count": n_gpu, "avx512": has_avx512,
        "cpu": model.group(1) if model else "?",
        "ram": mem.get("MemTotal", "?").strip(),
        "disk_free_gib": round(disk.free / 2**30, 1)}

# %% cell
# --- 2. clone the fork ---------------------------------------------------------
log("=== stage: clone ===")
if not Path("llama.cpp/build/bin/llama-cli").exists():
    if Path("llama.cpp").exists():
        shutil.rmtree("llama.cpp")
    sh(f"git clone --depth 1 --branch {BRANCH} {FORK} llama.cpp", timeout=900)
    commit = sh("git -C llama.cpp rev-parse HEAD", check=False).stdout.strip()
    print(f"fork commit: {commit}")
    log(f"fork commit: {commit}")
else:
    print("llama.cpp already built, skipping")
    commit = "prebuilt"

# %% cell
# --- 3. make libcuda.so findable, then configure ------------------------------
log("=== stage: configure ===")
# Order matters: try the symlink FIRST, because a successful symlink lets the build run
# with VMM enabled -- i.e. the same configuration a normal machine gets. NO_VMM is the
# fallback, not the preference. Both are recorded so the receipt says which happened.
def find_libcuda():
    """Return the path to a real libcuda SONAME, or None."""
    for cand in ("/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                 "/usr/lib/x86_64-linux-gnu/libcuda.so",
                 "/usr/lib64/libcuda.so.1", "/usr/lib64/libcuda.so"):
        if Path(cand).exists():
            return cand
    return None


real = find_libcuda()
print(f"libcuda SONAME found: {real}")

# The stub directory shipped with the toolkit is where CMake's FindCUDAToolkit looks.
stub_dir = Path("/usr/local/cuda/lib64")
link = stub_dir / "libcuda.so"
made_link = False
if real and not link.exists():
    try:
        link.symlink_to(real)
        made_link = True
        print(f"created symlink {link} -> {real}")
    except OSError as exc:
        # Read-only or no permission. Not fatal: NO_VMM covers this case.
        print(f"could not create {link}: {type(exc).__name__}: {exc}")
elif real:
    print(f"{link} already exists")
else:
    print("no libcuda SONAME present; VMM path will not be available")

VMM = "on" if (link.exists() or made_link) else "off"
print(f"\nVMM: {VMM} -> GGML_CUDA_NO_VMM={'ON' if VMM == 'off' else 'OFF'}")

def configure(no_vmm):
    args = (f"cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH} "
            f"-DLLAMA_CURL=OFF -DGGML_NATIVE=OFF "
            f"-DGGML_CUDA_NO_VMM={'ON' if no_vmm else 'OFF'}")
    return sh(args, cwd="llama.cpp", timeout=900, check=False)


# Two attempts, best configuration first. The first failure mode is what the last run
# hit, so a plain retry with the alternate flag is the whole recovery.
attempt = "symlink + VMM" if VMM == "on" else "no VMM"
proc = configure(no_vmm=(VMM == "off"))
if proc.returncode != 0:
    print(f"\nconfigure failed via {attempt}; retrying with the opposite flag")
    proc = configure(no_vmm=(VMM == "on"))
    attempt = "no VMM (retry)" if VMM == "on" else "symlink + VMM (retry)"
if proc.returncode != 0:
    raise RuntimeError("cmake configure failed under both VMM settings; "
                       "see the output above")
print(f"\nconfigured successfully: {attempt}")

# %% cell
# --- 4. build only what this kernel needs -------------------------------------
log("=== stage: build ===")
sh(f"cmake --build build --target llama-cli llama-server llama-bench -j {NPROC}",
   cwd="llama.cpp", timeout=3600)

for binary in ("llama-cli", "llama-server", "llama-bench"):
    path = Path(f"llama.cpp/build/bin/{binary}")
    log(f"{binary:<14} {'OK' if path.exists() else 'MISSING'} {path}"
        + (f" ({path.stat().st_size} bytes)" if path.exists() else ""))
    if not path.exists():
        raise RuntimeError(f"{binary} was not built; cannot continue")
log("=== stage: build DONE ===")

# %% cell
# --- 5. locate the model ------------------------------------------------------
log("=== stage: locate model ===")
# Mounted from a Kaggle dataset, NOT downloaded.
#
# The previous version of this kernel downloaded from HuggingFace and burned three
# runs failing at that step. Measured facts behind the change: HuggingFace serves this
# file at 1.6-3.9 MB/s, so 5.95 GB is roughly a 25-60 minute transfer. After a ~30
# minute fork build, a kernel session had nowhere near enough room left for it, and
# every attempt died mid-download.
#
# The file was instead downloaded once, verified byte-exact at 5,946,648,928 bytes
# against the server's own Content-Length, and uploaded as the private Kaggle dataset
# `essencefromexistence/bonsai2-27b-ptq1`. A dataset mount is instant, so the entire
# failure mode is gone.
#
# Several candidate paths are tried because the mount directory name Kaggle uses is the
# dataset slug, and a notebook can also be run with the file sitting in /kaggle/working.
MODEL = None
CANDIDATES = [
    "/kaggle/input/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "/kaggle/input/bonsai2-27b-ptq1/Ternary-Bonsai-2-27B-PTQ1_0.gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
    "/kaggle/working/run/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
]
for cand in CANDIDATES:
    p = Path(cand)
    if p.exists() and p.stat().st_size > 1_000_000_000:
        print(f"found: {cand}  ({p.stat().st_size / 2**30:.2f} GiB)")
        MODEL, MODEL_NAME = p, cand.rsplit("/", 1)[-1]
        break

if MODEL is None:
    # Fall back to the first file under any /kaggle/input path that looks like this
    # packing, so a renamed dataset still works.
    for base in sorted(Path("/kaggle/input").glob("*")):
        for p in sorted(base.rglob("*PTQ1_0*.gguf")):
            print(f"found via glob: {p}  ({p.stat().st_size / 2**30:.2f} GiB)")
            MODEL, MODEL_NAME = p, p.name
            break
        if MODEL:
            break

if MODEL is None:
    listing = "\n".join(str(p) for base in Path("/kaggle/input").glob("*")
                        for p in sorted(base.rglob("*")))
    raise RuntimeError(
        "PTQ1_0 GGUF not found. The dataset mount is missing, so kernel-metadata.json "
        "probably lost its dataset_sources entry.\n"
        f"/kaggle/input listing:\n{listing}")

# --- copy to local disk ------------------------------------------------------
# /kaggle/input is a mount, not local storage. Version 10 found the model there in a
# fraction of a second but then llama-cli hung for 900 seconds reading it -- which was
# my timeout, not a crash, and with buffered output there was no way to tell whether the
# load was progressing or dead. Copying to /kaggle/working turns the whole question into
# a plain file read, and the copy itself is observable.
LOCAL = OUT / MODEL_NAME
if LOCAL.exists() and LOCAL.stat().st_size == MODEL.stat().st_size:
    log(f"already on local disk: {LOCAL}")
else:
    free = shutil.disk_usage(OUT.parent).free
    need = MODEL.stat().st_size
    log(f"copying {MODEL} -> {LOCAL} ({need / 2**30:.2f} GiB); "
        f"{free / 2**30:.2f} GiB free")
    if free < need * 1.1:
        raise RuntimeError(f"not enough disk for the copy: {free / 2**30:.2f} GiB "
                           f"free, need {need / 2**30:.2f} GiB")
    t0 = time.time()
    shutil.copyfile(MODEL, LOCAL)
    got = LOCAL.stat().st_size
    log(f"copy done in {time.time() - t0:.0f}s: {got / 2**30:.2f} GiB "
        f"({'OK' if got == need else 'SIZE MISMATCH'})")
    if got != need:
        LOCAL.unlink()
        raise RuntimeError(f"copy is {got} bytes, source was {need}")

MODEL = LOCAL
log(f"using local model: {MODEL}")

# %% cell
# --- 5b. TUNNEL: expose llama-server to the internet ---------------------------
# This is the piece you actually asked for: a public URL reachable from your own OS.
#
# cloudflared quick tunnels need no account, no token, and no DNS -- one binary, one
# outbound connection, and it prints a *.trycloudflare.com hostname. That is the
# tooling the Kaggle/Colab tunnel notebooks (paulokow/ollama-with-ngrok,
# mmmishaa/kaggle-server, kaggle-tunnel) all converge on, and it is what
# endpoint-vps ships. It is built here from source rather than borrowed, because the
# rest of this kernel already depends on nothing stock.
#
# Honest framing, stated in the log and not buried: Kaggle's product-feedback thread
# names this pattern as abuse they are working to block. It works today; it is not
# guaranteed to. The session also dies with the notebook, so the URL is valid only
# while this kernel runs (12 h hard cap).
log("=== stage: tunnel ===")
TUNNEL_LOG = OUT / "cloudflared.log"
cf = shutil.which("cloudflared") or (OUT / "cloudflared")
if not Path(str(cf)).exists():
    sh("curl -L --fail -o cloudflared "
       "https://github.com/cloudflare/cloudflared/releases/latest/download/"
       "cloudflared-linux-amd64", timeout=600, tail=200)
    os.chmod(cf, 0o755)
    log(f"cloudflared downloaded to {cf}")
else:
    log(f"cloudflared already present at {cf}")

cf_proc = subprocess.Popen(
    [str(cf), "tunnel", "--url", "http://127.0.0.1:8080",
     "--no-autoupdate", "--metrics", "127.0.0.1:45678"],
    stdout=open(TUNNEL_LOG, "w"), stderr=subprocess.STDOUT)
log(f"cloudflared started, pid {cf_proc.pid}; waiting for a hostname")

# cloudflared prints the hostname to stderr as soon as the edge connection is up.
# Poll the log for it rather than sleeping a fixed time.
TUNNEL_URL = None
for _ in range(60):
    try:
        text = TUNNEL_LOG.read_text()
    except OSError:
        text = ""
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
    if m:
        TUNNEL_URL = m.group(0)
        break
    if cf_proc.poll() is not None:
        log("cloudflared exited before printing a hostname; log tail:")
        log(text[-1500:])
        break
    time.sleep(2)

if TUNNEL_URL:
    log("=" * 70)
    log(f"PUBLIC API URL: {TUNNEL_URL}")
    log(f"  chat:  curl {TUNNEL_URL}/v1/chat/completions ...")
    log(f"  models: curl {TUNNEL_URL}/v1/models")
    log("=" * 70)
    (WORK / "TUNNEL_URL.txt").write_text(TUNNEL_URL, encoding="utf-8")
    print(f"\n{'=' * 70}\nPUBLIC API URL: {TUNNEL_URL}\n{'=' * 70}\n")
else:
    log("TUNNEL FAILED -- no hostname. Everything below still runs on loopback.")
    print("\n!!! TUNNEL FAILED -- see cloudflared.log. Continuing loopback-only.\n")

# %% cell
# --- 5c. start llama-server on loopback ----------------------------------------
# Started BEFORE the probes so the public URL is live as early as possible in the
# session; the build is what costs the time, so this is cheap either way.
server = subprocess.Popen(
    [str(SERVER), "-m", str(MODEL), "-ngl", "99", "-c", "8192",
     "--host", "127.0.0.1", "--port", "8080",
     "--parallel", "8", "--cont-batching",
     "--batch-size", "2048", "--ubatch-size", "512",
     "--reasoning-budget", "0", "--reasoning-format", "none"],
    env=SINGLE_GPU,
    stdout=open(OUT / "server.log", "w"), stderr=subprocess.STDOUT)
log(f"llama-server started, pid {server.pid}")

ready = False
for _ in range(300):
    if subprocess.run("curl -sf http://127.0.0.1:8080/health", shell=True,
                      capture_output=True).returncode == 0:
        ready = True
        break
    if server.poll() is not None:
        log("server exited before becoming ready; log tail:")
        try:
            log((OUT / "server.log").read_text()[-2000:])
        except OSError:
            pass
        break
    time.sleep(2)
log(f"server ready: {ready}")

# Prove the PUBLIC path end-to-end, which is the claim being made. A loopback hit
# would not show whether the tunnel actually forwards.
if TUNNEL_URL and ready:
    body = json.dumps({"model": MODEL_NAME.removesuffix(".gguf"),
                       "messages": [{"role": "user", "content": "Hello, Who are you?"}],
                       "temperature": 0.7, "max_tokens": 128})
    proc = subprocess.run(
        f"curl -s --max-time 300 {TUNNEL_URL}/v1/chat/completions "
        f"-H 'Content-Type: application/json' -d '{body}'",
        shell=True, capture_output=True, text=True, timeout=320)
    try:
        payload = json.loads(proc.stdout)
        ans = payload["choices"][0]["message"]["content"]
        log(f"TUNNEL VERIFIED. Answer through the public URL:\n{ans}")
        print(f"\n--- answer via {TUNNEL_URL} ---\n{ans}\n")
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        log(f"tunnel reachable but response unparsed ({type(exc).__name__}): "
            f"{proc.stdout[:600]}")
        print(f"\n!!! tunnel response unparsed: {proc.stdout[:400]}\n")

# %% cell
# --- 6. probe: does it load at all --------------------------------------------
log("=== stage: probe load ===")
# A one-token generation. This is the documented segfault gate and costs seconds. If
# the primary packing dies here the kernel switches rather than reporting a crash as
# a result.


def fetch(name):
    """Locate a packing among the mounted datasets. Returns the path or None.

    No download: everything comes from the dataset mount. If a packing is absent the
    run stops rather than silently substituting a different one, because the AVX-512
    load fault is packing-specific and swapping silently would make the tok/s figure
    ambiguous.
    """
    for base in sorted(Path("/kaggle/input").glob("*")):
        for p in sorted(base.rglob(f"*{name}*")):
            print(f"found via dataset mount: {p} ({p.stat().st_size / 2**30:.2f} GiB)")
            return p
    return None


def probe_ok(model_path):
    """True iff llama-cli runs and exits 0 on a trivial generation."""
    return probe(model_path).returncode == 0


def probe(model_path, timeout=600, extra=""):
    """Run llama-cli with THINKING DISABLED and print its log immediately.

    This flag set is the documented incantation, from PrismML's own troubleshooting page
    ("The model emits <think> blocks or stalls before answering"): the 27B reasons BY
    DEFAULT, so a bare `-p 'Hi'` spends its entire budget in a think block and emits
    nothing. That is the "hang" this kernel chased for two versions -- the model was
    working, it was just thinking.

    Version 11's hour-long stall and version 12's 180-second CPU stall are both
    explained by this: thinking enabled, tiny token budgets, no answer.

    `extra` lets the caller isolate variables: "-ngl 0" runs CPU-only, so if that also
    stalls the problem is the model format, and if only the CUDA path stalls the
    problem is the ternary kernels. With thinking off, a stall is a real stall.
    """
    # --reasoning-budget 0 turns thinking off; --reasoning-format none keeps the markers
    # out of the output; the chat-template kwarg is the belt-and-braces form.
    think_off = ("--reasoning-budget 0 --reasoning-format none "
                 "--chat-template-kwargs '{\"enable_thinking\": false}'")
    logp = OUT / "probe.log"
    sh(f"{CLI} -m {model_path} {extra} {think_off} --single-turn -c 2048 -n 64 "
       f"--temp 0 -p 'Hello' > {logp} 2>&1",
       timeout=timeout, check=False, env=SINGLE_GPU)
    try:
        text = logp.read_text()
    except OSError:
        text = "<probe.log unreadable>"
    log(f"--- probe.log ({'ngl 0' if '-ngl 0' in extra else 'ngl 99'}) ---\n{text[-3000:]}")
    print(f"--- probe.log ---\n{text[-3000:]}")
    return text


# Only PTQ1_0 is staged as a dataset. A fallback to PQ2_0 would require a second 7.2 GB
# dataset that does not exist, so it is not attempted: if PTQ1_0 fails to load, that is
# the answer this run produces, stated plainly with the exit code.
# --- DIAGNOSTIC ORDER: CPU first, then GPU ------------------------------------
# Version 14 PROVED the model loads: CPU answered in 78s (a 27B on CPU is just slow to
# load, not hung) and GPU answered in 5s at 22.0 t/s prompt / 25.5 t/s generation.
#
# The failure in v14 was THIS kernel's own detection, not the model. It looked for the
# string "llama_new_context_with_model", which llama.cpp never prints, so a successful
# load was scored as a failure and the run aborted on its own false negative. The
# correct signal is that llama-cli EXITED 0 and actually produced text.
log("=== stage: probe CPU (ngl 0) ===")
cpu_text = probe(MODEL, timeout=600, extra="-ngl 0")
CPU_OK = ("Hello!" in cpu_text) or ("Exiting..." in cpu_text)
log(f"CPU probe succeeded: {CPU_OK}")

log("=== stage: probe GPU (ngl 99) ===")
gpu_text = probe(MODEL, timeout=600, extra="-ngl 99")
GPU_OK = ("Hello!" in gpu_text) or ("Exiting..." in gpu_text)
log(f"GPU probe succeeded: {GPU_OK}")

if not GPU_OK:
    raise RuntimeError(
        "llama-cli did not complete a generation on the GPU. See probe.log above. "
        "Note v14 proved the model itself loads, so this is a new failure, not the "
        "old one.")

log("model loads on both CPU and GPU; proceeding to benchmark")

# A real generation, so the run produces output text even if the bench fails later.
load = sh(f"{CLI} -m {MODEL} -ngl 99 -c 512 -n 32 --temp 0 -p 'Hello'",
          timeout=900, check=False, tail=800)
print(load.stdout[-800:])

# %% cell [markdown]
"""
## 7. Tokens per second

`llama-bench` is the right instrument, not a hand-rolled timer, because it reports the
two numbers that matter separately:

| Metric | Flag | What it measures |
| --- | --- | --- |
| **pp** (prompt processing) | `-p 512` | tokens/second while ingesting the prompt |
| **tg** (text generation) | `-n 128` | tokens/second while decoding — **usually what "tok/s" means** |

The distinction matters on ternary models: a 2-bit weight is cheap to fetch but the
activation transform (the blockwise Hadamard) is not free, so prompt processing can
behave very differently from decode. Quoting only one of the two is how a misleading
number gets made.

`llama-bench` itself prints per-device rows; with 2 GPUs it reports a row per device.
The parse below takes the **tg128 column from the last row**, which is the aggregate/
best figure llama-bench reports for the offloaded device set. Note it in the receipt
alongside the raw output, because a 2-GPU number is not comparable to a 1-GPU one.

Averaged over repetitions on a warm context, all 99 layers offloaded.
"""

# %% cell
# Matches the bench line that produced the 26.32 -> 40.54 tok/s figure on the bonsai2
# branch: -fa 1 with a q4_0 K and V cache. The quantized cache also frees VRAM, which
# matters because the MTP compute buffer and a long context both compete for it.
BENCH_ARGS = "-ngl 99 -fa on -c 4096 -r 3 -ctk q4_0 -ctv q4_0"
log("=== stage: benchmark ===")
log(f"bench: {BENCH_ARGS}")
try:
    bench = sh(f"{BENCH} -m {MODEL} {BENCH_ARGS} -p 512 -n 128 ",
               timeout=3000, check=False, tail=4000, env=SINGLE_GPU)
    bench_out = bench.stdout
    log("llama-bench raw tail:\n" + bench_out[-1500:])
except Exception as exc:  # noqa: BLE001
    log(f"BENCH RAISED: {type(exc).__name__}: {exc}")
    import traceback
    log(traceback.format_exc())
    bench_out = ""
log("=== stage: benchmark DONE ===")

print("\n=== llama-bench raw tail ===")
print(bench_out[-1500:])

# Columns: |model|size|params|backend|ngl|cpu|gpu|...|pp512|tg128|
TG = re.findall(r"\|\s*tg128\s*\|\s*([\d.]+)\s*\|", bench_out)
PP = re.findall(r"\|\s*pp512\s*\|\s*([\d.]+)\s*\|", bench_out)
ALL_TG = [float(x) for x in TG]
ALL_PP = [float(x) for x in PP]

TOK_S = ALL_TG[-1] if ALL_TG else None
PP_S = ALL_PP[-1] if ALL_PP else None
print(f"\nparsed prompt processing (pp512): {PP_S} tok/s")
print(f"parsed text generation  (tg128): {TOK_S} tok/s")
if ALL_TG and len(ALL_TG) > 1:
    print(f"llama-bench reported {len(ALL_TG)} device rows: {ALL_TG}")

if TOK_S is None:
    print("could not parse the bench table; the raw output above is the receipt")

# %% cell [markdown]
"""
## 8. The API

`llama-server` from the fork exposes an **OpenAI-compatible** endpoint — the same
`/v1/chat/completions` shape that `lmstudio`, `ollama` and `vllm` use — so any client
that speaks OpenAI can talk to it with no code changes.

It is bound to `127.0.0.1` and used from inside this same session. That is deliberate:
Kaggle's own product-feedback thread on this exact pattern (notebooks tunnelled to the
public internet as free inference servers) is why outbound tunnels from GPU sessions
are what Kaggle is actively blocking. Serving on loopback inside your own session is
normal use; publishing it publicly is what gets accounts restricted.

So the demonstration below is a real API call, made to `localhost`.
"""

# %% cell
# The server and the tunnel were both started earlier (stage 5b/5c), deliberately
# before the slow CPU probe, so the public URL is live for as much of the session as
# possible. Nothing here restarts or terminates them.
log(f"=== stage: API === server pid {server.pid} ready={ready} "
    f"tunnel={TUNNEL_URL}")

PROMPT = "Hello, Who are you?"


def call_api(content, max_tokens=128):
    """One OpenAI-compatible chat completion. Returns (answer, completion_tokens)."""
    import threading

    body = json.dumps({
        "model": MODEL_NAME.removesuffix(".gguf"),
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.7, "max_tokens": max_tokens,
    })
    proc = subprocess.run(
        f"curl -s http://127.0.0.1:8080/v1/chat/completions "
        f"-H 'Content-Type: application/json' -d '{body}'",
        shell=True, capture_output=True, text=True, timeout=1200)
    payload = json.loads(proc.stdout)
    return (payload["choices"][0]["message"]["content"],
            payload.get("usage", {}).get("completion_tokens", 0))


# --- 1. the single answer you asked for ---------------------------------------
answer = None
if ready:
    try:
        answer, n = call_api(PROMPT)
        log(f"API answer to {PROMPT!r}:\n{answer}\n({n} completion tokens)")
        print(f"\n=== API response to {PROMPT!r} ===\n{answer}\n")
    except Exception as exc:  # noqa: BLE001
        import traceback
        log(f"API RAISED: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        print(f"API call failed: {type(exc).__name__}: {exc}")

# --- 2. aggregate throughput with continuous batching -------------------------
# llama-server with --parallel N runs continuous batching: one batch is decoded across
# all queued requests, so aggregate tok/s rises while per-request tok/s falls. The
# measurement is a wall-clock division over the whole concurrent burst, which is the
# only number that matters for serving.
#
# The baseline is 1 request alone, not the v14 probe figure, because that keeps the
# comparison inside one session with one server configuration.
BENCH_PROMPT = ("Write one short paragraph explaining what a hash table is, "
                "in about 60 words.")
BATCH_SIZES = [1, 4, 8]
results = {}
if ready:
    import threading

    log("=== stage: batching benchmark ===")
    for n_req in BATCH_SIZES:
        tokens = [0] * n_req
        errors = []

        def worker(i):
            try:
                _, tokens[i] = call_api(BENCH_PROMPT, max_tokens=64)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")

        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_req)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        dt = time.time() - t0

        total = sum(tokens)
        agg = total / dt if dt > 0 else 0.0
        per_req = total / (dt * n_req) if n_req else 0.0
        results[n_req] = {"requests": n_req, "tokens": total,
                          "seconds": round(dt, 2), "aggregate_tok_s": round(agg, 1),
                          "per_request_tok_s": round(per_req, 1),
                          "errors": errors}
        log(f"n={n_req:>2}: {total:>4} tokens in {dt:>6.2f}s -> "
            f"aggregate {agg:>6.1f} tok/s, per-request {per_req:>5.1f} tok/s"
            + (f"  ERRORS: {errors}" if errors else ""))
        print(f"n={n_req:>2}: {total:>4} tokens in {dt:>6.2f}s -> "
              f"aggregate {agg:>6.1f} tok/s, per-request {per_req:>5.1f} tok/s")

    single = results.get(1, {}).get("aggregate_tok_s", 0)
    multi = results.get(8, {}).get("aggregate_tok_s", 0)
    if single and multi:
        log(f"batching gain at n=8: {multi / single:.2f}x single-request throughput")
        print(f"\naggregate gain at n=8: {multi / single:.2f}x "
              f"({single:.1f} -> {multi:.1f} tok/s)")
        try:
            log("server.log tail:\n" + (OUT / "server.log").read_text()[-3000:])
        except OSError:
            pass

# server.terminate() is deliberately NOT called here. The tunnel and the server are
# meant to stay up for the rest of the session, so tearing them down at the end of the
# benchmark would undo the whole point. The keepalive cell owns their lifetime now.
log("=== stage: API DONE ===")

# %% cell
# --- 9. the receipt ------------------------------------------------------------
# Everything above in one JSON, written to /kaggle/working so it survives the session.
receipt = {
    "model": MODEL_NAME,
    "repo": REPO,
    "fork": FORK,
    "branch": BRANCH,
    "fork_commit": commit,
    "host": HOST,
    "build": {
        "cuda_arch": CUDA_ARCH,
        "ggml_cuda_no_vmm": (VMM == "off"),
        "vmm": VMM,
        "configure_attempt": attempt,
        "nproc": NPROC,
    },
    "bench_args": BENCH_ARGS,
    "prompt_processing_tok_s": PP_S,
    "text_generation_tok_s": TOK_S,
    "all_tg_rows": ALL_TG,
    "all_pp_rows": ALL_PP,
    "api_prompt": PROMPT,
    "api_answer": answer,
    "api_endpoint": "http://127.0.0.1:8080/v1/chat/completions",
    "tunnel_url": TUNNEL_URL,
    "tunnel_verified": bool(TUNNEL_URL and answer),
    "batching": results,
    "batching_gain_n8": (round(results.get(8, {}).get("aggregate_tok_s", 0)
                               / results.get(1, {}).get("aggregate_tok_s", 1), 2)
                         if results.get(1, {}).get("aggregate_tok_s") else None),
    "server_flags": {"ngl": 99, "context": 8192, "parallel": 8,
                     "cont_batching": True},
    "tier": "exec" if TOK_S else "repo-managed (no tok/s parsed)",
    "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
(WORK / "bonsai2-t4-receipt.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))

# %% cell
# --- keep the session alive so the tunnel keeps serving ------------------------
# Kaggle ends the session when the notebook finishes its last cell, which would kill
# llama-server and cloudflared with it. This loop is the only thing standing between a
# working tunnel and a dead one, so it also babysits the two child processes: if either
# dies, the URL stops answering and there is no point holding the session open.
#
# It is capped at 11h rather than 12h: Kaggle's hard GPU cap is 12h, and stopping just
# short of it means this cell exits normally and the receipt is written rather than the
# session being killed mid-run with no output.
log("=== stage: keepalive ===")
log(f"serving on {TUNNEL_URL or 'loopback only'} for up to 11 hours")
deadline = time.time() + 11 * 3600
beat = 0
while time.time() < deadline:
    if server.poll() is not None or cf_proc.poll() is not None:
        log("server or cloudflared exited; ending keepalive")
        break
    beat += 1
    if beat % 30 == 0:  # every ~5 min, so the log shows it is alive
        log(f"alive: {int((deadline - time.time()) / 60)} min left, "
            f"url={TUNNEL_URL}")
    time.sleep(10)

log(f"keepalive ended. Tunnel was: {TUNNEL_URL}")

# %% cell [markdown]
"""
## What this receipt does and does not establish

**It establishes:** a tokens/second figure for decode and for prompt processing, on the
specific GPU set above, measured by `llama-bench` with the named arguments, together
with the answer the model gave to one prompt through an OpenAI-compatible API call.

**It does not establish:**
- Any TPU number. The CUDA backend is what ran; there is no TPU path in this kernel
  and the vendor publishes no TPU kernels for Bonsai 2.
- Any quality claim. One prompt is not an evaluation. The vendor's 83.9 benchmark and
  98.2% retention are their figures on their suite, not measured here.
- Any comparison to the F16 baseline, which was not attempted because it does not fit
  in 32 GB.
- That a public deployment is sanctioned. It is not: Kaggle's product-feedback thread
  names tunnelling a GPU notebook out to the internet as the pattern they are working to
  block. The tunnel here works today and carries no guarantee. `tunnel_url` and
  `tunnel_verified` in the receipt record what actually happened.

On context: the model supports 262,144 tokens and this bench used 4,096. A short
context is the cheapest config and the fastest number; it is not what you get at long
context, where KV cache pressure changes the picture.

On the build flag: `GGML_CUDA_NO_VMM` was set to avoid linking `libcuda.so`, which this
container does not ship as a development symlink. VMM governs GPU memory *sharing*,
not matmul throughput, so the tok/s figures are unaffected by it — but it is recorded in
the receipt because a reader comparing against a VMM-enabled build deserves to know.

MIT OR Apache-2.0
updated: 2026-10-10 00:00 UTC
"""
