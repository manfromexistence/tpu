"""Build/run sudoingX's pr-ptq1-mmv kernel branch on the T4, on the ORIGINAL file.

The claim being tested, from sudoingX/bonsai2-small-gpu, measured on an RTX 3060 12GB
with llama-bench r=3 on Ternary-Bonsai-2-27B-PTQ1_0.gguf -- the same file this hub
already has mounted as a Kaggle dataset:

    build                            tg128      pp512
    PrismML fork, prism-b10685       26.32      269.6
    pr-ptq1-mmv branch (bonsai2)     40.47      267.8

Same weights, same file, same architecture. The speedup is a memory-layout change in
the PTQ1_0 mat-vec path (planar-transposed activations: ~36 L1 wavefronts per
instruction down to 1), not a numerical approximation.

Why the prebuilt is tried FIRST: it is ~515 MB and 30 seconds. Building from source on
Kaggle's 4 cores costs ~32 minutes, which is worth avoiding if the binaries run.

The reason a fallback is needed at all: the published tarballs are built for sm86-sm89
(RTX 30/40) and sm120 (RTX 50). This is a T4, sm75. If the tarball carries no PTX for
sm75, the CUDA driver cannot JIT it and the run fails with "no kernel image is
available for execution on the device". That is a build-target question, not a defect
in the branch.

Both paths are recorded, so the receipt says which one produced the number.
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
say("=== bonsai2 kernel branch (sudoingX pr-ptq1-mmv) on Kaggle T4 ===")
say(f"cpus {os.cpu_count()} | disk {shutil.disk_usage(WORK).free / 2**30:.1f} GiB free")

# --- model --------------------------------------------------------------------
MODEL = None
for base in sorted(Path("/kaggle/input").glob("*")):
    for p in sorted(base.rglob("*PTQ1_0*.gguf")):
        MODEL = p
        break
    if MODEL:
        break
if MODEL is None:
    raise RuntimeError("Bonsai PTQ1_0 not found in /kaggle/input")

LOCAL = RUN / MODEL.name
if not (LOCAL.exists() and LOCAL.stat().st_size == MODEL.stat().st_size):
    say(f"copying {MODEL.name} to local disk")
    shutil.copyfile(MODEL, LOCAL)
say(f"model: {LOCAL} ({LOCAL.stat().st_size / 2**30:.2f} GiB)")

BRANCH = "bonsai2"
BENCH = None
BINARY_SOURCE = None

# --- 1. prebuilt tarball ------------------------------------------------------
# The published tarballs name their target sm: sm86-sm89 covers RTX 30/40, sm120 covers
# RTX 50. A T4 is sm75. CUDA forward-compat via PTX may still work, so this is tried
# rather than assumed.
say("=== stage: try prebuilt binaries ===")
TB = RUN / "bonsai2-bin.tar.gz"
if not TB.exists():
    sh("curl -L --fail --retry 3 -o " + str(TB) + " https://github.com/sudoingX/"
       "bonsai2-small-gpu/releases/download/r1/"
       "bonsai2-small-gpu-linux-x64-cuda12.4-sm86-sm89-285542d.tar.gz",
       timeout=1800, check=False)
if TB.exists() and TB.stat().st_size > 100000000:
    say(f"tarball: {TB.stat().st_size / 2**20:.0f} MB; extracting")
    sh(f"tar -xzf {TB} -C {RUN}", timeout=600, check=False)
    for cand in RUN.rglob("llama-bench"):
        BENCH = cand.parent / "llama-bench"
        BINARY_SOURCE = "prebuilt tarball (sm86-sm89)"
        break
    if BENCH:
        say(f"prebuilt llama-bench: {BENCH}")
else:
    say("prebuilt tarball unavailable")

# --- 2. source build ----------------------------------------------------------
if BENCH is None:
    say("=== stage: building from source (branch bonsai2) ===")
    if Path("llama.cpp").exists():
        shutil.rmtree("llama.cpp")
    sh(f"git clone --depth 1 --branch {BRANCH} "
       f"https://github.com/sudoingX/llama.cpp.git", timeout=900)
    # sm75 is pinned because prebuilt binaries target sm86+: on a T4 the driver cannot
    # JIT an sm86 cubin, so the source must be told what this silicon actually is.
    sh("cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 "
       "-DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_CUDA_NO_VMM=ON",
       cwd="llama.cpp", timeout=1200)
    sh("cmake --build build --target llama-bench llama-cli -j 4",
       cwd="llama.cpp", timeout=7200)
    BENCH = Path("llama.cpp/build/bin/llama-bench").resolve()
    BINARY_SOURCE = f"source build, branch {BRANCH}, sm75"

if not (BENCH.exists() and os.access(BENCH, os.X_OK)):
    for exe in ("llama-bench", "llama-cli"):
        p = BENCH.parent / exe
        if p.exists():
            os.chmod(p, 0o755)
    for versioned in BENCH.parent.glob("*.so.*.*.*"):
        base, _, rest = str(versioned.name).partition(".so.")
        for alias in (f"{base}.so.{rest.split('.')[0]}", f"{base}.so"):
            link = BENCH.parent / alias
            if not link.exists():
                link.symlink_to(versioned.name)

say(f"llama-bench: {BENCH} ({BINARY_SOURCE})")
if not (BENCH.exists() and os.access(BENCH, os.X_OK)):
    raise RuntimeError("no usable llama-bench")

# --- 3. benchmark -------------------------------------------------------------
# Flags match sudoingX's own measurement exactly: baseline is r=3, original file, no
# MTP, no --reasoning flags (llama-bench does raw decode). Comparing against his
# numbers only means anything if the flags match.
say("=== stage: llama-bench (original PTQ1_0 file, no MTP) ===")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
p = sh(f"{BENCH} -m {LOCAL} -p 512 -n 128 -r 3 -ngl 99",
       timeout=2400, check=False, env=env, tail=4000)
out = p.stdout or ""

tg = re.findall(r"\|\s*tg128\s*\|\s*([\d.]+)\s*\|", out)
pp = re.findall(r"\|\s*pp512\s*\|\s*([\d.]+)\s*\|", out)

receipt = {
    "platform": "kaggle",
    "accelerator": "T4 (1x, 15360 MiB, sm75)",
    "model": LOCAL.name,
    "model_bytes": LOCAL.stat().st_size,
    "binary_source": BINARY_SOURCE,
    "branch": BRANCH,
    "baseline_prism_b10685": {"tg128": 26.32, "pp512": 269.6},
    "reference_rtx3060_12gb": {"tg128": 40.47, "pp512": 267.8},
    "measured_tg128": [float(x) for x in tg],
    "measured_pp512": [float(x) for x in pp],
    "speedup_vs_baseline": (round(float(tg[-1]) / 26.32, 2) if tg else None),
    "exit_code": p.returncode,
    "no_kernel_image": "no kernel image" in out.lower(),
    "tier": "exec",
    "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
Path(WORK / "kernel-branch-receipt.json").write_text(json.dumps(receipt, indent=2))
say("=== verdict ===")
say(json.dumps(receipt, indent=2))

if receipt["no_kernel_image"]:
    say("\nRESULT: the binaries carry no sm75 image. The source build is the only "
        "path on a T4; re-run with the prebuilt step skipped.")
