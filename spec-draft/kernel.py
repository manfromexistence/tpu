"""Test Qwen3.5-0.8B (Q4_0) as a speculative draft for Bonsai 2 27B on a Kaggle T4.

The question this answers, and only this: does the draft pair load, and what does it cost
per token? Two outcomes, both informative:

  1. llama.cpp refuses -- its speculative.cpp validates vocabs token-by-token and names
     the first mismatched token. That is a definitive, cheap "no".
  2. It runs -- then the two numbers that matter are `draft acceptance rate` and tok/s,
     and they are compared against the 25.5 tok/s measured without a draft.

The draft is downloaded from HuggingFace rather than staged as a dataset: 537 MB, and
Kaggle's network is 1-2 GB/s, so it costs roughly 10 seconds. A dataset mount would add
a permanent artefact for no saving.

Q4_0 rather than BF16: 537 MB against 1485 MB, and the draft runs in the same T4 as the
5.54 GB target. Memory is not the constraint (16 GB card, ~6.3 GB used), but a smaller
draft is a faster draft, and draft speed is the whole point.
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
LOG = RUN / "spec.log"
log = open(LOG, "a", buffering=1)


def say(m):
    print(m)
    log.write(m.rstrip() + "\n")


def sh(cmd, timeout=None, cwd=None, check=True):
    say(f"\n$ {cmd}")
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout, check=False, cwd=cwd)
    out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
    log.write(out[-20000:] + "\n")
    say(out[-1500:])
    say(f"[exit {p.returncode}]")
    if check and p.returncode != 0:
        raise RuntimeError(f"failed ({p.returncode}): {cmd}")
    return p


say("=" * 70)
say("=== bonsai2 + qwen3.5-0.8b-q4_0 as speculative draft ===")

# --- 1. build cache ------------------------------------------------------------
# /kaggle/input is mounted read-only, so the cached binaries are COPIED out and chmod'd
# rather than symlinked -- a symlink there cannot be made executable. The soname links
# are recreated too, because the dataset upload dropped them.
BUILD_CACHE = "bonsai2-llamacpp-build"


def find_cache():
    """Locate the build cache's bin/llama-cli or root llama-cli under /kaggle/input.

    Two layouts are tried because the dataset was uploaded with `-r zip`, which strips
    the source directory prefix: the binaries land at the dataset ROOT, not under bin/.
    The pattern is built as a STRING and passed to glob as a string -- an earlier
    version built a list of patterns and passed the list, which raised TypeError.
    """
    root = Path("/kaggle/input")
    for depth in (1, 2, 3, 4):
        for suffix in ("bin/llama-cli", "llama-cli"):
            pat = "*/" * depth + suffix
            hits = [p for p in root.glob(pat) if BUILD_CACHE in str(p)]
            if hits:
                hits.sort(key=lambda p: len(str(p)))
                return hits[0].parent
    return None


cached = find_cache()
CLI = Path("llama.cpp/build/bin/llama-cli").resolve()
if cached and not CLI.exists():
    say(f"found cached build at {cached}")
    target = Path("llama.cpp/build/bin")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(cached.iterdir()):
        if src.is_file():
            shutil.copy2(src, target / src.name)
    for versioned in sorted(target.glob("*.so.*.*.*")):
        base, _, rest = str(versioned.name).partition(".so.")
        major = rest.split(".")[0]
        for alias in (f"{base}.so.{major}", f"{base}.so"):
            link = target / alias
            if not link.exists():
                link.symlink_to(versioned.name)
    for exe in ("llama-cli", "llama-server", "llama-bench"):
        p = target / exe
        if p.exists():
            os.chmod(p, 0o755)
    say(f"restored cache; llama-cli executable: {CLI.exists() and os.access(CLI, os.X_OK)}")
else:
    say("no cached build found")

if not (CLI.exists() and os.access(CLI, os.X_OK)):
    say("=== stage: build from source ===")
    if Path("llama.cpp").exists():
        shutil.rmtree("llama.cpp")
    sh("git clone --depth 1 --branch prism "
       "https://github.com/PrismML-Eng/llama.cpp.git", timeout=900)
    sh("cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 "
       "-DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_CUDA_NO_VMM=ON",
       cwd="llama.cpp", timeout=1200)
    sh("cmake --build build --target llama-cli -j 4", cwd="llama.cpp", timeout=7200)

say(f"llama-cli: {CLI}")

# --- 2. target model ----------------------------------------------------------
MODEL = None
for base in sorted(Path("/kaggle/input").glob("*")):
    for p in sorted(base.rglob("*PTQ1_0*.gguf")):
        MODEL = p
        break
    if MODEL:
        break
if MODEL is None:
    raise RuntimeError("Bonsai PTQ1_0 not found in any /kaggle/input path")

LOCAL_MODEL = RUN / MODEL.name
if not (LOCAL_MODEL.exists() and LOCAL_MODEL.stat().st_size == MODEL.stat().st_size):
    say(f"copying {MODEL} to local disk ({MODEL.stat().st_size / 2**30:.2f} GiB)")
    shutil.copyfile(MODEL, LOCAL_MODEL)
say(f"target: {LOCAL_MODEL} ({LOCAL_MODEL.stat().st_size / 2**30:.2f} GiB)")

# --- 3. draft model -----------------------------------------------------------
DRAFT = RUN / "Qwen3.5-0.8B-Q4_0.gguf"
if not (DRAFT.exists() and DRAFT.stat().st_size == 563036064):
    say("=== stage: download draft ===")
    # curl, not `hf download`. The hf client reported "Downloaded" with exit 0 while
    # leaving a 0-byte file and a live .lock behind -- it had been warned about
    # unauthenticated HF requests, and a silent truncation on a 537 MB file is exactly
    # the failure that leaves. curl -L --fail verifies the transfer, and the byte count
    # is checked immediately after rather than trusted.
    url = ("https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/"
           "Qwen3.5-0.8B-Q4_0.gguf")
    ok = False
    for attempt in range(1, 6):
        if DRAFT.exists():
            DRAFT.unlink()
        sh(f"curl -L --fail --retry 5 --retry-delay 3 --retry-all-errors "
           f"-o {DRAFT} {url}", timeout=2400, check=False)
        if DRAFT.exists() and DRAFT.stat().st_size == 563036064:
            say(f"draft verified on attempt {attempt}")
            ok = True
            break
        got = DRAFT.stat().st_size if DRAFT.exists() else 0
        say(f"attempt {attempt}: {got} bytes, expected 563036064 -- retrying")
    if not ok:
        raise RuntimeError("draft download failed after 5 attempts")
size = DRAFT.stat().st_size
say(f"draft: {DRAFT} ({size / 2**20:.0f} MB)")
if size != 563036064:
    raise RuntimeError(f"draft size {size} != 563036064")

# --- 4. discover the speculative flags ----------------------------------------
# The fork is a moving target; --model-draft alone may not be enough, and --spec-type
# naming has changed across versions. Reading the help and building the command from
# what is actually supported beats guessing and burning a run on a bad flag.
say("=== stage: flag discovery ===")
# Line-scan the help. A fancier block-level parser was tried and reverted: it captured
# 7 flags out of hundreds and reported --spec-type absent, when the simple scan has
# always found it. Simple and complete beats clever and partial.
help_text = sh(f"{CLI} --help 2>&1", timeout=120, check=False).stdout or ""
have = set()
for line in help_text.splitlines():
    m = re.match(r"\s*(--[a-z0-9-]+)", line)
    if m:
        have.add(m.group(1))
say(f"parsed {len(have)} flags")

# This fork moves the whole speculative-decode surface into a --spec-* namespace. The
# draft MODEL path is --spec-draft-model, not --model-draft and not --draft:
#   --spec-draft-model PATH   the draft model
#   --spec-draft-ngl N        layers to offload for the draft
#   --spec-draft-n-max N      tokens drafted per round
#   --spec-type draft-simple  the plain separate-draft-model path
# v11's log printed all 143 flags and listed --spec-draft-model explicitly; that is the
# source for this, not a guess.
draft_flags = [f for f in ("--spec-draft-model", "--spec-draft-hf") if f in have]
say(f"draft path flag     : {draft_flags or 'NONE FOUND'}")
say(f"draft alias (dead)  : {'--draft' in have}  <- deliberately NOT used")
say(f"spec-type present   : {'--spec-type' in have}")
say(f"spec-draft-n-max    : {'--spec-draft-n-max' in have}")
say(f"spec-draft-ngl      : {'--spec-draft-ngl' in have}")

if not draft_flags:
    raise RuntimeError(
        "this llama-cli build exposes no draft model path flag. Running anyway would "
        "attach no draft and report the baseline. Available --spec-draft-* flags: "
        f"{sorted(f for f in have if f.startswith('--spec-draft'))}")

cmd = [str(CLI), "-m", str(LOCAL_MODEL), "-ngl", "99", "-c", "2048",
       "-n", "64", "--temp", "0", "-p", "Hello, Who are you?",
       "--single-turn", "--reasoning-budget", "0",
       "--reasoning-format", "none",
       draft_flags[0], str(DRAFT)]
if "--spec-draft-ngl" in have:
    cmd += ["--spec-draft-ngl", "99"]
if "--spec-type" in have:
    # The fork renamed the generic draft type: `--spec-type draft` is rejected with
    # "unknown speculative type: draft". The accepted set, printed by the binary
    # itself on that failure, is:
    #   none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,
    #   ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache
    # draft-simple is the plain separate-draft-model path this experiment needs.
    cmd += ["--spec-type", "draft-simple"]
if "--spec-draft-n-max" in have:
    cmd += ["--spec-draft-n-max", "4"]

say("\nrunning:")
say(" ".join(cmd))
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
t0 = time.time()
p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                   check=False, env=env)
elapsed = time.time() - t0
log.write((p.stdout or "")[-20000:] + "\n")
log.write((p.stderr or "")[-20000:] + "\n")
say((p.stdout or "")[-3000:])
say((p.stderr or "")[-1500:])

# --- 5. verdict ---------------------------------------------------------------
out = (p.stdout or "") + (p.stderr or "")
accept = re.search(r"draft acceptance rate = ([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\)", out)
tok = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
pp = re.search(r"Prompt:\s*([\d.]+)\s*t/s", out)
refuse = re.search(r"vocab.*must match|vocab_type|token \d+ content differs", out, re.I)

receipt = {
    "platform": "kaggle",
    "accelerator": "T4 (1x)",
    "target": LOCAL_MODEL.name,
    "target_bytes": LOCAL_MODEL.stat().st_size,
    "draft": DRAFT.name,
    "draft_bytes": size,
    "draft_quant": "Q4_0",
    "vocab_check_refused": bool(refuse),
    "refuse_message": refuse.group(0) if refuse else None,
    "draft_acceptance": float(accept.group(1)) if accept else None,
    "accepted_tokens": int(accept.group(2)) if accept else None,
    "generated_tokens": int(accept.group(3)) if accept else None,
    "generation_tok_s": float(tok.group(1)) if tok else None,
    "prompt_tok_s": float(pp.group(1)) if pp else None,
    "baseline_tok_s_no_draft": 25.5,
    "speedup_vs_baseline": (round(float(tok.group(1)) / 25.5, 2)
                            if tok else None),
    "exit_code": p.returncode,
    "wall_seconds": round(elapsed, 1),
    "tier": "exec",
    "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
Path(WORK / "spec-receipt.json").write_text(json.dumps(receipt, indent=2))
say("=== verdict ===")
say(json.dumps(receipt, indent=2))
