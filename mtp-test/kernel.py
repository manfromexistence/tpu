"""OUT OF SCOPE for the TPU track as of 2026-10-20: this is a GPU (Kaggle T4)
kernel. It is kept for the record and its numbers are [sourced] references,
not results of this project -- see tpu/README.md section 6 and AGENTS.md
section 2 rule 7. Do not run it without the user's explicit go-ahead.

Try Qwen3.8-27B's built-in MTP head on the Bonsai 2 27B GGUF.

The claim being tested: `--spec-type draft-mtp --spec-draft-n-max 8` gives 50.44 tok/s
at 48.3% acceptance on a reference Qwen3.8-27B GGUF. That flag uses the model's OWN
multi-token-prediction head, which llama.cpp reads from the model file itself
(n_layer_nextn), so no separate draft model is needed and nothing new is downloaded.

The honest uncertainty, stated before the run: the reference result was measured on
Qwen's own Qwen3.8-27B GGUF. This notebook runs **Ternary Bonsai 2 27B (PTQ1_0)**,
a 5.54 GB ternary quantization of the same base, produced by PrismML. Whether the MTP
head survived that quantization is not documented anywhere I could find. So the
outcomes and what each means:

  * llama.cpp refuses with "model does not have a MTP head" -> Bonsai 2's PTQ1_0 ships
    without one, and the 7.0 GB grafted-MTP files are the only path. Definitive.
  * It runs -> the first tok/s figure for MTP speculation on Bonsai 2, which nobody
    has published.

Either way this is a real measurement, and it costs about six minutes because the build
cache and the model dataset are already mounted.

As of 2026-10-20 the kernel runs at **-c 8192** (the 8k window this track's target
is written against) and measures the **no-speculation baseline in the same session**
before the MTP run, so the speedup is same-file, same-card, same-context rather than a
number quoted from another machine.
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
LOG = RUN / "mtp.log"
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
    say(out[-1200:])
    say(f"[exit {p.returncode}]")
    if check and p.returncode != 0:
        raise RuntimeError(f"failed ({p.returncode}): {cmd}")
    return p


say("=" * 70)
say("=== bonsai2 + built-in MTP head (--spec-type draft-mtp) ===")

# --- build cache --------------------------------------------------------------
BUILD_CACHE = "bonsai2-llamacpp-build"


def find_cache():
    for depth in (1, 2, 3, 4):
        for suffix in ("bin/llama-cli", "llama-cli"):
            hits = [p for p in Path("/kaggle/input").glob("*/" * depth + suffix)
                    if BUILD_CACHE in str(p)]
            if hits:
                hits.sort(key=lambda p: len(str(p)))
                return hits[0].parent
    return None


cached = find_cache()
CLI = Path("llama.cpp/build/bin/llama-cli").resolve()
if cached and not (CLI.exists() and os.access(CLI, os.X_OK)):
    say(f"restoring cached build from {cached}")
    target = Path("llama.cpp/build/bin")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for src in sorted(cached.iterdir()):
        if src.is_file():
            shutil.copy2(src, target / src.name)
    for versioned in sorted(target.glob("*.so.*.*.*")):
        base, _, rest = str(versioned.name).partition(".so.")
        for alias in (f"{base}.so.{rest.split('.')[0]}", f"{base}.so"):
            link = target / alias
            if not link.exists():
                link.symlink_to(versioned.name)
    for exe in ("llama-cli", "llama-server", "llama-bench"):
        p = target / exe
        if p.exists():
            os.chmod(p, 0o755)
    say(f"cache restored; llama-cli usable: {CLI.exists() and os.access(CLI, os.X_OK)}")

if not (CLI.exists() and os.access(CLI, os.X_OK)):
    raise RuntimeError("no usable llama-cli")

# --- model --------------------------------------------------------------------
MODEL = None
for base in sorted(Path("/kaggle/input").glob("*")):
    for p in sorted(base.rglob("*PTQ1_0*.gguf")):
        MODEL = p
        break
    if MODEL:
        break
if MODEL is None:
    raise RuntimeError("Bonsai PTQ1_0 not found")

LOCAL = RUN / MODEL.name
if not (LOCAL.exists() and LOCAL.stat().st_size == MODEL.stat().st_size):
    shutil.copyfile(MODEL, LOCAL)
say(f"model: {LOCAL} ({LOCAL.stat().st_size / 2**30:.2f} GiB)")

# --- probe the file for an MTP head ------------------------------------------
# Does this GGUF even carry nextn layers? Reading it out of the model is the honest way
# to know, rather than inferring from Qwen's model card for a different file.
say("=== stage: inspect for MTP head ===")
probe = sh(f"{CLI} -m {LOCAL} -ngl 99 -c 512 -n 4 --temp 0 -p 'Hi' "
           f"--single-turn --reasoning-budget 0 --reasoning-format none "
           f"2>&1 | grep -iE 'mtp|nextn|block_count|model size' | head -20",
           timeout=600, check=False)
head_txt = probe.stdout
say(head_txt or "(no mtp/nextn line found in the load log)")

# --- two runs, same context, same session ------------------------------------
# The baseline is measured HERE, not quoted from another session: the 25.5 tok/s
# in the reference came from a different kernel, a different card and a
# different context, and the whole question is what MTP does to THIS file.
CTX = 8192
NTOK = 128
PROMPT = "Hello, Who are you?"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")


def run_stage(label, extra):
    say(f"=== stage: {label} ===")
    cmd = (f"{CLI} -m {LOCAL} -ngl 99 -c {CTX} -n {NTOK} --temp 0 "
           f"-p '{PROMPT}' --single-turn "
           f"--reasoning-budget 0 --reasoning-format none "
           f"{extra}").strip()
    say(cmd)
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=2400, check=False, env=env)
    elapsed = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    log.write(out[-20000:] + "\n")
    say(out[-3000:])
    return out, elapsed, p.returncode


def parse(out):
    accept = re.search(r"draft acceptance rate = ([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated", out)
    tok = re.search(r"Generation:\s*([\d.]+)\s*t/s", out)
    pp = re.search(r"Prompt:\s*([\d.]+)\s*t/s", out)
    no_head = "mtp" in out.lower() and ("does not have" in out.lower()
                                        or "no mtp" in out.lower()
                                        or "requires" in out.lower())
    return {
        "acceptance": float(accept.group(1)) if accept else None,
        "accepted": int(accept.group(2)) if accept else None,
        "generated": int(accept.group(3)) if accept else None,
        "gen_tok_s": float(tok.group(1)) if tok else None,
        "prompt_tok_s": float(pp.group(1)) if pp else None,
        "no_head": no_head,
    }


base_out, base_elapsed, base_rc = run_stage("baseline (no speculation)", "")
base = parse(base_out)
spec_out, spec_elapsed, spec_rc = run_stage("run --spec-type draft-mtp --spec-draft-n-max 8",
                                             "--spec-type draft-mtp --spec-draft-n-max 8")
spec = parse(spec_out)

speedup = (round(spec["gen_tok_s"] / base["gen_tok_s"], 2)
           if spec["gen_tok_s"] and base["gen_tok_s"] else None)

receipt = {
    "platform": "kaggle",
    "accelerator": "T4 (1x)",
    "model": LOCAL.name,
    "model_bytes": LOCAL.stat().st_size,
    "context_tokens": CTX,
    "generated_requested": NTOK,
    "prompt": PROMPT,
    "baseline": {
        "spec_type": None,
        "generation_tok_s": base["gen_tok_s"],
        "prompt_tok_s": base["prompt_tok_s"],
        "exit_code": base_rc,
        "wall_seconds": round(base_elapsed, 1),
    },
    "draft_mtp": {
        "spec_type": "draft-mtp",
        "spec_draft_n_max": 8,
        "generation_tok_s": spec["gen_tok_s"],
        "prompt_tok_s": spec["prompt_tok_s"],
        "draft_acceptance": spec["acceptance"],
        "accepted_tokens": spec["accepted"],
        "generated_tokens": spec["generated"],
        "exit_code": spec_rc,
        "wall_seconds": round(spec_elapsed, 1),
    },
    "file_has_mtp_head": bool(spec["acceptance"]) or (not spec["no_head"] and bool(spec["gen_tok_s"])),
    "rejected_no_mtp_head": spec["no_head"],
    "measured_speedup_vs_same_session_baseline": speedup,
    "reference_not_measured_here": "pich, RTX PRO 4000 24GB, Qwen3.8-27B GGUF: 50.44 tok/s @ 48.3% acceptance",
    "tier": "exec",
    "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
Path(WORK / "mtp-receipt.json").write_text(json.dumps(receipt, indent=2))
say("=== verdict ===")
say(json.dumps(receipt, indent=2))
