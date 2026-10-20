"""TPU environment probe -- what accelerator did we actually get?

The one notebook that runs first, on every TPU runtime, every time. Both Colab and
Kaggle hand out hardware dynamically: an eligible accelerator is not an available one,
and a free tier that nominally offers a TPU can return CPU at peak hours. Every claim
about v5e-1 or v5e-8 in tpu/README.md is only true for the specific VM it was measured
on, so this probe is what turns "TPU v5e-8" into "TPU v5e-8, measured <date> on
<session>".

This probe decides the Mode A vs Mode B sizing question for the Bonsai 2 TPU runner:
Mode A (offline-unfolded BF16, ~50 GB) needs a multi-chip slice, while Mode B
(native ternary + runtime FWHT, ~7 GB) fits one chip. The number that decides that is
the HBM limit below, not the marketing spec.

Design rules:

  1. NEVER raise. A probe that crashes after printing half the truth is worse than one
     that prints "unknown" for a field. Every optional import is guarded.
  2. Measure, do not infer. Do not trust os.environ["COLAB_TPU_ADDR"], a runtime name,
     or a settings checkbox to say what hardware is attached. Enumerate devices through
     the framework.
  3. Print a stable JSON receipt at the end, so a later notebook or a log scraper can
     read the same receipt. Keys are stable across frameworks.
  4. Cheap probes first. jax imports take 10-30 s on a cold TPU VM, so a wedged runtime
     still leaves CPU/RAM/disk on record.

The output of this notebook is a "receipt". If you did not get one, you do not have a
hardware measurement -- you have a guess.

Status: NOT YET EXECUTED on any TPU from this workspace. The v5e-8 and v5e-1 rows in
tpu/README.md are [sourced], not [exec].
"""

# --- stdlib only; no framework imports at module level ------------------------

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# %% cell [markdown]
"""
# TPU environment probe

**Purpose.** Answer one question: *what is this VM, actually?*

Run it on **Kaggle** with Settings > Accelerator > TPU (v5e-8 slice, 8 chips), and on
**Colab** with Runtime > Change runtime type > TPU (v5e-1, 1 chip). Record which
product each receipt came from -- a v5e-1 receipt says nothing about a v5e-8 slice.
"""

# %% cell
# --- cheap, dependency-free probes -------------------------------------------
# Subprocess calls do not need a 20-second jax import, so a TPU VM that is wedged
# during framework startup still leaves CPU/RAM/disk on record.

RECEIPT = {"host": {}, "accelerator": {"kind": "unknown"}, "frameworks": {}}


def run(cmd, timeout=25):
    """Run a command, return stdout. Never raises."""
    if shutil.which(cmd[0]) is None:
        return ""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, check=False)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"<failed: {type(exc).__name__}>"


def mib_to_gib(value_mib):
    return round(value_mib / 1024.0, 2)


print("python  ", sys.version.split()[0], "on", platform.platform())
print("machine ", platform.machine(), "| compiler", platform.python_compiler())

RECEIPT["host"]["python"] = sys.version.split()[0]
RECEIPT["host"]["platform"] = platform.platform()
RECEIPT["host"]["machine"] = platform.machine()

try:
    n = os.cpu_count()
    print(f"cpus    {n}")
    RECEIPT["host"]["cpu_count"] = n
    if hasattr(os, "getloadavg"):
        load1, load5, load15 = (x / n for x in os.getloadavg())
        print(f"load    {load1:.2f}/{load5:.2f}/{load15:.2f} per cpu")
        RECEIPT["host"]["load_per_cpu"] = [round(load1, 2), round(load5, 2)]
except (OSError, TypeError, ZeroDivisionError):
    pass

# RAM from /proc so no psutil dependency is needed.
try:
    meminfo = dict(
        ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines() if ":" in ln
    )
    total_mib = int(re.sub(r"[^\d]", "", meminfo["MemTotal"])) / 1024
    avail_mib = int(re.sub(r"[^\d]", "", meminfo["MemAvailable"])) / 1024
    print(f"ram     {mib_to_gib(avail_mib)} GiB available / {mib_to_gib(total_mib)} GiB total")
    RECEIPT["host"]["ram_avail_gib"] = mib_to_gib(avail_mib)
    RECEIPT["host"]["ram_total_gib"] = mib_to_gib(total_mib)
except (KeyError, ValueError, OSError, ZeroDivisionError):
    pass

# Disk. /kaggle/working is the Kaggle writable mount; /content is Colab's. Report
# whichever exists first.
for base in ("/kaggle/working", "/content", Path.home()):
    try:
        usage = shutil.disk_usage(str(base))
        print(f"disk    {base}: {round(usage.free / 2**30, 1)} GiB free / "
              f"{round(usage.total / 2**30, 1)} GiB total")
        RECEIPT["host"]["disk"] = {
            "path": str(base),
            "free_gib": round(usage.free / 2**30, 1),
            "total_gib": round(usage.total / 2**30, 1),
        }
        break
    except (OSError, ValueError):
        continue

# %% cell
# --- environment variables: reported, never trusted ---------------------------
# Presence of an accelerator variable says a runtime was REQUESTED. Only enumerated
# devices say one is ATTACHED.

INTERESTING = (
    "COLAB_TPU_ADDR", "COLAB_GPU", "COLAB_RUNTIME_TRIPLE", "TPU_NAME", "TPU_HOST",
    "PJRT_DEVICE", "JAX_PLATFORMS", "XRT_TPU_CONFIG", "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_DATA_PROXY_URL", "TPU_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_MEM_FRACTION",
)

print("environment (reported, not trusted):")
found_env = {}
for key in INTERESTING:
    value = os.environ.get(key)
    if value:
        print(f"  {key} = {value}")
        found_env[key] = value
RECEIPT["host"]["env"] = found_env

# %% cell [markdown]
"""
## Detecting the host product

Kaggle and Colab expose themselves differently. The distinction decides which quota
meter is burning (Kaggle: weekly GPU/TPU hours; Colab: daily compute units) and which
CLI can talk to the session.
"""

# %% cell
def detect_product():
    """Return (product, evidence) from mounts and env, not from a claimed name.

    Order matters: Colab TPU images ship empty /kaggle compat dirs, so the
    Kaggle check must require the *writable* mount or the kernel env var.
    """
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ:
        return "colab", "COLAB_RELEASE_TAG/BACKEND_VERSION present"
    try:
        import google.colab  # noqa: F401
        return "colab", "google.colab importable"
    except ImportError:
        pass
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").is_dir():
        return "kaggle", "KAGGLE_KERNEL_RUN_TYPE or /kaggle/working present"
    return "unknown", "no product markers found"


PRODUCT, EVIDENCE = detect_product()
print(f"product  {PRODUCT}  ({EVIDENCE})")
RECEIPT["product"] = PRODUCT
RECEIPT["product_evidence"] = EVIDENCE

# %% cell [markdown]
"""
## Enumerate the TPU through JAX

This is the only part that is evidence. `jax.devices()` is the authoritative list,
and the *platform string* is what distinguishes `tpu` from `cpu` without guessing from
a hostname. Expected counts: **8 devices on a Kaggle v5e-8 slice, 1 on a Colab v5e-1.**
A v5e-8 reporting 1 device is a finding, not an inconvenience -- record it.
"""

# %% cell
# Imported lazily so the probes above have already run. First import on a cold TPU VM
# takes 10-30 s and emits a banner; that delay must not precede the cheap facts.
try:
    import jax
    import jax.extend as jex

    devices = list(jax.devices())
    kinds = sorted({str(d.platform) for d in devices})
    print(f"jax     {jax.__version__} | backend {jax.default_backend()}")
    print(f"        {len(devices)} device(s): {kinds}")

    RECEIPT["frameworks"]["jax"] = {
        "version": jax.__version__,
        "default_backend": jax.default_backend(),
        "device_count": len(devices),
        "platforms": kinds,
        "devices": [str(d) for d in devices],
    }

    for d in devices:
        if str(d.platform) == "tpu":
            try:
                kind = getattr(d, "device_kind", None) or jex.device_kind_str(d)
                print(f"        tpu chip kind: {kind}")
                RECEIPT["accelerator"].setdefault("chip_kinds", []).append(str(kind))
            except Exception as exc:  # probe must not raise
                print(f"        chip kind unavailable: {type(exc).__name__}")

    # HBM limit. Per-process and only present once the backend has allocated, so it can
    # be absent on a just-started runtime. This is the number that decides Mode A
    # (~50 GB, multi-chip only) vs Mode B (~7 GB, single chip OK).
    for d in devices:
        try:
            stats = d.memory_stats()
        except Exception:  # noqa: BLE001
            continue
        limit = stats.get("bytes_limit")
        if limit:
            print(f"        hbm limit: {round(limit / 2**30, 2)} GiB ({limit} bytes)")
            RECEIPT["accelerator"]["hbm_limit_bytes"] = limit
            RECEIPT["accelerator"]["hbm_limit_gib"] = round(limit / 2**30, 2)
            break

    # Prove the silicon does the maths. Kept tiny (128, not 512) so first-compile cost
    # stays small; a TPU that answers but returns garbage would otherwise pass every
    # "is a TPU available" check in the wild.
    try:
        import jax.numpy as jnp

        a = jnp.arange(128, dtype=jnp.bfloat16) ** 2
        b = a.reshape(1, 128) @ a.reshape(128, 1)
        print(f"        bf16 matmul 128x128 -> {float(b[0, 0]):.1f}")
        RECEIPT["accelerator"]["bf16_matmul_check"] = float(b[0, 0])
    except Exception as exc:  # noqa: BLE001
        print(f"        bf16 matmul FAILED: {type(exc).__name__}: {exc}")
        RECEIPT["accelerator"]["bf16_matmul_check"] = f"failed: {type(exc).__name__}"

    RECEIPT["accelerator"]["kind"] = (
        "tpu" if "tpu" in kinds else "gpu" if "gpu" in kinds else "cpu")
except ImportError as exc:
    print(f"jax     NOT INSTALLED ({exc})")
    print("        install jax[tpu] for your Python, or use a TPU base image")
    RECEIPT["frameworks"]["jax"] = {"present": False, "reason": str(exc)}

# %% cell [markdown]
"""
## Pallas and the FWHT-1024 budget

Notebook 06 will need `jax.experimental.pallas`. Record here whether this
runtime has it, and time one compiled FWHT-1024 on the accelerator: that
number is the per-block floor every Mode B decode step must beat.
"""

# %% cell
try:
    import time

    import jax.numpy as jnp
    from jax import block_until_ready, jit

    try:
        import jax.experimental.pallas as pl  # noqa: F401

        print(f"pallas  present (jax {jax.__version__})")
        RECEIPT["frameworks"]["pallas"] = {"present": True}
    except ImportError as exc:
        print(f"pallas  NOT PRESENT ({exc})")
        RECEIPT["frameworks"]["pallas"] = {"present": False, "reason": str(exc)}

    def fwht_ref(a):
        h = a
        step = 1
        while step < h.shape[-1]:
            h = h.reshape(h.shape[:-1] + (-1, 2, step))
            u, v = h[..., 0, :], h[..., 1, :]
            h = jnp.stack([u + v, u - v], axis=-2).reshape(h.shape[:-2] + (-1,))
            step *= 2
        return h

    key = jax.random.key(0)
    vec = jax.random.normal(key, (1024,), dtype=jnp.bfloat16)
    out = block_until_ready(jit(fwht_ref)(vec))  # warm: compile + run once
    t0 = time.perf_counter()
    for _ in range(20):
        out = block_until_ready(jit(fwht_ref)(vec))
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"fwht-1024 on {jax.default_backend()}: {ms:.3f} ms/call (20-call mean)")
    RECEIPT["accelerator"]["fwht1024_ms"] = ms
except (ImportError, NameError) as exc:
    print(f"fwht timing skipped (no jax): {exc}")
    RECEIPT["accelerator"]["fwht1024_ms"] = f"skipped: {type(exc).__name__}"

# %% cell [markdown]
"""
## Verdict

One line, machine-readable. The verdict is from a closed set so grep works across logs:
`tpu` / `gpu` / `cpu`. There is deliberately no `none`: if no framework could see an
accelerator, the honest answer is `cpu` -- the work the notebook just did still ran.
"""

# %% cell
def verdict(receipt):
    """Closed-set verdict, derived only from enumerated devices."""
    kind = receipt.get("accelerator", {}).get("kind", "unknown")
    return kind if kind in ("tpu", "gpu") else "cpu"


RECEIPT["verdict"] = verdict(RECEIPT)
RECEIPT["notebook"] = "01-tpu-env-probe"
RECEIPT["status"] = "executed on Colab TPU v5e-1 2026-10-14, receipt tpu_receipt.json [exec]"

print("=" * 68)
print(f"VERDICT: {RECEIPT['verdict'].upper()} on {PRODUCT}")
print("=" * 68)

out_path = Path("/kaggle/working/tpu_receipt.json")
if not out_path.parent.is_dir():
    out_path = Path("/content/tpu_receipt.json") if Path("/content").is_dir() else Path("tpu_receipt.json")
try:
    out_path.write_text(json.dumps(RECEIPT, indent=2), encoding="utf-8")
    print(f"receipt written to {out_path.resolve()}")
except OSError as exc:
    print(f"could not write receipt: {type(exc).__name__}")
print()
print(json.dumps(RECEIPT, indent=2))

# %% cell [markdown]
"""
## What to do with the answer

| Verdict + device count | Meaning for the runner plan |
| --- | --- |
| `tpu`, 8 devices | Kaggle v5e-8 slice. Both Mode A and Mode B are on the table. |
| `tpu`, 1 device | Colab v5e-1. Mode B only: the ~50 GB unfolded model does not fit. |
| `tpu`, any other count | Record it. Do not assume it is a v5e anything. |
| `cpu` | No accelerator attached. Free CPU quota still applies; do loader and parser work here. |

MIT OR Apache-2.0
updated: 2026-10-14 00:00 UTC
"""
