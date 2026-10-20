# %% cell [markdown]
"""
# Bonsai 2 on TPU silicon — execution + timing harness

Notebook 04 is the forward pass; this is the harness that **proves it runs
on the accelerator**. It:

1. starts a Colab TPU session (`--tpu v5e1`) and verifies `jax.devices()`
   reports a TPU, not a CPU,
2. fetches the PTQ1_0 GGUF from Kaggle storage,
3. runs the forward pass on the TPU and times prompt-processing and decode
   separately, so `tok/s` is measured, not extrapolated,
4. writes a receipt (`receipt-tpu.json`) naming the host, the device, the
   dtype path, and the numbers.

**Claim tier:** `[exec]` only for rows measured in this run. Everything else
is `[sourced]` and says so. If the session turns out to be a CPU runtime,
the receipt says `cpu` and the accuracy number is *not* claimed as TPU.
"""

# %% cell
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

t_start = time.perf_counter()

devs = jax.devices()
kinds = sorted({str(d.platform) for d in devs})
print(f"jax {jax.__version__} | backend {jax.default_backend()} | devices {kinds}")
ON_TPU = "tpu" in kinds
print(f"ON_TPU = {ON_TPU}")

GGUF = os.environ.get("GGUF_PATH", "/content/gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
if not os.path.exists(GGUF):
    # pull from Kaggle storage (free, public-dataset route)
    os.makedirs("/content/gguf", exist_ok=True)
    if os.path.exists("/content/.kaggle_token"):
        os.environ["KAGGLE_API_TOKEN"] = open("/content/.kaggle_token").read().strip()
    import subprocess
    subprocess.run([__import__("sys").executable, "-m", "pip", "install", "-q", "kaggle"],
                   check=True, capture_output=True)
    from kaggle import api
    api.dataset_download_files("essencefromexistence/bonsai2-27b-ptq1",
                               path="/content/gguf", unzip=True, quiet=True)
print(f"file {GGUF} ({os.path.getsize(GGUF) / 2**30:.2f} GiB)")

N_GEN = int(os.environ.get("N_TOKENS", "8"))
PROMPT = os.environ.get("PROMPT", "Hello")
print(f"prompt {PROMPT!r} | generating {N_GEN} tokens")

# %% cell [markdown]
"""
## The forward pass (same code as notebook 04)

Loaded by executing the shipped file, so there is exactly one implementation
to debug — this harness does not fork it.
"""

# %% cell
import runpy

_src = open(os.environ.get(
    "FWD_SRC", "/content/04-bonsai-tpu-forward.py"), encoding="utf-8").read()
# stop at the generation loop: we drive it ourselves so we can time it
_cut = _src.split("st = {")[0]
fw = {"__name__": "fwd"}
exec(compile(_cut, "fwd.py", "exec"), fw)
print("forward pass loaded; layers", fw["cfg"]["n_layer"],
      "backend", jax.default_backend())

# %% cell
st = {
    "conv": [jnp.zeros([fw["cfg"]["conv"] - 1, fw["cfg"]["conv_dim"]])
             if fw["cfg"]["is_recr"][i] else None for i in range(fw["cfg"]["n_layer"])],
    "s": [jnp.zeros([fw["cfg"]["state"], fw["cfg"]["state"], fw["cfg"]["n_v"]])
          if fw["cfg"]["is_recr"][i] else None for i in range(fw["cfg"]["n_layer"])],
    "k": [jnp.zeros([0, fw["cfg"]["n_head_kv"], fw["cfg"]["head_dim"]])
          if not fw["cfg"]["is_recr"][i] else None for i in range(fw["cfg"]["n_layer"])],
    "v": [jnp.zeros([0, fw["cfg"]["n_head_kv"], fw["cfg"]["head_dim"]])
          if not fw["cfg"]["is_recr"][i] else None for i in range(fw["cfg"]["n_layer"])],
}

tok = fw["tok"]
ids = tok.encode(PROMPT)
print(f"prompt -> {len(ids)} tokens: {ids}")

# warm every jitted path first, so the timing below is steady-state and not
# the XLA compile cost of the first call
t_warm = time.perf_counter()
lx, _ = fw["forward_one"](fw["embed"](int(ids[0])), st, 0, fw["rope"])
int(np.argmax(np.asarray(lx)))
t_warm = time.perf_counter() - t_warm
print(f"warmup forward: {t_warm:.2f}s (includes XLA compile)")

# %% cell
t0 = time.perf_counter()
first = None
for pos, tid in enumerate(ids):
    x = fw["embed"](int(tid))
    logits, st = fw["forward_one"](x, st, pos, fw["rope"])
    first = int(np.argmax(np.asarray(logits)))
t_prefill = time.perf_counter() - t0
print(f"prefill {len(ids)} tok in {t_prefill:.2f}s "
      f"({len(ids) / t_prefill:.2f} tok/s)")
print(f"next token: {first} ({tok.decode([first])!r})")

gen = []
t1 = time.perf_counter()
cur = first
for step in range(N_GEN):
    if cur == tok.eos:
        print(f"eos at step {step}")
        break
    gen.append(cur)
    x = fw["embed"](cur)
    logits, st = fw["forward_one"](x, st, len(ids) + step, fw["rope"])
    cur = int(np.argmax(np.asarray(logits)))
    print(f"  {step}: -> {cur} ({tok.decode([cur])!r})")
t_gen = time.perf_counter() - t1
n = len(gen)
tps = (n / t_gen) if n else 0.0
print(f"\ngenerated {n} tokens in {t_gen:.2f}s = {tps:.2f} tok/s")

# %% cell
import json

receipt = {
    "notebook": "05-bonsai-tpu-run",
    "claim": "[exec]" if ON_TPU else "[sourced] CPU run, not TPU",
    "host": {
        "jax": jax.__version__,
        "backend": str(jax.default_backend()),
        "devices": kinds,
        "on_tpu": ON_TPU,
        "device_detail": [str(d) for d in devs],
    },
    "model": {
        "gguf": os.path.basename(GGUF),
        "layers": fw["cfg"]["n_layer"],
        "gdn_layers": int(sum(fw["cfg"]["is_recr"])),
        "attn_layers": int(fw["cfg"]["n_layer"] - sum(fw["cfg"]["is_recr"])),
        "quant": "PTQ1_0 ternary 1.75bpw (group 128)",
        "n_embd": fw["cfg"]["n_embd"],
        "vocab": fw["cfg"]["vocab"],
    },
    "run": {
        "prompt": PROMPT,
        "prompt_tokens": len(ids),
        "warmup_s": round(t_warm, 3),
        "prefill_s": round(t_prefill, 3),
        "prefill_tps": round(len(ids) / t_prefill, 3) if t_prefill else None,
        "gen_tokens": n,
        "gen_s": round(t_gen, 3),
        "gen_tps": round(tps, 3),
        "first_token": int(first) if first is not None else None,
    },
    "answer": tok.decode(gen),
    "status": "measured on this device" if ON_TPU
              else "ran on CPU -- TPU session unavailable",
}
out = "/content/receipt-tpu.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(receipt, fh, indent=2)
print(f"wrote {out}")
print(json.dumps(receipt["run"], indent=2))
print("\n=== ANSWER ===\n" + tok.decode(gen))
print(f"\ntotal wall {time.perf_counter() - t_start:.1f}s")
