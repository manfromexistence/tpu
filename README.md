# Dx TPU workspace (`G:\Dx\tpu`)

TPU work for the Bonsai 2 27B ternary model: a from-scratch JAX/Pallas forward
pass, a fused ternary kernel, the cost model that says where the time goes, and
the multi-token mechanism that is the only route to 50 tok/s on one v5e-1.

**Bonsai 2 27B runs on a Colab TPU v5e-1** (first run 2026-10-16: greedy-decoded
coherent English for 40 tokens at 0.27 tok/s). Today it runs at:

| what | number | receipt |
| --- | --- | --- |
| single-stream, 8k window, one stream | **7.45 tok/s** (134.3 ms/step) | `lane-aligned-ternary-kernel/receipt-gqa-b1-f16-tc1024.json` |
| single-stream, 8k window, with the decode fold | **7.70 tok/s** (129.9 ms/step) | `lane-aligned-ternary-kernel/receipt-fold2-b1-f16-tc1024.json` |
| aggregate, batch 80 (the older per-layer path) | **281.01 tok/s** | `lane-aligned-ternary-kernel/receipt-b80-f16-tc1024.json` |

The aggregate number is the batched path on the **per-layer** step: the fused
whole-model step does not fit in HBM at batch 80 (its compile-time temporaries
want 20-22 GB against a 15.75 GiB budget), so batch 80 runs the per-layer path
and the single-stream numbers are the fused 64-layer step. llama.cpp cannot use TPU silicon at
all -- it has no TPU backend, so on a TPU VM it silently runs on the host CPU --
which is why this is a from-scratch JAX forward pass rather than a build of the
existing runner. See `TPU-STATUS.md` for what is verified, what the throughput
ceiling is, and the bugs that were fixed on the way.

**The one hard rule for this folder: always run the actual Bonsai 2 model.** A
session that only times a single tensor is a wasted session, whatever it
measures. See section 7.

---

## 1. What was measured on 2026-10-21

Everything below ran on a Colab TPU v5e-1, the real 64-layer model end to end,
same prompt (`what is the capital of franch, india and bangaldesh and give me
tps`, 17 prompt tokens, typos included), `max_t` 8192, state float16,
out-tile TC 1024, projection fusion on, grouped-query attention on.

### 1.1 The decode fold: three arms and an oracle

The shipped decode line is

```python
w = (jnp.concatenate([t80, t40, h8], 0) - 1).astype(jnp.bfloat16) * sc
```

and the cast and the multiply are **one expression**, so it promotes back to
fp32: the MXU was being fed a **4-byte-per-element** weight tile where it needs
2 -- and that tile is "the largest single term in the projection's memory
traffic" by the kernel's own comment. The `- 1` and the `* sc` are two more
full passes over it, and neither belongs there: the scale is per out-column
(so it belongs on the accumulator) and the `- 1` is a constant offset (so
`sum_in (level-1)*x = sum_in level*x - sum_in x`, and `sum_in x` is one scalar
per token).

| arm | what it does | ms/step | tok/s | text vs the fp32 oracle |
| --- | --- | --- | --- | --- |
| `FOLD=0` | the shipped decode | 134.1 | 7.45 | **identical** |
| `FOLD=1` | the level in bf16, the scale on the accumulator, the `- 1` folded into a per-token correction | **119.5** | **8.37** | **diverges** |
| `FOLD=2` | the scale inside the bf16 cast: the MXU operand is `bf16((level-1)*sc)`, a bf16 tile | 129.9 | 7.70 | **identical** |
| `FOLD=3` | the scale pre-rounded to bf16, the tile never leaves bf16 | **not measured** | -- | **provably identical** (0 differing words of 2048, `fold_test.py`) |

The **oracle** is the from-scratch fp32 reference forward pass
(`reference-forward-pass/`) run on the same prompt: it produces byte-identical
text to `FOLD=0`. So:

- **`FOLD=2` is the same model, 3.1 % faster.** It is the one to adopt.
- **`FOLD=1` is a different model**: 10.9 % faster, and its output is not the
  model's. It stays a knob (default off) and must never be quoted as "the same
  model, faster". The mechanism behind the divergence is not settled -- the
  obvious explanation (the cancellation in `sum_in level*x` costs a few bits)
  does not survive an arithmetic check, so the ledger records the measurement
  and not the story.

Receipts: `receipt-fold0-`, `receipt-fold1-`, `receipt-fold2-b1-f16-tc1024.json`,
all in `lane-aligned-ternary-kernel/`. The algebra is proved on CPU by
`lane-aligned-ternary-kernel/fold_test.py` (worst relative difference 2.3e-07).

**`FOLD=3` is the one to bet on.** The Pallas documentation is explicit:
*"Even if you pass 32-bit operands to a matrix multiplication, they will be
rounded to bfloat16 unless float32 precision is requested."* So the shipped
path's fp32 tile was being rounded by the hardware before the MXU saw it, and
the operand the MXU actually gets is `bf16((level-1)*sc)` -- which, since
`level-1` is in {-1, 0, 1}, equals `(level-1)*bf16(sc)` exactly. Pre-rounding
the scale to bf16 and multiplying in bf16 therefore produces a **bit-identical
operand** (0 differing words of 2048, checked on CPU) while the tile stays bf16
end to end, which is `FOLD=1`'s width. Expected: `FOLD=1`'s 119.5 ms with the
shipped path's exact text. Not yet measured -- the pool has been 503ing for
hours.

### 1.2 The chunked path: exact on the real model

48 of the model's 64 layers are gated-delta-net, whose recurrence is
inherently sequential -- so k tokens need k passes over the weights, and the
weights are the whole cost of a step. The chunked (WY) form of the same
recurrence computes k tokens with matmuls, against the *same* unpacked weights.

`tpu/multi-token-verification/` holds the derivation and a CPU contract
(**60 cases** for the recurrence: chunk lengths 1-32, five decay profiles
including one that underflows, numpy and JAX, outputs and final state, worst
absolute difference **2.086e-06**; and **12 cases** for the attention and the conv
window, worst absolute difference **0.000e+00** -- bit-identical). The kernel grew a `CHUNK`/`CHUNKCHECK` pair, and the gate runs the
real model: k tokens are replayed in **one chunked pass** from the state before
them, and the per-position argmaxes are compared with the sequential ones.

```text
receipt-chunk4-b1-f16-tc1024.json
  133.9 ms/step | 7.47 tok/s | first token 271 | text identical to the shipped path
  chunk check: 3/3 argmaxes agree, exact=true, first disagreement: None
  sequential : 271, 248068, 198, 760
  chunked   : 248068, 198, 760, 1156
```

The chunk's position `j` is the sequential path's position `j+1`, which is
exactly what the derivation says it must be: the chunk's output after token `j`
is the sequential state after token `j`. **The chunked path is exact on the
real model.**

Four bugs got it there, and each is a trap worth naming:

1. **The decay was passed as a factor where the recurrence wanted its log.**
   The shipped path multiplies the state by `exp(gate)`; the chunked form
   accumulates the factors. Passing the log made the whole model emit token 0.
2. **The causal mask was expanded to the wrong rank.** The chunked score tensor
   is five axes, and a four-axis mask broadcasts against the *group* axis
   instead of the query axis.
3. **The check's own reference was broken.** It fed the hidden state into the
   next step; the model always feeds the **embedding** of the previous token.
   Both sides were wrong, which is why the check "agreed" on zeros.
4. **The cumulative decay underflows.** The model's decay is strong enough that
   `A_t` flushes to zero inside a chunk, after which `A_t/A_s` cannot be
   recovered, the upper triangle overflows, and `0 * inf = NaN` lands in the
   output. The fix is the log domain: accumulate the log decay and take the
   ratio as `exp(cum_t - cum_s)`, which is `<= 1` for `s <= t` and never
   underflows.

---

## 2. The 50 tok/s path

**Why it is the only route.** The step reads 6.37 GiB of packed weights per
token and the measured 8k step is 134.1 ms, of which **~109 ms is the
projection's per-element work** -- the step is ~15x off its own HBM roofline
of 9.0 ms, and the MXU is ~99 % idle at batch 1. That cost is per *weight*, and
per-weight work does not shrink when more tokens ride along with it. A
verification pass computes k tokens against the same unpacked weights and
amortises it k ways. Everything else on this track has been measured and ruled
out: launch count (36 % fewer launches moved the step 0.5 %), HBM bandwidth
(86 % of spec), program count, DMA count, and the unpack's memory traffic
(cutting it made the step *slower*).

**The arithmetic.** At the measured pass times, with an acceptance rate *a*
(accepted tokens per pass):

| pass | a=3 | a=4 | a=6 | a=7 |
| --- | --- | --- | --- | --- |
| 134.1 ms (the shipped path) | 22.4 | 29.8 | 44.7 | 52.2 |
| 129.9 ms (`FOLD=2`, the same model) | 23.1 | 30.8 | 46.2 | 53.9 |
| 119.5 ms (`FOLD=1`, a different model) | 25.1 | 33.4 | 50.2 | 58.6 |

Inferact's open-source megakernel for **Qwen3.8-27B** -- the same base
architecture as Bonsai 2 -- reports **3-6 accepted tokens per step** with its
DFlash2 draft, and 1,515 tok/s at acceptance length 6 on 4x TPU v7 [sourced:
`inferact.ai/blog/tpu-megakernels`, `github.com/Inferact/tpu-megakernels`;
their hardware, not ours]. So 50 tok/s on one v5e-1 needs the **top of the
published acceptance range**, and every other term in the table above is a
measured number.

**The draft is a download, not a graft.** Two candidates, both for this base
architecture:

| draft | size | what it is |
| --- | --- | --- |
| `z-lab/Qwen3.8-27B-DFlash2` | 1.92 B parameters, 3.58 GB | the block-diffusion drafter Inferact's Qwen megakernel uses; base model `Qwen/Qwen3.8-27B` |
| `sudoingx/Ternary-Bonsai-2-27B-PTQ1_0-MTP-GGUF` | 5.87 GB (the `mtp-lean` file) / 6.53 GB (the full one) | this model's own MTP head, in the same ternary format |

The MTP head is the memory-friendly one: the target's peak is already 14.55 GiB
of the 15.75 GiB budget, so a 3.58 GB draft in the same process does not fit and
the draft would have to run as a separate process with its own state.

**What is left, in order:** the chunk length (8 and 16; the compile-time HLO
temporaries grow with the chunk), the amortization ratio (the chunk pass's wall
time against k sequential steps -- now recorded in the receipt's `chunk_check`),
then the draft and its **measured** acceptance rate, which is the one number
that decides whether this track reaches 50 or stops at 30.

---

## 3. Running a session

The Colab CLI is the only way in, and the pool is frequently unavailable
(`503 Service Unavailable` on the assign endpoint, for hours at a time).
`v6e1` is rejected for entitlement -- v5e1 is the only TPU this account has.

```powershell
# the CLI (it is not part of the image)
uv tool install google-colab-cli

# the 120 s auth-timeout patch, into the CLI's own site-packages
$sp = "$env:APPDATA\uv\tools\google-colab-cli\Lib\site-packages"
Copy-Item 'G:\Dx\tpu\colab-auth-timeout-patch\colab_timeout_patch.py' $sp -Force
Copy-Item 'G:\Dx\tpu\colab-auth-timeout-patch\zz_colab_timeout.pth'   $sp -Force

colab new -s dx-fuse5 --tpu v5e1     # 503 until the pool opens
```

Then, in the session:

```powershell
colab upload 'G:\Dx\tpu\lane-aligned-ternary-kernel\kernel.py' '/content/kernel.py'
colab upload 'G:\Dx\tpu\reference-forward-pass\kernel.py'         '/content/04.py'
colab exec -s dx-fuse5 -f setup_fast.py --timeout 2400   # pin + weights
colab exec -s dx-fuse5 -f run_fold_ab.py  --timeout 5400  # the fold A/B
colab download '/content/fold1/receipt-b1-f16-tc1024.json' 'G:\Dx\tpu\lane-aligned-ternary-kernel\receipt-fold1-b1-f16-tc1024.json'
```

**The environment is not optional.** Stock Colab (jax 0.7.2 / libtpu 0.0.21.1)
cannot compile a Pallas kernel at all: *"Unsupported version: expected <= 7 but
got 8"*. The pin is **jax 0.11.2 + libtpu 0.0.50**, and a recycled session VM
reverts to the stock stack, so every session re-pins. `setup_fast.py` does it
idempotently and also fetches the 5.54 GiB of weights from Hugging Face
(Kaggle 403s from Colab IPs). A Pallas core halt wedges the runtime and only
`colab restart-kernel` clears it; compile failures do not, so compile-only
variants can be batched into one process.

The runner scripts (`setup_fast.py`, `run_fold_ab.py`, `run_chunk_check.py`)
live in `G:\Temp\UserTemp\opencode\colab_run\`, which is **scratch space and
gets swept** -- the recipes above are the durable copy.

---

## 4. Directory map

Split three ways: what the track still uses is at the top, everything else says
what it is. Nothing was deleted, and `git mv` kept every folder's history.
**Renamed 2026-10-21** -- the numbers were work-order prefixes and are gone;
the names now say what each folder is.

### The live track (at the root of `tpu/`)

| Folder | What it is |
| --- | --- |
| `lane-aligned-ternary-kernel/` | **the kernel**: lane-aligned tiles, the fused 64-layer step, the grouped attention, the KV-dtype knob, the decode fold, the chunked path. 36 receipts |
| `multi-token-verification/` | the chunked gated-delta recurrence, the chunked attention and the conv window: the derivation, the CPU contract (60 + 12 cases), and the exactness gate |
| `projection-cost-breakdown/` | the cost model -- where a projection's time goes (the decode is 82 % of it, and it is data movement) |
| `pallas-matmul-findings/` | the Pallas API facts and the nine proof scripts (BlockSpec index maps, the grid rules, the `Bad rhs type` fix) |
| `reference-forward-pass/` | the correctness proof -- the from-scratch 27B forward pass, and the bugs it found. **Also the oracle**: it is the only thing that can say which of two texts is the model's |
| `ternary-2bit-weight-format/` | the 2-bit `PQ2_0` format: a verified numpy decode and the port design (`TODO.md` TPU-08) |
| `colab-auth-timeout-patch/` | the 120 s auth-timeout patch every session needs |

### `archive/` -- the superseded generations, kept as history

`environment-pallas-probe`, `gguf-format-and-sign-metadata`,
`hadamard-transform-tests`, `run-harness`, `batched-kv-decode`,
`lax-scan-over-layers`, `int8-weights`, `fused-ternary-matmul`,
`fully-fused-64-layer-model`, `windowed-dma-in-blocks`, and the
`llamacpp-{baseline-kernel,fast-kernel,cached-kernel,four-model-comparison,tunnel-server}`
and `tunnel-connectivity-test` generations. Each still carries its own
receipt; the ladder they form is the track's history.

### `out-of-scope-gpu/` -- GPU kernels, ruled out by AGENTS.md section 2 rule 7

`gpu-llamacpp-t4-baseline` (2xT4, 7.5 tok/s), `gpu-mtp-head-availability-test`,
`gpu-draft-model-speculation-test`, `gpu-draft-model-two-t4-test`,
`gpu-mtp-head-graft-test`. Kept for the record; a GPU run is not a result of
this project.

### `tools/` -- not kernels

`dq-cli/` (the `dq` download/engine CLI source, Rust) and `bin/` (local build
outputs, never committed).

---

## 5. The kernel convention

Each kernel folder owns `kernel.py` as the source of truth (cells split by
`# %% cell`), and where the folder keeps a notebook it also owns
`make_notebook.py`, `notebook.ipynb` and `kernel-metadata.json`. After editing
`kernel.py`, regenerate the notebook and re-run `--check` before claiming
anything is done. The live kernel folder also keeps its receipts and
`fold_test.py` (the decode fold's CPU proof).

## 6. Claim tiers

Every performance or status claim in this folder carries a tier: `[exec]` (you
ran it, name the receipt), `[sourced]` (name the report), `[extrapolated]`
(name the basis). Never write a bare superlative. A CPU reference is not a
TPU measurement and says so.

## 7. Hard rules

- **Always run the actual Bonsai 2 model.** Every TPU session runs the real
  64-layer model end to end. A session that only times one tensor is a wasted
  session, no matter how clean the measurement looks. Cost breakdowns,
  ablations and sweeps are taken **inside that same run**, after the model
  number is in the log -- never instead of it. This rule outranks any plan
  elsewhere in this folder that says to benchmark first. See
  `DO-NOT-RETEST.md` section 0.
- **TPU only.** Every result this folder reports is a TPU result. A GPU run
  (Kaggle T4, any CUDA host) is **not** a result of this folder, may not be
  quoted as this folder's throughput, and may not be used to decide what the
  TPU work does next. An agent needs the user's explicit go-ahead before
  starting any GPU run; a GPU is otherwise allowed only as a *correctness
  oracle* (llama.cpp's text as a cross-check), and then it is labelled as an
  oracle, never as a speed claim. Rationale, measured: llama.cpp has no TPU
  backend, so a GPU run cannot inform the TPU kernel, and the TPU's own
  bottleneck is the per-weight-byte ternary unpack (`DO-NOT-RETEST.md` section
  10) -- a GPU number does not move it. **The Kaggle GPU kernels under this
  folder are out of scope since 2026-10-20 and are kept only for the record:
  `out-of-scope-gpu/gpu-llamacpp-t4-baseline/`,
  `out-of-scope-gpu/gpu-mtp-head-availability-test/`,
  `out-of-scope-gpu/gpu-draft-model-speculation-test/`,
  `out-of-scope-gpu/gpu-draft-model-two-t4-test/`,
  `out-of-scope-gpu/gpu-mtp-head-graft-test/`. Do not run them.**
- **The oracle decides, not a tolerance.** When two paths disagree, the fp32
  reference forward pass says which one is the model. A faster path that
  changes the output is a different model, and is labelled as one.
- Never commit `colab-auth-timeout-patch.ps1`, tokens, cookies, or
  service-account keys. `.gitignore` already excludes
  `/tpu/colab-auth-timeout-patch.ps1`; keep it that way.
- Never paste a secret into a kernel, notebook, log, or chat quote.
- The hub `G:\Dx\.gitignore` is deny-everything: a new file needs its
  allow-list line in the same commit.
- Do not delete build outputs or caches to free space; report instead.

MIT OR Apache-2.0
updated: 2026-10-22 00:35 UTC
