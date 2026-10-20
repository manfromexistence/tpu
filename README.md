# Dx TPU workspace (`G:\Dx\tpu`)

On-device and cloud-GPU/TPU work for the Bonsai 2 27B ternary model:
Kaggle and Colab access, kernel history, and a JAX/Pallas TPU runner.
A new agent starts here and should be productive without asking for
credentials.

**Bonsai 2 27B runs on a Colab TPU v5e-1** (notebook 04, first run 2026-10-16:
greedy-decoded coherent English for 40 tokens at 0.27 tok/s), and the current
best is **281.01 tok/s aggregate at batch 80** with every ternary projection
served by a fused PTQ1_0 Pallas kernel that unpacks the weights in place
(notebook 14, `[exec]` 2026-10-18). That is 9.5x the 29.66 tok/s notebook 11
baseline, from one change: store the packed tiles so the **out-columns land on
the 128 lanes** (`x @ w` over `(32, rows)`) instead of the 2-16-wide in-dim
axis. The decoded weights are bit-identical; the rest came from the batch
ladder -- step time is 245.6 ms at B=32 and 284.7 ms at B=80. llama.cpp cannot use TPU silicon at all --
it has no
TPU backend, so on a TPU VM it silently runs on the host CPU -- which is
why this is a from-scratch JAX forward pass rather than a build of the
existing runner. See `TPU-STATUS.md` for what is verified, what the
throughput ceiling is, and the 9 bugs that were fixed on the way.

**The one hard rule for this folder: always run the actual Bonsai 2 model.** A
session that only times a single tensor is a wasted session, whatever it
measures. See §6.

## 1. First five minutes

Kaggle auth lives outside the hub (never copy it in):

```powershell
. 'G:\Ai\.secrets\kaggle\colab-env-patch.ps1'   # sets KAGGLE_USERNAME + KAGGLE_KEY
kaggle kernels list --mine --csv
```

Colab runs through the `Colab1` WSL1 distro (no starting scripts,
`G:\Wsl\Colab1`), account `emonmahabubhosen@gmail.com`:

```powershell
wsl -d Colab1 -u root -- google-colab-cli auth status
```

Check the live numbers before planning TPU/GPU work: Kaggle GPU quota
resets monthly (18.48h used / 11.52h left / 30h on 2026-10-13), TPU
quota was 20h untouched. Re-check with `kaggle competitions list`
or the web console; do not quote these numbers as current.

## 2. Directory map

The folder is split three ways, so that what the track still uses is at the top and
everything else says what it is. **Reorganised 2026-10-20**; nothing was deleted, and
`git mv` kept every folder's history.

### The live track (at the root of `tpu/`)

| Folder | What it is |
| --- | --- |
| `14-lane-aligned-ternary-kernel/` | **the kernel**: lane-aligned tiles, the fused 64-layer step, the grouped attention, the KV-dtype knob. 281.01 tok/s aggregate, 7.45 tok/s single-stream at an 8k window |
| `13-projection-cost-breakdown/` | the cost model -- where a projection's time goes (the decode is 82 % of it, and it is data movement) |
| `07-pallas-matmul-findings/` | the Pallas API facts and the nine proof scripts (BlockSpec index maps, the grid rules, the `Bad rhs type` fix) |
| `04-reference-forward-pass/` | the correctness proof -- the from-scratch 27B forward pass, and the nine bugs it found |
| `pq2-2bit-weight-format/` | the 2-bit `PQ2_0` format: a verified numpy decode and the port design (the next lever, `TODO.md` TPU-08) |
| `colab-env-patch/` | the 120 s auth-timeout patch every session needs |

### `archive/` -- the superseded generations, kept as history

`01-environment-and-pallas-probe`, `02-gguf-format-and-sign-metadata`, `03-hadamard-transform-tests`,
`05-run-harness`, `06-batched-kv-decode`, `08-lax-scan-over-layers`, `09-int8-weights`,
`10-fused-ternary-matmul`, `11-fully-fused-64-layer-model`, `12-windowed-dma-in-blocks`, and the
`bonsai2-{kernel,fast,cached,4way,tunnel}` / `tunnel-connectivity-test` generations. Each still
carries its own receipt; the ladder they form is the track's history
(`29.66 -> 281.01 tok/s`).

### `out-of-scope-gpu/` -- GPU kernels, ruled out by AGENTS.md section 2 rule 7

`gpu-llamacpp-t4-baseline` (2xT4, 7.5 tok/s), `gpu-mtp-head-availability-test`, `gpu-draft-model-speculation-test`, `gpu-draft-model-two-t4-test`,
`gpu-mtp-head-graft-test`. Kept for the record; a GPU run is not a result of this project.

### `tools/` -- not kernels

`dq-cli/` (the `dq` download/engine CLI source, Rust) and `bin/` (local build
outputs, never committed).

Kaggle datasets used: `essencefromexistence/bonsai2-27b-ptq1`,
`essencefromexistence/bonsai2-llamacpp-build`. Kaggle account:
`essencefromexistence`.

## 3. Kernel convention (must follow)

Each kernel folder owns exactly four files:

- `kernel.py` — the source of truth, cells split by `# %% cell`
- `make_notebook.py` — builds `notebook.ipynb` from `kernel.py`;
  `python make_notebook.py --check` must print `up to date`
- `notebook.ipynb` — generated, never hand-edited
- `kernel-metadata.json` — `enable_tpu: true` for the TPU track

After editing `kernel.py`, regenerate the notebook and re-run
`--check` before claiming anything is done.

## 4. The TPU-runner track (in order)

Ground truth for the ternary math is the PrismML-Eng/llama.cpp
`prism` branch: PQ2_0 decode, the `prism.hadamard.*` metadata
contract, and foldable linears. Two modes: **A** = offline BF16
(~50 GB), **B** = native ternary + runtime FWHT (~7 GB).

| # | Notebook | Does | Status |
| --- | --- | --- | --- |
| 01 | `tpu-env-probe` | JAX/Pallas presence, FWHT-1024 timing | `[exec]` Colab TPU v5e-1 2026-10-14, receipt in folder; Kaggle kernel pushed, run queued |
| 02 | `gguf-prism-metadata` | GGUF fork reader, sign-vector inventory | `[exec]` Colab TPU v5e-1 2026-10-14 against PTQ1_0, receipt in folder; push blocked by Kaggle 1-TPU slot |
| 03 | `fwht-kernel-test` | Signed-FWHT reference + control tests | `[exec]` all 4 tests PASS on Colab TPU v5e-1 2026-10-14, receipt in folder; push blocked by Kaggle 1-TPU slot |
| 04 | `bonsai-tpu-forward` | From-scratch 27B forward pass; the correctness proof | `[exec]` v5e-1 2026-10-17, 0.27 tok/s single-stream, coherent 40 tokens |
| 05 | `bonsai-tpu-run` | Run harness for the forward pass | `[exec]` 2026-10-17 |
| 06 | `bonsai-tpu-fast` | Fixed-KV batched decode | `[exec]` batch 16 = 3.5 tok/s, 13x notebook 04 |
| 07 | `pallas-fusion` | Why the fused matmul failed, and the grid API that works | `[exec]` HF fused-INT8 control at `ca674d6`; four retractions in FINDINGS |
| 08 | `bonsai-scan` | One `lax.scan` over all 64 layers | `[exec]` batch 4 = 1.5 tok/s |
| 09 | `bonsai-int8` | int8 weights | `[exec]` batch 8 = 4.0 tok/s |
| 10 | `bonsai-fused` | Fused PTQ1_0 ternary matmul in Pallas | `[exec]` tile vs the model's matrix maxdiff **0.0**; 4.04 ms per 52.4 M params, flat in batch |
| 11 | `bonsai-fused-full` | The real 64-layer model, fully fused | `[exec]` **batch 32 = 29.66 tok/s**, gate `first token 11`; 15.31 -> 29.66 in one session with no algorithmic change |
| 12 | `bonsai-bigdma` | 8 in-blocks per grid step | `[exec]` 13.30 tok/s -- the per-DMA-overhead hypothesis, refuted |
| 13 | `cost-sweep` | What the fused kernel's cost actually is | `[exec]` decode is 82 % of the projection and it is **data movement**: `concatenate` 1.68x, int16 1.34x, broadcast order 4.2x worse |
| 14 | `bonsai-transposed` | Lane-aligned tiles: out-columns on the 128 lanes | `[exec]` **batch 80 = 281.01 tok/s**, f16 state, TC 1024, gate `first token 11`; 9.5x notebook 11, text byte-identical |

Measured on Colab TPU v5e-1 (`TPU v5 lite`, jax 0.7.2, 15.75 GiB HBM
limit, Pallas present): **FWHT-1024 = 0.295 ms/call** (20-call mean,
the per-block floor for Mode B). From PTQ1_0: `block_size` 1024,
`transform` normalized-sylvester-walsh-hadamard, `sign_mode` explicit,
28,672 sign values = 5120+6144+17408 exactly, all +-1; 401 forward
weights mapped, signed widths {5120, 6144, 17408}, plain widths
{1024, 10240, 12288, 248320}, inverse = `token_embd.weight` only.

Next notebooks (not yet written): `04-pq2-decode` (PQ2_0 block
decode vs `prism` branch), `05-forward-pass` (Mode A BF16 reference),
`06-pallas-kernel` (fused `Wq @ FWHT(s.x)` Pallas kernel).

## 5. Claim tiers

Every performance or status claim in this folder carries a tier:
`[exec]` (you ran it, name the log), `[sourced]` (name the report),
`[extrapolated]` (name the basis). TPU v5e-8 / v5e-1 rows are
`[sourced]` until a kernel in §4 runs them. Never write a bare
superlative. Real sign vectors come only from notebook 02; until
then every FWHT test uses random signs and says so.

## 6. Hard rules

- **Always run the actual Bonsai 2 model.** Every TPU session runs the real
  64-layer model end to end. A session that only times one tensor is a wasted
  session, no matter how clean the measurement looks. Cost breakdowns,
  ablations and sweeps are taken **inside that same run**, after the model
  number is in the log -- never instead of it. This rule outranks any plan
  elsewhere in this folder that says to benchmark first. See
  `DO-NOT-RETEST.md` §0.
- **TPU only.** Every result this folder reports is a TPU result. A GPU run (Kaggle
  T4, any CUDA host) is **not** a result of this folder, may not be quoted as this
  folder's throughput, and may not be used to decide what the TPU work does next.
  An agent needs the user's explicit go-ahead before starting any GPU run; a GPU is
  otherwise allowed only as a *correctness oracle* (llama.cpp's text as a
  cross-check), and then it is labelled as an oracle, never as a speed claim.
  Rationale, measured: llama.cpp has no TPU backend, so a GPU run cannot inform the
  TPU kernel, and the TPU's own bottleneck is the per-weight-byte ternary unpack
  (`DO-NOT-RETEST.md` section 10) -- a GPU number does not move it.
  **The Kaggle GPU kernels under this folder are out of scope since 2026-10-20 and
  are kept only for the record: `out-of-scope-gpu/gpu-llamacpp-t4-baseline/`, `out-of-scope-gpu/gpu-mtp-head-availability-test/`, `out-of-scope-gpu/gpu-draft-model-speculation-test/`,
  `out-of-scope-gpu/gpu-draft-model-two-t4-test/`, `out-of-scope-gpu/gpu-mtp-head-graft-test/`. Do not run them.**
- Never commit `colab-env-patch.ps1`, tokens, cookies, or service-account keys.
  `.gitignore` already excludes `/tpu/colab-env-patch.ps1`; keep it that way.
- Never paste a secret into a kernel, notebook, log, or chat quote.
- The hub `G:\Dx\.gitignore` is deny-everything: a new file needs
  its allow-list line in the same commit.
- Do not delete build outputs or caches to free space; report instead.

MIT OR Apache-2.0
updated: 2026-10-20 20:40 UTC
