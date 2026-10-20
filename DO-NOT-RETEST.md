# DO-NOT-RETEST: settled facts for the Bonsai 2 TPU track

TPU sessions are the scarce resource. Everything on this page has already
been measured or proven, with the evidence named. **Do not spend a session
re-deriving any of it.** If a change would contradict one of these, say so in
the commit message and point at the measurement that has been overturned --
do not silently re-run it.

If you are about to run something on the TPU, check this page first. If your
question is already answered here, you are about to waste a session.

## 0. The standing rule: always run the actual model

**Every TPU session runs the real 64-layer Bonsai 2 model end to end.** A
session that only times one tensor is a wasted session, however clean the
measurement looks.

This rule comes from the person paying for the sessions. It **outranks any
plan anywhere in this repo that says otherwise** -- including §7 below and
`07-pallas-matmul-findings/FINDINGS.md` §8 as they were first written.

What it means in practice:

- The first thing a session prints is the full-model result: aggregate tok/s,
  ms/step, the correctness gate, and the generated text.
- Cost breakdowns, ablations and sweeps are taken **inside that same run**,
  after the model number is in the log -- never instead of it.
- A single-tensor measurement is allowed only as one step of a run that also
  runs the model, and its number is reported next to the model number.
- "Isolating the cost first is cheaper" is not a reason to skip the model. The
  model run is the cheap one: nb 11's tile build was 8.7 s and compile plus
  first step was 10.0 s (§7). The whole model traverses in 2.09 s per forward
  pass.

Why this is a rule and not a preference -- single-layer projections have been
wrong on this track three times:

| Projection | What happened |
| --- | --- |
| nb 10: 4.04 ms/52.4 M params -> "~59 tok/s at batch 128" | retracted: assumed a 27.0 GB layout that does not fit (§6) |
| nb 12: "the k-loop is the bottleneck" | confounded: `KB` 1->8 *and* `TO` 128->64 changed together |
| nb 12: per-in-block cost "scales with work" | an artifact of reading one tensor in isolation |

Measuring the model costs the same and cannot be wrong in this way.

## 1. Environment

| Fact | Evidence |
| --- | --- |
| Pallas needs jax 0.11.2 + libtpu 0.0.50; stock Colab fails with a Mosaic version mismatch | `07-pallas-matmul-findings/FINDINGS.md` §1 |
| A core halt wedges the runtime across processes; only `colab restart-kernel` clears it | `07-pallas-matmul-findings/FINDINGS.md` §1 |
| Compile errors do **not** wedge the core, so compile-only variants batch into one process | same |
| The `colab` 120 s auth timeout needs the `.pth` patch in `colab-env-patch/` | `colab-env-patch/colab_timeout_patch.py` |
| A recycled session VM reverts to the stock stack (jax 0.7.2 / libtpu 0.0.21.1), which cannot compile any Pallas kernel (`Failed to deserialize the Mosaic module: expected <= 7 but got 8`) | session log 2026-10-18; re-pin with `colab_run/pin_stack.py` + `restart-kernel` (~2 min) |
| v6e1 has no quota; T4 GPU returns `Service Unavailable` | session log |
| Weights come from HF, not the Kaggle API (Kaggle 403s from Colab IPs) | `dl_hf.py` |

## 2. Pallas API facts

| Fact | Evidence |
| --- | --- |
| Use `pltpu.PrefetchScalarGridSpec` + `CompilerParams(dimension_semantics=...)`; plain `grid=` halts the core | `07-pallas-matmul-findings/FINDINGS.md` §2 |
| `("parallel","parallel","arbitrary")` is accepted; `"sequential"` is rejected | same |
| **`pl.BlockSpec` index maps return block indices, not element offsets** | `07-pallas-matmul-findings/FINDINGS.md` §4.1 |
| `u16.view(jnp.float16)` (i16->f16 bitcast) is unsupported: pass scales as a separate input | `proof_min_align_128.py` |
| `pl.dot` is removed in jax 0.12; use `@` / `jnp.dot` | same |
| Minimum dynamic-slice alignment inside a Pallas program is 128 | `proof_min_align_128.py` |
| Computed LHS, in-kernel transpose, `slice`, `concat`, `cast` all legalize | `07-pallas-matmul-findings/FINDINGS.md` §3 |

## 3. Two failure signatures that mean "fix it locally, do not re-run"

These two cost the most time in the whole track, and both are recognisable
from the error text alone. Neither is worth a session.

### `Bad rhs type: 128, 128`

The process imported notebook 04, which runs

```python
jax.config.update("jax_default_matmul_precision", "highest")
```

at import time. Fix locally -- reset the config after the exec and pin
`precision=jax.lax.Precision.DEFAULT` on the `dot_general`. It is a one-line
fix and it is already written into `archive/10-fused-ternary-matmul/kernel.py`.

### `BoundsCheck 2 [deref of opnd#0] for dma.hbm_to_vmem` (then a wedged core)

A `BlockSpec` index map is passing element offsets where Pallas wants block
indices. Audit every index map before running. Cost of getting this wrong is
a wasted session plus a forced restart.

## 4. Model facts (no session needed to use these)

| Fact | Evidence |
| --- | --- |
| PTQ1_0 block is 28 bytes: `qs[24]`, `qh[2]`, fp16 `d`; 128 trits | `ggml-common.h:214` |
| Within a block: idx 0-79 = qs bytes 0-15 power-major; 80-119 = qs bytes 16-23 power-major; 120-127 = `qh` **interleaved** per power | `04-reference-forward-pass/kernel.py:90-96` |
| The model's weight matrix is `decode(...).reshape(-1).reshape(dims, "F")` -- **not** `decode(...).reshape(dims, "F")` | `04-reference-forward-pass/kernel.py:280-281` |
| Those two forms differ in 89% of entries on layer-0 qkv | `07-pallas-matmul-findings/FINDINGS.md` §4.3 |
| Tiled reconstruction of the raw blocks == the model's weight matrix, exactly | `archive/10-fused-ternary-matmul/kernel.py` run log, maxdiff 0.0 |
| Block (out `o`, in-block `k`) is at flat block index `o*BPC + k` | `07-pallas-matmul-findings/FINDINGS.md` §4.3 |

## 5. Measured performance (do not re-measure)

Layer-0 `blk.0.attn_qkv.weight`, 52.4 M params, `[5120, 10240]`:

| Layout | Block | ms | Correct |
| --- | --- | --- | --- |
| 28-byte GGUF blocks, in-kernel unpack | `(128, 128)` | 4.04-4.18 | yes |
| host 2-bit repack, 32-byte rows | `(128, 32)` | 30.6-30.7 | yes |

`[exec]` on v5e-1, `archive/10-fused-ternary-matmul/kernel.py`.

- **The cost is flat in batch.** 4.04 ms at B=1, 4.18 ms at B=128.
- **A host 2-bit repack measured 7.5x slower -- but see §7 before treating that
  as a fact about 2-bit packing.** Its extraction used a broadcast-vector shift,
  which notebook 13 has since measured as 4.2x slower than the same trits
  concatenated, for reasons that have nothing to do with the packing width.

**Lane-aligned decode (notebook 14, 2026-10-18).** Moving the out-columns onto
the 128 lanes made the same layer-0 projection **4.4x faster end to end on the
real model**: 29.66 -> 130.25 tok/s at batch 32. Do not reconcile that against
the notebook 13 cells above -- those were measured on the old, lane-starved
layout. See §8.

## 6. A correction: the "59 tok/s at batch 128" projection was against an impossible layout

`archive/10-fused-ternary-matmul/README.md` projected ~59 tok/s at batch 128 by scaling the
measured 4.18 ms/52.4 M params up to the model's ~27.0 B ternary parameters.
That arithmetic is right, but it assumed the measured layout -- **128-byte
rows** -- which used 100 bytes of padding per 28-byte block. Held for the
whole model that layout needs

```text
210.7 M blocks x 128 B = 27.0 GB
```

and v5e-1 has 15.75 GiB. **The 128-byte-row layout cannot hold the model.**
The fastest *measured* configuration does not fit in HBM, so it was never
reachable end to end. Any projection must be made against a layout that
actually fits:

| Layout | Bytes/block | Model total | Fits v5e-1 |
| --- | --- | --- | --- |
| 28 B rows, raw GGUF blocks | 28 | 5.90 GB + 0.84 GB scales | **yes** |
| 32 B rows, host 2-bit repack | 32 | 6.74 GB + 0.84 GB scales | yes |
| 128 B rows (what was measured fast) | 128 | 27.0 GB | **no** |

So the open question is not "is the kernel correct" -- it is settled -- but
**how the 28-byte-row layout performs**, because that is the only one that
fits without a host repack.

## 7. The fused full model: the open question is settled, and it was data movement

Five runs of the real 64-layer model on Colab TPU v5e-1, jax 0.11.2 /
libtpu 0.0.50, every one printing the same correctness gate (first token `11`,
`','`) and the same generated text as notebook 04.

| config | tok/s (B=32) | ms/step | receipt |
| --- | --- | --- | --- |
| nb 11, `stack(...).reshape` decode, TC 128, int32 | 15.31 | 2090.6 | `archive/11-fully-fused-64-layer-model/receipt-baseline-dec-stack-tc128.json` |
| nb 12, eight in-blocks per step | 13.30 | 2406 | `archive/12-windowed-dma-in-blocks/receipt-colab-tpu.json` |
| nb 11, decode spelled `concatenate`, TC 128, int32 | 20.35 | 1572.5 | `archive/11-fully-fused-64-layer-model/receipt-dec-cat-int32-tc128.json` |
| nb 11, + out-tile TC 512 | 24.68 | 1297 | log only (`colab_run/11-tc512.log`) |
| nb 11, + decode at int16 width, TC 512 | **29.66** | **1078.7** | `archive/11-fully-fused-64-layer-model/receipt-dec-cat-int16-tc512.json` |

The last three rows differ from the first by **nothing but how the decode is
spelled and how wide its lanes are** -- same weights, same tile order, same
kernels, same output text.

**Do not retest "the k-loop is the bottleneck".** It is refuted twice over:
cutting the sequential step count 4x made the model 15 % slower (nb 12), and
notebook 13 varied `TO` with everything else fixed -- 128 -> 1024 is 3200 -> 400
programs -- for **1.18x**, saturating by `TO = 512`. Program count, DMA count and
k-loop length are all excluded as causes.

**Do not retest "a cheaper unpack will not show up in the wall clock".** The
statement survives, but it was about the wrong kind of cheaper. Notebook 13 Q5
prices four decodes that produce *identical trits in identical columns* on real
layer-0 weights:

| change to the decode | what it cuts | decode ms, TO=512 | effect on the model |
| --- | --- | --- | --- |
| `(3*((q*p)&255))>>8` for the 5-op shift chain | instructions (~1.4x fewer) | -- | none (nb 11) |
| `concatenate(-1)` for `stack(axis=1).reshape` | a layout conversion | 3.126 -> 1.866 | **1.33x** |
| int16 lanes for int32 | element width | 1.866 -> 1.393 | **1.20x** |
| one broadcast per group (byte-major) | nothing: adds a real interleave | 7.916 | 4.2x slower than `concatenate`, same trits |

So the decode is neither instruction-bound nor program-bound. It is
**data-movement bound inside the VPU**, and the way to make it faster is to stop
asking Mosaic to reorder columns -- not to write fewer operations. An
interleaving reshape costs 4.2x more than the same trits concatenated.

**This also re-explains the parked 2-bit repack** (§5). The repack extracted
trits with `(b >> 2j) & 3` over a broadcast axis, which is the same class of
column interleave as the `dec3` row above, and it measured 7.5x slower. That is
evidence about the *extraction pattern*, not about 2-bit packing or the row
stride: a 2-bit repack is not disproven, and the next attempt at one needs an
extraction that is a concatenation rather than a broadcast.

Excluded as causes: HBM bandwidth (709 GB/s measured on this slice against
~820 GB/s spec), the MXU (batch 16 -> 32 costs 4 %), DMA count, program count,
and decode instruction count. The decode chain is now *measured* as the dominant
cost of the projection -- 3.13 ms of 3.83 ms at TO=512 before these changes --
rather than unseparated.

**And one host-code bug, now fixed:** `TC` was doing two jobs (out-tile height
and in-block width, `BPC = IN // TC`). At TC=128 the two numbers coincide, which
is why it survived every run on this track. At TC=512 the tile build aborts with
`attn_qkv.weight 11468800 != 2867200`. The repaired chain is verified at TC
128/512/1024 by `G:\Temp\UserTemp\opencode\colab_run\check11.py`, a no-TPU numpy
check that lives outside the repo because `tpu/*/` allow-lists only `kernel.py`,
`make_notebook.py`, `kernel-metadata.json`, `notebook.ipynb`, `receipt-*.json`
and `*.md` (`.gitignore` lines 38-47).

**Also settled: scoped VMEM is 16 MB and unpack intermediates scale with the
in-block width.** A block covering all `BPC` in-blocks needs 74 MB and fails
with `E1001: CompileTimeScopedVmemOom`. Keep the per-step tile small
(`(TO, KB, 28)` was 32 KB per intermediate).

**And: restart the kernel between runs.** `colab exec` reuses the process, so
the previous run's 6 GB of tiles stay resident and produce a
`RuntimeBufferAllocationFailure` about 80 MB short of the true limit. That
signature is not evidence about the batch size.

## 8. The lane axis is the lever (notebook 14, measured 2026-10-18)

**Settled: for every projection on this track, the out-columns must sit on
the 128 lanes.** Pallas maps only the last two dims to the 8x128 register
file, and Mosaic's layout solver puts a matmul RHS's free axis (N =
out-columns) on the lanes. Notebook 11 computed `w @ x` from `(rows, 28)`
tiles, which put 2-16 *in-dims* on the lanes. Flipping the tile to
`(32, rows)` and computing `x @ w` is the change that took the real model
from 29.66 to 281.01 tok/s.

| Fact | Evidence |
| --- | --- |
| `x @ w` over `(32, rows)` decodes bit-identically to `w @ x` over `(rows, 28)` | `colab_run/check14.py`, 4 real geometries, maxdiff 0.0 |
| the flipped block indexing reproduces `y = x @ W` on a simulated grid | `colab_run/check14b.py`, max err 7.6e-05 |
| real model, lane flip alone: 29.66 -> 130.25 tok/s (4.4x) | `14-lane-aligned-ternary-kernel/README.md` (log; cache-on receipt 130.30) |
| 28-byte rows cannot be a Pallas block dim (needs %8==0 or == the array dim), so the flip costs +14 % bytes | pad 28 -> 32; 6.37 GiB measured on device |
| rebuilding the pallas callable per geometry (~400 calls/step) costs **nothing**; `CACHE` was a wrong hypothesis | 130.25 vs 130.30 tok/s, 0.04 % |
| step time is nearly flat in batch at this layout: 245.6 ms (B=32) -> 284.7 ms (B=80) | `receipt-b80-f16-tc1024.json` |
| f16 state storage with the recurrence still fp32 changed no output token in 32 steps | nb 14 runs, byte-identical text |

Best measured configuration: **batch 80, f16 state storage, TC 1024 = 281.01
tok/s** -- 9.5x the lane-starved baseline. The wall at batch 80 is memory
(15.67 of 15.75 GiB), not the kernel.

## 9. The fused 64-layer step, and the constants trap that ate 2 h 12 m

| Fact | Evidence |
| --- | --- |
| all 64 layers in one `jax.jit` compiles in **139.6 s** and runs the model at **180.6 ms/step** (batch 1, `max_t` 8192): **5.54 tok/s**, first token 271, output byte-identical to the per-layer-jit run | `14-lane-aligned-ternary-kernel/receipt-b1-f16-tc1024-fused.json` against `receipt-b1-f16-tc1024-franch.json` (4.62 tok/s) |
| **Do not close over the weight tiles.** JAX then captures the whole weight set as constants: *"A large amount of constants were captured during lowering (6.95GB total)"*. The compile never finished and the kernel connection dropped 2 h 12 m later with the job still running | session log 2026-10-19 13:11-15:23 |
| pass the per-layer weight pytree as an **argument** and the same program compiles in 139.6 s | same |
| a persistent compile cache (`jax_compilation_cache_dir=/content/jaxcache`) makes the rerun's prefill 4.6 s instead of a fresh compile | same run, prefill line |
| the correctness gate value is **prompt-dependent**: `Hello` -> first token 11 `','`; a 17-token prompt -> 271. A check that expects one value for every prompt will abort a good run | `14-lane-aligned-ternary-kernel/receipt-b1-f16-tc1024-fused-smoke.json`, `receipt-b1-f16-tc1024-fused.json` |

## 10. The step is **not** launch-overhead-bound: the projections' own work is the cost

**This section was wrong on 2026-10-18 and was corrected by measurement on
2026-10-20. Read the correction, not the original claim.**

| Fact | Evidence |
| --- | --- |
| 180.6 ms/step at batch 1 (`max_t` 8192) against 284.7 ms/step at batch 80 (`max_t` 64): 80 sequences cost 1.58x one sequence | `14-lane-aligned-ternary-kernel/receipt-b1-f16-tc1024-fused.json`, `receipt-b80-f16-tc1024.json` |
| the step launches ~400 projections (48 GDN layers x 6 + 16 attention layers x 7) | `14-lane-aligned-ternary-kernel/kernel.py` |
| **CORRECTION: the launches are not the cost.** Merging the projections that share an input took the launch count from 400 to 256 -- **36 % fewer -- and the step time moved from 182.3 ms to 181.3 ms, i.e. 0.5 %.** The first token is `271` in both runs and the 64 generated tokens are byte-identical, so the fusion is exact | `14-lane-aligned-ternary-kernel/receipt-fuse0-b1-f16-tc1024.json`, `receipt-fuse1-b1-f16-tc1024.json` |
| what the step really pays for is the **per-weight-byte VPU work**: the ternary unpack of 6.37 GiB of packed bytes, every token, for every sequence. That is why 80 streams share one unpack and one stream pays it alone | same receipts; `13-projection-cost-breakdown/README.md:28` (the unpack is 82 % of a projection) |
| the traffic floor is 6.37 GiB / 709 GB/s = 9.0 ms, so the step is ~20x off the HBM roofline | `13-projection-cost-breakdown/README.md:28` |
| **consequence:** 50 tok/s (20 ms/step) needs the unpack itself to be ~9x faster, or a pass that processes several tokens against the same unpacked weights (`TODO.md` `TPU-03`, `TPU-07`) | same |

## 11. The single-chat fp32 context ceiling is 45k, and 64k needs a q4 KV cache that does not exist

| Fact | Evidence |
| --- | --- |
| `max_t` 45056 runs: 2.86 tok/s, 350.1 ms/step, peak 14.01 GiB | `14-lane-aligned-ternary-kernel/receipt-b1-f16-tc1024-45056.json` |
| `max_t` 49152 is **140 MB short** of loading `jit_attn_step` (needs 2.44 GB, 2.30 GB free) | `14-lane-aligned-ternary-kernel/receipt-mem-ceiling.json` |
| `max_t` 65536 needs 3.26 GB with 184.53 MB free | same |
| fp32 KV is 128 KiB/token/sequence; q4 would be 16 KiB, making 64k fit at ~11.3 GiB -- **not implemented**: there is no q4 KV path and no blocked or paged attention | same |

## 12. The Colab v5e1 pool is quota-limited; the unlock is a re-auth

| Fact | Evidence |
| --- | --- |
| 25 consecutive `503 Service Unavailable` on `colab new --tpu v5e1`, then `Session READY` on the first attempt after the account was re-authorised | 2026-10-19 12:31-13:08 |
| a second round the same evening: 3 failures, then OK on try 4 | 2026-10-19 18:42-18:47 |
| the account itself is fine: `colab new` with no accelerator assigns a CPU session immediately, and `--tpu v6e1` is refused with *"You may not have quota or entitlement"* | 2026-10-19 probes |
| the 120 s auth timeout patch is still required for the re-auth flow itself | `colab-env-patch/colab_timeout_patch.py`, `colab-env-patch/zz_colab_timeout.pth` |
| **after any session re-create the GGUF is gone** (5.54 GiB, `/content/gguf/`) -- re-download before the next run, or the kernel opens a file that is not there | run logs 2026-10-19 |

## 16. Scope: this track is TPU-only, and the Kaggle GPU numbers are references, not results

| Fact | Evidence |
| --- | --- |
| **Decision (2026-10-20):** the project runs Bonsai 2 27B -- and other good models -- **on TPU silicon, at maximum speed**. A GPU measurement answers a different question, does not move the TPU kernel, and is not a result of this track | `AGENTS.md` section 2 rule 7; `tpu/README.md` section 6 |
| the four Kaggle GPU kernels this track had queued (MTP-head test, draft-model + n-gram, 2xT4, MTP graft) were **deleted**; Kaggle's API has no cancel, so deletion was the only way to stop them. Their source stays in this repo and the facts they produced are cited below | 2026-10-20 |
| the 50.44 tok/s @ 48.3 % acceptance and the 25.5 tok/s figures that appear in this repo are **[sourced] references from other hardware** -- they are not track results and may not be quoted as this project's throughput | `tpu/TPU-STATUS.md`; `TODO.md` (the GPU items were withdrawn) |
| the fact that survives: Bonsai 2's GGUFs carry no MTP head, while the base `Qwen/Qwen3.8-27B` does | section 14 |

## 17. The 2x2 projection/attention matrix (2026-10-20) and what it refutes

Every arm ran the real 64-layer model at an 8k window, batch 1, 64 tokens, and
every arm was numerically exact (first token `271`, identical generated text).

| arm | ms/step | tok/s | receipt |
| --- | --- | --- | --- |
| shipped | 159.7 | 6.26 | `14-lane-aligned-ternary-kernel/receipt-base-b1-f16-tc1024.json` |
| `NOCONCAT=1` | 189.2 | 5.28 | `receipt-nc-b1-f16-tc1024.json` |
| `GQA=1` | **134.3** | **7.45** | `receipt-gqa-b1-f16-tc1024.json` |
| both | 163.9 | 6.10 | `receipt-both-b1-f16-tc1024.json` |

| Fact | Why it is settled |
| --- | --- |
| **the kv-head expansion costs ~25 ms/step at an 8k window** | `K[:, :, _KIDX, :]` is (B, T, 24, 256) -- 201 MB per layer per step, 3.2 GB per step; contracting the query groups against the kv heads directly reads the KV once instead of six times, and saves 25.4 ms |
| **the concatenation is not the traffic problem** | removing both concatenations made the step **18.5 % slower**: the fourteen groups are 5x16, 5x8 and 4x2 rows, and the MXU pads each to K=128, so 14 x 128 = 1792 K-slots instead of 128 |
| the unpacked rows of a tile are the **in-dims**, not the out-columns | contracting a 16-row group against the whole (128, QK) x failed with *"contracting dimensions to have the same shape, got (128,) and (16,)"* |
| **the window ceiling is set by the whole-model jit's compile-time temporaries, not by the KV dtype** | `KVDTYPE=float16` at `max_t` 65536 fails with *"CompileTimeHbmOom: ... HLO temporaries (22.68G) exceeds available HBM (15.75G)"*, and at 131072 with *"RuntimeBufferAllocationFailure: Error allocating device buffer: Attempting to allocate 512.00M. That was not possible. There are 184.60M free"*; `KVDTYPE=float32` at 65536 lost its connection during the (very slow) compile, so that point is **inconclusive**, not a measurement |
| the KV cache dtype is now selectable and the attention reads it in its own precision | `KVDTYPE=fp32|fp16|bf16` in `14-lane-aligned-ternary-kernel/kernel.py`: the write casts into the cache dtype and the query is cast to match, so the matmul runs in the cache's precision and no full-size f32 upcast of the cache is materialised -- that temporary would eat the whole saving. A true 4-bit cache needs the dequant fused into the attention (a flash-style Pallas kernel): a 4-bit cache read through an f32 einsum is upcast straight back to 32 bits per element |
| the kv head is the same index in both operands of the grouped attention | the first version used `h` for the query group and `h` for the cache and JAX accepted it silently, producing garbage (first token 220, output `" 3 3 3 ..."`); the corrected form uses `h` in both, with the query head index `h*G + g`, the same order as the expanded `_KIDX` path |
| **do not re-run the "it is the VMEM traffic" theory** | the one experiment that could have tested it says the opposite |
| **the fused whole-model jit fits only the single-stream regime** | `CompileTimeHbmOom: HLO temporaries (20.08G) exceeds available HBM (15.75G)`, module `jit_full_step`, at batch 80 as well as at a 45k window; the aggregate path (281.01 tok/s) needs the per-layer jit (`FUSE=0`) |
| the kv-head expansion is **80x larger at batch 80** (201 MB per layer per step at 8k becomes 16 GB per step at batch 80), so the grouped attention matters more there, not less | the 8k batch-1 saving was 25.4 ms; the batch-80 measurement is the open one, and it is blocked by the row below |
| **the shipped kernel cannot run batch 80 at all** | the whole-model jit needs 20.1 GB of HLO temporaries against 15.75 GiB, and `FUSE=0` does not change that (it governs the projection fusion, not the jit). The 281.01 tok/s aggregate number is the *per-layer-jit* kernel's, preserved in this repository's history at `22e94a1`; see `TODO.md` `TPU-10` |

## 13. Corrections to claims made on this track

- *"The decode step scales with MAX_T, not only with the batch"* (in
  `14-lane-aligned-ternary-kernel/README.md`) is **confounded**: the long-context runs also
  used the smaller batches, so neither term is isolated. The context cost at
  fixed batch has never been measured -- `TODO.md` `TPU-06`.
- *"Context is expensive, batch is not"* (said in conversation on 2026-10-19) is
  **withdrawn**: the 638/662 ms step times were measured at batch 48 and 32,
  which are smaller than the 284.7 ms run at batch 80.
- *"The step is launch-overhead-bound"* (section 10 as it was written on
  2026-10-18) is **withdrawn**: 36 % fewer launches moved the step time 0.5 %.

## 14. Bonsai 2's GGUFs carry no MTP head (measured from the files themselves)

| Fact | Evidence |
| --- | --- |
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, `-PQ2_0.gguf` and `-F16.gguf` each report **851 tensors, 64 blocks, and zero tensors named `nextn`/`mtp`**, and none has a `qwen35.nextn_block_count` key | GGUF headers read with 40 MB byte-range requests, 2026-10-20 |
| the architecture is `qwen35` with `context_length 262144`, 24/4 heads, `full_attention_interval 4` (48 GDN + 16 attention), and the `prism.hadamard.*` metadata | same |
| the base model is `Qwen/Qwen3.8-27B` and *does* have the head: unsloth publishes it as a standalone file `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (18 tensors, `qwen35.nextn_predict_layers: 1`, of which 15 are the `blk.64.*` block) | HF API, 2026-10-20 |
| **so** llama.cpp's `--spec-type draft-mtp` cannot work on the target file as shipped, and the head has to come from the base model | same |

## 15. Pallas `num_scalar_prefetch` does not fit this kernel's grid

| Fact | Evidence |
| --- | --- |
| setting `num_scalar_prefetch=1` on this kernel's `PrefetchScalarGridSpec` fails to compile: *"Pytree for `in_specs` and `inputs` do not match ... `in_specs` is a tuple of length 3 but `inputs` is a tuple of length 2"* | run log 2026-10-20 19:52; the knob is `PREFETCH` in `14-lane-aligned-ternary-kernel/kernel.py`, default 0 |
| the prefetch mechanism wants the prefetched dimension to be a **scalar** dim; this kernel's grid is `(TOUT, 1, BPC)` with the parallel dim first | same |
| the ceiling was small anyway: the weight DMA is 0.29 ms of a 3.83 ms projection (7.6 %), and the A/B above shows the step is VPU-bound rather than DMA-exposed | `13-projection-cost-breakdown/README.md:63-76` |
| **do not** spend a session on this lever without a new reason | - |

MIT OR Apache-2.0
updated: 2026-10-21 01:45 UTC
