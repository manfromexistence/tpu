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
`07-pallas-fusion/FINDINGS.md` §8 as they were first written.

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
| Pallas needs jax 0.11.2 + libtpu 0.0.50; stock Colab fails with a Mosaic version mismatch | `07-pallas-fusion/FINDINGS.md` §1 |
| A core halt wedges the runtime across processes; only `colab restart-kernel` clears it | `07-pallas-fusion/FINDINGS.md` §1 |
| Compile errors do **not** wedge the core, so compile-only variants batch into one process | same |
| The `colab` 120 s auth timeout needs the `.pth` patch in `env/` | `env/colab_timeout_patch.py` |
| v6e1 has no quota; T4 GPU returns `Service Unavailable` | session log |
| Weights come from HF, not the Kaggle API (Kaggle 403s from Colab IPs) | `dl_hf.py` |

## 2. Pallas API facts

| Fact | Evidence |
| --- | --- |
| Use `pltpu.PrefetchScalarGridSpec` + `CompilerParams(dimension_semantics=...)`; plain `grid=` halts the core | `07-pallas-fusion/FINDINGS.md` §2 |
| `("parallel","parallel","arbitrary")` is accepted; `"sequential"` is rejected | same |
| **`pl.BlockSpec` index maps return block indices, not element offsets** | `07-pallas-fusion/FINDINGS.md` §4.1 |
| `u16.view(jnp.float16)` (i16->f16 bitcast) is unsupported: pass scales as a separate input | `proof_min_align_128.py` |
| `pl.dot` is removed in jax 0.12; use `@` / `jnp.dot` | same |
| Minimum dynamic-slice alignment inside a Pallas program is 128 | `proof_min_align_128.py` |
| Computed LHS, in-kernel transpose, `slice`, `concat`, `cast` all legalize | `07-pallas-fusion/FINDINGS.md` §3 |

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
fix and it is already written into `10-bonsai-fused/kernel.py`.

### `BoundsCheck 2 [deref of opnd#0] for dma.hbm_to_vmem` (then a wedged core)

A `BlockSpec` index map is passing element offsets where Pallas wants block
indices. Audit every index map before running. Cost of getting this wrong is
a wasted session plus a forced restart.

## 4. Model facts (no session needed to use these)

| Fact | Evidence |
| --- | --- |
| PTQ1_0 block is 28 bytes: `qs[24]`, `qh[2]`, fp16 `d`; 128 trits | `ggml-common.h:214` |
| Within a block: idx 0-79 = qs bytes 0-15 power-major; 80-119 = qs bytes 16-23 power-major; 120-127 = `qh` **interleaved** per power | `04-bonsai-tpu-forward/kernel.py:90-96` |
| The model's weight matrix is `decode(...).reshape(-1).reshape(dims, "F")` -- **not** `decode(...).reshape(dims, "F")` | `04-bonsai-tpu-forward/kernel.py:280-281` |
| Those two forms differ in 89% of entries on layer-0 qkv | `07-pallas-fusion/FINDINGS.md` §4.3 |
| Tiled reconstruction of the raw blocks == the model's weight matrix, exactly | `10-bonsai-fused/kernel.py` run log, maxdiff 0.0 |
| Block (out `o`, in-block `k`) is at flat block index `o*BPC + k` | `07-pallas-fusion/FINDINGS.md` §4.3 |

## 5. Measured performance (do not re-measure)

Layer-0 `blk.0.attn_qkv.weight`, 52.4 M params, `[5120, 10240]`:

| Layout | Block | ms | Correct |
| --- | --- | --- | --- |
| 28-byte GGUF blocks, in-kernel unpack | `(128, 128)` | 4.04-4.18 | yes |
| host 2-bit repack, 32-byte rows | `(128, 32)` | 30.6-30.7 | yes |

`[exec]` on v5e-1, `10-bonsai-fused/kernel.py`.

- **The cost is flat in batch.** 4.04 ms at B=1, 4.18 ms at B=128.
- **A host 2-bit repack is 7.5x slower.** Do not propose one again without
  explaining what changed.

## 6. A correction: the "59 tok/s at batch 128" projection was against an impossible layout

`10-bonsai-fused/README.md` projected ~59 tok/s at batch 128 by scaling the
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

## 7. The fused full model, measured twice -- and one refuted hypothesis

Both runs executed the real 64-layer model on Colab TPU v5e-1, jax 0.11.2 /
libtpu 0.0.50, and both printed the same correctness gate (first token `11`,
`','`) and the same generated text as notebook 04. Receipts:
`11-bonsai-fused-full/receipt-colab-tpu.json`,
`12-bonsai-bigdma/receipt-colab-tpu.json`.

| config | tok/s (B=32) | ms/step | steps/forward |
| --- | --- | --- | --- |
| nb 11, one in-block per step | **15.30** | **2091** | 1.64 M |
| nb 12, eight in-blocks per step | 13.30 | 2406 | 410 K |

**Do not retest "the k-loop is the bottleneck".** It is refuted. Cutting the
sequential step count 4x made the model 15 % slower, so the ~2 s per forward
pass does not scale with the number of weight DMAs.

**Do not retest "a cheaper unpack will show up in the wall clock".** NB-11's
decode was replaced by `(3*((q*p) & 255)) >> 8`, proven bit-identical to the
old shift chain across all 256 byte values x 5 powers (0 mismatches), ~1.4x
fewer ops on the hot path. Measured effect: none.

Excluded so far: HBM bandwidth (2.8 GB/s effective vs ~820 GB/s peak), the MXU
(batch 16 -> 32 costs 4 %), DMA count (above), unpack instruction count at the
1.4x scale (above). Still unseparated: the unpack/cast/scale chain itself
versus Mosaic's scheduling of it. **Settle that inside the next full-model
run** (§0), with the ablations of `07-pallas-fusion/FINDINGS.md` §7 taken
after the model number is in the log.

**Also settled: scoped VMEM is 16 MB and unpack intermediates scale with the
in-block width.** A block covering all `BPC` in-blocks needs 74 MB and fails
with `E1001: CompileTimeScopedVmemOom`. Keep the per-step tile small
(`(TO, KB, 28)` was 32 KB per intermediate).

**And: restart the kernel between runs.** `colab exec` reuses the process, so
the previous run's 6 GB of tiles stay resident and produce a
`RuntimeBufferAllocationFailure` about 80 MB short of the true limit. That
signature is not evidence about the batch size.

MIT OR Apache-2.0
updated: 2026-10-18 03:42 UTC

