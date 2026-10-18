# Pallas fusion on Colab TPU v5e-1: findings

Goal: fuse PTQ1_0 ternary decode into the matmul so no bf16/fp32 weight ever
touches HBM — the 27.5-108 GB/step traffic that caps notebooks 06/08/09 at
3.5-4.0 tok/s. This file records exactly what works, what does not, and the
evidence. All results `[exec]` on Colab TPU v5e-1, jax/jaxlib 0.11.2, libtpu
0.0.50, unless noted.

**Status 2026-10-18: the fused kernel works and is numerically validated on
real layer-0 weights.** §1-§3 and §6 stand as written. §4 and §5 of the first
revision blamed a Mosaic lowering bug; that conclusion was wrong. It is
retracted in §4, and §7 records what the working kernel costs.

## 1. Environment prerequisites (both mandatory)

- **Pallas needs jax 0.11.2 + libtpu 0.0.50.** Stock Colab (jax 0.7.2,
  libtpu 0.0.21.1) fails with a Mosaic version mismatch
  (`Unsupported version: expected <= 7 but got 8`) and, after upgrading jax
  alone, `Pallas TPU requires a recent libtpu version (at least 0.0.46)`.
  Fix scripts: `upgrade_jax.py` then `upgrade_libtpu.py`, then
  `colab restart-kernel`.
- **A kernel restart is mandatory after any core halt.** A failed Pallas
  kernel (core halt) leaves the TPU runtime wedged: every subsequent Pallas
  call then fails with `RuntimeUnexpectedCoreHalt` / "program continuator
  has halted unexpectedly", even for kernels that previously worked. After a
  halt, `colab restart-kernel` restores normal operation. This masqueraded as
  "multi-program grids are broken" for a long time.
- Compile failures (`failed to legalize`, `Bad rhs type`) do **not** wedge the
  core, so compile-only variants can be batched into one process. Runtime
  halts must be isolated one per restart.

## 2. The correct Pallas grid API on this stack

Two forms of `pl.pallas_call` behave very differently:

- **Plain `in_specs=/out_specs=/grid=` (no `compiler_params`):** a grid with
  more than one program halts the core, and even a trivial `pl.kernel` /
  `pltpu.emit_pipeline` example from the JAX docs halts.
  Evidence: `test_pid.py`, `test_grid_ok.py`, `test_plkernel.py`.
- **`grid_spec=pltpu.PrefetchScalarGridSpec(...)` +**
  **`compiler_params=pltpu.CompilerParams(dimension_semantics=...)`:**
  **works.** The canonical JAX `matmul` from the docs (tiled, pipelined,
  3-D grid, accumulator scratch, `program_id`) compiles and is bit-exact on
  this stack. Evidence: `test_base_mm.py` ("BASELINE docs matmul OK
  maxdiff 0.0000").

So: use the `PrefetchScalarGridSpec` + `dimension_semantics` form, and
restart the kernel after any halt. `("parallel", "parallel", "arbitrary")` is
accepted; `"sequential"` is rejected on this stack.

## 3. What legalizes in isolation

Building the fused kernel's pieces separately (all with the correct API
form) shows each is fine. `test_bisect_mm.py` / `test_bisect4.py`:

- computed LHS inline (`x_ref[...] * 2.0` then `jnp.dot`) -> OK
- computed LHS written to a scratch ref then dotted -> OK
- `jnp.swapaxes` on an operand inside the kernel -> OK
- LHS built with `slice`/`astype` -> OK
- LHS built with `transpose(0,2,1)+reshape` -> OK
- LHS built with `jnp.concatenate` -> OK

Also verified independently:

- Pallas single-pass PTQ1_0 unpack is **bit-exact** vs the proven XLA
  decoder (`proof_unpack_exact.py`, maxdiff `0.000e+00`).
- Minimum dynamic-slice alignment inside a Pallas program is 128
  (`test_align.py`: ALIGN=4/32 fail, 128 works); u8 reductions require a cast
  to int32.
- `u16.view(jnp.float16)` (i16->f16 bitcast) is **not** supported by Mosaic:
  pass scales as a separate pre-decoded input.
- `pl.dot` is removed in jax 0.12; use `@` / `jnp.dot`.

## 4. Retraction: the blocker was two bugs in my harness

The first revision of this file recorded "failed to legalize operation
'tpu.matmul'" as a Mosaic lowering bug and listed the eliminated causes. That
diagnosis was wrong. Both failure modes had mundane causes in the surrounding
harness, and both are now fixed.

### 4.1 `pl.BlockSpec` index maps return **block** indices, not element offsets

This was the whole of the `RuntimeUnexpectedCoreHalt` /
`BoundsCheck 2 [deref of opnd#0] for dma.hbm_to_vmem` family, and therefore
the whole of the "multi-program grids are broken" theory.

Pallas multiplies the index your index-map returns by the block shape to get
the element offset. Passing an element offset (e.g. `(m*BPC + k)*TC`) makes
the row offset get multiplied by 128 a second time, which walks far outside
the array and halts the core. Two more rows of arithmetic did not look like a
bug because the numbers stayed in range for `m = k = 0`.

Correct index maps for the fused kernel, with the packed tile array laid out
in block order:

```python
pl.BlockSpec((TC, B),   lambda m, n, k: (k, 0)),            # x^T  [IN, B]
pl.BlockSpec((TC, 128), lambda m, n, k: (m * BPC + k, 0)),  # rep  [ROWS, 128]
pl.BlockSpec((TC,),     lambda m, n, k: (m * BPC + k,)),    # scale [ROWS]
out_specs=pl.BlockSpec((TC, B), lambda m, n, k: (m, 0)),    # out  [OUT, B]
grid=(TOUT, 1, BPC)
```

### 4.2 A global `jax.config` side effect from importing notebook 04

`tpu/04-bonsai-tpu-forward/kernel.py:47` runs

```python
jax.config.update("jax_default_matmul_precision", "highest")
```

at import time. Any process that imports the GGUF reader from notebook 04 —
which every test harness does — silently gets `precision=highest` on every
matmul, and Mosaic then rejects the bf16 MXU op with

```
Mosaic failed to compile TPU kernel ... Bad rhs type: 128, 128
```

That is the error the first revision attributed to the unpack. Stubbing,
slicing, transposing and concatenating could never have fixed it: the kernel
was already legal. Fix: reset the config after the exec, and pin
`precision=jax.lax.Precision.DEFAULT` explicitly on the `dot_general` so the
kernel does not depend on ambient state.

```python
exec(compile(src.split("n_layers = ws.meta")[0], "04m.py", "exec"), ns)
jax.config.update("jax_default_matmul_precision", "default")
```

### 4.3 A reference-matrix reshape trap

The two obvious spellings of the Fortran reshape are **not** the same
function, and the 89%-of-elements-apart difference looks exactly like a
broken kernel:

```python
dec.reshape((IN, OUT), order="F")              # form A  -- NOT what the model uses
dec.reshape(-1).reshape((IN, OUT), order="F")  # form B  -- what the model uses
```

Measured on layer-0 qkv: `maxdiff 1.371e-01`, **46,780,136 of 52,428,800
entries differ (89%)**. Form B sends destination `(i, j)` to `flat[i + j*IN]`
of the C-flattened decode; form A sends it to element `i + j*IN` of the
source traversed in F order. `tpu/04-bonsai-tpu-forward/kernel.py:280-281`
(`layer_weights`) uses form B via `flat[o:o+n].reshape(dims, order="F")`.

Comparing the fused kernel against form A produced a `rel ~ 2` disagreement
that survived several rounds of debugging and very nearly got the working
kernel rewritten. Anything that reshapes a decoded weight must use form B.

## 5. What is now proven

Harness: `tpu/10-bonsai-fused/kernel.py`, one run on Colab TPU v5e-1.
Reference is the model's own convention (§4.3 form B).

| Check | Result |
| --- | --- |
| Tiled reconstruction vs model convention | **maxdiff 0.0** (exact) |
| Fused kernel vs `W_ref @ x`, B=128 | maxdiff 1.65e-02 on `ref|max|` 5.46 |
| Fused kernel vs `W_ref @ x`, B=32 | maxdiff 1.22e-02 |
| Fused kernel vs `W_ref @ x`, B=1 | maxdiff 6.31e-03 |
| Kernel faithfulness to its host layout | maxdiff 1.70e-02 |
| Accumulation test (all-ones LHS) | 2.19e-01 on `ref|max|` 86.2 |
| Real GGUF blocks matched to the decode, all 409600 | 409600/409600 exact |

The kernel — one program per (out-tile, in-block), unpack PTQ1_0 to bf16 in
VMEM, MXU-accumulate over the in-block dim, W as the MXU LHS — compiles, runs
and is numerically correct. **W as the MXU LHS and an in-kernel unpack are
both legal.** The earlier claims that a computed-operand layout "is not
MXU-consumable" and that "the in-kernel transpose is illegal" are withdrawn.

## 6. The HF fused INT8 pattern also works (control experiment)

`proof_hf_int8_fused_works.py` reproduces the Hugging Face "Fused INT8
Weight-Only Quantization in Pallas" kernel verbatim (jax 0.11.2, libtpu
0.0.50, v5e-1): int8 weights + bf16 scales, in-kernel cast to bf16, then
`jax.lax.dot_general(..., (((1,),(0,)),((),())))`. Result: `build+exec 0.1s`,
maxdiff 0.25, 0.28 ms for 512x512.

This is a control, not a path: it was never the blocker, and it is not
faster. It stays in the record because it proves the fused dequant-matmul
pattern is sound on this runtime.

## 7. Measured cost of the fused ternary matmul

Layer 0 `blk.0.attn_qkv.weight`, `dims [5120, 10240]` -> IN 5120, OUT 10240,
TC 128, BPC 40, TOUT 80, ROWS 409600, raw 11,468,800 bytes (`OUT*BPC*28`).

| Variant | B | ms | notes |
| --- | --- | --- | --- |
| V1: 28-byte GGUF blocks, in-kernel unpack | 128 | 4.18 | correct |
| V1 | 32 | 4.07 | correct |
| V1 | 1 | 4.04 | correct |
| V2: host 2-bit repack, trivial decode | 128 | 30.73 | correct, **7.5x slower** |

Two facts matter:

1. **The cost is flat in batch size.** 4.04 ms at B=1 and 4.18 ms at B=128
   means the time is set entirely by the per-forward-pass weight work, not by
   the matmul. Batching amortises it for free.
2. **V2 is 7.5x slower, so no host repack is needed or wanted.** The GGUF
   28-byte blocks go straight in. V2's block is 32 bytes per row, below the
   128-byte minimum DMA alignment established in §3, which is the likely
   cause — but the conclusion holds either way: keep the packed 5.9 GB in HBM
   and unpack in-kernel.

Rates, `[exec]`:

| Quantity | Value |
| --- | --- |
| packed bytes streamed per traversal | 11.5 MB |
| effective packed-weight bandwidth | ~2.8 GB/s (vs ~820 GB/s HBM) |
| trits unpacked | 52.4 M |
| effective unpack rate | ~12.8 Gtrit/s |

So the kernel ran roughly 300x below the HBM roofline: compute and
program-overhead bound, not traffic bound. **Both live hypotheses have since
been separated by notebook 13, and the first one was right** (2026-10-18):

| Hypothesis | Measurement | Verdict |
| --- | --- | --- |
| the layout shuffles inside the unpack dominate | same trits in the same columns, `concatenate(-1)` for `stack(axis=1).reshape`: 3.126 -> 1.866 ms at TO=512 | **yes — 1.68x** |
| 3200 programs at ~1.3 us each is launch overhead | `TO` 128 -> 1024 is 3200 -> 400 programs: 3.834 -> 3.249 ms | 1.18x, saturating by 512 — minor |

The V2 row above argues *for* the first one, not against it. V2 did less
arithmetic and was 7.5x slower because it still had to place trits into
columns, and its extraction was a broadcast of exactly the kind notebook 13
measures at 4.2x worse than the concatenated spelling. V2's penalty is
evidence about the extraction pattern, not about 2-bit packing.

The fix, measured on the model rather than on one tensor: spelling the decode
with `concatenate` is 1.33x end to end, raising the out-tile to 512 is 1.21x,
and doing the decode in int16 lanes is 1.20x. Notebook 11 now measures
**29.66 tok/s** against the 15.31 in the table above.

## 8. Consequence for the throughput target

**Status: superseded by notebook 11, which is the projection that was actually
executed. Keep this section for the arithmetic, not as a plan.**

Extrapolating layer-0 qkv's measured cost per parameter to the ~27.0 B
ternary parameters in the 5.9 GB GGUF (x515) gives ~2.15 s of weight work per
forward pass, excluding attention, GDN, norms and sampling:

| Batch | Projected aggregate | Outcome |
| --- | --- | --- |
| 1 | ~0.5 tok/s | -- |
| 32 | ~15 tok/s | **measured 15.30** (nb 11) |
| 128 | ~59 tok/s | **retracted** -- assumed a 27.0 GB layout (`DO-NOT-RETEST.md` 6) |

Batch is the free lever but the GDN recurrence caps it near 32, so ~15 tok/s
was where the kernel landed in nb 11 -- better than notebook 09's 4.0 tok/s,
and nowhere near 100. The remaining gap was in the unpack itself (§7), and
**notebook 13 plus a second nb 11 session closed part of it: 29.66 tok/s now**,
with the decode's spelling and lane width the only changes to the kernel.

The headroom that remains is smaller than the 300x figure suggests, but it is
also no longer the binding constraint. At 29.66 tok/s the model runs
26.9 Gtrit / 1.079 s = **24.9 Gtrit/s**, against 38.5 Gtrit/s for notebook 13's
best decode cell and 182 Gtrit/s for its pure-DMA floor. Pricing the model's
513 `attn_qkv`-equivalents at that measured 1.393 ms per cell accounts for
~715 ms of the 1079 ms step, leaving ~364 ms of non-projection work (attention,
GDN, norms, the output projection, sampling).

That arithmetic decides the 100 tok/s question, and the answer is not "a faster
decode":

| target | step budget at batch 32 | verdict |
| --- | --- | --- |
| 100 tok/s | 320 ms | **out of reach at this batch**: the non-projection work alone is ~364 ms |

Batch is the only way through, because the kernel is flat in batch: the same
~1079 ms step carries 32 tokens, would carry 64 (~59 tok/s) and would carry 128
(~119 tok/s). The block is the GDN recurrence's f32 state, ~151 MB per sequence
per 48 layers: batch 64 needs bf16 state to fit next to the tiles, and batch 128
does not fit even then. So the concrete route is **batch 64 plus a ~2.6x faster
projection** (0.538 ms per `attn_qkv`-equivalent against 1.393 ms now) -- above
the 0.291 ms load floor, so possible; not demonstrated, so not claimed.

**How to pursue it -- rewritten 2026-10-18.** This used to read "run the
separating experiment, re-optimise, and *only then* wire the fused kernel into
the 64-layer scan". That order was wrong and it is retracted. It made
single-tensor benchmarking the default and the real model the reward at the
end, and this track has already been burned three times by single-layer
projections that did not survive contact with the model (this section's own
batch-128 row, `DO-NOT-RETEST.md` §6, and nb 12's confounded k-loop
refutation).

1. **Run the real 64-layer model first.** Print its tok/s, ms/step, the
   correctness gate and the generated text. This is the number that counts.
2. With that number in the log, take the separating measurements of §7 inside
   the same session.
3. Re-optimise the unpack with the answer, and confirm with **another real
   model run** -- not with a re-timed single tensor.

The model run is the cheap one: nb 11's tile build was 8.7 s, and compile plus
first step 10.0 s.

The hardware is capable (v5e: 197 TFLOP/s bf16, 393 TOPS int8, 820 GB/s). The
kernel is correct. The open question is purely how fast the unpack can be made,
and it is answered on the model.

MIT OR Apache-2.0
updated: 2026-10-18 04:20 UTC
