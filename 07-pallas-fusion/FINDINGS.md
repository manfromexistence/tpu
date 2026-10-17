# Pallas fusion on Colab TPU v5e-1: findings (2026-10-16)

Goal: fuse PTQ1_0 ternary decode into the matmul so no bf16/fp32 weight
ever touches HBM (the 27.5-108 GB/step traffic that caps notebooks 06/08/09
at 3.5-4.0 tok/s). This file records exactly what works, what does not, and
the evidence, so the next attempt starts from the frontier instead of from
zero. All results `[exec]` on Colab TPU v5e-1, jax/jaxlib 0.11.2, libtpu
0.0.50.

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
restart the kernel after any halt.

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

## 4. What does NOT legalize (the current blocker)

The assembled fused kernel -- one program per (out-tile, in-block), unpack
PTQ1_0 to bf16 in VMEM, MXU-accumulate -- compiles every part but fails at

    failed to legalize operation 'tpu.matmul'
    tpu.matmul ... <dimension_numbers = <[1], [0], [0], [1], [0, 0, 1, 1],
      [], []>, precision = fp32, transpose_lhs = false,
      transpose_rhs = false> :
      (vector<128x128xbf16>, vector<128x128xbf16>, vector<128x128xf32>)
      -> vector<128x128xf32>

Notably the operands are already the ideal shapes/dtypes ([128,128] bf16
into f32), and the same `jnp.dot` with the same block sizes works in the
docs `matmul`. Excluded as causes by direct experiment:

- op types: slice / transpose / concat / cast each legalize in isolation
- operand count: fails with 3, 4 and 5 inputs
- power-vector inputs: fails with powers unrolled inline (no array input)
- scratch materialization of LHS: still fails
- packed-data block rank: 1-D and 2-D blocks both fail
- unpack arithmetic: fails with the unpack stubbed to a trivial op

So the failure is specific to the *combination* (grid + indexed packed
inputs + accumulated MXU with a computed LHS). This is a Mosaic lowering
bug/limitation in jax 0.11.2 + libtpu 0.0.50 on v5e, not a shape or dtype
error in the kernel.

Two orientations were tried with the same result, and the second is
diagnostic:

- RHS = token matrix, LHS = unpacked W: fails legalize, `rhs_contracting=[0]`.
- LHS = token matrix, RHS = unpacked W transposed (the canonical docs
  orientation, so the only change from the working docs kernel is a computed
  RHS): fails legalize with `rhs_contracting=[1]` even though the contraction
  must be on rhs dim 0. The compiler is contracting the wrong RHS dimension
  and `transpose_rhs=false`, i.e. the `.T`/computed-operand layout is not
  being honoured -- the same root cause in both orientations.

Scripts (all in `G:\Temp\UserTemp\opencode\colab_run\`): `test_fused_mm.py`,
`test_fused_mm2.py`, `test_fused_mm3.py`, `test_fused_mm4.py`,
`test_bisect_mm.py`, `test_bisect2.py`, `test_bisect3.py`,
`test_bisect4.py`, `test_base_mm.py`.

## 5. Consequence for the throughput target

Measured ceiling without Pallas fusion: **4.0 tok/s aggregate** (notebook 09,
int8 + `lax.scan`, batch-8, correct output). The traffic roofline says a
fused decode could reach ~100+ tok/s aggregate, but the fused MXU kernel does
not compile on this stack. Remaining options, in order of feasibility:

1. Re-test the fused kernel on a jax/libtpu pair where the docs matmul and
   the computed-LHS matmul both lower (try a newer libtpu, or v6e) -- the
   kernel is written and only the lowering is in question.
2. Emit the unpack as a separate Pallas kernel (which is proven bit-exact)
   and feed a standard docs-style matmul from HBM: loses the traffic win
   (that is notebook 09), so not a path to 100.
3. Non-Pallas fusion via XLA only: measured to give ~5% per step
   (notebook 09), i.e. not a path to 100 either.

The hardware is capable (v5e: 197 TFLOP/s bf16, 820 GB/s); the block is the
Mosaic lowering of the fused kernel on this specific runtime.

MIT OR Apache-2.0
updated: 2026-10-16 00:00 UTC

## 6. The HF blog pattern DOES work on this stack (control experiment)

`proof_hf_int8_fused_works.py` reproduces the Hugging Face "Fused INT8
Weight-Only Quantization in Pallas" kernel verbatim (jax 0.11.2, libtpu
0.0.50, v5e-1): int8 weights + bf16 scales, in-kernel cast to bf16, then
`jax.lax.dot_general(..., (((1,),(0,)),((),())))`. Result:
`build+exec 0.1s`, maxdiff 0.25, 0.28 ms for 512x512. So the fused
dequant-matmul pattern itself is sound on this runtime; the block is the
ternary PTQ1_0 unpack specifically (complex int32 bit extraction +
transpose + concat) -- its output layout is not MXU-consumable even after
materializing to a scratch ref.

This pins the remaining work to ONE of:
  (a) simplify the PTQ1_0 unpack so the bf16 result is produced in a
      contiguous, non-transposed layout the MXU accepts (e.g. re-pack the
      trits on the host into [K, N] int8 first, then reuse the HF kernel);
  (b) try a different jax/libtpu pair where the computed-operand lowering
      works (the kernel is otherwise complete);
  (c) accept 4.0 tok/s on TPU, or move to GPU + llama.cpp (25+ tok/s T4,
      100+ on H100 today).
