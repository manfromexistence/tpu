"""Exact HF blog fused INT8 weight-only dequant matmul on this stack."""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

print("jax:", jax.__version__, "backend:", jax.default_backend())
M, K, N = 512, 512, 512
bm, bk, bn = 128, 256, 128
X = jax.random.normal(jax.random.key(0), (M, K), jnp.bfloat16)
W = jax.random.normal(jax.random.key(1), (K, N), jnp.bfloat16)
W_int8 = (W * 127).astype(jnp.int8)
Scales = (jnp.abs(W) / 127.0).astype(jnp.bfloat16)
nsteps = K // bk


def fused_dequant_matmul_kernel(x_ref, w_int8_ref, scale_ref, z_ref, acc_ref,
                               nsteps):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    w_bf16 = w_int8_ref[...].astype(jnp.bfloat16) * scale_ref[...]
    acc_ref[...] += jax.lax.dot_general(
        x_ref[...], w_bf16, (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32)
    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


@jax.jit
def quantized_matmul(X, W_int8, Scales):
    kernel_fn = functools.partial(fused_dequant_matmul_kernel, nsteps=nsteps)
    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((M, N), X.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda m, n, k: (m, k)),
                pl.BlockSpec((bk, bn), lambda m, n, k: (k, n)),
                pl.BlockSpec((bk, bn), lambda m, n, k: (k, n)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda m, n, k: (m, n)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(M // bm, N // bn, K // bk),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(X, W_int8, Scales)


print("building...")
t0 = time.perf_counter()
o = quantized_matmul(X, W_int8, Scales)
jax.block_until_ready(o)
print(f"build+exec {time.perf_counter()-t0:.1f}s")
ref = np.asarray(X.astype(jnp.float32) @ (W_int8.astype(jnp.float32)
              * Scales.astype(jnp.float32)))
got = np.asarray(o).astype(np.float32)
d = float(np.abs(ref - got).max())
print(f"HF int8 fused vs ref maxdiff: {d:.3e} "
      f"{'OK' if d < 1.0 else 'MISMATCH'}")
t0 = time.perf_counter()
for _ in range(5):
    jax.block_until_ready(quantized_matmul(X, W_int8, Scales))
ms = (time.perf_counter() - t0) / 5 * 1000
print(f"512x512 int8 fused matmul: {ms:.2f}ms")
