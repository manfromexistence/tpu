"""Baseline vs computed-operand Pallas matmul, bisect the failure."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

print("jax:", jax.__version__, "backend:", jax.default_backend())
M = K = N = 512


def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)
    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...],
                            preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


def mm(x, y, bm=128, bk=128, bn=128):
    m, k = x.shape
    _, n = y.shape
    return pl.pallas_call(
        functools.partial(matmul_kernel, nsteps=k // bk),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                      pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, k // bk),
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(x, y)


x = jax.random.normal(jax.random.key(0), (M, K), jnp.bfloat16)
y = jax.random.normal(jax.random.key(1), (K, N), jnp.bfloat16)
try:
    z = mm(x, y)
    jax.block_until_ready(z)
    ref = np.asarray(x @ y)
    d = float(np.abs(np.asarray(z) - ref).max())
    print(f"BASELINE docs matmul OK, maxdiff {d:.4f}")
except Exception as e:
    print(f"BASELINE FAIL: {type(e).__name__}: {str(e)[:200]}")
