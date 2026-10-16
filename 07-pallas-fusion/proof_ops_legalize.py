"""Which LHS op breaks MXU legalization? All shapes [128,128]."""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

print("jax:", jax.__version__)
TC = 128
B = 128


def try_kernel(name, make_lhs, nsc=1):
    def kernel(rep_ref, xt_ref, o_ref, acc_ref, *extra, nsteps):
        @pl.when(pl.program_id(1) == 0)
        def _():
            acc_ref[...] = jnp.zeros_like(acc_ref)
        just = rep_ref[...]  # [TC, 6*TC]
        w = make_lhs(just, extra)
        acc_ref[...] += jnp.dot(w, xt_ref[...],
                                preferred_element_type=jnp.float32)

        @pl.when(pl.program_id(1) == nsteps - 1)
        def _():
            o_ref[...] = acc_ref[...].astype(o_ref.dtype)

    rep = jax.random.normal(jax.random.key(0), (TC, 6 * TC), jnp.float32)
    xt = jax.random.normal(jax.random.key(1), (TC, B), jnp.bfloat16)
    specs = [
        pl.BlockSpec((TC, 6 * TC), lambda j, k: (0, 0)),
        pl.BlockSpec((TC, B), lambda j, k: (0, 0)),
    ]
    try:
        f = pl.pallas_call(
            functools.partial(kernel, nsteps=4),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0, in_specs=specs,
                out_specs=pl.BlockSpec((TC, B), lambda j, k: (j, 0)),
                scratch_shapes=[pltpu.VMEM((TC, B), jnp.float32)],
                grid=(4, 4)),
            out_shape=jax.ShapeDtypeStruct((4 * TC, B), jnp.bfloat16),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary")),
        )
        o = f(rep, xt)
        jax.block_until_ready(o)
        print(f"{name}: OK")
    except Exception as e:
        print(f"{name}: FAIL {str(e)[:100]}")


# V1: pure slice+cast (no transpose, no concat)
try_kernel("V1 slice+cast",
           lambda j, e: j[:, :TC].astype(jnp.bfloat16) * 1.0)
# V2: with transpose (like unpack)
try_kernel("V2 transpose+reshape",
           lambda j, e: j.reshape(TC, 6, TC).transpose(0, 2, 1).reshape(TC, 6 * TC)[:, :TC].astype(jnp.bfloat16))
# V3: with concat (like unpack)
try_kernel("V3 concat",
           lambda j, e: jnp.concatenate(
               [j[:, :64].astype(jnp.bfloat16),
                j[:, 64:TC].astype(jnp.bfloat16)], -1))
