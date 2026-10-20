import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

print("backend:", jax.default_backend())


def k(out_ref):
    i = pl.program_id(0)
    out_ref[0:128] = jnp.full((128,), i, jnp.float32)


try:
    f = pl.pallas_call(
        k, out_shape=jax.ShapeDtypeStruct((256,), jnp.float32), grid=(2,),
        out_specs=pl.BlockSpec((128,), lambda i: (i * 128,)))
    o = f()
    jax.block_until_ready(o)
    a = np.asarray(o).reshape(2, 128)
    print(f"grid=(2,) pid OK: prog0={a[0,0]} prog1={a[1,0]}")
except Exception as e:
    print(f"grid=(2,) FAIL: {type(e).__name__}: {str(e)[:150]}")


def k2(out_ref):
    a, b = pl.program_id(0), pl.program_id(1)
    out_ref[0:128] = jnp.full((128,), a * 4 + b, jnp.float32)


try:
    f2 = pl.pallas_call(
        k2, out_shape=jax.ShapeDtypeStruct((4 * 4 * 128,), jnp.float32),
        grid=(4, 4),
        out_specs=pl.BlockSpec((128,), lambda a, b: ((a * 4 + b) * 128,)))
    o2 = f2()
    jax.block_until_ready(o2)
    a2 = np.asarray(o2).reshape(16, 128)
    print("grid=(4,4) pid OK:", [int(a2[i, 0]) for i in range(16)])
except Exception as e:
    print(f"grid=(4,4) FAIL: {type(e).__name__}: {str(e)[:150]}")
