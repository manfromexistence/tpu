import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

print("backend:", jax.default_backend())
N = 4096


def probe(ALIGN):
    def k(x_ref, out_ref):
        def body(b, acc):
            return acc + jnp.sum(x_ref[pl.ds(b * ALIGN, ALIGN)],
                                 keepdims=True)

        out_ref[0:1] = jax.lax.fori_loop(0, N // ALIGN, body,
                                         jnp.zeros((1,), jnp.float32))

    x = jnp.asarray(np.arange(N, dtype=np.float32))
    try:
        f = pl.pallas_call(
            k, out_shape=jax.ShapeDtypeStruct((1,), jnp.float32), grid=(1,),
            in_specs=[pl.BlockSpec((N,), lambda i: (0,))],
            out_specs=pl.BlockSpec((1,), lambda i: (0,)))
        o = f(x)
        jax.block_until_ready(o)
        return f"ALIGN={ALIGN} f32 OK sum={float(o[0]):.0f}"
    except Exception as e:
        return f"ALIGN={ALIGN} f32 FAIL: {str(e)[:90]}"


for a in (4, 32, 128):
    print(probe(a))


def probe_u8(ALIGN):
    def k(x_ref, out_ref):
        def body(b, acc):
            return acc + jnp.sum(x_ref[pl.ds(b * ALIGN, ALIGN)])

        out_ref[0:1] = jax.lax.fori_loop(0, N // ALIGN, body,
                                         jnp.zeros((1,), jnp.int32))

    x = jnp.asarray(np.arange(N, dtype=np.uint8))
    try:
        f = pl.pallas_call(
            k, out_shape=jax.ShapeDtypeStruct((1,), jnp.int32), grid=(1,),
            in_specs=[pl.BlockSpec((N,), lambda i: (0,))],
            out_specs=pl.BlockSpec((1,), lambda i: (0,)))
        o = f(x)
        jax.block_until_ready(o)
        return f"ALIGN={ALIGN} u8 OK sum={int(o[0])}"
    except Exception as e:
        return f"ALIGN={ALIGN} u8 FAIL: {str(e)[:90]}"


for a in (32, 64, 128):
    print(probe_u8(a))
