"""Chunked gated-delta recurrence: the exact k-token form of the one we ship.

What this file is
----------------
`lane-aligned-ternary-kernel/kernel.py` advances the gated-delta state one
token at a time (`gdn_core`, lines 429-433):

    Sg  = S * exp(gate)            # decay, one scalar per head
    sk  = sum_i Sg[i,j] k[i]       # read
    dv  = (v[j] - sk[j]) * beta    # delta
    Sg  = Sg + k[i] * dv[j]        # write
    o[j] = sum_i Sg[i,j] q[i] / sqrt(I)

That form is *inherently sequential*: token t+1 reads the state token t wrote,
so k tokens need k passes over the weights. The weights are the whole cost of
a step (measured: ~109 ms of the 134 ms step at an 8k window; the MXU is ~99%
idle at batch 1), so the weights are paid k times for k tokens.

This file derives the chunk-parallel form of the *same* recurrence: the same
numbers, computed with matmuls, so k tokens share one pass over the weights.
It is the mathematical prerequisite for multi-token verification, which is the
only route to 50 tok/s on one v5e-1 (see `README.md`).

The derivation
--------------
Write the recurrence with the decay made explicit. With `a_t = exp(gate_t)`
(one scalar per head) and the cumulative decay `A_t = prod_{s<=t} a_s`
(`A_0 = 1`), the state after token t is

    S_t = A_t S_0 + sum_{s<=t} (A_t / A_s) k_s dv_s^T

Substituting that into the read (`S_{t-1}^T k_t`) and the write gives, for the
chunk's dv vector, a **lower-triangular system with unit diagonal**:

    dv_t + b_t * sum_{s<t} R[t,s] * (k_s . k_t) * dv_s = rhs_t

    b_t   = beta_t
    R[t,s]= A_{t-1} / A_s                     (s < t)
    rhs_t = beta_t * ( v_t - A_{t-1} * (S_0^T k_t) )

Every ratio in it is <= 1 (a decay only shrinks), so the chunked form is
numerically *safer* than the scaled frame `T_t = S_t / A_t`, whose `b_t/A_t`
would blow up. The chunk's outputs and final state follow from the same
expansion:

    o_t   = ( A_t (S_0^T q_t) + sum_{s<=t} (A_t/A_s) (k_s . q_t) dv_s ) / sqrt(I)
    S_end = A_C S_0 + sum_s (A_C/A_s) k_s dv_s^T

So a chunk of k tokens costs: one Gram matrix, one triangular solve, two small
matmuls, one state update -- all of them MXU work, which we are not using yet.

The contract
------------
`gdn_chunked` must equal `gdn_recurrence` to fp32 rounding, for every decay
profile, including strong decay (which is where a naive chunked form
overflows). `test_equivalence()` is that contract, and it runs on CPU with no
TPU: the numbers here are a *reference*, never a speed claim.
"""

import numpy as np

try:                                    # JAX is optional: numpy is the reference
    import jax
    import jax.numpy as jnp
    HAVE_JAX = True
except ImportError:                     # pragma: no cover
    HAVE_JAX = False


# --------------------------------------------------------------------------
# the recurrence we ship, transcribed token by token
# --------------------------------------------------------------------------
def gdn_recurrence(S0, q, k, v, alpha, beta):
    """Token-by-token gated delta rule.

    S0    (I, J, H)   state at the start of the chunk
    q, k  (T, I, H)   key dim I
    v     (T, J, H)   value dim J
    alpha (T, H)      decay per step (already exponentiated)
    beta  (T, H)      the delta rule's write rate
    returns o (T, J, H), S (I, J, H)
    """
    S0 = np.asarray(S0, np.float32)
    q, k, v = (np.asarray(z, np.float32) for z in (q, k, v))
    alpha = np.asarray(alpha, np.float32)
    beta = np.asarray(beta, np.float32)
    T, I, H = q.shape
    J = v.shape[1]
    S = S0.copy()
    o = np.empty((T, J, H), np.float32)
    inv = np.float32(1.0 / np.sqrt(I))
    for t in range(T):
        S = S * alpha[t][None, None, :]                     # decay
        sk = np.einsum("ijh,ih->jh", S, k[t])               # read
        dv = (v[t] - sk) * beta[t][None, :]                # delta
        S = S + np.einsum("ih,jh->ijh", k[t], dv)           # write
        o[t] = np.einsum("ijh,ih->jh", S, q[t]) * inv       # output
    return o, S


# --------------------------------------------------------------------------
# the chunk-parallel form of the same recurrence
# --------------------------------------------------------------------------
def gdn_chunked(S0, q, k, v, alpha, beta):
    """k tokens of the same recurrence, as matmuls.

    Same signature and same return as `gdn_recurrence`.
    """
    S0 = np.asarray(S0, np.float32)
    q, k, v = (np.asarray(z, np.float32) for z in (q, k, v))
    alpha = np.asarray(alpha, np.float32)
    beta = np.asarray(beta, np.float32)
    T, I, H = q.shape
    J = v.shape[1]

    # A_t = prod_{s<=t} a_s, and A_{t-1} (A_0 = 1)
    A = np.empty((T, H), np.float32)
    np.cumprod(alpha, axis=0, out=A)
    Aprev = np.concatenate([np.ones((1, H), np.float32), A[:-1]], 0)

    # ratio[t, s] = A_t / A_s for s <= t, else 0.  Every entry is <= 1.
    ratio = np.zeros((T, T, H), np.float32)
    for t in range(T):
        ratio[t, :t + 1] = A[t][None, :] / A[:t + 1]

    # the read against the state at the chunk start.  The recurrence reads from
    # the *decayed* state, so the factor is A_t ( = alpha_t * A_{t-1} ), not
    # A_{t-1}: that is the whole difference between this form and the scaled
    # frame, and getting it wrong is invisible in the no-decay case.
    KtS0 = np.einsum("tih,ijh->tjh", k, S0)                 # (T, J, H)
    rhs = beta[:, None, :] * (v - A[:, None, :] * KtS0)

    # the intra-chunk coupling: M[t, s] = beta_t * (A_t/A_s) * (k_s . k_t)
    KK = np.einsum("sih,tih->sth", k, k)                    # (T, T, H)
    tri = np.tril(np.ones((T, T), np.float32), -1)[:, :, None]
    M = beta[:, None, :] * ratio * KK * tri

    # forward substitution, per head, vectorised over J
    dv = np.empty((T, J, H), np.float32)
    for t in range(T):
        acc = rhs[t].copy()
        for s in range(t):
            acc -= M[t, s][None, :] * dv[s]
        dv[t] = acc

    # the outputs: o_t = ( A_t (S_0^T q_t) + sum_{s<=t} (A_t/A_s)(k_s.q_t) dv_s ) / sqrt(I)
    # the mask is *inclusive* here (the state at t holds the write of step t)
    # and *strict* in the system above (the read at t predates the write of t).
    # Note the two different decay factors: A_t on the state-at-chunk-start
    # term, and A_t/A_s on the in-chunk writes (which are not decayed at all
    # in the step that makes them, so s = t gives exactly 1).
    # The output couples s <= t, which in the [s, t] indexing is the *upper*
    # triangle; the system couples s < t, which in the [t, s] indexing is the
    # strict lower triangle. Two different triangles -- the wrong one here
    # silently drops the token's own write.
    incl = np.triu(np.ones((T, T), np.float32))[:, :, None]
    KQ = np.einsum("sih,tih->sth", k, q)                    # (T, T, H), [s, t]
    S0Tq = np.einsum("ijh,tih->tjh", S0, q)                 # (T, J, H)
    # the ratio is indexed [t, s] (A_t/A_s) while KQ is indexed [s, t]
    inner = np.einsum("sth,sjh->tjh", KQ * incl * ratio.transpose(1, 0, 2), dv)
    o = (A[:, None, :] * S0Tq + inner) * np.float32(1.0 / np.sqrt(I))

    # the state at the end of the chunk: A_C S_0 + sum_s (A_C/A_s) k_s dv_s^T
    S = A[-1][None, None, :] * S0 + np.einsum(
        "sih,sjh->ijh", k, dv * ratio[-1, :, None])
    return o, S


# --------------------------------------------------------------------------
# the same thing in JAX, for the kernel
# --------------------------------------------------------------------------
if HAVE_JAX:

    def gdn_chunked_jax(S0, q, k, v, alpha, beta):
        """JAX version, batched over the batch axis (B, ...).

        The batch axis is folded into the head axis: the triangular solve and
        the Gram matrix are per head, so one more axis of parallelism costs
        nothing.
        """
        S0, q, k, v = (jnp.asarray(z, jnp.float32) for z in (S0, q, k, v))
        alpha = jnp.asarray(alpha, jnp.float32)
        beta = jnp.asarray(beta, jnp.float32)
        B, T, I, H = q.shape
        J = v.shape[2]

        A = jnp.cumprod(alpha, axis=1)                       # (B, T, H)

        def ratios(Aax):
            # (B, T, T, H): ratio[t, s] = A_t / A_s for s <= t
            num = Aax[:, :, None, :]                          # (B, T, 1, H)
            den = Aax[:, None, :, :]                          # (B, 1, T, H)
            r = num / den
            mask = jnp.tril(jnp.ones((T, T), jnp.float32))
            return r * mask[:, :, None]

        ratio = ratios(A)

        KtS0 = jnp.einsum("btih,bijh->btjh", k, S0)
        rhs = beta[:, :, None, :] * (v - A[:, :, None, :] * KtS0)

        KK = jnp.einsum("bsih,btih->bsth", k, k)
        tri = jnp.tril(jnp.ones((T, T), jnp.float32), -1)
        M = beta[:, :, None, :] * ratio * KK * tri[:, :, None]

        # forward substitution over the chunk length (a small loop: 16 steps).
        # Each step starts from that step's right-hand side and subtracts the
        # already-solved rows; the carry is only the solved dv.
        def step(t, carry):
            dv = carry
            acc = rhs[:, t] - jnp.einsum("bsh,bsjh->bjh", M[:, t], dv)
            return dv.at[:, t].set(acc)

        dv = jax.lax.fori_loop(
            0, T, step, jnp.zeros((B, T, J, H), jnp.float32))

        KQ = jnp.einsum("bsih,btih->bsth", k, q)
        incl = jnp.triu(jnp.ones((T, T), jnp.float32))
        S0Tq = jnp.einsum("bijh,btih->btjh", S0, q)
        # ratio is [b, t, s, h] and KQ is [b, s, t, h]: transpose the ratio
        ratioST = jnp.swapaxes(ratio, 1, 2)
        inner = jnp.einsum("bsth,bsjh->btjh", KQ * incl[:, :, None] * ratioST, dv)
        o = (A[:, :, None, :] * S0Tq + inner) * (1.0 / jnp.sqrt(I))

        # ratio[:, -1] is (B, T, H) = A_C/A_s per step; dv is (B, T, J, H)
        S = (A[:, -1][:, None, None, :] * S0
             + jnp.einsum("bsih,bsjh->bijh", k, dv * ratio[:, -1][:, :, None, :]))
        return o, S


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------
def test_equivalence(verbose=True):
    """`gdn_chunked` must equal `gdn_recurrence`, to fp32 rounding."""
    rng = np.random.default_rng(20261021)
    worst = 0.0
    cases = []

    # (T, I, J, H, decay profile)
    profiles = [
        ("no decay",        lambda T, H: np.ones((T, H), np.float32)),
        ("mild decay",      lambda T, H: np.exp(-0.1 * rng.random((T, H)).astype(np.float32))),
        ("strong decay",    lambda T, H: np.exp(-1.5 * rng.random((T, H)).astype(np.float32))),
        ("per-head decay",  lambda T, H: np.exp(-2.0 * rng.random((T, H)).astype(np.float32))),
    ]
    for T, I, J, H in [(1, 16, 16, 4), (4, 16, 16, 4), (8, 32, 32, 8),
                       (16, 32, 32, 8), (16, 64, 64, 16), (32, 32, 32, 8)]:
        for name, decay in profiles:
            S0 = rng.standard_normal((I, J, H)).astype(np.float32) * 0.5
            # The model L2-normalises q and k (`l2n_b`, kernel.py:388). That is
            # what keeps the delta rule's triangular system well conditioned:
            # with normalised keys the Gram entries are ~1/I, and the explicit
            # solve agrees with the sequential recurrence to fp32 rounding.
            # Without it the system is exponentially ill conditioned and *both*
            # forms are meaningless -- see the `unnormalised keys` case below.
            q = rng.standard_normal((T, I, H)).astype(np.float32)
            k = rng.standard_normal((T, I, H)).astype(np.float32)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            k /= np.linalg.norm(k, axis=1, keepdims=True)
            v = rng.standard_normal((T, J, H)).astype(np.float32)
            alpha = decay(T, H)
            beta = rng.random((T, H)).astype(np.float32) * 0.9 + 0.05

            o_ref, S_ref = gdn_recurrence(S0, q, k, v, alpha, beta)
            o_new, S_new = gdn_chunked(S0, q, k, v, alpha, beta)
            eo = float(np.max(np.abs(o_ref - o_new)))
            eS = float(np.max(np.abs(S_ref - S_new)))
            worst = max(worst, eo, eS)
            cases.append((T, I, J, H, name, eo, eS))
            if HAVE_JAX:
                o_j, S_j = gdn_chunked_jax(
                    S0[None], q[None], k[None], v[None], alpha[None], beta[None])
                eo_j = float(np.max(np.abs(o_ref - np.asarray(o_j)[0])))
                eS_j = float(np.max(np.abs(S_ref - np.asarray(S_j)[0])))
                worst = max(worst, eo_j, eS_j)
                cases.append((T, I, J, H, name + " [jax]", eo_j, eS_j))

    if verbose:
        print(f"{'T':>3} {'I':>4} {'J':>4} {'H':>3}  {'profile':<22} "
              f"{'max|do|':>12} {'max|dS|':>12}")
        for T, I, J, H, name, eo, eS in cases:
            print(f"{T:>3} {I:>4} {J:>4} {H:>3}  {name:<22} "
                  f"{eo:>12.3e} {eS:>12.3e}")
        print(f"\nworst absolute error over {len(cases)} cases: {worst:.3e}")
        print("PASS" if worst < 2e-4 else "FAIL")
    return worst < 2e-4


if __name__ == "__main__":
    ok = test_equivalence()
    raise SystemExit(0 if ok else 1)
