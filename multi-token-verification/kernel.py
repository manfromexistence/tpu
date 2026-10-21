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
    alpha (T, H)      the decay FACTOR per step (already exponentiated -- the
                      model computes a log-decay and exponentiates it)
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
def gdn_chunked(S0, q, k, v, log_alpha, beta):
    """k tokens of the same recurrence, as matmuls.

    Same signature and same return as `gdn_recurrence`, except that the decay
    is passed as its LOG: the cumulative product of the factors underflows to
    zero inside a chunk when the model's decay is strong, and the ratio
    A_t/A_s cannot be recovered from a product that has already flushed to
    zero. The log domain never underflows -- the ratio is exp(cum_t - cum_s),
    which is <= 1 for s <= t.
    """
    S0 = np.asarray(S0, np.float32)
    q, k, v = (np.asarray(z, np.float32) for z in (q, k, v))
    log_alpha = np.asarray(log_alpha, np.float32)
    beta = np.asarray(beta, np.float32)
    T, I, H = q.shape
    J = v.shape[1]

    # cum_t = sum_{s<=t} log a_s  (the log of A_t)
    cum = np.cumsum(log_alpha, axis=0)

    # ratio[t, s] = A_t / A_s for s <= t, else 0.  Every entry is <= 1.
    # The denominator is guarded: the model's decay can underflow the
    # cumulative product to zero inside a chunk.
    ratio = np.zeros((T, T, H), np.float32)
    for t in range(T):
        ratio[t, :t + 1] = np.exp(cum[t][None, :] - cum[:t + 1])

    # the read against the state at the chunk start.  The recurrence reads from
    # the *decayed* state, so the factor is A_t ( = alpha_t * A_{t-1} ), not
    # A_{t-1}: that is the whole difference between this form and the scaled
    # frame, and getting it wrong is invisible in the no-decay case.
    KtS0 = np.einsum("tih,ijh->tjh", k, S0)                 # (T, J, H)
    rhs = beta[:, None, :] * (v - np.exp(cum)[:, None, :] * KtS0)

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
    o = (np.exp(cum)[:, None, :] * S0Tq + inner) * np.float32(1.0 / np.sqrt(I))

    # the state at the end of the chunk: A_C S_0 + sum_s (A_C/A_s) k_s dv_s^T
    S = np.exp(cum[-1])[None, None, :] * S0 + np.einsum(
        "sih,sjh->ijh", k, dv * ratio[-1, :, None])
    return o, S


# --------------------------------------------------------------------------
# the same thing in JAX, for the kernel
# --------------------------------------------------------------------------
if HAVE_JAX:

    def gdn_chunked_jax(S0, q, k, v, log_alpha, beta):
        """JAX version, batched over the batch axis (B, ...).

        The batch axis is folded into the head axis: the triangular solve and
        the Gram matrix are per head, so one more axis of parallelism costs
        nothing.
        """
        S0, q, k, v = (jnp.asarray(z, jnp.float32) for z in (S0, q, k, v))
        log_alpha = jnp.asarray(log_alpha, jnp.float32)
        beta = jnp.asarray(beta, jnp.float32)
        B, T, I, H = q.shape
        J = v.shape[2]

        cum = jnp.cumsum(log_alpha, axis=1)                  # (B, T, H): log A_t

        def ratios(cum):
            # (B, T, T, H): ratio[t, s] = A_t / A_s = exp(cum_t - cum_s)
            # for s <= t. The log domain never underflows, and the exponent is
            # <= 0 there, so the ratio is <= 1. The upper triangle is selected
            # away with where rather than multiplied by a mask: 0 * inf is
            # NaN, and that is how the first chunked run on the TPU failed.
            d = cum[:, :, None, :] - cum[:, None, :, :]         # (B, T_t, T_s, H)
            mask = jnp.tril(jnp.ones((T, T), jnp.float32))
            return jnp.where(mask[:, :, None], jnp.exp(d), 0.0)

        ratio = ratios(cum)

        KtS0 = jnp.einsum("btih,bijh->btjh", k, S0)
        rhs = beta[:, :, None, :] * (v - jnp.exp(cum)[:, :, None, :] * KtS0)

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
        o = (jnp.exp(cum)[:, :, None, :] * S0Tq + inner) * (1.0 / jnp.sqrt(I))

        # ratio[:, -1] is (B, T, H) = A_C/A_s per step; dv is (B, T, J, H)
        S = (jnp.exp(cum[:, -1])[:, None, None, :] * S0
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
        # the model's own decay is strong enough to underflow the cumulative
        # product inside a chunk: the first chunked run on the TPU produced
        # NaN for exactly this reason
        ("underflowing",    lambda T, H: np.exp(-30.0 * rng.random((T, H)).astype(np.float32))),
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
            # the chunked form takes the LOG of the decay (see its docstring)
            o_new, S_new = gdn_chunked(S0, q, k, v, np.log(alpha), beta)
            eo = float(np.max(np.abs(o_ref - o_new)))
            eS = float(np.max(np.abs(S_ref - S_new)))
            worst = max(worst, eo, eS)
            cases.append((T, I, J, H, name, eo, eS))
            if HAVE_JAX:
                o_j, S_j = gdn_chunked_jax(
                    S0[None], q[None], k[None], v[None],
                    np.log(alpha)[None], beta[None])
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


# --------------------------------------------------------------------------
# the chunk boundary: the causal conv window and the chunked attention
# --------------------------------------------------------------------------
def conv_window(hist, chunk):
    """The causal depthwise conv window for a chunk of tokens.

    The conv is `CONV` taps ending at the current token. For a chunk of k
    tokens the first rows must reach *back* into the previous chunk's tail, so
    the window for row j is

        [ hist tail ] ++ [ chunk[0 .. j] ]

    `hist` holds the last CONV-1 tokens before the chunk; `chunk` holds the k
    new ones, each (CONV_DIM,).

    Returns (window, new_hist): the window is (k, CONV, CONV_DIM) and the new
    history is the last CONV-1 tokens of hist ++ chunk.
    """
    hist = np.asarray(hist)
    chunk = np.asarray(chunk)
    k = chunk.shape[0]
    conv = hist.shape[0] + 1                 # CONV
    both = np.concatenate([hist, chunk], 0)  # (CONV-1+k, CONV_DIM)
    window = np.stack([both[j:j + conv] for j in range(k)], 0)
    new_hist = both[-(conv - 1):] if conv > 1 else both[:0]
    return window, new_hist


def attention_mask(pos, k, total):
    """The causal mask for k queries at positions pos .. pos+k-1.

    Row j of the mask allows the cache positions 0 .. pos+j.
    """
    qpos = pos + np.arange(k)
    return np.arange(total)[None, :] <= qpos[:, None]


def attn_token_by_token(q, k_, v, K, V, pos, n_heads, n_kv, hd):
    """The kernel's attention, one token at a time (the reference).

    The cache layout is the kernel's: (T, n_kv, hd) -- time first, then the kv
    heads (`K`/`V` in `attn_core`, kernel.py:466-467).
    """
    T = K.shape[0]
    G = n_heads // n_kv
    iss = np.float32(1.0 / np.sqrt(hd))
    Kj = K.copy()
    Vj = V.copy()
    outs = []
    for j in range(q.shape[0]):
        Kj[pos + j] = k_[j]
        Vj[pos + j] = v[j]
        qr = q[j].reshape(n_kv, G, hd)
        scores = np.einsum("hgd,shd->hgs", qr, Kj).astype(np.float32) * iss
        scores = np.where(attention_mask(pos + j, 1, T), scores, np.float32(-1e30))
        p = np.exp(scores - scores.max(-1, keepdims=True))
        p = p / p.sum(-1, keepdims=True)
        o = np.einsum("hgs,shd->hgd", p, Vj)
        outs.append(o.reshape(n_heads, hd))
    return np.stack(outs, 0), Kj, Vj


def attn_chunk(q, k_, v, K, V, pos, n_heads, n_kv, hd):
    """The same attention for k tokens: one cache write, one masked pass."""
    T = K.shape[0]
    k = q.shape[0]
    G = n_heads // n_kv
    iss = np.float32(1.0 / np.sqrt(hd))
    Kc = K.copy()
    Vc = V.copy()
    Kc[pos:pos + k] = k_
    Vc[pos:pos + k] = v
    # the head axis of q is h*G+g, so the heads split into (kv head, group);
    # the time axis must not be folded into that split
    qr = q.reshape(k, n_kv, G, hd).transpose(1, 2, 0, 3)      # (kv, group, k, hd)
    scores = np.einsum("hgtd,shd->hgts", qr, Kc).astype(np.float32) * iss
    allow = attention_mask(pos, k, T)          # (k, T)
    # scores is (kv head, group, query in chunk, cache position): the mask
    # broadcasts as (1, 1, k, T)
    scores = np.where(allow[None, None, :, :], scores, np.float32(-1e30))
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    o = np.einsum("hgts,shd->hgtd", p, Vc)                    # (kv, group, k, hd)
    return o.transpose(2, 0, 1, 3).reshape(k, n_heads, hd), Kc, Vc


def test_attention_and_conv(verbose=True):
    """The chunked attention and the conv window must equal the token-by-token
    forms."""
    rng = np.random.default_rng(20261021)
    worst = 0.0
    cases = []

    for n_heads, n_kv, hd in [(24, 4, 32), (8, 2, 16)]:
        for T, pos, k in [(64, 40, 4), (64, 40, 16), (128, 100, 8), (32, 0, 8)]:
            q = rng.standard_normal((k, n_heads, hd)).astype(np.float32)
            k_ = rng.standard_normal((k, n_kv, hd)).astype(np.float32)
            v = rng.standard_normal((k, n_kv, hd)).astype(np.float32)
            K = rng.standard_normal((T, n_kv, hd)).astype(np.float32)
            V = rng.standard_normal((T, n_kv, hd)).astype(np.float32)
            o_ref, Kr, Vr = attn_token_by_token(q, k_, v, K, V, pos, n_heads, n_kv, hd)
            o_new, Kc, Vc = attn_chunk(q, k_, v, K, V, pos, n_heads, n_kv, hd)
            eo = float(np.max(np.abs(o_ref - o_new)))
            eK = float(np.max(np.abs(Kr - Kc)))
            eV = float(np.max(np.abs(Vr - Vc)))
            worst = max(worst, eo, eK, eV)
            cases.append((n_heads, n_kv, hd, T, pos, k, eo, eK, eV))

    # the conv window across a chunk boundary
    conv_cases = []
    for conv, dim, k in [(4, 16, 4), (4, 16, 16), (3, 8, 8), (5, 32, 6)]:
        hist = rng.standard_normal((conv - 1, dim)).astype(np.float32)
        chunk = rng.standard_normal((k, dim)).astype(np.float32)
        window, new_hist = conv_window(hist, chunk)
        # the token-by-token window for each row of the chunk
        both = np.concatenate([hist, chunk], 0)
        ref = np.stack([both[j:j + conv] for j in range(k)], 0)
        e = float(np.max(np.abs(window - ref)))
        eh = float(np.max(np.abs(new_hist - both[-(conv - 1):])))
        worst = max(worst, e, eh)
        conv_cases.append((conv, dim, k, e, eh))

    if verbose:
        print(f"{'heads':>6} {'kv':>4} {'hd':>4} {'T':>5} {'pos':>5} {'k':>4} "
              f"{'max|do|':>12} {'max|dK|':>12} {'max|dV|':>12}")
        for nh, nkv, hd, T, pos, k, eo, eK, eV in cases:
            print(f"{nh:>6} {nkv:>4} {hd:>4} {T:>5} {pos:>5} {k:>4} "
                  f"{eo:>12.3e} {eK:>12.3e} {eV:>12.3e}")
        print(f"\n{'conv':>6} {'dim':>5} {'k':>4} {'max|dwin|':>12} {'max|dhist|':>12}")
        for conv, dim, k, e, eh in conv_cases:
            print(f"{conv:>6} {dim:>5} {k:>4} {e:>12.3e} {eh:>12.3e}")
        print(f"\nworst absolute error: {worst:.3e}")
        print("PASS" if worst < 2e-4 else "FAIL")
    return worst < 2e-4


if __name__ == "__main__":
    ok = test_equivalence()
    ok = test_attention_and_conv() and ok
    raise SystemExit(0 if ok else 1)
