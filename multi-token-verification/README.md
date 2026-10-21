# multi-token verification — the chunked gated-delta form, and the route to 50 tok/s

**Status: the mathematics is done and verified on CPU (2026-10-21). The TPU
path is not.** What is measured here is a *reference*, on CPU, in numpy and JAX.
No number in this folder is a TPU throughput claim; the TPU receipts live in
`../lane-aligned-ternary-kernel/`.

## What this folder is

`../lane-aligned-ternary-kernel/` advances the gated-delta state **one token at
a time** (`gdn_core`, `kernel.py:429-433`):

```text
Sg  = S * exp(gate)            # decay, one scalar per head
sk  = sum_i Sg[i,j] k[i]       # read
dv  = (v[j] - sk[j]) * beta    # delta
Sg  = Sg + k[i] * dv[j]        # write
o[j] = sum_i Sg[i,j] q[i] / sqrt(I)
```

That form is inherently sequential: token `t+1` reads the state token `t` wrote,
so *k* tokens need *k* passes over the weights. And the weights are the whole
cost of a step:

| what | measured | receipt |
| --- | --- | --- |
| step, 8k window, kv-head expansion removed | 134.3 ms (7.45 tok/s) | `receipt-gqa-b1-f16-tc1024.json` |
| of which the projection's per-element work | ~109 ms (~81 %) | `../projection-cost-breakdown/README.md` |
| of which the attention over the 8k window | ~25 ms | same |
| HBM traffic floor for one pass over the weights | 9.0 ms | `../projection-cost-breakdown/README.md` |

The step is **~15x off its own memory roofline** and the MXU is ~99 % idle at
batch 1 (`../lane-aligned-ternary-kernel/kernel.py:154-159`). The cost is
per-weight work, and **per-weight work does not depend on how many tokens ride
along with it**. That is the whole reason this folder exists: give the step
*k* tokens and the weight cost is amortised *k* ways.

No other lever on this kernel does that. Cutting 36 % of the launches moved the
step by 0.5 %; cutting the unpack's memory traffic made it *slower*.

## The derivation

With the decay made explicit — `a_t = exp(gate_t)` (one scalar per head),
`A_t = prod_{s<=t} a_s`, `A_0 = 1` — the state after token `t` is

```text
S_t = A_t S_0 + sum_{s<=t} (A_t / A_s) k_s dv_s^T
```

Substituting that back into the read gives, for the chunk's `dv`, a
**lower-triangular system with unit diagonal**:

```text
dv_t + beta_t * sum_{s<t} (A_t / A_s) (k_s . k_t) dv_s
      = beta_t * ( v_t - A_t (S_0^T k_t) )
```

and the chunk's outputs and final state follow from the same expansion:

```text
o_t   = ( A_t (S_0^T q_t) + sum_{s<=t} (A_t/A_s) (k_s . q_t) dv_s ) / sqrt(I)
S_end = A_C S_0 + sum_s (A_C/A_s) k_s dv_s^T
```

Every ratio is `<= 1` (a decay only shrinks), so this frame is numerically
*safer* than the scaled frame `T_t = S_t / A_t`, whose `beta_t / A_t` blows up
as the chunk's total decay accumulates.

Three details are easy to get wrong, and each one produced a wrong answer
before it produced a right one — they are the reason this folder has a test
rather than a claim:

1. **The read sees the *decayed* state.** The recurrence reads from
   `a_t S_{t-1}`, so the coupling factor is `A_t`, not `A_{t-1}`. With no
   decay the two are identical, so the bug hides until a decay profile is
   tested.
2. **Two different triangles.** The system couples `s < t`; the output couples
   `s <= t` (the state at `t` holds the write of step `t`). Using one mask for
   both silently drops the token's own write.
3. **The decay ratio is transposed between the two matmuls.** `A_t/A_s` is
   indexed `[t, s]`; the Gram and the `k.q` matrices are indexed `[s, t]`.

## The contract, and its result

`kernel.py` holds six functions and two contracts:

| function | what it is |
| --- | --- |
| `gdn_recurrence` | the recurrence we ship, transcribed token by token |
| `gdn_chunked` | the same recurrence as matmuls (numpy reference) |
| `gdn_chunked_jax` | the same thing in JAX, batched over the batch axis |
| `attn_token_by_token` | the kernel's attention, one token at a time |
| `attn_chunk` | the same attention for k tokens: one cache write, one masked pass |
| `conv_window` | the causal conv window across a chunk boundary |

`test_equivalence()` and `test_attention_and_conv()` are the contracts.
**Run: `python kernel.py`** (numpy only; with JAX installed it also checks the
JAX form).

```text
worst absolute error over 48 cases: 7.153e-07      <- the gated-delta recurrence
PASS
 heads   kv   hd     T   pos    k      max|do|      max|dK|      max|dV|
    24    4   32    64    40    4    0.000e+00    0.000e+00    0.000e+00
    24    4   32    64    40   16    0.000e+00    0.000e+00    0.000e+00
     8    2   16   128   100    8    0.000e+00    0.000e+00    0.000e+00
 conv   dim    k    max|dwin|   max|dhist|
     4    16    4    0.000e+00    0.000e+00
     4    16   16    0.000e+00    0.000e+00
```

**The recurrence: 48 cases** — chunk lengths 1-32, four decay profiles (none /
mild / strong / per-head), numpy and JAX, outputs and final state. The worst
absolute difference is **7.15e-07**, which is fp32 rounding, not a modelling
difference.

**The attention and the conv: bit-identical** (0.0 everywhere). The chunked
attention is the same arithmetic with the k queries in one pass, so it *must*
be bit-identical; it is not, one of three bugs shows up:

1. **The head axis must not be folded into the time axis.** The query's heads
   are `h*G+g`, so a chunk of k tokens is `(k, n_heads, hd)` and splitting the
   heads into `(kv head, group)` has to keep the time axis separate. A plain
   `reshape(n_kv, G, k, hd)` silently interleaves the two, and the error looks
   like a wrong attention mask.
2. **The mask broadcasts as (1, 1, k, T)** against a score tensor shaped
   `(kv head, group, query in chunk, cache position)` — one `None` too many
   turns it into a five-axis tensor and numpy then refuses the `einsum`.
3. **The cache write happens once, before any read** — so the chunked form
   *can* see the later tokens of its own chunk, and the causal mask is what
   stops it. A mask that is off by one position shows up as a plausible-looking
   number rather than an error.

One numerical property worth keeping: the test uses **L2-normalised** keys and
queries, because the model does (`l2n_b`, `../lane-aligned-ternary-kernel/kernel.py:388`).
That is what keeps the delta rule's triangular system well conditioned — with
normalised keys the Gram entries are `~1/I`. Without normalisation the system is
exponentially ill conditioned and *both* forms are meaningless, so the
normalisation is a correctness requirement of this mechanism, not a detail.

## The plan from here

The chunked GDN is the hard part of the recurrence; the rest is plumbing that
the existing kernel already supports:

| step | what | why it is cheap |
| --- | --- | --- |
| 1 | move `gdn_chunked_jax` into the kernel's `gdn_core` | the kernel is already batched (`B` is an argument); the projections become a `(k, IN)` matmul against the **same** unpacked weights |
| 2 | the 16 attention layers take `k` queries in one pass | the GQA path already reads the KV tiles once; the mask becomes `arange(T) <= pos + arange(k)` |
| 3 | a draft proposes `k` tokens | `z-lab/Qwen3.8-27B-DFlash2` (the draft Inferact uses for this exact base architecture) or the MTP head from the community GGUF |
| 4 | verify: accept the longest matching prefix + 1 | standard speculative verification, greedy |

**The arithmetic that has to come out right**, with the measured 134.3 ms
pass (the unpack does not shrink; the attention is paid once for the chunk
instead of once per token, so a `k`-token pass is ~135-150 ms):

| acceptance | tokens per pass | tok/s at 140 ms/pass | tok/s at 115 ms/pass |
| --- | --- | --- | --- |
| 3 tokens | 3 | 21 | 26 |
| 4 tokens | 4 | 29 | 35 |
| 6 tokens | 6 | 43 | 52 |
| 7 tokens | 7 | 50 | 61 |

Inferact's own megakernel for **Qwen3.8-27B** — the same base architecture as
Bonsai 2 (hybrid gated-delta-net + grouped-query-attention) — reports **3-6
accepted tokens per step** with its DFlash2 draft, and 1,515 tok/s on 4x TPU v7
at acceptance length 6 (`inferact.ai/blog/tpu-megakernels`, and the open-source
`Inferact/tpu-megakernels`). So the acceptance rate needed for 50 tok/s is at
the **top of the published range**, and the second column (115 ms/pass) is what
the unpack work in `TPU-11` buys.

**The honest position:** 50 tok/s single-stream on one v5e-1 is reachable with
this mechanism *if* the draft's acceptance reaches ~6-7 tokens per pass, and
the unpack gets ~20 % cheaper. With the draft alone and no unpack work, the
same arithmetic gives 40-50 tok/s. Below an acceptance of ~4, the ceiling is
~30 tok/s and the honest answer is that 50 tok/s needs a cheaper unpack, not a
bigger draft.

## References

- `inferact.ai/blog/tpu-megakernels` and `github.com/Inferact/tpu-megakernels`
  — the open-source TPU megakernels for Kimi K3, **Qwen3.8-27B** and
  Muse-Spark; the techniques that matter here are the grid-less single program,
  `run_scoped` VMEM lifetimes, `make_async_copy(...).start()/.wait()` for
  cross-layer weight prefetch, and the fused "1 anchor + 7 draft rows"
  verification.
- Gated DeltaNet-2, arXiv 2605.22791v1, eq. 20-21 — the chunkwise WY form of
  the delta rule (`T = tril(E K^T, -1)`, `A = (I+T)^-1`). This folder derives
  the same structure against *our* recurrence instead of adopting its notation,
  which is what the equivalence test pins down.
- `flash-linear-attention`'s `chunk_gated_delta_rule` — the same algorithm in
  Triton, and the reason the blocked triangular inverse is the standard
  implementation.

MIT OR Apache-2.0
updated: 2026-10-21 05:10 UTC
