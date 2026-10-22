# 14-lane-aligned-ternary-kernel: out-columns on the 128 lanes

Runs the **actual 64-layer Bonsai 2 27B** end to end with the same fused
PTQ1_0 Pallas kernel as notebook 11, but with the weight tiles stored
**transposed**. One change -- which axis carries the out-columns -- took the
model from 29.66 to 281.01 tok/s.

## Status

**Executed 2026-10-18 on Colab TPU v5e-1. 281.01 tok/s aggregate at batch 80
with f16 state storage and TC 1024, 284.7 ms/step.** That is 9.5x the
notebook 11 baseline of 29.66 tok/s and 2.8x the 100 tok/s target this track
was aiming at.

- correctness gate: first token `11` (`','`) in every run
- generated text byte-identical to notebook 11's receipted string:
  `", I'm a student in the University of the West of England (UWE) Bristol,
  and I are writing a formal letter to the Dean of the School"`
- `jax 0.11.2` / `libtpu 0.0.50`; tile build 11.5 s, 6.37 GiB on device

**Single stream (batch 1) was measured later, on 2026-10-19: 5.54 tok/s
(180.6 ms/step) at an 8k context window, with the 64-layer step fused into one
jitted program** -- see *The fused step* below. The two numbers are different
quantities and must not be read as alternatives: 281.01 tok/s is 80 sequences
advancing in lockstep, 5.54 tok/s is one conversation.

## The change

Notebook 11 decoded each packed GGUF block as `(in-dims, out-columns)` and
computed `w @ x`: the in-dims (2, 8 or 16 of them) sat on the lane axis.
Pallas maps only the **last two dims** to the 8x128 register file, and
Mosaic's layout solver puts the RHS free axis (N = out-columns) on the
**lanes**. So every decode op in notebook 11 ran on 2-16 lanes of 128.

Notebook 14 instead stores each packed block as `(32, rows)` -- 32-byte rows
on the sublanes, one byte of every in-block per row -- so the projection is
`x @ w` and the out-columns land on the 128 lanes:

| | notebook 11 | notebook 14 |
| --- | --- | --- |
| packed tile | `(rows, 28)` | `(32, rows)`, 28 real bytes + 4 pad |
| projection | `w @ x` | `x @ w` |
| decode spelling | `concatenate(..., axis=-1)` | `concatenate(..., axis=0)` |
| lanes carry | 2 / 8 / 16 in-dims | up to 1024 out-columns |
| weight bytes | 5.90 GB + 0.84 GB scales | +14 % (32-byte rows), 6.37 GiB measured |

The in-dim order is identical in both spellings, so the decoded weights are
**bit-identical** -- proven without a TPU (`colab_run/check14.py`, 4 real
geometries, maxdiff 0.0) -- and the tile indexing was proven against a
simulated grid (`colab_run/check14b.py`: `y = x @ W`, max err 7.6e-05). The
28 -> 32 byte pad costs 14 % more weight bytes; that is the only price.

## Measured (every run is the real model)

| batch | state | TC | tok/s | ms/step | receipt |
| --- | --- | --- | --- | --- | --- |
| 32 | f32 | 512 | 130.25 | 245.7 | log only (cache 0; receipt overwritten) |
| 32 | f32 | 512 | 130.30 | 245.6 | `receipt-b32-f32-tc512.json` |
| 32 | f16 | 512 | 134.67 | 237.6 | `receipt-b32-f16-tc512.json` |
| 48 | f16 | 512 | 187.09 | 256.6 | `receipt-b48-f16-tc512.json` |
| 48 | f16 | 1024 | 203.01 | 236.4 | `receipt-b48-f16-tc1024.json` |
| 64 | f16 | 512 | 217.49 | 294.3 | `receipt-b64-f16-tc512.json` |
| 64 | f16 | 1024 | 239.25 | 267.5 | `receipt-b64-f16-tc1024.json` |
| 80 | f16 | 512 | 266.37 | 300.3 | `receipt-b80-f16-tc512.json` |
| 80 | f16 | 1024 | 279.46 | (repeat) | log only |
| **80** | **f16** | **1024** | **281.01** | **284.7** | `receipt-b80-f16-tc1024.json` |

### Single stream (batch 1), 2026-10-19 -- one conversation, not 80

| window | kernel | prompt | tok/s | ms/step | peak | receipt |
| --- | --- | --- | --- | --- | --- | --- |
| 8192 | per-layer jit | `what is the capital of franch, india and bangaldesh and give me tps` | 4.62 | 216.3 | 9.53 GiB | `receipt-b1-f16-tc1024-franch.json` |
| 8192 | **fused step** | same prompt, 17 tokens | **5.54** | **180.6** | 14.46 GiB | `receipt-b1-f16-tc1024-fused.json` |
| 8192 | fused step | `Hello` (1 token) | 5.55 | 180.3 | 9.28 GiB | `receipt-b1-f16-tc1024-fused-smoke.json` |
| 45056 | per-layer jit | `Hello, who are you?` | 2.86 | 350.1 | 14.01 GiB | `receipt-b1-f16-tc1024-45056.json` |

The two 8k rows with the same prompt are the only apples-to-apples pair in
this file: same window, same batch, same 17-token prompt, same 64 generated
tokens, **byte-identical output** (`first token 271` in both, same sample
string). The difference is the kernel: 216.3 -> 180.6 ms/step, **+20 %**.

The 45056 row is the largest fp32 window that runs at all, and it does so
slowly. `receipt-mem-ceiling.json` records the ceiling: 49152 is 140 MB
short of loading `jit_attn_step` and 65536 wants 3.26 GB with 184 MB free.

What each knob was worth, holding the rest fixed:

| Comparison | Effect |
| --- | --- |
| lane flip alone: nb 11 29.66 -> nb 14 32:f32:TC512 130.25 | **4.4x** |
| state f32 -> f16 at 32 / TC512 | +3.4 %, and halves state memory |
| batch 32 -> 80 at f16 / TC1024 | 2.09x aggregate; step 245.6 -> 284.7 ms (+16 % for 2.5x the batch) |
| TC 512 -> 1024 at 80 / f16 | +5.5 % |
| pallas-callable cache off -> on at 32:f32 | 130.25 -> 130.30 -- **no effect, hypothesis refuted** |

Run-to-run spread on the best config: 279.46 vs 281.01 in the same process,
0.5 %.

## The decode fold (implemented 2026-10-21, not yet measured)

`FOLD=1` moves two operations off the per-weight path, and with them the tile's
element width.

The shipped line is

```python
w = (jnp.concatenate([t80, t40, h8], 0) - 1).astype(jnp.bfloat16) * sc
```

Two things about it:

- `.astype(bfloat16) * sc` promotes **back to fp32** — the cast and the
  multiply are one expression — so the MXU is fed a **4-byte-per-element**
  weight tile where it needs 2. The tile is the largest single term in the
  projection's memory traffic by this kernel's own comment, so it is carried at
  twice the width it needs.
- The `- 1` and the `* sc` are two more full passes over that tile, and
  neither of them belongs there: the scale is **per out-column**, so it belongs
  on the accumulator, and the `- 1` is a constant offset, so
  `sum_in (level-1)*x = sum_in level*x - sum_in x` and `sum_in x` is **one
  scalar per token**.

The fold leaves the level in bf16 (0, 1 and 2 are exact in bf16), applies the
scale to the fp32 accumulator after the matmul, and subtracts the per-token
`sum_in x` inside the kernel -- so the cancellation happens at full precision
and only the result is rounded to bf16, the same rounding the shipped path
ends with.

**Executed on CPU:** `fold_test.py` reproduces the kernel's decode spelling
and compares the two projections -- worst relative difference **2.3e-07**, fp32
rounding. Run: `python fold_test.py`.

**Not yet measured on the TPU.** `FOLD` defaults to **0**, so the shipped path
is unchanged until a receipt says otherwise. The gate is the first token
(still `271`); and because the shipped path hands the MXU a bf16 rounding of
`(level-1)*sc` while the fold keeps the scale in fp32, the fold is the *more*
accurate of the two and the two may not agree bit for bit. If the greedy text
changes, the comparison against `../reference-forward-pass/` decides which is
right. See `TODO.md` `TPU-11`.

## FOLD=3: the same model with the tile never leaving bf16 (2026-10-22)

`FOLD=1` (the scale on the accumulator) is 10.9 % faster but its output is not
the model's. `FOLD=2` (the scale inside the bf16 cast) reproduces the shipped
text exactly. `FOLD=3` is `FOLD=2`'s operand with the tile kept in bf16
throughout: the scale is **pre-rounded to bf16** (a (1, TC) array, once) and the
multiply happens in bf16.

Why that is the same model, bit for bit. The Pallas documentation is explicit:
*"Even if you pass 32-bit operands to a matrix multiplication, they will be
rounded to bfloat16 unless float32 precision is requested."* The shipped path
hands the MXU an **fp32** tile, so the hardware rounds it before the MXU sees
it, and the operand the MXU actually gets is `bf16((level-1)*sc)`. Because
`level-1` is in {-1, 0, 1}, multiplying by it is exact, so

    bf16((level-1) * sc) == (level-1) * bf16(sc)

which is exactly what `FOLD=3` computes. `fold_test.py` checks it: **0
differing words of 2048**. (That the rounding is a single bf16 rounding and not
a 3-pass scheme was already confirmed by the measurement: `FOLD=2`'s text
matched the shipped path's, and the fp32 oracle agreed with both.)

### What the TPU said (2026-10-22, the real 64-layer model, one session)

| arm | ms/step | tok/s | first token | text |
| --- | --- | --- | --- | --- |
| `FOLD=0` (shipped) | 134.1 | 7.45 | 271 | the model |
| `FOLD=1` | 119.5 | 8.37 | 271 | **diverges** |
| `FOLD=2` | 131.1 | 7.63 | 271 | **identical** |
| `FOLD=3` | 136.2 | 7.34 | 271 | **identical** |

`FOLD=3`'s text is the shipped text, exactly as the CPU proof predicted, so the
operand identity held on the hardware too. **But it is not faster**, and that
refutes the hypothesis it was built on: the tile's *width* is worth about 3 %
(`FOLD=2` against `FOLD=0`), while the explicit bf16 multiply costs more than
the narrower tile saves. The TPU's vector lanes are 32-bit, so a bf16
elementwise multiply is no cheaper than an fp32 one -- the width buys
bandwidth, and the multiply buys nothing.

**So the cost is the per-weight scale multiply itself.** `FOLD=1` removes it
(10.9 % faster) and changes the model's output; `FOLD=2` and `FOLD=3` keep the
output and keep the multiply (3 % and nothing). The two levers are in tension:
the speed comes from taking the scale off the per-weight path, and the shipped
text requires the scale to be inside the operand. `FOLD=2` is the same model
3 % faster and is the one to keep; `FOLD=1` is the fast one and is not the
model.

## The concatenation pass (experiment queued 2026-10-20)

The unpack produces fourteen power groups (5 x (16, TC) from `qs[:16]`, 5 x (8, TC) from
`qs[16:]`, 4 x (2, TC) from `qh`). The shipped path concatenates them twice -- into
(80, TC) and then (128, TC) -- and each concatenation is a full read+write pass over the
unpacked weights, which is the largest single term in the projection's memory traffic
(~25x the packed tile bytes per tile). `NOCONCAT=1` serves the fourteen groups with
fourteen separate matmuls instead, skipping both concatenations; at batch 1 the MXU is
~99 % idle, so the smaller K dimension costs no measurable arithmetic. The A/B
(`NOCONCAT=0` against `NOCONCAT=1`, 8k window, batch 1) is queued on a TPU session; its
receipts land as `receipt-nc0-b1-f16-tc1024.json` and
`receipt-nc1-b1-f16-tc1024.json`.

## The fused step (2026-10-19)

`full_step` runs all 64 layers inside **one** `jax.jit`, and the 61-chunk vocab
loop is a second jitted program. Before this, each layer was its own
dispatch, so a step paid ~64 host dispatches plus ~122 in the logits loop.

**The rule that made it work: the weights are an argument, not a closure.**
The first attempt closed over `TILES`, and JAX reported *"A large amount of
constants were captured during lowering (6.95GB total)"* -- the entire weight
set became a constant in the jaxpr. The compile never finished; the kernel
connection dropped 2 h 12 m later with the job still running. Passing the
per-layer weight pytree as an argument (`W_t`) cut the compile to **139.6 s**,
and a persistent compile cache (`/content/jaxcache`) took the rerun's prefill
to **4.6 s**.

`num_scalar_prefetch=0` and the `pl.ANY` memory space are still untested here,
so the DMA/compute overlap lever is open -- `TODO.md` `TPU-02`.

## Honest limits

- f16 state is **storage only**: the GDN recurrence still runs in fp32 and the
  state is cast on the way in and out. It is not a bf16 recurrence.
- "Byte-identical text" holds for these 32 greedy steps from this prompt. It
  is not a claim that f16 storage can never change an output.
- 281.01 tok/s is 32 steps x 80 sequences / 9.11 s. It is aggregate
  throughput, not latency: 284.7 ms/step, 3.56 ms/token.
- Batch 80 stops at the **memory wall** (measured peak 15.67 GiB of
  15.75 GiB), not at a throughput wall. Batch 96 was not attempted; the head
  room is 0.08 GiB.
- The aggregate is 80 greedy continuations of one prompt, not 80 users.

## How to run

`colab_run/run14.sh "80:float16" 1024 32 1` (args: `RUNS`, `TC`, `N_TOKENS`,
`CACHE`) against `colab_run/setup5.sh`'s pinned environment. A recycled VM
must be re-pinned first -- see `DO-NOT-RETEST.md` 1 -- otherwise the stock
jax 0.7.2 / libtpu 0.0.21.1 pair cannot compile any Pallas kernel.

**Prompting (added 2026-10-18):** `PROMPT=<text>` is now prefilled in full,
one token per step, before generation starts; `N_TOKENS` bounds the generated
continuation and `MAX_T` the context. Before this, only the first prompt token
was consumed, which is why every earlier run repeated the same continuation. A
single-token prompt (`PROMPT=Hello`) takes the exact path the receipts above
used, so they remain reproducible. Two receipts with a 19-token prompt on this
build: `receipt-prompt-b80-f16-tc1024.json` (batch 80, 64 generated, 268.3
tok/s) and `receipt-prompt-b64-f16-tc1024.json` (batch 64, 128 generated,
207.1 tok/s -- long enough to reach the answered part of the continuation).

**Context cost at fixed batch, measured 2026-10-20 (batch 1, the fused step, one
prompt throughout):** `max_t` 83 -> **152.3 ms/step (6.57 tok/s)**, `max_t` 1024 ->
153.6 ms (6.51 tok/s), `max_t` 8192 -> **180.9 ms (5.53 tok/s)** -- receipts
`receipt-ctx64-b1-f16-tc1024.json`, `receipt-ctx1024-b1-f16-tc1024.json`,
`receipt-ctx8192-b1-f16-tc1024.json`. **The whole 8k window costs 19 %** at batch
1, which makes the unpack, not the context, the thing to fix for single-stream
throughput. The `max_t` 45056 point needs `FUSE=0`: with the fused step it does not
compile (*"CompileTimeHbmOom: ... HLO temporaries (18.92G) exceeds available HBM
(15.75G)"*).

**Context cost -- corrected 2026-10-20, because the curve is confounded.**
The per-step ms on this build: 284.7 (`max_t` 64, batch 80) -> 298 (`max_t` 96,
batch 80) -> 309 (`max_t` 160, batch 64) -> 638 (`max_t` 320, batch 48) -> 662
(`max_t` 512, batch 32). It used to be read here as "the step scales with
`MAX_T`, not only with the batch". **That reading does not survive its own
table:** the two longest-context runs used the *smaller* batches (48 and 32
against 80), so batch cannot be the sole cause, and the context cost at fixed
batch has never been measured. A claim made in conversation on 2026-10-19 --
that those jumps were "batch-driven, not context" -- is withdrawn for the same
reason. The honest statement is that **both** terms are in the curve and
neither has been isolated; `TODO.md` `TPU-06` files the measurement.

What is solid, because it was measured at fixed batch and fixed window: a
long-context run trades throughput for reach. Same 41-token prompt, batch 48 /
256 generated at `max_t` 320 measured 75.2 tok/s
(`receipt-dxprompt-b48-f16-tc1024.json`), batch 80 / `max_t` 96 / 44
generated measured 268.2 tok/s (`receipt-dxprompt-b80-f16-tc1024.json`),
and batch 32 / 460 generated at `max_t` 512 measured 48.3 tok/s
(`receipt-essay-b32-f16-tc1024.json`).

MIT OR Apache-2.0
updated: 2026-10-21 12:35 UTC
