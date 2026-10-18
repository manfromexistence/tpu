# 14-bonsai-transposed: out-columns on the 128 lanes

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
| 32 | f32 | 512 | 130.30 | 245.6 | `receipt-14-b32-f32-tc512.json` |
| 32 | f16 | 512 | 134.67 | 237.6 | `receipt-14-b32-f16-tc512.json` |
| 48 | f16 | 512 | 187.09 | 256.6 | `receipt-14-b48-f16-tc512.json` |
| 48 | f16 | 1024 | 203.01 | 236.4 | `receipt-14-b48-f16-tc1024.json` |
| 64 | f16 | 512 | 217.49 | 294.3 | `receipt-14-b64-f16-tc512.json` |
| 64 | f16 | 1024 | 239.25 | 267.5 | `receipt-14-b64-f16-tc1024.json` |
| 80 | f16 | 512 | 266.37 | 300.3 | `receipt-14-b80-f16-tc512.json` |
| 80 | f16 | 1024 | 279.46 | (repeat) | log only |
| **80** | **f16** | **1024** | **281.01** | **284.7** | `receipt-14-b80-f16-tc1024.json` |

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
build: `receipt-14-prompt-b80-f16-tc1024.json` (batch 80, 64 generated, 268.3
tok/s) and `receipt-14-prompt-b64-f16-tc1024.json` (batch 64, 128 generated,
207.1 tok/s -- long enough to reach the answered part of the continuation).

**Context costs step time (measured 2026-10-18):** the decode step scales with
`MAX_T`, not only with the batch. Per-step ms on this build: 284.7 (`max_t` 64,
batch 80) -> 298 (`max_t` 96, batch 80) -> 309 (`max_t` 160, batch 64) -> 638
(`max_t` 320, batch 48). A long-context run therefore trades throughput for
reach: same 41-token prompt, batch 48 / 256 generated at `max_t` 320 measured
75.2 tok/s (`receipt-14-dxprompt-b48-f16-tc1024.json`), while batch 80 /
`max_t` 96 / 44 generated measured 268.2 tok/s
(`receipt-14-dxprompt-b80-f16-tc1024.json`).

MIT OR Apache-2.0
updated: 2026-10-18 07:00 UTC
