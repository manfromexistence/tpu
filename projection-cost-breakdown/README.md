# 13 — cost breakdown, taken **second**, inside a real model run

**This folder is not a standalone session.** By the standing rule
(`tpu/README.md` §6, `DO-NOT-RETEST.md` §0) every TPU session runs the real
64-layer Bonsai 2 model end to end, and the numbers here are taken in that same
session **after the model number is in the log**. A session that ran only this
file would be a wasted session, however clean the measurements looked.

The rule exists because this track has been burned three times by single-tensor
measurements that did not survive contact with the model: nb 10's "~59 tok/s at
batch 128" (a 27.0 GB layout that does not fit), `DO-NOT-RETEST.md` §6's own
retraction, and nb 12's k-loop refutation, which is confounded. This folder is
meant to be *appended to* a model run, not to replace one.

## Status

**Executed 2026-10-18, three passes, on Colab TPU v5e-1, after the model run.**
Receipt: `receipt-cost-sweep-pass3.json`. It settled the question it was written
to settle, and the answer became a 1.94x change to the model.

**Later the same day, notebook 14 changed the tile layout** (out-columns on the
128 lanes) and the real model went 29.66 -> 281.01 tok/s. The per-cell timings
below are properties of the old, lane-starved layout -- do not reconcile them
against notebook 14's 284.7 ms/step. See `DO-NOT-RETEST.md` §8.

| Question | Answer | Evidence |
| --- | --- | --- |
| Is the Colab slice bandwidth-limited? | No — 709 GB/s read, 623 GB/s copy, ~86 % of the v5e-1 spec | Q0 |
| Does cost scale with program count? | No — `TO` 128 -> 1024 is 3200 -> 400 programs for **1.18x**, saturating by 512 | Q2 |
| How much is the weight DMA? | 0.50 ms of 3.83 ms at TO=128 | Q3 |
| How much is the decode arithmetic? | **3.13 ms of 3.83 ms at TO=512, 82 %** | Q4 |
| What inside the decode costs? | the **spelling**: `stack(axis=1).reshape` prices as a layout conversion | Q5 |
| Can it be made cheaper? | Yes: `concatenate(-1)` **1.68x**, int16 lanes **1.34x**, both identical values | Q5 |
| Is a byte-major broadcast order better? | No — **4.2x worse** than the concatenated spelling | Q5 |

## What the breakdown answers

`DO-NOT-RETEST.md` §7 and `07-pallas-matmul-findings/FINDINGS.md` §7 both stopped at the
same question: the fused kernel ran ~300x below the HBM roofline and ~50x below
the VPU, every arithmetic explanation had been excluded by measurement, and the
two survivors — **program count** and **the unpack/cast/scale chain** — had never
been separated.

It prices the stages of the *proven* kernel in order, on the weights the model
uses: weight DMA alone, then DMA + unpack, then DMA + unpack + matmul. Each
stage is the previous one plus one thing, so the difference between two rows of
the receipt is the cost of the thing that was added.

It also measures something nobody had measured on this track: **raw HBM
bandwidth of the Colab v5e-1 slice**. That came back at 709 GB/s, so the ~2 s per
forward pass was never a property of the slice.

## Measured (all three passes, TO=512 unless noted)

Layer-0 `blk.0.attn_qkv.weight`, 52.4 M trits, B=32, `REPS=10`.

| Cell | ms | Gtrit/s | vs nb 11 |
| --- | --- | --- | --- |
| read f32 (sum) — 256 MB | 0.362 | 709.6 GB/s | Q0 |
| copy f32 (`x*2`) — 512 MB moved | 0.823 | 622.4 GB/s | Q0 |
| read int8 (sum) — 128 MB | 0.185 | 693.5 GB/s | Q0 |
| dispatch floor (4 KB op) | 0.082 | 82 us/call | Q0 |
| `dot` TO=128 (the nb 11 config) | 3.834 | 13.68 | 1.00x |
| `dot` TO=256 | 3.443 | 15.23 | 1.11x |
| `dot` TO=512 | 3.288 | 15.94 | 1.17x |
| `dot` TO=1024 | 3.249 | 16.14 | 1.18x |
| `dma` TO=128 | 0.500 | 104.86 | 7.67x |
| `dma` TO=512 | 0.291 | 180.44 | 13.19x |
| `dec` (stack) TO=512 | 3.129 | 16.75 | 1.23x |
| `cast` TO=512 (cast + sum only) | 0.390 | 134.46 | 9.83x |
| `dec_nc` TO=512 (decode, no final concat) | 3.120 | 16.81 | 1.23x |
| `dec2` TO=512 (`concatenate` spelling) | 1.866 | 28.09 | 2.06x |
| `dec2` TO=1024 | 1.822 | 28.78 | 2.11x |
| `dec3` TO=512 (byte-major broadcast) | 7.916 | 6.62 | 0.48x |
| `dec16` TO=512 (int16, identical values) | **1.393** | 37.64 | 2.75x |
| `dec16` TO=1024 | **1.361** | **38.52** | 2.82x |

Read this as three facts:

1. **`dma` is 0.29 ms of a 3.83 ms projection.** Everything else is arithmetic
   and data movement *inside* the VPU. HBM was never the problem, and the
   projection has no 300x mystery left in it.
2. **`dec_nc` ≈ `dec`.** Removing the final `concatenate` of the three power
   groups changes nothing, so the trailing concat is free. The cost is the
   `stack(axis=1).reshape` *inside* each group, and `dec2` — which produces the
   identical columns — is 1.68x cheaper. **The reshape prices as a layout
   conversion, not as its instruction count.**
3. **`dec3` is 4.2x worse than `dec2`** for the same 128 trits in a different
   (byte-major) column order. That is the control: when the decode asks Mosaic
   for a genuine interleave, it pays an order of magnitude. It also re-explains
   the parked 2-bit repack (§`DO-NOT-RETEST.md` §7), whose extraction was a
   broadcast of exactly this shape.

`cast` at 0.390 ms against `dma` at 0.291 ms also says the int32 widening of the
payload is nearly free; the money is all after it.

### What the first pass got wrong

Pass 1's `dec2` used `P5v`/`P4v` broadcast constants captured from module scope,
and this stack rejects those (`captures constants [i32[1,1,5], i32[1,1,4]]` —
pass them as inputs or build them in-kernel). Pass 3's `dec16` builds its power
vectors from `jnp.arange`, so the same shape of variant now compiles. The
failure was caught by the per-cell `try/except` and cost nothing.

## `TO` is the lever nobody had pulled

`TO` is the out-tile height. In every configuration before this one it was pinned
to `TC = 128`, the number of trits in a PTQ1_0 block. That is an accident of
the reference layout, not a constraint: the weight block is `(TO, 28)` against
an array of `(ROWS, 28)`, which is legal for any `TO` divisible by 8, because
the block's last dim equals the array's last dim (`FINDINGS.md` §3).

`FINDINGS.md:228` already names **"a 256-row out tile"** as *the* separating
experiment. Notebook 12 did not run it: nb 12 moved `KB` 1→8 **and** `TO`
128→64 in the same change, so it cut the step count 4x while doubling the
program count, and its 15 % regression is evidence about `TO` rather than about
the k-loop.

Pulled now, on real weights: `TO` is worth **1.18x** and stops mattering at 512.
Not nothing — it is applied in nb 11, together with the decode fixes — but the
program-count theory is dead, and `TO=64` is illegal anyway (a rank-1 `(64,)`
f32 scale block is below the 128-element floor).

## What was checked locally, before spending anything

`G:\Temp\UserTemp\opencode\colab_run\check13.py` and `check13b.py` pull the
load-bearing functions out of the shipped `kernel.py` with `ast` — so they test
the shipped source, not a copy — and run them with numpy. (They live outside the
repo because `tpu/*/` allow-lists only `kernel.py`, `make_notebook.py`,
`kernel-metadata.json`, `notebook.ipynb`, `receipt-*.json` and `*.md`.)

| Check | Result |
| --- | --- |
| `host_tiles(TO)` row/scale geometry, TO 8..256 | 0 mismatches, bit-exact |
| `perm_ctg()` maps `dec_contig` order onto `dec_stack` order | `True` |
| `perm_ctg()` is a bijection | `True` |
| `dec_cat` produces the same columns as `dec_stack` | `True` |
| both against an independent decode written from the format spec | `True` |
| `3 ** arange(5)` is the power vector the decode assumes | `True` |

Those checks found real bugs: byte/power swapped in two of the three groups of
the first `perm_ctg`, and — in the check itself — a reference written in the
wrong column order and a `.view(np.uint16)` applied to a float32 array, which
doubles the element count instead of reinterpreting it. That is the class of
mistake that would otherwise have been discovered on the TPU as an unexplained
numerical mismatch.

## Cells

| Cell | `mode` | Prices |
| --- | --- | --- |
| Q1 `TO=128` | `dot` | the nb 11 configuration — cross-checks nb 10's 4.04 ms |
| Q2 `TO` 64…1024 | `dot` | whether cost scales with program count |
| Q3 | `dma` | the weight DMA and grid overhead alone |
| Q4 | `dec` | what the unpack/cast/scale chain adds |
| Q5 | `cast`, `dec_nc`, `dec2`, `dec3`, `dec16` | what *inside* the unpack costs |

`dma`, `dec`, `cast` and `dec_nc` are ablations: they price a stage, they are not
numerically checked. `dec2`/`dec3`/`dec16` produce the same 128 trits per block
as `dec_stack` — `dec2` in the same order, `dec3` in byte-major order, which is
free only because `x` is permuted to match (`perm_ctg`) — and all of them are
verified equal to a format-spec reference locally.

Compile failures do not wedge the core (`FINDINGS.md` §1), so every cell is
wrapped and a bad shape cannot cost the session. Runtime halts do wedge it, so
the proven configuration is measured first and every line is flushed.

## Run — the model is step 1, this is step 2

Both steps are in **one** TPU session. Step 1 is the real model and its number
is the one that counts; step 2 is this breakdown. The `restart-kernel` between
them is mandatory: `colab exec` reuses the process, and nb 11's 6 GB of tiles
would otherwise still be resident (`DO-NOT-RETEST.md` §7).

```bash
S=<session>

# 1. THE MODEL. Do not skip, do not reorder.
colab restart-kernel -s $S
colab exec -s $S --timeout 2400 --env TC=512 -f /mnt/g/Dx/tpu/11-fully-fused-64-layer-model/kernel.py

# 2. the cost breakdown, same session
colab restart-kernel -s $S
sleep 12
colab exec -s $S --timeout 900 -f /mnt/g/Dx/tpu/projection-cost-breakdown/kernel.py
```

Step 2 takes about 25 seconds: compile dominates, and the measured kernel time is
milliseconds. `RECEIPT=/content/receipt-colab-tpu-13.json` by default.

Folding these cells into `archive/11-fully-fused-64-layer-model/kernel.py` so the model and its
breakdown are one file is the preferred end state. It is not done yet, and this
README does not claim it is.

MIT OR Apache-2.0
updated: 2026-10-18 05:45 UTC
