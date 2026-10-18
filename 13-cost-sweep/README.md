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

## What the breakdown answers

`DO-NOT-RETEST.md` §7 and `07-pallas-fusion/FINDINGS.md` §7 both stop at the
same question: the fused kernel is ~300x below the HBM roofline and ~50x below
the VPU, every arithmetic explanation has already been excluded by measurement,
and the two survivors — **program count** and **the unpack/cast/scale chain** —
have never been separated.

It prices the stages of the *proven* kernel in order, on the weights the model
uses: weight DMA alone, then DMA + unpack, then DMA + unpack + matmul. Each
stage is the previous one plus one thing, so the difference between two rows of
the receipt is the cost of the thing that was added.

It also measures something nobody has measured on this track: **raw HBM
bandwidth of the Colab v5e-1 slice**. If that number is low, the ~2 s per
forward pass is a property of the slice we are given and no kernel work will
move it. If it is high, the loss is ours. Every other reading is conditional on
that number.

## `TO` is the lever nobody pulled

`TO` is the out-tile height. In every configuration so far it has been pinned
to `TC = 128`, the number of trits in a PTQ1_0 block. That is an accident of
the reference layout, not a constraint: the weight block is `(TO, 28)` against
an array of `(ROWS, 28)`, which is legal for any `TO` divisible by 8, because
the block's last dim equals the array's last dim (`FINDINGS.md` §3).

`FINDINGS.md:228` already names **"a 256-row out tile"** as *the* separating
experiment. It has never been run. Notebook 12 did not run it either: nb 12
moved `KB` 1→8 **and** `TO` 128→64 in the same change, so it cut the step count
4x while doubling the program count. Its 15 % regression is therefore evidence
about `TO`, not about the k-loop — and `DO-NOT-RETEST.md` §7's refutation of
"the k-loop is the bottleneck" rests on that confounded measurement.

## What was checked locally, before spending anything

`G:\Temp\UserTemp\opencode\colab_run\check13.py` pulls `host_tiles` and
`perm_ctg` out of `kernel.py` with `ast` and checks the shipped source:

| Check | Result |
| --- | --- |
| `host_tiles(TO)` row/scale geometry, TO 8..256 | 0 mismatches, bit-exact |
| `perm_ctg()` maps `dec_contig` order onto `dec_stack` order | `True` |
| `perm_ctg()` is a bijection | `True` |

The permutation check also found a real bug in the first draft of `perm_ctg`
(byte and power swapped in two of the three groups). That is the class of
mistake that would otherwise have been discovered on the TPU as an unexplained
numerical mismatch.

## Cells

| Cell | `mode` | Prices |
| --- | --- | --- |
| Q1 `TO=128` | `dot` | the nb 11 configuration — cross-checks nb 10's 4.04 ms |
| Q2 `TO` 64…1024 | `dot` | whether cost scales with program count |
| Q3 | `dma` | the weight DMA and grid overhead alone |
| Q4 | `dec` | what the unpack/cast/scale chain adds |
| Q5 | `dec2` | whether a contiguous decode order beats the 5-way `stack` |

`dma` and `dec` are ablations: they price a stage, they are not numerically
checked. `dec2` produces the same 128 trits per block as `dec_stack` in a
different order, which is free because `x` is permuted to match (`perm_ctg`).

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
colab exec -s $S --timeout 2700 -f /mnt/g/Dx/tpu/11-bonsai-fused-full/kernel.py

# 2. the cost breakdown, same session
colab restart-kernel -s $S
sleep 12
colab exec -s $S --timeout 900 -f /mnt/g/Dx/tpu/13-cost-sweep/kernel.py
```

Step 2 takes about two minutes: compile dominates, and the measured kernel time
is milliseconds. `RECEIPT=/content/receipt-colab-tpu-13.json` by default.

Folding these cells into `11-bonsai-fused-full/kernel.py` so the model and its
breakdown are one file is the preferred end state. It is not done yet, and this
README does not claim it is.

## Honest status

`kernel.py` is written and syntax-checked; its two load-bearing host functions
are verified locally. **Nothing in this folder has been executed on a TPU yet.**
The numbers it will produce do not exist, and no projection from this folder
should be quoted as if they did.

MIT OR Apache-2.0
updated: 2026-10-18 03:42 UTC
