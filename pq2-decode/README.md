# `tpu/pq2-decode/` — the 2-bit `PQ2_0` weight format, pinned down locally

`prism-ml/Ternary-Bonsai-2-27B-gguf` publishes **two** encodings of the same model:

| file | weight encoding | size | the decode |
| --- | --- | --- | --- |
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 1.75 bpw, base-3 modular arithmetic | 5.93 GB | `floor(3*((byte*p) & 0xFF)/256) - 1` — the power `p` depends on the **row** |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 2.625 bpw, 2-bit fields | 7.25 GB | `q = (qs[j+m] >> (l*2)) & 3; y = (q-1)*d` — **no row dependence** |

Both carry the identical model: same `qwen35` architecture, same
`prism.hadamard.*` metadata (block 1024, the same 401 folded weight names, the same
28,672 sign values), same 851 tensors.

**Why the second one matters on a TPU.** The PTQ1_0 value is a modular multiply
per element whose power is a property of the row — that is *why* the shipped kernel
has to concatenate the five power groups into one (128, TC) weight block before it
can multiply. The 2-bit form is a shift, a mask, a subtract and a scale, with no row
dependence at all, so the four 2-bit planes can be contracted as four clean matmuls
and the concatenation disappears structurally rather than by a trick.

## What is here

`tq2_decode.py` — the format, as read out of the PrismML fork's own
`ggml-common.h` / `ggml-quants.c`, with three implementations that are checked
against each other:

* `decode_superblock` — the C reference loop, transcribed literally;
* `decode_vectorized` — the same result without the triple loop (the shape a
  Pallas kernel wants);
* `planes` — the same data split into the four 2-bit planes.

**Verified locally, with no TPU:** all three agree on 200 random super-blocks.

```
$ python tpu/pq2-decode/tq2_decode.py
TQ2_0 decode: the C loop, the vectorised form and the four-plane split agree on
200 random super-blocks
```

## The two things a port still has to decide

1. **The planes are strided by 32 in the in-dim axis**, so contracting them as
   four matmuls needs the in-dims permuted so a plane is contiguous. That is a
   basis permutation: the weight rows are permuted, and the activation's Hadamard
   basis and sign vector are permuted to match. Both are host-side.
2. **The price is 1.4 GB of extra HBM** (6.37 → ~7.8 GB of tiles on device),
   which lowers the fp32-KV context ceiling from ~45k to ~38k tokens, and the
   int8 MXU path additionally needs int8 activations — a precision change that
   has to be re-validated against the model's correctness gate.

This is `TODO.md` item **TPU-08**; the 1M-context arithmetic is in
`HELP.md` section 4.

## The port, step by step

Written before the code, so the next session executes rather than re-derives.

### 1. The tile layout changes shape

| | PTQ1_0 (shipped) | PQ2_0 |
| --- | --- | --- |
| values per byte-row | 128 | 256 |
| bytes per byte-row | 28 (+4 pad = 32) | 84 (+4 pad = 88) |
| padding overhead | +14 % | **+4.8 %** |
| in-dim order inside a row | 5 power groups: 80 + 40 + 8 | 2 chunks × 4 planes × 32 bytes |

The Pallas block dimension must be a multiple of 8, so a byte-row is padded from
84 to 88 (28 → 32 today). The 256 in-dims of one row are laid out
`(chunk, plane, byte)`: plane `p` of chunk `c` occupies in-dims
`c*128 + p*32 .. c*128 + p*32 + 31`.

### 2. The decode becomes four planes, and the planes are strided

The PTQ1_0 decode produces five power groups and **concatenates** them into one
(128, TC) block, because the power is a property of the row. The 2-bit decode
produces four planes with **no row dependence**, but each plane is strided by 32
in the in-dim axis, so a plane is not contiguous in the output.

Two ways to make a plane contiguous, and the choice matters:

* **permute the basis** (host-side, once): the weight rows are permuted so a
  plane is contiguous, and the activation's transform is permuted to match --
  the Hadamard basis `R` and the sign vector `ROW` are both permuted by the same
  permutation. The four planes then become four contiguous `(rows, TC)` blocks
  and the kernel does four matmuls with no concatenation at all.
* **keep the strided order** and let the MXU read the strided operand: possible,
  but the operand is then not a clean `(rows, TC)` block and the packing
  discipline Pallas wants is lost.

The first is the plan: it is host-side work (a permutation in the tile build
plus a permuted `R`/`ROW`), and it makes the kernel path *simpler* than
today's.

### 3. The reader change

The staging module (`colab_run/04-bonsai-tpu-forward.py`, uploaded as
`/content/04.py`) reads the GGUF and stages the raw weight blocks. For PQ2_0 it
must stage the 84-byte super-blocks instead of the 28-byte PTQ1_0 blocks; the
tensor list and the `prism.hadamard.*` metadata are identical, so the change is
local to the block reader. The kernel's `build_layer` then transposes the
super-blocks into the `(88, rows)` tile with the permuted row order.

### 4. What has to be true when it lands

* the correctness gate is unchanged: first token `11` for the prompt `Hello`,
  `271` for the 17-token prompt, and the generated text byte-identical to
  `receipt-14-b1-f16-tc1024-fused.json`;
* `unfold(fold(z))` still round-trips (`kernel.py` prints `max|R@R - I| = 0.0`);
* the step time at 8k / batch 1 is below the PTQ1_0 number, and the receipts
  record `proj_format: pq2_0` so the two are never compared without saying which
  is which.

### 5. The price, again, because it is the only thing that can sink this

1.4 GB more on device: the tiles go from 6.37 GiB to ~7.8 GiB, which drops the
fp32-KV context ceiling from ~45k to ~38k tokens. The int8 MXU path (which
halves the unpacked traffic, and is the only way to approach the HBM floor)
additionally needs int8 activations -- a precision change, so it needs its own
validation against the gate.

MIT OR Apache-2.0
updated: 2026-10-20 23:25 UTC
