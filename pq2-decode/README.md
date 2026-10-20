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

MIT OR Apache-2.0
updated: 2026-10-20 23:20 UTC
