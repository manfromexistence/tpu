"""The PQ2_0 (TQ2_0) block decode, in numpy, verified against the C reference.

The format, from the PrismML fork's ggml-common.h / ggml-quants.c:

    // 2-bit quantization, 16 blocks of 16 elements, 2.625 bits per weight
    typedef struct {
        uint8_t scales[QK_K/16];   // 16 bytes
        uint8_t qs[QK_K/4];        // 64 bytes: four 2-bit fields each
        ggml_half d;               // the scale
        ggml_half dmin;            // (unused by the simple decode)
    } block_tq2_0;                 // 84 bytes for 256 elements

    // dequantize_row_tq2_0:
    for (j = 0; j < sizeof(x->qs); j += 32)
      for (l = 0; l < 4; l++)
        for (m = 0; m < 32; m++) {
          int8_t q = (x[i].qs[j + m] >> (l*2)) & 3;
          *y++ = (float)(q - 1) * d;
        }

Note what this buys over PTQ1_0: the PTQ1_0 value is
`floor(3*((byte*p) & 0xFF)/256) - 1`, a modular multiply per element whose
`p` depends on the ROW -- which is what forces the group concatenation. The
2-bit form is a shift, a mask, a subtract and a scale, with no row dependence
at all, so the four 2-bit planes can be contracted as four clean matmuls.

Verified locally (no TPU): the reference loop, a vectorised form and a
four-plane split agree on 200 random super-blocks.

See `tpu/TODO.md` item TPU-08 and `tpu/lane-aligned-ternary-kernel/README.md`.
"""
import numpy as np

QK = 256           # elements per super-block
BLK = QK // 4      # 64 bytes of 2-bit quants
SCALES = QK // 16  # 16 bytes of sub-block scales (unused by the simple decode)
SCALE_OFF = SCALES + BLK


def decode_superblock(raw: bytes) -> np.ndarray:
    """One 84-byte super-block -> 256 float32 values, in the file's order."""
    assert len(raw) == 84, len(raw)
    qs = np.frombuffer(raw[SCALES:SCALES + BLK], dtype=np.uint8)
    d = np.frombuffer(raw[SCALE_OFF:SCALE_OFF + 2], dtype=np.float16)[0]
    out = np.empty(QK, dtype=np.float32)
    pos = 0
    for j in range(0, BLK, 32):
        for l in range(4):
            for m in range(32):
                q = (int(qs[j + m]) >> (l * 2)) & 3
                out[pos] = (q - 1) * np.float32(d)
                pos += 1
    assert pos == QK
    return out


def decode_vectorized(raw: bytes) -> np.ndarray:
    """The same thing without the triple loop -- the shape a Pallas kernel wants.

    The C loop's order is: for each 32-byte chunk, the four 2-bit planes in
    turn, each plane's 32 bytes in order. So the value order is
    (chunk, plane, byte) -- and the four planes of one chunk are strided by 32
    in the output, which is exactly the "four planes" structure the kernel can
    contract as four matmuls.
    """
    assert len(raw) == 84, len(raw)
    qs = np.frombuffer(raw[SCALES:SCALES + BLK], dtype=np.uint8)
    d = np.float32(np.frombuffer(raw[SCALE_OFF:SCALE_OFF + 2], dtype=np.float16)[0])
    qb = qs.reshape(2, 32)
    codes = (qb[:, :, None] >> (2 * np.arange(4, dtype=np.uint8))) & 3
    vals = (codes.transpose(0, 2, 1).astype(np.float32) - 1.0) * d
    return vals.reshape(-1)


def planes(raw: bytes):
    """The four 2-bit planes of a super-block, each (64,) in output order.

    Each plane is strided by 32 in the in-dim axis, so a kernel that wants
    them contiguous has to permute the basis (and the weight rows) -- that is
    the "pre-permute the Hadamard basis" step, and it is a host-side repack.
    """
    assert len(raw) == 84, len(raw)
    qs = np.frombuffer(raw[SCALES:SCALES + BLK], dtype=np.uint8)
    d = np.float32(np.frombuffer(raw[SCALE_OFF:SCALE_OFF + 2], dtype=np.float16)[0])
    qb = qs.reshape(2, 32)
    codes = (qb[:, :, None] >> (2 * np.arange(4, dtype=np.uint8))) & 3
    out = codes.transpose(0, 2, 1)
    return [(out[:, p, :].reshape(-1).astype(np.float32) - 1.0) * d for p in range(4)]


def _random_block(rng) -> bytes:
    raw = bytearray(bytes(rng.integers(0, 256, 84, dtype=np.uint8)))
    # a random 2-byte pattern can be a NaN in fp16; use a valid scale
    scale = np.array([0.5 + 2.0 * rng.random()], dtype=np.float16)
    raw[SCALE_OFF:SCALE_OFF + 2] = scale.tobytes()
    return bytes(raw)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for trial in range(200):
        raw = _random_block(rng)
        a = decode_superblock(raw)
        b = decode_vectorized(raw)
        assert np.array_equal(a, b), (trial, a[:8], b[:8])
        # the four planes, placed back at their (chunk, plane, byte) positions,
        # must equal the flat decode
        pl = planes(raw)
        flat = np.empty(256, dtype=np.float32)
        for c in range(2):
            for p in range(4):
                flat[c * 128 + p * 32: c * 128 + (p + 1) * 32] = \
                    pl[p][c * 32:(c + 1) * 32]
        assert np.array_equal(flat, a), trial
    print("TQ2_0 decode: the C loop, the vectorised form and the four-plane "
          "split agree on 200 random super-blocks")

    # a worked example: one byte, four codes, d = 1.0
    raw = bytearray(84)
    raw[SCALES] = 0b11_10_01_00        # low field first: 0,1,2,3
    raw[SCALE_OFF:SCALE_OFF + 2] = np.array([1.0], dtype=np.float16).tobytes()
    v = decode_superblock(bytes(raw))
    print("first 8 values for byte 0b11_10_01_00, d=1.0:", v[:8])
    print("expected: code 0 -> -1, 1 -> 0, 2 -> +1, 3 -> +2, then the next plane")
