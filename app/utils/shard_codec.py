"""
Lossless compression for int16 spectral cubes in training shards: band-to-band delta, then zstd.

Written by scripts/recompress_shards.py; read by the spectral-stem trainer's webdataset decoder
(the `.i16dz` member). Raw `.npy` shards stay readable alongside.

Why: shards leave AWS for the airgapped H200, where S3 egress costs $0.109/GB and the VM disk
(~540 MB/s) caps patches per second. Neighbouring bands are strongly correlated, so storing each
band as its difference from the previous one leaves small numbers that zstd packs well.
Measured on SpectralEarth shards (2026-10-14): 1.91x smaller (zstd alone: 1.35x), decode
~140 MiB/s per core.

Gotchas:
  - Differences are taken in int16 and wrap modulo 2^16 (nodata -32768 next to a bright band
    overflows); the cumulative sum on decode wraps the same way, so the round trip is exact.
  - Delta runs along axis 0 (bands); shape and dtype travel in a fixed header.
"""

import struct

import numpy as np
import zstandard

MAGIC = b"I16DZ1"
# zstd level 3: ~1.91x at ~130 MiB/s encode; level 9 gains 2.6% for 2.5x the encode time.
LEVEL = 3


def encode_i16dz(a: np.ndarray) -> bytes:
    """(C, ...) int16 -> bytes: header (magic, ndim, shape) + zstd(band delta)."""
    if a.dtype != np.int16 or a.ndim < 1:
        raise ValueError(f"expected an int16 array with >= 1 dim, got {a.dtype} {a.shape}")
    delta = np.diff(a, axis=0, prepend=np.zeros((1,) + a.shape[1:], np.int16))   # wraps in int16
    header = MAGIC + struct.pack(f"<B{a.ndim}I", a.ndim, *a.shape)
    return header + zstandard.ZstdCompressor(level=LEVEL).compress(np.ascontiguousarray(delta).tobytes())


def decode_i16dz(data: bytes) -> np.ndarray:
    """Inverse of encode_i16dz. Raises ValueError on a wrong header or a truncated payload."""
    if data[:len(MAGIC)] != MAGIC:
        raise ValueError("not an I16DZ1 blob")
    ndim = data[len(MAGIC)]
    shape = struct.unpack_from(f"<{ndim}I", data, len(MAGIC) + 1)
    payload = data[len(MAGIC) + 1 + 4 * ndim:]
    raw = zstandard.ZstdDecompressor().decompress(payload, max_output_size=2 * int(np.prod(shape)))
    delta = np.frombuffer(raw, np.int16)
    if delta.size != int(np.prod(shape)):
        raise ValueError(f"payload holds {delta.size} values, header says {shape}")
    return np.cumsum(delta.reshape(shape), axis=0, dtype=np.int16)                 # wraps back
