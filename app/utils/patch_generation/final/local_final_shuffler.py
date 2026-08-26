"""
Final-stage shuffle for segmentation shards, on a local filesystem.

The intermediate stage writes one shard per scene so a killed Colab session
resumes rather than restarts. That leaves each shard holding ~270 spatially
sequential tiles of one scene, and the trainers shuffle only at shard level -
so a batch would be drawn from very few scenes. This stage fixes that.

Differences from `FinalShuffler`, which does the same job for the
reconstruction lane:

1. **Plain `tarfile`, not webdataset's reader.** webdataset routes every shard
   name through its URL opener, which has no handler for a bare path and reads
   a Windows ``C:\\...`` drive letter as a scheme. Three separate attempts to
   work around that failed. For a local-filesystem job the URL layer buys
   nothing, so this reads and writes tars directly. It also removes the torch
   dependency (no DataLoader), which matters on a CPU Colab session.
2. **Mixing is explicit.** `FinalShuffler` gets most of its mixing from
   `resampled=True` plus however many DataLoader workers happen to be
   configured - set workers to 0 and the shuffling quietly collapses. Here the
   interleave width is a parameter, so mixing cannot depend on a knob that
   looks like a performance setting.
3. **One complete pass.** `resampled=True` samples shards *with replacement* to
   hit a target count, which is why the hyperspectral trainer suppresses a
   "duplicate file name in tar" warning. For a fixed dataset we want every
   patch exactly once and none twice.

Input shards are left alone, so the output can be verified before anything is
deleted.
"""

import glob
import os
import random
import tarfile
from typing import Dict, Iterable, Iterator, List


class LocalFinalShuffler:
    """Mix per-scene shards into evenly sized, scene-interleaved shards."""

    def __init__(
        self,
        source_dir: str,
        dest_dir: str,
        shuffle_size: int = 200,
        group_size: int = 8,
        shard_size_bytes: int = 1 << 30,
        seed: int = 42,
    ):
        # shuffle_size costs shuffle_size x patch size in RAM. Segmentation
        # patches are ~8.2 MB, so 200 is ~1.6 GB - sized for Colab.
        # group_size is how many shards are read round-robin at once; it is
        # the main source of cross-scene mixing, not a speed setting.
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.shuffle_size = shuffle_size
        self.group_size = group_size
        self.shard_size_bytes = shard_size_bytes
        self.seed = seed

        self.sources = sorted(glob.glob(os.path.join(source_dir, "*.tar")))
        if not self.sources:
            raise FileNotFoundError(f"no .tar shards under {source_dir}")

        os.makedirs(dest_dir, exist_ok=True)
        self.shard_pattern = os.path.join(dest_dir, "final_shard_%05d.tar")

    # --- reading ---------------------------------------------------------

    @staticmethod
    def _records(tar: tarfile.TarFile) -> Iterator[Dict]:
        """Yield webdataset samples from an open tar.

        Members are named `<key>.<ext>` and a sample's members are contiguous,
        so a change of key ends the current sample. Keys here are
        `<scene_id>#row_coord:R#col_coord:C` and contain no dots.
        """
        key, parts = None, {}
        for member in tar:
            if not member.isfile():
                continue
            name, _, ext = member.name.partition(".")
            if name != key:
                if parts:
                    yield {"__key__": key, **parts}
                key, parts = name, {}
            parts[ext] = tar.extractfile(member).read()
        if parts:
            yield {"__key__": key, **parts}

    def _interleaved(self, paths: List[str]) -> Iterator[Dict]:
        """Round-robin across `group_size` shards at a time."""
        for start in range(0, len(paths), self.group_size):
            tars = [tarfile.open(p) for p in paths[start:start + self.group_size]]
            try:
                streams = [self._records(t) for t in tars]
                while streams:
                    for stream in list(streams):
                        try:
                            yield next(stream)
                        except StopIteration:
                            streams.remove(stream)
            finally:
                for t in tars:
                    t.close()

    @staticmethod
    def _shuffled(stream: Iterable[Dict], size: int, rng: random.Random) -> Iterator[Dict]:
        """Window shuffle: hold `size` samples, emit a random one each time."""
        buffer: List[Dict] = []
        for item in stream:
            buffer.append(item)
            if len(buffer) >= size:
                i = rng.randrange(len(buffer))
                buffer[i], buffer[-1] = buffer[-1], buffer[i]
                yield buffer.pop()
        rng.shuffle(buffer)
        yield from buffer

    # --- writing ---------------------------------------------------------

    def write_shards(self) -> None:
        """One complete pass: every patch out exactly once, order mixed."""
        import webdataset as wds  # TarWriter only; heavy, keep it lazy

        rng = random.Random(self.seed)
        paths = list(self.sources)
        rng.shuffle(paths)

        written, index, handle, tar = 0, 0, None, None
        for sample in self._shuffled(self._interleaved(paths), self.shuffle_size, rng):
            if handle is None:
                handle = open(self.shard_pattern % index, "wb")
                tar = wds.TarWriter(handle)
            tar.write(sample)
            written += 1
            if handle.tell() >= self.shard_size_bytes:
                tar.close()
                handle.close()
                handle, index = None, index + 1
        if handle is not None:
            tar.close()
            handle.close()

        print(f"[enmap_seg] shuffled {written} patches from "
              f"{len(self.sources)} shards into {index + 1} -> {self.dest_dir}")

    def __repr__(self) -> str:
        gb = sum(os.path.getsize(p) for p in self.sources) / 1e9
        return (
            f"LocalFinalShuffler({len(self.sources)} shards, {gb:.1f} GB"
            f" -> {self.dest_dir}, buffer={self.shuffle_size},"
            f" interleave={self.group_size})"
        )
