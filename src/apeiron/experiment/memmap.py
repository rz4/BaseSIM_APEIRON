"""Memory-mapped arrays for data too large to read through.

A window that does not fit in memory should be paged in by the operating
system on demand rather than read into the process. That needs a file whose
bytes are the array -- HDF5 is generally not one, being chunked and often
compressed, so the file a harness fetches is not the file it wants to map.

The mapped file is therefore *derived*: converted once from whatever was
fetched and then reused forever, which is the same bargain the store already
makes for downloads.

The format is ``.npy``. It is not a choice worth making twice: the header is
self-describing (shape, dtype, order), the array is contiguous after it,
``numpy`` maps it with one call, and it is one file, so landing it atomically
is the rename the store already does. A custom container would need its own
header, its own sidecar, and its own way of keeping the two consistent.

Writing works without holding the array in memory::

    with writing(dest, shape=(1000, 4, 128, 128), dtype="float32") as out:
        for i, frame in enumerate(source_frames()):
            out[i] = frame          # goes to disk, not to RAM

and reading gives a view, not a copy::

    frames = read(path)             # nothing has been read yet
    batch = frames[10:18]           # only these pages are faulted in
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

SUFFIX = ".npy"


@contextmanager
def writing(dest: str | Path, shape: Sequence[int], dtype: Any) -> Iterator[np.memmap]:
    """Open a new mapped array for writing and flush it on exit.

    The array is created at its full size immediately and filled through the
    returned view, so converting a large file never holds it in memory.
    """
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape)
    )
    try:
        yield out
    finally:
        out.flush()
        del out


def read(path: str | Path) -> np.memmap:
    """Map an array read-only. Pages are faulted in as they are touched.

    The result is read-only and backed by the file. Slice it before handing it
    to torch -- ``torch.from_numpy`` on a mapped array shares memory and
    refuses a read-only buffer, so the usual move is::

        torch.from_numpy(np.ascontiguousarray(frames[a:b]))

    which copies only the slice.
    """
    mapped = np.load(Path(path), mmap_mode="r")
    if not isinstance(mapped, np.memmap):  # pragma: no cover - defensive
        raise TypeError(f"{path} did not map to an array")
    return mapped


def describe(path: str | Path) -> dict[str, Any]:
    """Shape and dtype of a mapped file.

    Mapping does not read the array, so this costs a header read no matter how
    large the file is.
    """
    mapped = read(path)
    return {
        "shape": list(mapped.shape),
        "dtype": str(mapped.dtype),
        "bytes": int(mapped.nbytes),
    }
