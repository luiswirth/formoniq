# formoniq-cutile

Tile-based GPU kernels for the formoniq FEEC engine, written in cuTile Rust.

Two things run on the device: the batched evaluation of element matrices, and the
matrix-free application of an operator inside a Krylov solve. They are the same
arithmetic. An element matrix over Whitney forms reads the geometry of its cell
only through the volume and the multiform Gramian, and depends on the latter
linearly, so

```text
vec(M_c) = vol_c · A · vec(Λᵏ g_c⁻¹)
```

with `A` a constant matrix per dimension and grade. Stacking the cells turns
assembly into one matrix product, and it turns the operator apply into a batched
small dense matvec whose element matrices never reach memory.

That second reading is where the time is. A sparse matvec reads one coefficient
per nonzero, tens per row for a grade-1 Whitney operator; the matrix-free apply
reads the `binom(n,k)²` numbers of the cell geometry, shared by the whole element
matrix, and recomputes the rest. Trading memory traffic for arithmetic is the
wrong direction on a CPU and the right one on a GPU.

## Layout

The crate is split so that the mathematics can be checked without a GPU.

- `spec` states what each kernel computes, as ordinary Rust on flat slices in
  the device's layout. It is compiled and tested everywhere.
- `kernels` holds the cuTile kernels, one per function in `spec`.
- `device` holds the device-resident vector and the operator built on it.

`CellOperator` flattens a formoniq operator into the buffers a launch needs, and
applies it on the host through `spec`. Its equality with the assembled operator
is a test, so the decomposition into kernels is settled before any device is
involved; a device run can then only disagree by a transcription error in one
kernel, each of which has its own reference to be diffed against.

## Race freedom without atomics

Assembly's natural direction is to scatter each cell's rows into the global
vector, which two cells sharing a face do to the same entry. cuTile partitions a
mutable output into disjoint tiles, so that scatter is not expressible without
opting out of the ownership discipline.

The apply is therefore two stages. The first writes each cell's contribution to
its own slot of a dense per-cell array, disjoint by construction. The second
gathers, one degree of freedom per tile program, through the inverted
local-to-global map (`DofSegments`). No atomics appear, and the race is absent
rather than synchronized away.

## Building

The cuTile stack needs the CUDA toolkit at build time, so it sits behind an
off-by-default feature:

```sh
cargo test -p formoniq-cutile                   # spec and host reference
cargo test -p formoniq-cutile --features cuda   # adds the device kernels
```

The device tests require an NVIDIA GPU of compute capability 8.0 or above,
CUDA 13.3, and `CUDA_TOOLKIT_PATH` set. They are skipped, not failed, when no
device is present.
