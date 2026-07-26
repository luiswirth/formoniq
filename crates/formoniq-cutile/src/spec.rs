//! What the kernels compute, written as ordinary Rust.
//!
//! Every device kernel in this crate has exactly one function here stating its
//! arithmetic, on flat slices in the layout the device uses. These are the
//! contract: the GPU kernel is correct when it agrees with its specification,
//! and the specification is correct when the pipeline built from it agrees with
//! [`formoniq::matfree::ElementOperator`].
//!
//! Keeping the two apart is what makes the mathematics testable without a GPU.
//! The specification is compiled and checked everywhere, so the decomposition
//! into kernels --- which is the part a hardware run cannot help you get right
//! --- is settled before any device is involved. What the GPU adds is speed,
//! and a transcription that can be diffed against a reference line by line.
//!
//! # Layouts
//!
//! Device arrays are **row-major**, with the parallel axis leading, which is
//! how a tile program indexes them. That is the transpose of nalgebra's
//! column-major convention, and the conversion happens once, at the boundary,
//! in [`crate::CellOperator`].
//!
//! An element matrix is flattened **column-major within itself**, index
//! $r = i + "nrows" j$, matching
//! [`ElMatKernel`](formoniq::operators::kernel::ElMatKernel), whose constant
//! tensor is built in that order.

/// The shapes every kernel is parameterized by.
///
/// One value rather than a list of loose integers, since a mismatch between any
/// two of them is a silent wrong answer rather than a crash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shapes {
  /// The number of cells: the parallel axis, and the only large dimension.
  pub ncells: usize,
  /// $P^2$, the entries of a cell's multiform Gramian.
  pub ngramian: usize,
  /// Rows of an element matrix: the local row degrees of freedom.
  pub nrows: usize,
  /// Columns of an element matrix: the local column degrees of freedom.
  pub ncols: usize,
}

impl Shapes {
  /// The entries of one element matrix, $"nrows" dot "ncols"$.
  pub fn nelmat(&self) -> usize {
    self.nrows * self.ncols
  }
}

/// The batched element matrices: `elmats[c, r] = sum_p gramians[c, p] coeff[r, p]`.
///
/// One matrix product, $G A^top$, and the whole arithmetic of assembly. `coeff`
/// is the constant tensor, shared by every cell; `gramians` is the entire
/// per-cell geometry, the cell volume already folded in.
///
/// - `coeff`: `nelmat × ngramian`, row-major
/// - `gramians`: `ncells × ngramian`, row-major
/// - `elmats`: `ncells × nelmat`, row-major
pub fn elmat_batch(shapes: Shapes, coeff: &[f64], gramians: &[f64], elmats: &mut [f64]) {
  let nelmat = shapes.nelmat();
  assert_eq!(coeff.len(), nelmat * shapes.ngramian);
  assert_eq!(gramians.len(), shapes.ncells * shapes.ngramian);
  assert_eq!(elmats.len(), shapes.ncells * nelmat);

  for icell in 0..shapes.ncells {
    let gramian = &gramians[icell * shapes.ngramian..(icell + 1) * shapes.ngramian];
    for ientry in 0..nelmat {
      let row = &coeff[ientry * shapes.ngramian..(ientry + 1) * shapes.ngramian];
      elmats[icell * nelmat + ientry] = row.iter().zip(gramian).map(|(a, g)| a * g).sum::<f64>();
    }
  }
}

/// Each cell's local contribution to the matvec, before any of them are
/// combined: `locals[c, i] = sum_j M_c[i, j] x[col_dofs[c, j]]`.
///
/// The element matrix is rebuilt from the constant tensor and discarded, never
/// reaching memory. Writing to a *dense* per-cell array rather than accumulating
/// into the global vector is what keeps the output disjoint across cells, so
/// the tile programs need no atomics and the ownership model accepts the kernel
/// as written.
///
/// Takes the cell's degrees of freedom *already gathered*, `cellx[c, j] =
/// x[col_dofs[c, j]]`, which [`gather`] produces. Splitting the gather out
/// leaves this kernel entirely regular: every read is a partition load and
/// every write lands in the cell's own slot.
///
/// - `cellx`: `ncells × ncols`, row-major
/// - `locals`: `ncells × nrows`, row-major
pub fn cell_matvec(
  shapes: Shapes,
  coeff: &[f64],
  gramians: &[f64],
  cellx: &[f64],
  locals: &mut [f64],
) {
  let (nrows, ncols) = (shapes.nrows, shapes.ncols);
  assert_eq!(cellx.len(), shapes.ncells * ncols);
  assert_eq!(locals.len(), shapes.ncells * nrows);

  for icell in 0..shapes.ncells {
    let gramian = &gramians[icell * shapes.ngramian..(icell + 1) * shapes.ngramian];
    let local_x = &cellx[icell * ncols..(icell + 1) * ncols];

    for irow in 0..nrows {
      let mut sum = 0.0;
      for (jcol, &xj) in local_x.iter().enumerate() {
        // Column-major within the element matrix: entry (irow, jcol).
        let entry = irow + nrows * jcol;
        let row = &coeff[entry * shapes.ngramian..(entry + 1) * shapes.ngramian];
        let value: f64 = row.iter().zip(gramian).map(|(a, g)| a * g).sum();
        sum += value * xj;
      }
      locals[icell * nrows + irow] = sum;
    }
  }
}

/// Sum the per-cell contributions into the global vector, one degree of freedom
/// at a time: `y[i] = sum_(t in segment i) locals[indices[t]]`.
///
/// The transpose reading of the local-to-global map. Assembly's natural
/// direction is to scatter each cell's rows outward, which two cells sharing a
/// face do to the same entry; gathering instead makes every output element the
/// property of exactly one tile program, so the race is not avoided by
/// synchronization but absent by construction.
///
/// `offsets` has length `ndofs + 1` and is non-decreasing, `indices` holds
/// positions into the flat `locals` array, `c * nrows + i`.
pub fn gather_sum(locals: &[f64], offsets: &[u32], indices: &[u32], y: &mut [f64]) {
  assert_eq!(offsets.len(), y.len() + 1);
  for (idof, out) in y.iter_mut().enumerate() {
    let (begin, end) = (offsets[idof] as usize, offsets[idof + 1] as usize);
    *out = indices[begin..end]
      .iter()
      .map(|&t| locals[t as usize])
      .sum();
  }
}

/// [`gather_sum`] with every segment padded to a common width, so the inner
/// extent is a compile-time constant.
///
/// A tile program wants a fixed shape, and a segment whose length varies with
/// the valence of a degree of freedom does not have one. Padding to the maximum
/// valence buys that shape back. The padding entries point at one extra zero
/// appended to `locals`, so the kernel contains no branch and no masking: every
/// lane gathers and the padding contributes zero.
///
/// - `table`: `ndofs × width`, row-major, entries index into `locals`
/// - `locals`: the per-cell contributions **followed by one trailing zero**
///
/// The cost is the waste of gathering the padding, which is the gap between
/// average and maximum valence. On a structured mesh that is small; on a badly
/// graded unstructured one it is not, and the compressed [`gather_sum`] is the
/// right kernel there instead.
pub fn gather_sum_padded(locals: &[f64], table: &[u32], width: usize, y: &mut [f64]) {
  assert_eq!(table.len(), y.len() * width);
  for (idof, out) in y.iter_mut().enumerate() {
    *out = table[idof * width..(idof + 1) * width]
      .iter()
      .map(|&t| locals[t as usize])
      .sum();
  }
}

/// `out[t] = src[indices[t]]`: an indexed read, and the only irregular access
/// in the crate.
///
/// Isolated into its own kernel deliberately. The tile model addresses memory
/// through partitions of a tensor, which express a *regular* access and nothing
/// else; reading `src` at an arbitrary index needs cuTile's pointer path, which
/// is the documented local opt-out from the ownership discipline. Confining it
/// here means exactly one kernel is unsafe and every other one is checked, and
/// the one that is unsafe is the one where the mesh's irregularity actually
/// lives.
///
/// The output is disjoint by construction --- one entry per lane, written once
/// --- so even the opt-out kernel has no race to reason about. What it gives up
/// is only the bounds guarantee on the read.
pub fn gather(src: &[f64], indices: &[u32], out: &mut [f64]) {
  assert_eq!(out.len(), indices.len());
  for (out, &index) in out.iter_mut().zip(indices) {
    *out = src[index as usize];
  }
}

/// Sum each fixed-width row: `y[i] = sum_w gathered[i, w]`.
///
/// The second half of the padded gather, once [`gather`] has made the access
/// regular. A reduction along the minor axis of a tile, with an output element
/// per tile program, so it is race-free in the safe API.
pub fn segment_reduce(gathered: &[f64], width: usize, y: &mut [f64]) {
  assert_eq!(gathered.len(), y.len() * width);
  for (idof, out) in y.iter_mut().enumerate() {
    *out = gathered[idof * width..(idof + 1) * width].iter().sum();
  }
}

/// $y <- alpha x + beta y$, elementwise.
pub fn axpby(alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) {
  assert_eq!(x.len(), y.len());
  for (y, x) in y.iter_mut().zip(x) {
    *y = alpha * x + beta * *y;
  }
}

/// $x <- alpha x$, elementwise.
pub fn scale(alpha: f64, x: &mut [f64]) {
  for x in x.iter_mut() {
    *x *= alpha;
  }
}

/// The inner product $angle.l x, y angle.r$, as a full reduction.
///
/// The one operation with no disjoint output: every element contributes to one
/// scalar. On the device it is a reduction to per-tile partial sums followed by
/// a reduction of those, which is why [`dot_partials`] exists separately.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
  assert_eq!(x.len(), y.len());
  x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// The first stage of [`dot`]: one partial sum per tile of `block` elements.
///
/// Stated separately because the device performs the reduction in two stages
/// and the *sum order therefore differs* from the flat [`dot`]. Floating-point
/// addition is not associative, so the two agree only to rounding, and a test
/// comparing them must say so rather than demand equality.
pub fn dot_partials(x: &[f64], y: &[f64], block: usize, partials: &mut [f64]) {
  assert_eq!(x.len(), y.len());
  assert_eq!(partials.len(), x.len().div_ceil(block));
  for (ipartial, out) in partials.iter_mut().enumerate() {
    let begin = ipartial * block;
    let end = (begin + block).min(x.len());
    *out = x[begin..end]
      .iter()
      .zip(&y[begin..end])
      .map(|(a, b)| a * b)
      .sum();
  }
}
