//! The cuTile kernels, one per function in [`crate::spec`].
//!
//! Each kernel here computes exactly what its specification computes, and the
//! device tests assert that. Read the two side by side: the specification says
//! what, this says how it maps onto tiles.
//!
//! # Where the safety boundary falls
//!
//! Every kernel but one writes a *disjoint* output: an element matrix belongs
//! to its cell, a reduced degree of freedom to itself. cuTile takes the mutable
//! output as a partition, so that disjointness is what the launcher checks, and
//! there is no atomic anywhere in the crate.
//!
//! [`gather`] is the exception, and it is where the mesh's irregularity is
//! confined. A tile program addresses memory through partitions of a tensor,
//! which is a regular access by construction; reading at an index the mesh
//! chose is not regular, and cuTile expresses it through the pointer path,
//! which is `unsafe`. Its *output* is still one element per lane, written once,
//! so what the opt-out gives up is the bounds guarantee on the read, not the
//! absence of races.
//!
//! # Shapes
//!
//! Tile extents are compile-time constants, so the shapes of an operator ---
//! $binom(n,k)^2$ Gramian entries, $binom(n+1,k+1)$ local degrees of freedom
//! --- arrive as generic parameters and are monomorphized at JIT time from the
//! runtime dimension and grade. This is the one place where formoniq's runtime
//! `Degree` has to become a compile-time value, and the JIT is what makes that
//! a specialization rather than a restriction.

#[cutile::module]
pub mod tile {
  use cutile::core::*;

  /// $y <- alpha x + beta y$. See [`crate::spec::axpby`].
  #[cutile::entry()]
  fn axpby<const S: [i32; 1]>(
    y: &mut Tensor<f64, S>,
    alpha: f64,
    x: &Tensor<f64, { [-1] }>,
    beta: f64,
  ) {
    let ta = alpha.broadcast(y.shape());
    let tb = beta.broadcast(y.shape());
    let tx = load_tile_like(x, y);
    let ty = y.load();
    y.store(ta * tx + tb * ty);
  }

  /// $x <- alpha x$. See [`crate::spec::scale`].
  #[cutile::entry()]
  fn scale<const S: [i32; 1]>(x: &mut Tensor<f64, S>, alpha: f64) {
    let ta = alpha.broadcast(x.shape());
    let tx = x.load();
    x.store(ta * tx);
  }

  /// One partial inner product per tile block. See [`crate::spec::dot_partials`].
  ///
  /// The reduction to a single scalar is finished off the device, over an array
  /// whose length is the block count. That second stage is small, and keeping
  /// it separate is what lets this kernel stay a pure per-tile reduction with a
  /// disjoint output.
  #[cutile::entry()]
  fn dot_partials<const B: i32>(
    out: &mut Tensor<f64, { [1] }>,
    x: &Tensor<f64, { [-1] }>,
    y: &Tensor<f64, { [-1] }>,
  ) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let block = pid.0;

    let shape: Shape<{ [B] }> = const_shape![B];
    let xp: Partition<f64, { [B] }> = x.partition(shape);
    let yp: Partition<f64, { [B] }> = y.partition(shape);

    let tx: Tile<f64, { [B] }> = xp.load([block]);
    let ty: Tile<f64, { [B] }> = yp.load([block]);
    let product: Tile<f64, { [B] }> = tx * ty;

    let sum: Tile<f64, { [] }> = reduce_sum(product, 0i32);
    out.store(sum.reshape(const_shape![1]));
  }

  /// Every element matrix of a tile of cells, as one matrix multiply. See
  /// [`crate::spec::elmat_batch`].
  ///
  /// $M = G A^top$ with $G$ the cells' Gramians and $A$ the constant tensor,
  /// which is small enough to be loaded whole by every tile program. The whole
  /// of assembly, on tensor cores.
  #[cutile::entry()]
  fn elmat_batch<const BM: i32, const NG: i32, const NE: i32>(
    elmats: &mut Tensor<f64, { [BM, NE] }>,
    gramians: &Tensor<f64, { [-1, NG] }>,
    coeff: &Tensor<f64, { [NE, NG] }>,
  ) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let block = pid.0;

    let gp: Partition<f64, { [BM, NG] }> = gramians.partition(const_shape![BM, NG]);
    let cp: Partition<f64, { [NE, NG] }> = coeff.partition(const_shape![NE, NG]);

    let g: Tile<f64, { [BM, NG] }> = gp.load([block, 0i32]);
    let c: Tile<f64, { [NE, NG] }> = cp.load([0i32, 0i32]);
    let ct: Tile<f64, { [NG, NE] }> = c.transpose();

    let zero: f64 = 0.0;
    let acc: Tile<f64, { [BM, NE] }> = broadcast_scalar(zero, const_shape![BM, NE]);
    elmats.store(mma(g, ct, acc));
  }

  /// Each cell's contribution to the matvec. See [`crate::spec::cell_matvec`].
  ///
  /// The element matrix is never formed. Splitting the constant tensor by local
  /// column, $A_j$ being the block of rows $"NR" j$ through $"NR"(j+1)$, gives
  ///
  /// $
  ///   "locals"[c, :] = sum_j (G[c, :] A_j^top) dot "cellx"[c, j],
  /// $
  ///
  /// a short loop of tensor-core products against tiles of cells, accumulated
  /// in registers. The output slot belongs to the cell, so the tile programs
  /// are disjoint and nothing is synchronized.
  #[cutile::entry()]
  fn cell_matvec<const BM: i32, const NG: i32, const NR: i32, const NC: i32>(
    locals: &mut Tensor<f64, { [BM, NR] }>,
    gramians: &Tensor<f64, { [-1, NG] }>,
    cellx: &Tensor<f64, { [-1, NC] }>,
    coeff: &Tensor<f64, { [-1, NG] }>,
  ) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let block = pid.0;

    let gp: Partition<f64, { [BM, NG] }> = gramians.partition(const_shape![BM, NG]);
    let xp: Partition<f64, { [BM, NC] }> = cellx.partition(const_shape![BM, NC]);
    let wp: Partition<f64, { [NR, NG] }> = coeff.partition(const_shape![NR, NG]);

    let g: Tile<f64, { [BM, NG] }> = gp.load([block, 0i32]);
    let xc: Tile<f64, { [BM, NC] }> = xp.load([block, 0i32]);

    let zero: f64 = 0.0;
    let mut acc: Tile<f64, { [BM, NR] }> = broadcast_scalar(zero, const_shape![BM, NR]);

    for j in 0i32..NC {
      let w: Tile<f64, { [NR, NG] }> = wp.load([j, 0i32]);
      let wt: Tile<f64, { [NG, NR] }> = w.transpose();

      let empty: Tile<f64, { [BM, NR] }> = broadcast_scalar(zero, const_shape![BM, NR]);
      let column: Tile<f64, { [BM, NR] }> = mma(g, wt, empty);

      let row0: Tile<i32, { [] }> = scalar_to_tile(0i32);
      let colj: Tile<i32, { [] }> = scalar_to_tile(j);
      let xj: Tile<f64, { [BM, 1] }> = extract(xc, [row0, colj]);
      let xjb: Tile<f64, { [BM, NR] }> = xj.broadcast(const_shape![BM, NR]);

      acc = acc + column * xjb;
    }

    locals.store(acc);
  }

  /// Sum each degree of freedom's fixed-width row of contributions. See
  /// [`crate::spec::segment_reduce`].
  #[cutile::entry()]
  fn segment_reduce<const BM: i32, const W: i32>(
    y: &mut Tensor<f64, { [BM] }>,
    gathered: &Tensor<f64, { [-1, W] }>,
  ) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let block = pid.0;

    let gp: Partition<f64, { [BM, W] }> = gathered.partition(const_shape![BM, W]);
    let g: Tile<f64, { [BM, W] }> = gp.load([block, 0i32]);

    let sum: Tile<f64, { [BM] }> = reduce_sum(g, 1i32);
    y.store(sum);
  }

  /// `out[t] = src[indices[t]]`. See [`crate::spec::gather`].
  ///
  /// The crate's one `unsafe` kernel, and the only place a pointer appears.
  /// An indexed read is not a partition of anything, so it goes through
  /// cuTile's pointer path: broadcast the base pointer over the tile, offset it
  /// by the index tile, load under a bounds mask. The mask is what keeps the
  /// tail of a non-dividing length in range; the value it substitutes never
  /// reaches the output, since those lanes are masked on the store too.
  ///
  /// # Safety
  ///
  /// `src` must be at least as long as the largest entry of `indices` allows,
  /// and `out` at least `len`. Both hold by construction for the buffers
  /// [`crate::CellOperator`] builds: the index tables are inverted
  /// local-to-global maps, whose entries are positions in the arrays they index.
  #[cutile::entry()]
  unsafe fn gather<const B: i32>(
    out_ptr: *mut f64,
    src_ptr: *mut f64,
    idx_ptr: *mut i32,
    len: i32,
  ) {
    let pid: (i32, i32, i32) = get_tile_block_id();
    let start: i32 = pid.0 * B;

    let shape: Shape<{ [B] }> = const_shape![B];
    let offsets: Tile<i32, { [B] }> = iota(shape) + broadcast_scalar(start, shape);
    let len_tile: Tile<i32, { [B] }> = broadcast_scalar(len, shape);
    let mask: Tile<bool, { [B] }> = lt_tile(offsets, len_tile);

    let idx_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(idx_ptr);
    let idx_ptrs: PointerTile<*mut i32, { [B] }> =
      idx_base.reshape(const_shape![1]).broadcast(shape);
    let indices: (Tile<i32, { [B] }>, Token) = load_ptr_tko(
      idx_ptrs.offset_tile(offsets),
      ordering::Weak,
      None::<scope::TileBlock>,
      Some(mask),
      Some(0i32),
      None,
      Latency::<0>,
    );

    let src_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(src_ptr);
    let src_ptrs: PointerTile<*mut f64, { [B] }> =
      src_base.reshape(const_shape![1]).broadcast(shape);
    let values: (Tile<f64, { [B] }>, Token) = load_ptr_tko(
      src_ptrs.offset_tile(indices.0),
      ordering::Weak,
      None::<scope::TileBlock>,
      Some(mask),
      Some(0.0f64),
      None,
      Latency::<0>,
    );

    let out_base: PointerTile<*mut f64, { [] }> = pointer_to_tile(out_ptr);
    let out_ptrs: PointerTile<*mut f64, { [B] }> =
      out_base.reshape(const_shape![1]).broadcast(shape);
    store_ptr_tko(
      out_ptrs.offset_tile(offsets),
      values.0,
      ordering::Weak,
      None::<scope::TileBlock>,
      Some(mask),
      None,
    );
  }
}
