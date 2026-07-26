//! Device-resident vectors and operators: the solve that never comes back to
//! the host.
//!
//! _DISCLAIMER: This module was written without access to a CUDA machine and
//! has never been compiled. The mathematics it drives is tested (see
//! [`crate::spec`] and [`crate::CellOperator`]); the host-side cuTile API calls
//! here are transcribed from the upstream examples and are expected to need
//! correction on first build._
//!
//! # Why the vector has to live here
//!
//! A Krylov method's iterates are read and written every step. Copying them
//! across the bus each time would cost more than the matvec: at ten million
//! degrees of freedom a single vector is eighty megabytes, which is milliseconds
//! over PCIe against microseconds of kernel. So the vector stays on the device
//! for the whole solve, and only the scalars a stopping criterion needs come
//! back.
//!
//! That is what [`iterative::InnerProductSpace`] is for. The Krylov methods ask
//! for five operations and nothing else, so implementing them here puts
//! `cg` and `minres` on the device unchanged --- no GPU variant of either
//! method exists, and none should.

use crate::{CellOperator, spec::Shapes};

use iterative::InnerProductSpace;

use cuda_core::Device;
use cutile::error::Error;
use cutile::tensor::{IntoPartition, Tensor, ToHostVec};

use std::sync::Arc;

/// The stream and device a solve runs on, shared by every vector and operator
/// taking part in it.
///
/// Held by [`Arc`] because a vector must outlive the expression that made it
/// and there is no useful ordering among them.
#[derive(Clone)]
pub struct Context {
  device: Arc<Device>,
  stream: Arc<cuda_core::Stream>,
}

impl Context {
  /// Open a device by ordinal and create a stream on it.
  pub fn new(ordinal: usize) -> Result<Self, Error> {
    let device = Device::new(ordinal)?;
    let stream = device.new_stream()?;
    Ok(Self {
      device: Arc::new(device),
      stream: Arc::new(stream),
    })
  }

  pub fn stream(&self) -> &cuda_core::Stream {
    &self.stream
  }

  pub fn device(&self) -> &Device {
    &self.device
  }
}

/// A vector living in device memory for the duration of a solve.
///
/// The [`InnerProductSpace`] implementation is what a Krylov method sees, and
/// it is the whole interface: no entry is ever indexed, so nothing here has to
/// support one.
///
/// [`Self::dot`] is the only operation that has to produce a host value. It
/// reduces to one partial sum per tile block on the device and finishes the sum
/// on the host, which is a synchronization point per inner product, two per
/// conjugate-gradient step. Eliminating it needs a device-side second stage and
/// a residual test that does not read the scalar, and is the obvious next
/// optimization rather than a correctness matter.
pub struct DeviceVector {
  context: Context,
  data: Option<Arc<Tensor<f64>>>,
  dim: usize,
}

/// The tile width every elementwise kernel is launched at.
///
/// One value, because these kernels are bandwidth-bound and the width only has
/// to be large enough to saturate a memory transaction.
const BLOCK: usize = 256;

impl DeviceVector {
  /// Upload a host slice.
  pub fn upload(context: &Context, host: &[f64]) -> Result<Self, Error> {
    let data = cutile::api::from_slice(host).sync_on(context.stream())?;
    Ok(Self {
      context: context.clone(),
      data: Some(Arc::new(data)),
      dim: host.len(),
    })
  }

  /// Read the vector back, which a solve does once, at the end.
  pub fn download(&self) -> Result<Vec<f64>, Error> {
    self
      .tensor()
      .to_host_vec()
      .sync_on(self.context.stream())
      .map_err(Into::into)
  }

  fn tensor(&self) -> &Arc<Tensor<f64>> {
    self.data.as_ref().expect("vector was left moved out")
  }

  /// Take the tensor out for a launch that consumes it, leaving the vector
  /// momentarily empty.
  ///
  /// cuTile launches move their mutable argument in and hand it back, which is
  /// how the ownership contract survives the launch boundary. An in-place
  /// operation on the host side therefore has to move out and move back, and
  /// the window in between is what this documents.
  fn take(&mut self) -> Arc<Tensor<f64>> {
    self.data.take().expect("vector was left moved out")
  }
}

impl Clone for DeviceVector {
  fn clone(&self) -> Self {
    let data = self
      .tensor()
      .dup()
      .sync_on(self.context.stream())
      .expect("device allocation");
    Self {
      context: self.context.clone(),
      data: Some(Arc::new(data)),
      dim: self.dim,
    }
  }
}

impl InnerProductSpace for DeviceVector {
  /// Unimplementable as stated: a zero vector needs a device to live on, and
  /// the trait's constructor takes only a dimension.
  ///
  /// This is a real limitation of the current interface, not an oversight here.
  /// The fix is for the trait to carry an allocation context, which the host
  /// vector would instantiate at the unit type; until then a device solve must
  /// be entered through a wrapper that has one.
  fn zeros(_dim: usize) -> Self {
    unimplemented!("a device vector needs a context; see the type's documentation")
  }

  fn dim(&self) -> usize {
    self.dim
  }

  fn dot(&self, other: &Self) -> f64 {
    let nblocks = self.dim.div_ceil(BLOCK);
    let partials = cutile::api::zeros::<f64>(&[nblocks])
      .sync_on(self.context.stream())
      .expect("device allocation")
      .partition([1]);

    let (partials, _, _) =
      crate::kernels::tile::dot_partials(partials, self.tensor().clone(), other.tensor().clone())
        .generics(vec![BLOCK.to_string()])
        .sync_on(self.context.stream())
        .expect("dot_partials launch");

    let partials: Vec<f64> = partials
      .unpartition()
      .to_host_vec()
      .sync_on(self.context.stream())
      .expect("partial download");
    partials.into_iter().sum()
  }

  fn axpby(&mut self, alpha: f64, x: &Self, beta: f64) {
    let y = self.take().partition([BLOCK]);
    let (y, _) = crate::kernels::tile::axpby(y, alpha, x.tensor().clone(), beta)
      .generics(vec![BLOCK.to_string()])
      .sync_on(self.context.stream())
      .expect("axpby launch");
    self.data = Some(Arc::new(y.unpartition()));
  }

  fn scale(&mut self, alpha: f64) {
    let x = self.take().partition([BLOCK]);
    let x = crate::kernels::tile::scale(x, alpha)
      .generics(vec![BLOCK.to_string()])
      .sync_on(self.context.stream())
      .expect("scale launch");
    self.data = Some(Arc::new(x.unpartition()));
  }
}

/// A [`CellOperator`] with its buffers resident on the device.
///
/// Uploaded once. An apply is four launches --- gather, element matvec, gather,
/// reduce --- and touches no host memory at all, which is what lets a whole
/// Krylov solve run without a transfer except the inner products.
pub struct DeviceOperator {
  context: Context,
  shapes: Shapes,
  width: usize,
  ndofs_row: usize,
  coeff: Arc<Tensor<f64>>,
  gramians: Arc<Tensor<f64>>,
  col_dofs: Arc<Tensor<i32>>,
  gather_table: Arc<Tensor<i32>>,
}

impl DeviceOperator {
  /// Upload every buffer the kernels read.
  ///
  /// The index maps narrow to `i32` because that is what a tile program indexes
  /// with; a mesh large enough to overflow it would exceed device memory long
  /// before, but the conversion is checked rather than assumed.
  pub fn upload(context: &Context, op: &CellOperator) -> Result<Self, Error> {
    let to_i32 = |slice: &[u32]| -> Vec<i32> {
      slice
        .iter()
        .map(|&i| i32::try_from(i).expect("index fits in i32"))
        .collect()
    };

    let stream = context.stream();
    Ok(Self {
      context: context.clone(),
      shapes: op.shapes(),
      width: op.padded().width(),
      ndofs_row: op.ndofs_row(),
      coeff: Arc::new(cutile::api::from_slice(op.coeff()).sync_on(stream)?),
      gramians: Arc::new(cutile::api::from_slice(op.gramians()).sync_on(stream)?),
      col_dofs: Arc::new(cutile::api::from_slice(&to_i32(op.col_dofs())).sync_on(stream)?),
      gather_table: Arc::new(
        cutile::api::from_slice(&to_i32(op.padded().table())).sync_on(stream)?,
      ),
    })
  }

  pub fn shapes(&self) -> Shapes {
    self.shapes
  }
}

impl iterative::LinearOperator for DeviceOperator {
  type Space = DeviceVector;

  fn dim(&self) -> usize {
    self.ndofs_row
  }

  fn apply(&self, _x: &DeviceVector) -> DeviceVector {
    todo!(
      "the four launches: gather -> cell_matvec -> gather -> segment_reduce. \
       Their arithmetic is CellOperator::apply_rect, which is tested; what is \
       missing is the launch plumbing, and writing it blind would only be \
       guessing at the host API."
    )
  }
}
