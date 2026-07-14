//! Point location in a coordinate mesh.
//!
//! To evaluate a reconstructed field (a Whitney interpolation of a cochain) at
//! an arbitrary point $x$, one must first find the cell containing $x$ and the
//! barycentric coordinates of $x$ within it -- the geometric half of the FEEC
//! "representation formula". A linear scan over all cells is $O(N)$ per query,
//! which is fatal when sampling a field on a grid.
//!
//! [`PointLocator`] builds a **bounding-volume hierarchy** (BVH) once and then
//! answers containment queries in $O(log N)$: a binary tree of axis-aligned
//! bounding boxes (AABBs), each internal node's box enclosing its children's.
//! A query descends the tree, skipping any subtree whose box excludes $x$, and
//! tests barycentric containment only against the handful of candidate cells
//! in the surviving leaves. Each cell's inverse affine map is cached, so the
//! global-to-barycentric map is a single mat-vec.
//!
//! Point location is inherently a *coordinate* (extrinsic) operation -- it has
//! no meaning without an embedding -- so it lives in the `coord` layer, apart
//! from the intrinsic metric machinery. On an embedded manifold (intrinsic
//! dimension below ambient, e.g. a surface in space) a query point is accepted
//! only if it also lies within tolerance of the cell's affine hull.

use super::{simplex::SimplexRefExt, Coord, CoordRef};
use crate::{
  point::{local2bary, MeshPoint},
  topology::{complex::Complex, handle::SimplexIdx},
  Dim,
};

use common::linalg::nalgebra::{Matrix, Vector};

/// An axis-aligned bounding box in $RR^d$.
#[derive(Debug, Clone)]
struct Aabb {
  min: Vector,
  max: Vector,
}
impl Aabb {
  fn empty(dim: Dim) -> Self {
    Self {
      min: Vector::from_element(dim, f64::INFINITY),
      max: Vector::from_element(dim, f64::NEG_INFINITY),
    }
  }
  fn union(&self, other: &Aabb) -> Aabb {
    Aabb {
      min: self.min.inf(&other.min),
      max: self.max.sup(&other.max),
    }
  }
  fn extend_point(&mut self, p: CoordRef) {
    self.min = self.min.inf(&p.clone_owned());
    self.max = self.max.sup(&p.clone_owned());
  }
  /// Whether $x$ lies in the box, grown by an absolute tolerance.
  fn contains(&self, x: CoordRef, eps: f64) -> bool {
    (0..self.min.len()).all(|i| x[i] >= self.min[i] - eps && x[i] <= self.max[i] + eps)
  }
  fn diagonal(&self) -> f64 {
    (&self.max - &self.min).norm()
  }
}

/// A node of the BVH, stored in a flat post-order array (the root is last).
enum BvhNode {
  /// A leaf covering `order[start..start + count]`.
  Leaf {
    bbox: Aabb,
    start: usize,
    count: usize,
  },
  /// An internal node with two children by node index.
  Internal {
    bbox: Aabb,
    left: usize,
    right: usize,
  },
}
impl BvhNode {
  fn bbox(&self) -> &Aabb {
    match self {
      BvhNode::Leaf { bbox, .. } | BvhNode::Internal { bbox, .. } => bbox,
    }
  }
}

/// The cached affine data of one cell: the barycentric coordinates of a global
/// point are $lambda(x) = "local2bary"(A^+ (x - v_0))$, and the residual
/// $norm(x - v_0 - A A^+ (x - v_0))$ is its distance from the cell's affine
/// hull (nonzero only on embedded manifolds).
struct CellAffine {
  kidx: usize,
  base: Coord,
  /// The spanning vectors $A$ (ambient x intrinsic).
  spanning: Matrix,
  /// The pseudo-inverse $A^+$ (intrinsic x ambient).
  inv: Matrix,
}

/// A bounding-volume hierarchy over the cells of a coordinate mesh, answering
/// point-containment queries in logarithmic time. Build once, query many times.
pub struct PointLocator {
  cell_dim: Dim,
  embedded: bool,
  cells: Vec<CellAffine>,
  /// Permutation of `0..cells.len()` grouping cells into BVH leaves.
  order: Vec<usize>,
  nodes: Vec<BvhNode>,
  root: usize,
  eps: f64,
}

const LEAF_SIZE: usize = 4;

impl PointLocator {
  /// Build the hierarchy over the cells of `topology` with the geometry
  /// `coords`.
  pub fn new(topology: &Complex, coords: &super::mesh::MeshCoords) -> Self {
    let cell_dim = topology.dim();
    let ambient = coords.dim();
    let embedded = cell_dim < ambient;

    let mut cells = Vec::with_capacity(topology.nsimplices(cell_dim));
    let mut boxes = Vec::with_capacity(cells.capacity());
    let mut centroids = Vec::with_capacity(cells.capacity());

    let mut total = Aabb::empty(ambient);
    for cell in topology.cells().handle_iter() {
      let simp = cell.coord_simplex(coords);
      let mut bbox = Aabb::empty(ambient);
      for v in simp.coord_iter() {
        bbox.extend_point(v);
      }
      total = total.union(&bbox);
      centroids.push(0.5 * (&bbox.min + &bbox.max));
      boxes.push(bbox);
      cells.push(CellAffine {
        kidx: cell.kidx(),
        base: simp.base_vertex().into_owned(),
        spanning: simp.linear_transform(),
        inv: simp.inv_linear_transform(),
      });
    }

    let eps = 1e-9 * total.diagonal().max(1.0);

    let mut order: Vec<usize> = (0..cells.len()).collect();
    let mut nodes = Vec::new();
    let root = if cells.is_empty() {
      nodes.push(BvhNode::Leaf {
        bbox: Aabb::empty(ambient),
        start: 0,
        count: 0,
      });
      0
    } else {
      build(&mut nodes, &mut order, 0, &boxes, &centroids)
    };

    Self {
      cell_dim,
      embedded,
      cells,
      order,
      nodes,
      root,
      eps,
    }
  }

  /// The point of the manifold at the global coordinate `x`: the containing
  /// cell and the barycentric coordinates within it. `None` if `x` lies
  /// outside the mesh.
  ///
  /// This is the inverse of the embedding, and the bridge from a coordinate
  /// domain into the intrinsic chart the fields actually live on.
  pub fn locate<'a>(&self, x: impl Into<CoordRef<'a>>) -> Option<MeshPoint> {
    let x = x.into();
    if self.cells.is_empty() {
      return None;
    }

    let mut stack = vec![self.root];
    while let Some(node) = stack.pop() {
      let node = &self.nodes[node];
      if !node.bbox().contains(x, self.eps) {
        continue;
      }
      match node {
        BvhNode::Leaf { start, count, .. } => {
          for &prim in &self.order[*start..*start + *count] {
            if let Some(bary) = self.try_barycentric(prim, x) {
              let cell = SimplexIdx::new(self.cell_dim, self.cells[prim].kidx);
              return Some(MeshPoint::new(cell, bary));
            }
          }
        }
        BvhNode::Internal { left, right, .. } => {
          stack.push(*left);
          stack.push(*right);
        }
      }
    }
    None
  }

  /// The barycentric coordinates of `x` in cell `prim`, if `x` lies inside it
  /// (and, on an embedded manifold, within tolerance of its affine hull).
  fn try_barycentric(&self, prim: usize, x: CoordRef) -> Option<Coord> {
    let cell = &self.cells[prim];
    let offset = x - &cell.base;
    let local = &cell.inv * &offset;

    if self.embedded {
      let residual = (&offset - &cell.spanning * &local).norm();
      if residual > self.eps {
        return None;
      }
    }

    let bary = local2bary(&local);
    let inside = bary.iter().all(|&b| b >= -self.eps && b <= 1.0 + self.eps);
    inside.then_some(bary)
  }
}

/// Build a subtree over `order[lo_slice]` (a mutable permutation slice with
/// absolute start `offset`), returning its node index. Post-order: children
/// are pushed before their parent, so the root is the last node.
fn build(
  nodes: &mut Vec<BvhNode>,
  order_slice: &mut [usize],
  offset: usize,
  boxes: &[Aabb],
  centroids: &[Coord],
) -> usize {
  let bbox = order_slice
    .iter()
    .map(|&i| boxes[i].clone())
    .reduce(|a, b| a.union(&b))
    .unwrap();

  if order_slice.len() <= LEAF_SIZE {
    nodes.push(BvhNode::Leaf {
      bbox,
      start: offset,
      count: order_slice.len(),
    });
    return nodes.len() - 1;
  }

  // Split along the longest axis of the centroid bounds.
  let mut cbounds = Aabb::empty(centroids[order_slice[0]].len());
  for &i in order_slice.iter() {
    cbounds.extend_point(centroids[i].as_view());
  }
  let axis = (0..cbounds.min.len())
    .max_by(|&a, &b| {
      let ext = &cbounds.max - &cbounds.min;
      ext[a].partial_cmp(&ext[b]).unwrap()
    })
    .unwrap();

  // Degenerate centroid spread: make a leaf rather than recurse forever.
  if (cbounds.max[axis] - cbounds.min[axis]) <= f64::EPSILON {
    nodes.push(BvhNode::Leaf {
      bbox,
      start: offset,
      count: order_slice.len(),
    });
    return nodes.len() - 1;
  }

  order_slice
    .sort_unstable_by(|&a, &b| centroids[a][axis].partial_cmp(&centroids[b][axis]).unwrap());
  let mid = order_slice.len() / 2;
  let (left_slice, right_slice) = order_slice.split_at_mut(mid);
  let left = build(nodes, left_slice, offset, boxes, centroids);
  let right = build(nodes, right_slice, offset + mid, boxes, centroids);

  nodes.push(BvhNode::Internal { bbox, left, right });
  nodes.len() - 1
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::gen::cartesian::CartesianMeshInfo;

  /// The BVH locator agrees with the brute-force linear scan on every query,
  /// and reports points outside the mesh as such.
  #[test]
  fn locator_matches_brute_force() {
    for dim in 1..=3 {
      let (topology, coords) = CartesianMeshInfo::new_unit(dim, 3).compute_coord_complex();
      let locator = PointLocator::new(&topology, &coords);

      // A grid of probe points covering the unit cube and a margin outside it.
      // A distinct per-axis phase keeps the points off cell faces and off the
      // triangulation diagonals, where the strict and tolerant tests would
      // legitimately disagree on measure-zero boundary sets.
      let samples: usize = 5;
      let phase = [0.041, 0.113, 0.237];
      let probe = |i: usize, d: usize| -0.2 + 1.4 * (i as f64 + phase[d]) / samples as f64;
      for flat in 0..samples.pow(dim as u32) {
        let x = Coord::from_iterator(
          dim,
          (0..dim).map(|d| probe(flat / samples.pow(d as u32) % samples, d)),
        );

        let brute = coords.find_cell_containing(&topology, x.as_view());
        let found = locator.locate(&x);

        match (brute, &found) {
          (Some(_), Some(loc)) => {
            // The located cell must actually contain x, and its barycentric
            // coordinates must reconstruct the point.
            let simp = loc.chart(&topology).coord_simplex(&coords);
            assert!(simp.is_global_inside(x.as_view()), "dim={dim} x={x:?}");
            let reconstructed = simp.bary2global(loc.bary.as_view());
            assert!((reconstructed - &x).norm() < 1e-9);
          }
          (None, None) => {}
          (a, b) => panic!(
            "disagreement at x={x:?}: brute={} bvh={}",
            a.is_some(),
            b.is_some()
          ),
        }
      }
    }
  }
}
