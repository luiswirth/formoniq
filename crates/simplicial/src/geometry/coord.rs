//! The extrinsic layer: a coordinate realization of the mesh.
//!
//! An embedding is *one* geometry among several (invariant 2), and everything
//! here is downstream of it: vertex coordinates, the affine parametrization of a
//! cell, point location. A manifold given by Regge edge lengths has none of it,
//! and the intrinsic geometry (invariant 2) must never ask for it.
//!
//! Ambient coordinates are therefore kept apart, by type, from the
//! [`Bary`](crate::atlas::Bary) and [`Local`](crate::atlas::Local) coordinates
//! of a chart: those are intrinsic and exist on every geometry, and the maps
//! between the two worlds are the parametrizations in
//! [`simplex`].

pub mod locate;
pub mod mesh;
pub mod simplex;

pub use coorder::{Ambient, Coord, CoordRef};

use crate::linalg::{RowVector, RowVectorView, Vector, VectorView};

use self::mesh::MeshCoords;
use super::cell_volume;
use crate::topology::complex::Complex;

pub type TangentVector = Vector;
pub type TangentVectorRef<'a> = VectorView<'a>;

pub type CoTangentVector = RowVector;
pub type CoTangentVectorRef<'a> = RowVectorView<'a>;

/// The cotangent of the angle between two vectors, $cot theta = (u dot w) \/
/// |u times w|$, read off their dot product and the (dimension-agnostic) Gram
/// identity $|u times w|^2 = |u|^2 |w|^2 - (u dot w)^2$ rather than an actual
/// cross product -- so this works in an embedding of any dimension, not just
/// $RR^3$. The `max(0.0)` guards roundoff at a near-degenerate (collinear)
/// corner, where the identity's right-hand side can dip fractionally below
/// zero.
fn cot_angle(u: &Vector, w: &Vector) -> f64 {
  let dot = u.dot(w);
  let cross_sq = u.norm_squared() * w.norm_squared() - dot * dot;
  dot / cross_sq.max(0.0).sqrt()
}

/// Discrete mean curvature magnitude $|H(v)|$ at every vertex of an embedded
/// 2-dimensional simplicial surface, via the cotangent-Laplacian
/// mean-curvature-normal identity (Meyer, Desbrun, Schröder, Barr 2003)
/// $Delta_(L B) x (v) = (1)/(2A(v)) sum_(j in N(v)) (cot alpha_(v j) + cot
/// beta_(v j)) (x_j - x_v) = -2H(v) n(v)$: applying the Laplace-Beltrami
/// operator to the embedding's own coordinate function returns the mean
/// curvature normal directly, so no curvature-specific estimator is needed
/// beyond the mesh's own cotangent weights. Only the magnitude is returned --
/// the sign of $H$ needs an outward-normal convention this function has no
/// use for, since [`vertex_curvature_radius`] consumes it through $H^2$
/// alone.
///
/// Extrinsic, unlike [`super::vertex_gaussian_curvature`]: the cotangent
/// weight is read off the embedded edge vectors, not the metric alone, so
/// this lives on the `coord` side and needs an actual [`MeshCoords`], not
/// just the intrinsic edge lengths. $A(v)$ is the same barycentric lumped area
/// [`super::vertex_gaussian_curvature`] uses, so the two combine into
/// consistent principal curvatures in [`vertex_curvature_radius`].
pub fn vertex_mean_curvature(topology: &Complex, coords: &MeshCoords) -> Vec<f64> {
  assert_eq!(
    topology.dim(),
    2,
    "Mean curvature is a 2D-surface quantity."
  );
  let nvertices = topology.skeleton_raw(0).len();

  let mut areas = vec![0.0; nvertices];
  let mut laplacian = vec![Vector::zeros(coords.dim()); nvertices];
  for cell in topology.cells().handle_iter() {
    let verts = &cell.simplex().vertices;
    let p = [
      coords.coord(verts[0]),
      coords.coord(verts[1]),
      coords.coord(verts[2]),
    ];
    let vol = cell_volume(&coords.cell_metric(cell));
    for &v in verts {
      areas[v] += vol / 3.0;
    }
    for apex in 0..3 {
      let (o1, o2) = ((apex + 1) % 3, (apex + 2) % 3);
      let u = p[o1] - p[apex];
      let w = p[o2] - p[apex];
      let cot = cot_angle(&u, &w);
      let weighted = (p[o1] - p[o2]) * cot;
      laplacian[verts[o1]] += &weighted;
      laplacian[verts[o2]] -= &weighted;
    }
  }

  laplacian
    .into_iter()
    .zip(areas)
    .map(|(l, area)| l.norm() / (4.0 * area))
    .collect()
}

/// A vertex's local radius of curvature: $1 \/ max(|kappa_1|, |kappa_2|)$,
/// the distance along the normal at which the offset map $x |-> x + t n(x)$
/// first develops a fold -- a focal point of the normal congruence, where its
/// differential $I - t S$ ($S$ the shape operator) degenerates. This is the
/// principled bound on how far a surface may be displaced along its normal
/// before self-intersecting: exactly "as far as the surface is locally big"
/// at that point.
///
/// Recovered from the two curvature invariants via $kappa_(1,2) = H
/// plus.minus sqrt(H^2 - K)$, so $max_i |kappa_i| = |H| + sqrt(max(H^2 - K,
/// 0))$; the clamp guards the near-umbilic vertex where discretization noise
/// can push $H^2$ fractionally below $K$ though the true value is $0$.
///
/// Infinite at a flat vertex ($H = K = 0$) and, unconditionally, at a
/// boundary vertex: both [`super::vertex_gaussian_curvature`] and
/// [`vertex_mean_curvature`] are natural (Neumann) boundary quantities there,
/// not curvature -- provably so, since the coordinate function is exactly
/// linear yet $integral_diff.Omega phi_i thin n_x thin d s != 0$ for a
/// boundary test function even on a flat domain. Using either at the rim
/// would clamp displacement near a flat edge for no geometric reason, so a
/// caller relies on an independent upper bound there (e.g. the mesh's own
/// coordinate extent) rather than this curvature-based one.
pub fn vertex_curvature_radius(topology: &Complex, coords: &MeshCoords) -> Vec<f64> {
  let gauss = super::vertex_gaussian_curvature(topology, &coords.to_edge_lengths_sq(topology));
  let mean = vertex_mean_curvature(topology, coords);
  let boundary: std::collections::HashSet<usize> =
    topology.boundary_vertices().into_iter().collect();
  (0..gauss.len())
    .map(|v| {
      if boundary.contains(&v) {
        return f64::INFINITY;
      }
      let (k, h) = (gauss[v], mean[v]);
      let kappa_max = h + (h * h - k).max(0.0).sqrt();
      kappa_max.recip()
    })
    .collect()
}

/// The **reach** of the embedded surface at every vertex: the largest $r$ such
/// that the normal offset by any $|d| <= r$ is still an embedding, so no fold
/// and no self-intersection.
///
/// Federer's reach, and it has two halves. The *local* one is the curvature
/// radius ([`vertex_curvature_radius`]): offset past the focal point and the
/// surface folds through itself. The *non-local* one is the bottleneck -- how
/// far the surface is from a different sheet of itself, which curvature cannot
/// see at all. A thin flat plate has infinite curvature radius and reach
/// $t \/ 2$: its two faces meet in the middle however flat they are. Bounding a
/// normal displacement by curvature alone is therefore only half a bound, and
/// it is exactly the half that thin features are not covered by.
///
/// Both halves come out of one quantity, the tangent-ball radius. The inner
/// medial ball at $p$ is tangent there and empty of surface, so its center
/// $c = p - r n$ satisfies $|q - c| >= r$ for every surface point $q$, which is
///
/// $r <= (|q - p|^2)/(-2 n dot (q - p))$ for every $q$ with $n dot (q - p) < 0$,
///
/// and the mirrored statement with $-n$ bounds the outer ball. The minimum over
/// both is the distance to the medial axis -- the local feature size -- and it
/// degenerates to the normal curvature radius as $q -> p$, so it subsumes the
/// local half rather than sitting beside it. On a sphere of radius $R$ every
/// $q$ returns exactly $R$; on the thin plate the opposite face returns
/// $t \/ 2$.
///
/// Since $r >= |q - p| \/ 2$, a $q$ farther than twice the running best cannot
/// improve it, which bounds the search to a ball that shrinks as the estimate
/// does: a thin feature is found immediately and terminates the walk early. The
/// initial bound is the curvature radius, itself capped by `max_reach` (pass
/// the object's own extent), because a reach exceeding the object is not a
/// meaningful cap and an uncapped one would make a flat mesh scan globally.
///
/// Defined for an embedded *surface* whose normal field exists: `INFINITY`
/// (no bound) at every vertex of a complex that is not 2-dimensional or not
/// orientable. That is not a dodge -- a non-orientable surface has no
/// continuous normal field, so "displace along the normal" has no meaning to
/// bound in the first place.
pub fn vertex_reach(topology: &Complex, coords: &MeshCoords, max_reach: f64) -> Vec<f64> {
  use rayon::prelude::*;

  let nvertices = topology.nsimplices(0);
  let unbounded = vec![f64::INFINITY; nvertices];
  if topology.dim() != 2 || !topology.is_orientable() {
    return unbounded;
  }

  let points: Vec<Vector3> = (0..nvertices).map(|v| embed3(coords, v)).collect();
  let Some(normals) = oriented_vertex_normals(topology, &points) else {
    return unbounded;
  };

  // A grid sized by the mean edge length: fine enough that a shell is a thin
  // layer, coarse enough that a cell holds a few vertices.
  let mut edge_total = 0.0;
  let mut edge_count = 0usize;
  for edge in topology.edges().handle_iter() {
    let vertices = &edge.simplex().vertices;
    edge_total += (points[vertices[1]] - points[vertices[0]]).norm();
    edge_count += 1;
  }
  let spacing = if edge_count > 0 {
    (edge_total / edge_count as f64).max(1e-12)
  } else {
    return unbounded;
  };
  let grid = PointGrid::new(&points, spacing);

  let curvature = vertex_curvature_radius(topology, coords);

  (0..nvertices)
    .into_par_iter()
    .map(|v| {
      let (p, n) = (points[v], normals[v]);
      let mut best = curvature[v].min(max_reach);
      // Shell by shell, stopping once even the nearest point of the next shell
      // is farther than twice the running bound.
      for shell in 0.. {
        if (shell as f64) * spacing > 2.0 * best {
          break;
        }
        let mut any = false;
        grid.for_each_in_shell(p, shell, |w| {
          any = true;
          if w == v {
            return;
          }
          let delta = points[w] - p;
          let along = n.dot(&delta);
          if along.abs() <= 1e-15 {
            return;
          }
          // Whichever side `w` lies on bounds that side's medial ball; the
          // reach is the smaller, since the displacement swings both ways.
          let radius = delta.norm_squared() / (2.0 * along.abs());
          best = best.min(radius);
        });
        if !any && (shell as f64) * spacing > grid.diagonal() {
          break;
        }
      }
      best
    })
    .collect()
}

type Vector3 = na::Vector3<f64>;

fn embed3(coords: &MeshCoords, vertex: usize) -> Vector3 {
  let c = coords.coord(vertex);
  Vector3::new(
    c.get(0).copied().unwrap_or(0.0),
    c.get(1).copied().unwrap_or(0.0),
    c.get(2).copied().unwrap_or(0.0),
  )
}

/// Area-weighted vertex normals of an orientable embedded surface, each cell
/// wound by the complex's coherent orientation so the 1-ring's face normals
/// agree instead of cancelling. `None` if the surface is not orientable, where
/// no such field exists. The global sign is the orientation's own and does not
/// matter to [`vertex_reach`], which minimizes over both sides.
fn oriented_vertex_normals(topology: &Complex, points: &[Vector3]) -> Option<Vec<Vector3>> {
  let orientation = topology.orientation()?;
  let mut normals = vec![Vector3::zeros(); points.len()];
  for cell in topology.cells().handle_iter() {
    let v = &cell.simplex().vertices;
    let (a, b, c) = (points[v[0]], points[v[1]], points[v[2]]);
    // Twice the area times the unit normal, so the sum is area-weighted with
    // no separate normalization per face.
    let face = (b - a).cross(&(c - a)) * orientation.sign(cell).as_f64();
    for &i in v {
      normals[i] += face;
    }
  }
  for normal in &mut normals {
    let length = normal.norm();
    if length > 1e-15 {
      *normal /= length;
    }
  }
  Some(normals)
}

/// A uniform bucket grid over the points, for the shell walk in
/// [`vertex_reach`]. Deliberately not a k-d tree: the query is "every point in
/// this shell", the shells are visited in order and abandoned early, and a
/// hashed uniform grid answers that in constant time per cell.
struct PointGrid {
  cells: std::collections::HashMap<[i32; 3], Vec<usize>>,
  spacing: f64,
  diagonal: f64,
}

impl PointGrid {
  fn new(points: &[Vector3], spacing: f64) -> Self {
    let mut cells: std::collections::HashMap<[i32; 3], Vec<usize>> =
      std::collections::HashMap::new();
    let (mut lo, mut hi) = (
      Vector3::repeat(f64::INFINITY),
      Vector3::repeat(f64::NEG_INFINITY),
    );
    for (i, p) in points.iter().enumerate() {
      cells.entry(Self::key(p, spacing)).or_default().push(i);
      lo = lo.inf(p);
      hi = hi.sup(p);
    }
    let diagonal = if points.is_empty() {
      0.0
    } else {
      (hi - lo).norm()
    };
    Self {
      cells,
      spacing,
      diagonal,
    }
  }

  fn key(p: &Vector3, spacing: f64) -> [i32; 3] {
    [
      (p[0] / spacing).floor() as i32,
      (p[1] / spacing).floor() as i32,
      (p[2] / spacing).floor() as i32,
    ]
  }

  fn diagonal(&self) -> f64 {
    self.diagonal
  }

  /// Every point in the cube shell at Chebyshev radius `shell` around `p`'s
  /// own cell. Shell 0 is that cell alone.
  fn for_each_in_shell(&self, p: Vector3, shell: i32, mut f: impl FnMut(usize)) {
    let center = Self::key(&p, self.spacing);
    let mut visit = |key: [i32; 3]| {
      if let Some(bucket) = self.cells.get(&key) {
        for &i in bucket {
          f(i);
        }
      }
    };
    if shell == 0 {
      visit(center);
      return;
    }
    for dx in -shell..=shell {
      for dy in -shell..=shell {
        for dz in -shell..=shell {
          // The shell's surface only: the interior was walked already.
          if dx.abs() != shell && dy.abs() != shell && dz.abs() != shell {
            continue;
          }
          visit([center[0] + dx, center[1] + dy, center[2] + dz]);
        }
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The unit sphere has constant curvature $K = H^2 = 1$ and curvature
  /// radius $1$ everywhere. The discrete estimators recover $|H|$ to within
  /// the barycentric lumped area's discretization error (cruder than a mixed
  /// Voronoi area, but simpler and reused as-is for [`vertex_curvature_radius`]).
  /// That same area error enters $kappa_max = |H| + sqrt(max(H^2-K,0))$
  /// asymmetrically -- squared through $H^2$, linear through $K$ -- so the
  /// radius estimate carries a larger, but *conservative* (radius
  /// underestimated, never overestimated), bias than $H$ alone. Underestimating
  /// the safe radius is exactly the safe direction for a fold-safety cap, so
  /// this is loose on purpose, not a correctness gap; the exact Gauss-Bonnet
  /// identity elsewhere is what checks correctness.
  /// On the unit sphere the reach is the radius, and it is the *curvature*
  /// half that says so: the medial axis is the center point. The tangent-ball
  /// formula returns exactly $R$ for every pair on a sphere, so this also
  /// checks the estimator against its one closed form.
  #[test]
  fn sphere_reach_is_its_radius() {
    let (topology, coords) = crate::gen::sphere::mesh_sphere_surface(3);
    let reach = vertex_reach(&topology, &coords, 10.0);
    for &r in &reach {
      assert!(r > 0.5 && r < 1.05, "expected reach ~ 1, got {r}");
    }
  }

  /// The half curvature cannot see. A thin flat slab has *infinite* curvature
  /// radius on its faces -- they are planes -- and reach $t \/ 2$, because the
  /// opposite face is what the offset runs into. This is the case that
  /// collapses a mesh when a displacement is bounded by curvature alone, and
  /// the assertion is that the bound now comes from the thickness rather than
  /// from the (absent) curvature.
  #[test]
  fn thin_slab_reach_is_half_its_thickness() {
    for &thickness in &[0.2, 0.05] {
      let (topology, coords) = slab(thickness);
      let curvature = vertex_curvature_radius(&topology, &coords);
      let reach = vertex_reach(&topology, &coords, 10.0);

      // The interior of a face is flat, so curvature alone would not bound it.
      let flat = curvature
        .iter()
        .filter(|r| r.is_infinite() || **r > 1.0)
        .count();
      assert!(flat > 0, "the slab's faces must be curvature-unbounded");

      let smallest = reach.iter().cloned().fold(f64::INFINITY, f64::min);
      let expected = thickness / 2.0;
      assert!(
        (smallest - expected).abs() < 0.2 * expected,
        "thickness {thickness}: expected reach ~ {expected}, got {smallest}"
      );
    }
  }

  /// A closed slab of the given thickness in $z$, triangulated on a coarse
  /// grid: two parallel faces plus the four sides, wound as one closed surface.
  fn slab(thickness: f64) -> (Complex, MeshCoords) {
    use crate::topology::{simplex::Simplex, skeleton::Skeleton};
    let n = 6;
    let half = thickness / 2.0;
    let mut points: Vec<Vector> = Vec::new();
    let index = |i: usize, j: usize, top: usize| top * (n + 1) * (n + 1) + j * (n + 1) + i;
    for top in 0..2 {
      let z = if top == 0 { -half } else { half };
      for j in 0..=n {
        for i in 0..=n {
          points.push(Vector::from_vec(vec![
            i as f64 / n as f64,
            j as f64 / n as f64,
            z,
          ]));
        }
      }
    }
    let mut quads: Vec<[usize; 4]> = Vec::new();
    for top in 0..2 {
      for j in 0..n {
        for i in 0..n {
          quads.push([
            index(i, j, top),
            index(i + 1, j, top),
            index(i + 1, j + 1, top),
            index(i, j + 1, top),
          ]);
        }
      }
    }
    // The four sides, closing the surface so it bounds a solid.
    for k in 0..n {
      quads.push([
        index(k, 0, 0),
        index(k + 1, 0, 0),
        index(k + 1, 0, 1),
        index(k, 0, 1),
      ]);
      quads.push([
        index(k, n, 0),
        index(k + 1, n, 0),
        index(k + 1, n, 1),
        index(k, n, 1),
      ]);
      quads.push([
        index(0, k, 0),
        index(0, k + 1, 0),
        index(0, k + 1, 1),
        index(0, k, 1),
      ]);
      quads.push([
        index(n, k, 0),
        index(n, k + 1, 0),
        index(n, k + 1, 1),
        index(n, k, 1),
      ]);
    }
    let cells = quads
      .into_iter()
      .flat_map(|q| {
        [
          Simplex::from_word(vec![q[0], q[1], q[2]]).1,
          Simplex::from_word(vec![q[0], q[2], q[3]]).1,
        ]
      })
      .collect();
    let complex = Complex::from_cells(Skeleton::new(cells));
    let coords = MeshCoords::from(crate::linalg::Matrix::from_columns(&points));
    (complex, coords)
  }

  #[test]
  fn sphere_mean_curvature_and_radius_match_unit_radius() {
    let (topology, coords) = crate::gen::sphere::mesh_sphere_surface(3);
    let mean = vertex_mean_curvature(&topology, &coords);
    let radius = vertex_curvature_radius(&topology, &coords);
    for &h in &mean {
      assert!((h - 1.0).abs() < 0.2, "expected |H| ~ 1, got {h}");
    }
    for &r in &radius {
      assert!(
        r < 1.05 && r > 0.5,
        "expected curvature radius in (0.5, 1.05), got {r}"
      );
    }
  }

  /// A flat unit-square grid is developable: zero mean curvature at every
  /// interior vertex (a boundary vertex's raw $H$ is a natural boundary
  /// term, not curvature -- see [`vertex_curvature_radius`]), so the
  /// curvature radius is unbounded everywhere, boundary included -- curvature
  /// must never clamp displacement on a flat surface.
  #[test]
  fn flat_grid_has_unbounded_curvature_radius() {
    let (topology, coords) = crate::gen::cartesian::CartesianGrid::new_unit(2, 4).triangulate();
    let coords = coords.embed_euclidean(3);
    let boundary: std::collections::HashSet<usize> =
      topology.boundary_vertices().into_iter().collect();
    let mean = vertex_mean_curvature(&topology, &coords);
    let radius = vertex_curvature_radius(&topology, &coords);
    for (v, &h) in mean.iter().enumerate() {
      if !boundary.contains(&v) {
        assert!(h.abs() < 1e-9, "expected interior H ~ 0, got {h}");
      }
    }
    for &r in &radius {
      assert!(
        r.is_infinite(),
        "expected an unbounded curvature radius, got {r}"
      );
    }
  }
}
