//! The reach of an embedded surface: how far it may be displaced along its
//! own normal and still be an embedding.
//!
//! The bound the `RR^3` bake needs and the one thing an offset cannot be
//! taken without. It lives here rather than with the intrinsic geometry for
//! the same reason the bake does: the normal field, the cross product and
//! the point grid are all statements about $RR^3$, where the core is
//! dimension-agnostic. Its local half, the curvature radius, is genuine
//! intrinsic-plus-embedding geometry in any dimension and stays in
//! [`regge::coord`](regge::coord::vertex_curvature_radius).

use regge::coord::{mesh::MeshCoords, vertex_curvature_radius};
use simplicial::{Dim, topology::complex::Complex};

extern crate nalgebra as na;

/// How far [`vertex_reach`] looks for a bottleneck, in mean edge lengths. Sets
/// the thickest feature the non-local term can detect, and with it the cost:
/// the walk visits $O("this"^2)$ occupied cells per vertex, since a surface
/// meets a box in a face rather than filling it.
const REACH_SEARCH_EDGES: i32 = 8;

/// The reach of the embedded surface at every vertex: the largest $r$ such
/// that the normal offset by any $|d| <= r$ is still an embedding, so no fold
/// and no self-intersection.
///
/// Federer's reach, and it has two halves. The local one is the curvature
/// radius ([`vertex_curvature_radius`]): offset past the focal point and the
/// surface folds through itself. The non-local one is the bottleneck, how
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
/// both is the distance to the medial axis, the local feature size, and it
/// degenerates to the normal curvature radius as $q -> p$, so it subsumes the
/// local half rather than sitting beside it. On a sphere of radius $R$ every
/// $q$ returns exactly $R$. On the thin plate the opposite face returns
/// $t \/ 2$.
///
/// Since $r >= |q - p| \/ 2$, a $q$ farther than twice the running best cannot
/// improve it, so the search is confined to a ball that shrinks as the estimate
/// does and a thin feature terminates the walk early. The initial bound is the
/// curvature radius, itself capped by `max_reach` (pass the object's own
/// extent), since a reach exceeding the object bounds nothing.
///
/// The shrinking ball alone does not bound the cost. On a convex surface
/// nothing ever reduces the estimate below the curvature radius, so the ball
/// stays as wide as the object while refinement shrinks the grid under it, and
/// the walk pays $O(V)$ cells per vertex to confirm an absence: $O(V^2)$
/// overall. The search is therefore also capped at a fixed number of mean edge
/// lengths, which restores $O(V)$, a surface meeting a box of side $s$ in
/// $O(s^2)$ cells.
///
/// The cap costs exactly this: a bottleneck wider than that neighborhood is
/// not seen, and the curvature radius alone bounds there. The cutoff is set by
/// the mesh's own resolution, since a gap spanning many times the local edge
/// length is not a thin feature at the scale the mesh resolves, and a feature
/// thinner than the mesh is not represented at all. The uncaught case is a
/// shape with two sheets far apart in edge lengths but close relative to the
/// object, such as a wide, finely meshed horseshoe.
///
/// Defined for an embedded surface whose normal field exists, returning
/// `INFINITY` at every vertex of a complex that is not 2-dimensional or not
/// orientable: a non-orientable surface has no continuous normal field, so
/// "displace along the normal" has no meaning to bound.
pub fn vertex_reach(topology: &Complex, coords: &MeshCoords, max_reach: f64) -> Vec<f64> {
  use rayon::prelude::*;

  let nvertices = topology.nsimplices(Dim::ZERO);
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
  let spacing = coords.to_edge_lengths_sq(topology).mesh_width_mean();
  if spacing <= 0.0 {
    return unbounded;
  }
  let spacing = spacing.max(1e-12);
  let grid = PointGrid::new(&points, spacing);

  let curvature = vertex_curvature_radius(topology, coords);

  (0..nvertices)
    .into_par_iter()
    .map(|v| {
      let (p, n) = (points[v], normals[v]);
      let mut best = curvature[v].min(max_reach);
      // Shell by shell, stopping once even the nearest point of the next shell
      // is farther than twice the running bound, or once the neighborhood is
      // exhausted, whichever comes first.
      for shell in 0..=REACH_SEARCH_EDGES {
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
/// agree instead of canceling. `None` if the surface is not orientable, where
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
  use simplicial::linalg::Vector;
  use simplicial::topology::complex::Complex;

  /// On the unit sphere the reach is the radius, and it is the curvature
  /// half that says so: the medial axis is the center point. The tangent-ball
  /// formula returns exactly $R$ for every pair on a sphere, so this also
  /// checks the estimator against its one closed form.
  #[test]
  fn sphere_reach_is_its_radius() {
    let (topology, coords) = regge::mesher::sphere::mesh_sphere_surface(3);
    let reach = vertex_reach(&topology, &coords, 10.0);
    for &r in &reach {
      assert!(r > 0.5 && r < 1.05, "expected reach ~ 1, got {r}");
    }
  }

  /// The half curvature cannot see. A thin flat slab has infinite curvature
  /// radius on its faces: they are planes, and reach $t \/ 2$, because the
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
    use simplicial::topology::{simplex::Simplex, skeleton::Skeleton};
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
    let coords = MeshCoords::from(simplicial::linalg::Matrix::from_columns(&points));
    (complex, coords)
  }
}
