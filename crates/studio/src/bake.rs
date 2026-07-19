//! The seam out: a simplicial complex, embedded in $RR^3$, reduced to the
//! primitives a rasterizer draws.
//!
//! Deliberately dimension-specific and coordinate-full -- winding and embedding
//! fixed at 3, the two things the core keeps out, because a graphics API needs
//! both. Downstream of here there are no FEEC types, only ambient geometry.
//!
//! This is where the viewer's *dimension* reduction lives, the mirror of the
//! *grade* reduction `scene.rs` performs: a $k$-form reduces to a render mark
//! via $min(k, n-k)$, and an $n$-manifold reduces to a render primitive via
//! $min(n, 2)$ -- a surface bakes to wound triangles, a curve to segments, a
//! point cloud to points, and anything above 2 to the 2-simplices of its
//! boundary, which is all of it an observer in $RR^3$ can see. The case
//! distinction is confined here, exactly as the grade's is confined to the
//! mark.
//!
//! The vertex table is split in two. A `BakedMesh` is the static half -- a
//! function of the mesh and its embedding, and of no field on it -- and
//! [`attributes`] is the other, one scalar per vertex. Switching fields (or
//! scrubbing a trajectory) therefore rewrites only the attribute stream, never
//! positions, normals, curvature or winding.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use simplicial::linalg::Vector;
use simplicial::{
  geometry::coord::{mesh::MeshCoords, vertex_reach},
  topology::{complex::Complex, handle::KSimplexIdx, simplex::Simplex},
  Dim,
};

/// Fraction of the local [`reach`](vertex_reach) allowed as normal
/// displacement: kept below 1 so a vertex never reaches its own medial axis,
/// where the offset would either fold through its focal point or meet the
/// sheet of surface facing it. The reach bounds both, which is why this is the
/// only safety fraction the displacement needs.
const REACH_SAFETY_FRACTION: f64 = 0.9;

/// The static half of a baked vertex: everything that is a function of the mesh
/// and its embedding, and of no field on it.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct BakedVertex {
  pub position: [f32; 3],
  /// Outward unit vertex normal: the axis of the scalar standing-wave
  /// displacement. Zero where the bake has no surface to displace (a curve in
  /// $RR^3$ has a rank-2 normal bundle, so no canonical axis), which makes the
  /// displacement the identity there rather than a case to exclude.
  pub normal: [f32; 3],
  /// The per-vertex cap on normal displacement magnitude: a fraction of the
  /// local curvature radius (`f32::INFINITY` where curvature imposes no bound),
  /// clamping the shared global wave amplitude down wherever it would otherwise
  /// fold a thin or sharply curved feature.
  pub max_displacement: f32,
}

/// A vertex of a segment mark: a baked vertex plus an opacity multiplier. The
/// wireframe, a line field's traced ribbons and a 1-manifold's own cells are
/// one mark drawn from one type.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct SegmentVertex {
  pub vertex: BakedVertex,
  /// Opacity multiplier in $[0, 1]$, for a mark that wants to fade part of
  /// itself out (none does today -- every producer emits 1). Kept as a
  /// per-vertex factor rather than a flat material constant, since a future
  /// mark's reason to fade (unlike a traced curve's arbitrary tracer cutoff,
  /// which is not such a reason) would vary along its length.
  pub opacity: f32,
}

/// A glyph mark, as one instance: an arrow lying flat in its surface cell.
///
/// Instanced rather than expanded into corners on the CPU. The quad's six
/// corners are `@builtin(vertex_index)` arithmetic in the vertex shader, and
/// every dimension of the arrow is a proportion of its own length
/// ([`GlyphMaterial`](crate::render::uniform::GlyphMaterial)), so nothing about
/// a corner needs storing -- only what distinguishes one arrow from another.
///
/// The one datum that looks per-corner is the barycentric coordinate the
/// fragment clips against, and it is not: the quad is planar and `global2bary`
/// is affine, so the coordinate over the quad is exactly its value at the
/// center plus two gradients. Three vectors reproduce what was six copies of
/// four floats.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GlyphInstance {
  /// The arrow's center in world space, on the undisplaced surface.
  pub center: [f32; 3],
  /// The arrow's length in world units: its lattice's realized spacing, and the
  /// scale every other dimension of the mark is a fraction of.
  pub length: f32,
  /// The unit direction the arrow points, in its cell's plane: the sharped
  /// field, pushed into ambient coordinates.
  pub direction: [f32; 3],
  /// Opacity multiplier in $[0, 1]$: the field's magnitude here against its
  /// peak, so a vanishing field's arrow fades rather than points arbitrarily.
  pub opacity: f32,
  /// The in-plane unit vector across the arrow, completing its frame.
  pub across: [f32; 3],
  pub _pad0: f32,
  /// The cell barycentric coordinate at the arrow's center, padded to four with
  /// ones so a cell of intrinsic dimension below three never trips the clip.
  pub bary_center: [f32; 4],
  /// Its gradients per world unit along [`Self::direction`] and
  /// [`Self::across`], padded with zeros, from which the fragment's coordinate
  /// is affine.
  pub bary_along: [f32; 4],
  pub bary_across: [f32; 4],
}

/// The primitives one complex's cells bake to, at dimension $min(n, 2)$.
/// Indices into [`BakedMesh::positions`].
#[derive(Debug, Clone)]
pub enum PrimBatch {
  /// Consistently wound triangles: an $n = 2$ mesh's cells, or the boundary
  /// 2-simplices of an $n >= 3$ one.
  Triangles(Vec<[u32; 3]>),
  /// An $n = 1$ mesh's cells.
  Segments(Vec<[u32; 2]>),
  /// An $n = 0$ mesh's cells. Baked, but carrying no mark yet -- a future mark,
  /// not a case to route around.
  Points(Vec<u32>),
}

/// Which cell one rendered triangle's field is read in, and where its corners
/// sit in that cell.
///
/// A reduced-grade field is discontinuous across cells -- only the *tangential*
/// part of a section is chart-independent, so incident cells genuinely disagree
/// at a shared vertex, and there is no single value to interpolate. The honest
/// readout is therefore per corner, in the corner's *own* cell: this map is what
/// the per-corner sampling is written against. For a surface mesh the cell is
/// the triangle itself; for a solid's rendered boundary face it is the unique
/// cell that face bounds. The dimension reduction stays confined here, in the
/// bake, exactly as the primitive choice is.
#[derive(Debug, Clone, Copy)]
pub struct CellCorner {
  /// The cell (colex rank in the top skeleton) whose chart the field is read in.
  pub cell: KSimplexIdx,
  /// Each corner's local vertex index within `cell`'s colex vertex tuple, in the
  /// rendered triangle's own (wound) corner order.
  pub local: [usize; 3],
}

/// One complex, baked: a shared vertex table plus the index streams over it.
///
/// One `BakedMesh` per complex, never merged across complexes. Cross-mesh
/// merging is batching -- a draw-call optimization, not an abstraction -- and it
/// destroys per-object identity (visibility, bounds, colormap, picking).
#[derive(Debug, Clone)]
pub struct BakedMesh {
  /// Static per-vertex data, one entry per mesh vertex.
  pub positions: Vec<BakedVertex>,
  pub cells: PrimBatch,
  /// The 1-skeleton, as the wireframe overlay over a filled surface. Empty
  /// where the cells already *are* the 1-skeleton ($n <= 1$), so a curve's
  /// edges are drawn once rather than twice.
  pub wireframe: Vec<[u32; 2]>,
  /// One entry per rendered triangle, in the same order as
  /// [`PrimBatch::Triangles`], mapping it to the cell its field is read in.
  /// Empty for a bake with no fill (a curve, a point cloud).
  pub cell_corners: Vec<CellCorner>,
}

impl BakedMesh {
  /// Bakes `topology` embedded by `coords`. Independent of any field on it; see
  /// [`attributes`] for the other half.
  pub fn new(topology: &Complex, coords: &MeshCoords) -> Self {
    let nvertices = topology.nsimplices(0);

    let ambient: Vec<na::Vector3<f64>> = (0..nvertices).map(|i| embed_r3(coords, i)).collect();

    // The one dimension dispatch: the cells' primitive, the overlay, and the
    // displacement frame (normal and curvature cap) are one decision, made
    // once, here.
    // Beyond the object's own extent a reach is not a meaningful bound, and an
    // unbounded one would make a flat mesh search globally for a bottleneck
    // that is not there.
    let centroid = ambient.iter().sum::<na::Vector3<f64>>() / (nvertices.max(1) as f64);
    let extent = ambient
      .iter()
      .map(|p| (p - centroid).norm())
      .fold(0.0, f64::max)
      .max(1e-6);

    let (cells, wireframe, normals, reach) = match topology.dim() {
      0 => (
        PrimBatch::Points((0..nvertices as u32).collect()),
        Vec::new(),
        vec![na::Vector3::zeros(); nvertices],
        vec![f64::INFINITY; nvertices],
      ),
      1 => (
        PrimBatch::Segments(cell_indices(topology)),
        Vec::new(),
        vec![na::Vector3::zeros(); nvertices],
        vec![f64::INFINITY; nvertices],
      ),
      2 => {
        let triangles = orient_triangles(&cell_indices(topology));
        let normals = vertex_normals(&triangles, &ambient);
        (
          PrimBatch::Triangles(triangles),
          skeleton_indices(topology, 1),
          normals,
          vertex_reach(topology, coords, extent),
        )
      }
      // An observer in $RR^3$ sees only the boundary of a solid, so that is
      // what is baked. The curvature cap is the boundary surface's own, which
      // this complex cannot report (its curvature estimators are those of an
      // $n$-manifold, not of its boundary), so the displacement is left
      // uncapped rather than clamped by a quantity that does not mean what the
      // cap needs it to.
      _ => {
        let triangles = orient_triangles(&boundary_indices(topology, 2));
        let normals = vertex_normals(&triangles, &ambient);
        (
          PrimBatch::Triangles(triangles),
          boundary_indices(topology, 1),
          normals,
          vec![f64::INFINITY; nvertices],
        )
      }
    };

    let positions = (0..nvertices)
      .map(|i| {
        let p = ambient[i];
        let n = normals[i];
        // A vertex whose incident triangles' normals cancel (or that has none)
        // has no displacement axis; normalizing would divide by zero, so the
        // zero vector stands, and the displacement is the identity there.
        let n = if n.norm() > 1e-12 { n.normalize() } else { n };
        BakedVertex {
          position: [p.x as f32, p.y as f32, p.z as f32],
          normal: [n.x as f32, n.y as f32, n.z as f32],
          max_displacement: (REACH_SAFETY_FRACTION * reach[i]) as f32,
        }
      })
      .collect();

    let cell_corners = match &cells {
      PrimBatch::Triangles(triangles) => cell_corners(topology, triangles),
      _ => Vec::new(),
    };

    Self {
      positions,
      cells,
      wireframe,
      cell_corners,
    }
  }

  /// The mesh's own vertices as segment vertices: the table the wireframe
  /// overlay (and a 1-manifold's cells) are drawn from, at full opacity.
  pub fn segment_vertices(&self) -> Vec<SegmentVertex> {
    self
      .positions
      .iter()
      .map(|&vertex| SegmentVertex {
        vertex,
        opacity: 1.0,
      })
      .collect()
  }

  /// The nearest hit of the ray `origin + t * dir` ($t > 0$) on the baked
  /// surface, as that `t`. `None` when the ray misses, and for a bake with no
  /// surface to hit -- a curve or a point cloud, whose measure-zero image in
  /// $RR^3$ no ray meets in general position, so a miss is the honest answer
  /// rather than a case to exclude.
  ///
  /// Ambient geometry on the bake, which is exactly what a bake is for and what
  /// `studio`'s extrinsic license covers. Against the *undisplaced* positions:
  /// the standing wave is a vertex-shader displacement, and the caller wants
  /// the point of the object, not of the frame it happened to be caught in.
  pub fn raycast(&self, origin: na::Vector3<f32>, dir: na::Vector3<f32>) -> Option<f32> {
    let PrimBatch::Triangles(triangles) = &self.cells else {
      return None;
    };
    triangles
      .iter()
      .filter_map(|&[a, b, c]| {
        let vertex = |i: u32| na::Vector3::from(self.positions[i as usize].position);
        ray_triangle(origin, dir, vertex(a), vertex(b), vertex(c))
      })
      .min_by(|a, b| a.total_cmp(b))
  }
}

/// Möller-Trumbore: the ray `origin + t * dir` against one triangle, as that
/// `t`. Two-sided -- a pick must land on a surface seen from its back exactly
/// as on one seen from its front, and a mesh whose winding says otherwise is
/// still a mesh the user is pointing at.
fn ray_triangle(
  origin: na::Vector3<f32>,
  dir: na::Vector3<f32>,
  a: na::Vector3<f32>,
  b: na::Vector3<f32>,
  c: na::Vector3<f32>,
) -> Option<f32> {
  const EPS: f32 = 1e-7;
  let (ab, ac) = (b - a, c - a);
  let pvec = dir.cross(&ac);
  let det = ab.dot(&pvec);
  // The ray runs parallel to the triangle's plane: it either misses or grazes
  // it edge-on, and neither is a point to pivot about.
  if det.abs() < EPS {
    return None;
  }
  let inv_det = 1.0 / det;
  let tvec = origin - a;
  let u = tvec.dot(&pvec) * inv_det;
  if !(-EPS..=1.0 + EPS).contains(&u) {
    return None;
  }
  let qvec = tvec.cross(&ab);
  let v = dir.dot(&qvec) * inv_det;
  if v < -EPS || u + v > 1.0 + EPS {
    return None;
  }
  let t = ac.dot(&qvec) * inv_det;
  (t > EPS).then_some(t)
}

/// The per-field half of the vertex table: the colormap scalar, one per entry of
/// [`BakedMesh::positions`] -- a scalar field's 0-form value, or a line field's
/// nodal magnitude tinting the surface its glyphs are drawn on. The one stream a
/// field change rewrites.
pub fn attributes(values: &[f64]) -> Vec<f32> {
  values.iter().map(|&v| v as f32).collect()
}

/// An ambient vector in the viewer's one ambient space: $RR^3$, so a mesh
/// embedded in fewer dimensions embeds as itself in the missing ones' zero
/// planes -- the codimension case, not a special case.
pub fn to_vec3(v: &Vector) -> na::Vector3<f64> {
  na::Vector3::from_iterator((0..3).map(|i| v.get(i).copied().unwrap_or(0.0)))
}

/// A vertex's ambient position.
fn embed_r3(coords: &MeshCoords, vertex: usize) -> na::Vector3<f64> {
  to_vec3(&coords.coord(vertex).view().into_owned())
}

/// The vertex tuples of the complex's cells, as indices into the baked vertex
/// table -- which is the vertex skeleton itself, so a simplex's vertices *are*
/// its indices.
fn cell_indices<const N: usize>(topology: &Complex) -> Vec<[u32; N]> {
  skeleton_indices(topology, topology.dim())
}

/// The [`CellCorner`] of each rendered triangle, in triangle order.
///
/// The owning cell is the one coface of the triangle's 2-simplex: itself when
/// the mesh is a surface (a 2-simplex is a cell), the unique cell it bounds when
/// the mesh is a solid whose boundary is what got baked. Reads the local indices
/// off the *wound* triangle, so they track the corner order the vertex streams
/// are built in.
fn cell_corners(topology: &Complex, triangles: &[[u32; 3]]) -> Vec<CellCorner> {
  let faces = topology.skeleton(2);
  triangles
    .iter()
    .map(|&tri| {
      // The rendered triangle is wound (its corner order is the winding pass's,
      // not colex); a `Simplex` is the colex-sorted vertex set, so sort for the
      // lookup while keeping `tri`'s order for the local-index map below.
      let mut sorted: Vec<usize> = tri.iter().map(|&v| v as usize).collect();
      sorted.sort_unstable();
      let cell = faces
        .handle_by_simplex(&Simplex::new(sorted))
        .cells()
        .next()
        .expect("a rendered triangle bounds a cell");
      let cell_vertices = &cell.simplex().vertices;
      let local = tri.map(|v| {
        cell_vertices
          .iter()
          .position(|&u| u == v as usize)
          .expect("a rendered triangle's vertex lies in the cell it bounds")
      });
      CellCorner {
        cell: cell.kidx(),
        local,
      }
    })
    .collect()
}

fn skeleton_indices<const N: usize>(topology: &Complex, dim: Dim) -> Vec<[u32; N]> {
  topology
    .skeleton_raw(dim)
    .iter()
    .map(|simp| index_tuple(&simp.vertices))
    .collect()
}

fn boundary_indices<const N: usize>(topology: &Complex, dim: Dim) -> Vec<[u32; N]> {
  topology
    .boundary_simplices(dim)
    .into_iter()
    .map(|idx| index_tuple(&idx.handle(topology).simplex().vertices))
    .collect()
}

fn index_tuple<const N: usize>(vertices: &[usize]) -> [u32; N] {
  assert_eq!(vertices.len(), N);
  std::array::from_fn(|i| {
    u32::try_from(vertices[i]).expect("a mesh with over 4 billion vertices is not renderable")
  })
}

/// A canonical undirected-edge key: the vertex pair sorted so `(a, b)` and
/// `(b, a)` hash identically.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
  if a < b {
    (a, b)
  } else {
    (b, a)
  }
}

fn directed_edges(t: [u32; 3]) -> [(u32, u32); 3] {
  [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
}

/// Consistently winds a triangle soup by flood-filling face adjacency: two
/// triangles sharing an edge are consistent iff they traverse that edge in
/// opposite directions.
///
/// A `Complex`'s cells carry no winding -- vertices are colex-sorted, per the
/// crate's one indexing convention -- so a triangle list read off one has an
/// essentially arbitrary, alternating winding per triangle. Per-vertex normals
/// of such a list are meaningless without first running this pass: consistent
/// orientation is topological data, recoverable from the manifold's face
/// adjacency, not from geometry (a 2-simplex in $RR^3$ has no signed
/// determinant to read a winding off of).
///
/// Fixes the orientation of one triangle per connected component arbitrarily;
/// the result is correct up to that per-component choice of global sign, which
/// is exactly the ambiguity a normal field has on an orientable surface.
/// Assumes a manifold surface (each edge shared by at most two triangles).
pub fn orient_triangles(triangles: &[[u32; 3]]) -> Vec<[u32; 3]> {
  let mut edge_to_tris: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
  for (ti, &t) in triangles.iter().enumerate() {
    for (a, b) in directed_edges(t) {
      edge_to_tris.entry(edge_key(a, b)).or_default().push(ti);
    }
  }

  let mut oriented = triangles.to_vec();
  let mut visited = vec![false; triangles.len()];
  for start in 0..triangles.len() {
    if visited[start] {
      continue;
    }
    visited[start] = true;
    let mut stack = vec![start];
    while let Some(ti) = stack.pop() {
      for (a, b) in directed_edges(oriented[ti]) {
        for &tj in &edge_to_tris[&edge_key(a, b)] {
          if visited[tj] {
            continue;
          }
          // Consistent winding traverses a shared edge in opposite directions;
          // if `tj` also has the directed edge `(a, b)`, it agrees with `ti`
          // and must be flipped.
          if directed_edges(oriented[tj]).contains(&(a, b)) {
            oriented[tj].swap(1, 2);
          }
          visited[tj] = true;
          stack.push(tj);
        }
      }
    }
  }
  oriented
}

/// Per-vertex normals of a triangle surface embedded in $RR^3$: the average of
/// the unit normals of a vertex's incident triangles.
///
/// Meaningful only on a consistently wound list; see [`orient_triangles`]. Not
/// itself renormalized -- at a crease or on a coarse mesh the average of unit
/// vectors falls short of one -- so a caller needing a unit axis normalizes
/// itself.
pub fn vertex_normals(
  triangles: &[[u32; 3]],
  positions: &[na::Vector3<f64>],
) -> Vec<na::Vector3<f64>> {
  let mut normals = vec![na::Vector3::zeros(); positions.len()];
  let mut counts = vec![0u32; positions.len()];
  for ivs in triangles {
    let vs = ivs.map(|i| positions[i as usize]);
    let triangle_normal = (vs[1] - vs[0]).cross(&(vs[2] - vs[0])).normalize();
    for &iv in ivs {
      normals[iv as usize] += triangle_normal;
      counts[iv as usize] += 1;
    }
  }
  for (normal, count) in normals.iter_mut().zip(counts) {
    if count > 0 {
      *normal /= f64::from(count);
    }
  }
  normals
}

#[cfg(test)]
mod tests {
  use super::*;
  use simplicial::geometry::coord::mesh::standard_coord_complex;

  /// The standard cell of every dimension the ambient reaches bakes, and bakes
  /// to the primitive $min(n, 2)$ names: a segment, a triangle, and a
  /// tetrahedron's four boundary triangles.
  #[test]
  fn standard_cell_bakes_at_every_dimension() {
    for dim in 0..=3 {
      let (topology, coords) = standard_coord_complex(dim);
      let coords = coords.embed_euclidean(3);
      let nvertices = topology.nsimplices(0);
      let baked = BakedMesh::new(&topology, &coords);

      assert_eq!(baked.positions.len(), nvertices);
      let ncells = topology.nsimplices(dim);
      match (dim, &baked.cells) {
        (0, PrimBatch::Points(p)) => assert_eq!(p.len(), ncells),
        (1, PrimBatch::Segments(s)) => assert_eq!(s.len(), ncells),
        (2, PrimBatch::Triangles(t)) => assert_eq!(t.len(), ncells),
        // The tetrahedron's boundary: its four faces.
        (3, PrimBatch::Triangles(t)) => assert_eq!(t.len(), 4),
        (d, b) => panic!("dim {d} baked to {b:?}"),
      }
      for &[a, b] in &baked.wireframe {
        assert!((a as usize) < nvertices && (b as usize) < nvertices);
      }
    }
  }

  /// The wireframe overlay is the 1-skeleton of a filled surface, and empty
  /// where the cells already are the 1-skeleton: a curve's edges are drawn
  /// once, not twice.
  #[test]
  fn wireframe_is_the_overlay_only_where_there_is_a_fill() {
    for dim in 0..=2 {
      let (topology, coords) = standard_coord_complex(dim);
      let coords = coords.embed_euclidean(3);
      let baked = BakedMesh::new(&topology, &coords);
      let expected = if dim == 2 { topology.nsimplices(1) } else { 0 };
      assert_eq!(baked.wireframe.len(), expected);
    }
  }

  /// Winding consistency, the property the normals depend on: every edge shared
  /// by two triangles is traversed in opposite directions by them. Checked on a
  /// closed surface (the tetrahedron's boundary) and on a mesh with boundary
  /// (the triforce), where an edge with one incident triangle constrains
  /// nothing.
  #[test]
  fn baked_triangles_are_consistently_wound() {
    let (tet, tet_coords) = standard_coord_complex(3);
    let cases = [
      (tet.clone(), tet_coords.embed_euclidean(3)),
      crate::demos::triforce(),
    ];
    for (topology, coords) in cases {
      let baked = BakedMesh::new(&topology, &coords);
      let PrimBatch::Triangles(triangles) = &baked.cells else {
        panic!("a 2- or 3-manifold bakes to triangles");
      };
      let mut seen: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
      for &t in triangles {
        for (a, b) in directed_edges(t) {
          if let Some(&(pa, pb)) = seen.get(&edge_key(a, b)) {
            assert_eq!(
              (pa, pb),
              (b, a),
              "edge ({a}, {b}) traversed the same way by both its triangles"
            );
          } else {
            seen.insert(edge_key(a, b), (a, b));
          }
        }
      }
    }
  }

  /// Every field of every Whitney basis gallery bakes, at every dimension the
  /// ambient reaches: the scene's grade reduction and the bake's dimension
  /// reduction compose without a hole, and each field samples to one colormap
  /// value per rendered corner, one surface displacement height per corner and
  /// one segment height per mesh vertex.
  #[test]
  fn every_whitney_basis_field_bakes() {
    use crate::scene::{nodal_heights, surface_corner_heights, surface_corner_values, Scene};
    for dim in 1..=3 {
      let scene = Scene::whitney_basis(dim);
      assert!(!scene.fields.is_empty());
      let baked = BakedMesh::new(&scene.topology, &scene.coords);
      assert_eq!(baked.positions.len(), scene.coords.nvertices());
      let ncorners = match &baked.cells {
        PrimBatch::Triangles(triangles) => 3 * triangles.len(),
        _ => 0,
      };
      assert_eq!(baked.cell_corners.len(), ncorners / 3);
      let cochains = scene
        .fields
        .iter()
        .map(|f| &f.cochain)
        .chain(scene.line_fields.iter().map(|f| &f.cochain));
      for cochain in cochains {
        let colors =
          surface_corner_values(&scene.topology, &scene.coords, cochain, &baked.cell_corners);
        assert_eq!(colors.len(), ncorners);
        let surface_heights =
          surface_corner_heights(&scene.topology, &scene.coords, cochain, &baked.cell_corners);
        assert_eq!(surface_heights.len(), ncorners);
        let heights = nodal_heights(&scene.topology, &scene.coords, cochain);
        assert_eq!(heights.len(), baked.positions.len());
      }
    }
  }
}
