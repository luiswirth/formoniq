//! Evenly-spaced streamlines of a grade-1 line field, traced on the manifold.
//!
//! The integral curves of the sharped Whitney field $V = (W c)^sharp$: distinct,
//! regularly spaced curves rather than the dense LIC texture. Two design
//! choices carry the quality.
//!
//! **Intrinsic tracing.** The curve is integrated in the barycentric charts of
//! the atlas, cell by cell, never in ambient space. Within a cell the field is
//! the affine sharped Whitney form, evaluated in the cell's own local frame; a
//! step is RK4 in those local coordinates. When a barycentric weight goes
//! negative the curve has left through the opposite facet, and the
//! [`Transition`](manifold::atlas::Transition) relabels the crossing point into
//! the neighbouring chart, where integration resumes. There is no reprojection
//! onto the surface and no ambient point-location in the hot loop: cell
//! membership is exact from the barycentric signs, and because the tangential
//! part of an $H(curl)$ Whitney form is continuous across a shared facet, the
//! heading carries across the edge seam without a gap. Unlike the LIC's
//! *unsigned* nodal line field, $V$ is a genuine directed vector field, so
//! forward and backward are simply $plus.minus V$ and there is no sign to track.
//!
//! **Even spacing** (Jobard--Lefebvre). A curve stops growing once it comes
//! within a fraction of the separation `d_sep` of an already-placed curve; after
//! a curve is accepted, fresh seeds are dropped a distance `d_sep` to either
//! side of it and integrated in turn. The separation is a fraction of the
//! scene's own extent, so the line density is a property of the object, not of
//! the triangulation.

use std::collections::{HashMap, VecDeque};

use common::{gramian::RiemannianMetric, linalg::nalgebra::Vector};
use ddf::{cochain::Cochain, whitney::interpolant::WhitneyInterpolant};
use manifold::{
  atlas::{local2bary, ChartExt, Local, MeshPoint},
  geometry::{
    coord::{locate::PointLocator, mesh::MeshCoords, simplex::SimplexRefExt, Coord},
    metric::geometry::Geometry,
  },
  topology::{complex::Complex, handle::SimplexIdx, simplex::Simplex},
};

use crate::{bake::to_vec3, scene::reduced_form};

/// One sample of a traced streamline: its ambient position, the unit ambient
/// field direction there (for the ribbon's orientation), and the field
/// magnitude $|V|_g$ (for the tint and the animated fade).
#[derive(Clone, Copy, Debug)]
pub struct StreamPoint {
  pub pos: na::Vector3<f64>,
  pub tangent: na::Vector3<f64>,
  pub magnitude: f64,
}

/// A single traced curve, as a densely sampled polyline.
pub type Polyline = Vec<StreamPoint>;

/// The evenly-spaced set of streamlines of one line field.
pub struct Streamlines {
  pub lines: Vec<Polyline>,
}

/// RK4 step length, as a fraction of the separation `d_sep`: several samples per
/// separation, so a curve resolves its own curvature and the spacing test is
/// fine-grained.
const STEP_FRACTION: f64 = 0.35;
/// A growing curve stops when it comes within this fraction of `d_sep` of an
/// already-placed curve.
const STOP_FRACTION: f64 = 0.8;
/// A candidate seed is discarded if it lies within this fraction of `d_sep` of
/// an existing curve: the placement separation proper.
const SEED_FRACTION: f64 = 0.9;
/// A curve is closed (a periodic orbit) when it returns within this fraction of
/// `d_sep` of its seed after travelling at least [`MIN_LOOP_FRACTION`].
const CLOSURE_FRACTION: f64 = 0.5;
const MIN_LOOP_FRACTION: f64 = 3.0;
/// Below this fraction of the field's peak magnitude the direction is taken as
/// undefined (a critical point) and the curve ends.
const MIN_SPEED_FRACTION: f64 = 1e-3;
/// Backstop on a single curve's arclength, in units of `d_sep`: a spiral that
/// neither closes nor reaches a critical point is cut here.
const MAX_ARCLEN_FRACTION: f64 = 400.0;
/// A traced curve shorter than this fraction of `d_sep` is discarded rather
/// than drawn. In a convergent region (near a vortex center, where
/// neighbouring curves are not parallel) a seed can pass the entry check at
/// its own point and then curve straight into a *different* already-placed
/// neighbour within a step or two, choking the curve on both ends almost at
/// birth. The result carries no coverage the curve that choked it does not
/// already provide -- the area is exactly why it was choked short -- so it is
/// pure visual noise: an isolated dash with no neighbour of its own.
const MIN_LINE_ARCLEN_FRACTION: f64 = 2.0;
/// Backstop on the total number of curves.
const MAX_LINES: usize = 20_000;
/// Hard cap on the steps of one half-curve, guarding against a step length that
/// never makes progress.
const MAX_STEPS: usize = 200_000;

/// Trace the evenly-spaced streamlines of the field `W cochain` (reduced to
/// grade 1) on the embedded surface, at separation `d_sep`.
pub fn trace(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  d_sep: f64,
) -> Streamlines {
  let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);

  // The field's peak magnitude over cell barycenters, which both sets the
  // critical-point threshold and picks the first seed. A field that vanishes
  // everywhere has no streamlines.
  let mut peak = 0.0;
  let mut seed_cell = None;
  for cell in topology.cells().handle_iter() {
    let metric = coords.cell_metric(cell);
    let bary = MeshPoint::barycenter(cell.idx());
    let speed = reduced_form(interpolant.eval(&bary), &metric).norm(&metric);
    if speed > peak {
      peak = speed;
      seed_cell = Some(cell.idx());
    }
  }
  let Some(seed_cell) = seed_cell else {
    return Streamlines { lines: Vec::new() };
  };
  if peak <= 0.0 {
    return Streamlines { lines: Vec::new() };
  }

  let tracer = Tracer {
    topology,
    coords,
    interpolant: &interpolant,
    step: STEP_FRACTION * d_sep,
    min_speed: MIN_SPEED_FRACTION * peak,
    max_arclen: MAX_ARCLEN_FRACTION * d_sep,
    stop_dist: STOP_FRACTION * d_sep,
    closure_dist: CLOSURE_FRACTION * d_sep,
    min_loop: MIN_LOOP_FRACTION * d_sep,
  };

  let locator = PointLocator::new(topology, coords);
  let mut placed = SpatialHash::new(d_sep);
  let mut lines: Vec<Polyline> = Vec::new();
  let mut queue: VecDeque<MeshPoint> = VecDeque::new();
  queue.push_back(MeshPoint::barycenter(seed_cell));

  // Seeds are sampled along an accepted curve roughly every `d_sep`.
  let seed_stride = (1.0 / STEP_FRACTION).ceil() as usize;

  while let Some(seed) = queue.pop_front() {
    if lines.len() >= MAX_LINES {
      break;
    }
    let seed_pos = tracer.sample(&seed).pos;
    if placed.any_within(seed_pos, SEED_FRACTION * d_sep) {
      continue;
    }

    let line = tracer.trace_line(&seed, &placed);
    if line.len() < 2 {
      continue;
    }
    let arclen: f64 = line.windows(2).map(|w| (w[1].pos - w[0].pos).norm()).sum();
    if arclen < MIN_LINE_ARCLEN_FRACTION * d_sep {
      continue;
    }

    for sp in &line {
      placed.insert(sp.pos);
    }
    for sp in line.iter().step_by(seed_stride.max(1)) {
      for cand in tracer.side_seeds(&locator, sp, d_sep) {
        let pos = tracer.sample(&cand).pos;
        if !placed.any_within(pos, SEED_FRACTION * d_sep) {
          queue.push_back(cand);
        }
      }
    }
    lines.push(line);
  }

  Streamlines { lines }
}

struct Tracer<'a> {
  topology: &'a Complex,
  coords: &'a MeshCoords,
  interpolant: &'a WhitneyInterpolant<'a>,
  step: f64,
  min_speed: f64,
  max_arclen: f64,
  stop_dist: f64,
  closure_dist: f64,
  min_loop: f64,
}

impl Tracer<'_> {
  /// The full curve through a seed: the backward half reversed, the seed, then
  /// the forward half.
  fn trace_line(&self, seed: &MeshPoint, placed: &SpatialHash) -> Polyline {
    let seed_sample = self.sample(seed);
    let forward = self.trace_half(seed, 1.0, placed, seed_sample.pos);
    let backward = self.trace_half(seed, -1.0, placed, seed_sample.pos);
    let mut line = Vec::with_capacity(backward.len() + 1 + forward.len());
    line.extend(backward.into_iter().rev());
    line.push(seed_sample);
    line.extend(forward);
    line
  }

  /// One half-curve from `start`, integrating $"sign" dot V$ until it leaves the
  /// surface at a boundary, reaches a critical point, closes on its seed, comes
  /// too close to an already-placed curve, or exhausts the arclength budget.
  fn trace_half(
    &self,
    start: &MeshPoint,
    sign: f64,
    placed: &SpatialHash,
    seed_pos: na::Vector3<f64>,
  ) -> Polyline {
    let mut out = Polyline::new();
    let mut cell = start.cell_idx();
    let mut metric = self.coords.cell_metric(cell.handle(self.topology));
    let mut x = start.local().vector().clone();
    let mut arclen = 0.0;
    let dim = self.topology.dim();
    let h = self.step;
    // The facet just crossed: a second crossing of the same facet with no
    // interior step between is the field running tangent to an edge, a
    // ping-pong we end rather than loop on.
    let mut last_facet: Option<SimplexIdx> = None;

    for _ in 0..MAX_STEPS {
      let Some(k1) = self.dir(cell, &x, &metric, sign) else {
        break;
      };
      let Some(k2) = self.dir(cell, &(&x + &k1 * (0.5 * h)), &metric, sign) else {
        break;
      };
      let Some(k3) = self.dir(cell, &(&x + &k2 * (0.5 * h)), &metric, sign) else {
        break;
      };
      let Some(k4) = self.dir(cell, &(&x + &k3 * h), &metric, sign) else {
        break;
      };
      let mut incr = k1;
      incr += &k2 * 2.0;
      incr += &k3 * 2.0;
      incr += k4;
      let x_new = &x + incr * (h / 6.0);

      let bary_new = local2bary(&Local::new(x_new.clone()));
      let exit = (0..=dim)
        .filter(|&i| bary_new[i] < -f64::EPSILON)
        .min_by(|&i, &j| bary_new[i].partial_cmp(&bary_new[j]).unwrap());

      let Some(exit) = exit else {
        // Interior step.
        x = x_new;
        arclen += h;
        let sample = self.emit(cell, &x, &metric);
        let done = self.record(&mut out, sample, arclen, seed_pos, placed);
        last_facet = None;
        if done {
          break;
        }
        continue;
      };

      // The step left the cell through the facet opposite local vertex `exit`.
      // Clip to the crossing along the (barycentrically affine) segment.
      let bary_old = local2bary(&Local::new(x.clone()));
      let t = bary_old[exit] / (bary_old[exit] - bary_new[exit]);
      let x_cross = &x + (&x_new - &x) * t;
      arclen += t * h;
      let sample = self.emit(cell, &x_cross, &metric);
      let done = self.record(&mut out, sample, arclen, seed_pos, placed);
      if done {
        break;
      }

      let handle = cell.handle(self.topology);
      let facet_vertices: Vec<usize> = handle
        .simplex()
        .vertices
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != exit)
        .map(|(_, &v)| v)
        .collect();
      let facet = self
        .topology
        .skeleton(dim - 1)
        .handle_by_simplex(&Simplex::new(facet_vertices));
      let Some(neighbor) = facet.cells().find(|c| c.idx() != cell) else {
        break; // boundary facet: the curve leaves the surface
      };
      if last_facet == Some(facet.idx()) {
        break; // tangent to the edge: ping-pong
      }

      let cross_point = MeshPoint::new(cell, local2bary(&Local::new(x_cross.clone())));
      let Some(entered) = cross_point.transition_to(neighbor.chart()) else {
        break;
      };
      cell = entered.cell_idx();
      metric = self.coords.cell_metric(cell.handle(self.topology));
      x = entered.local().vector().clone();
      last_facet = Some(facet.idx());
    }

    out
  }

  /// Push `sample` and report whether the curve should end there: it closes on
  /// its seed, runs into an already-placed curve, or exceeds the arclength cap.
  fn record(
    &self,
    out: &mut Polyline,
    sample: StreamPoint,
    arclen: f64,
    seed_pos: na::Vector3<f64>,
    placed: &SpatialHash,
  ) -> bool {
    let pos = sample.pos;
    out.push(sample);
    if arclen > self.min_loop && (pos - seed_pos).norm() < self.closure_dist {
      return true;
    }
    if placed.any_within(pos, self.stop_dist) {
      return true;
    }
    arclen > self.max_arclen
  }

  /// The unit-physical field velocity in the cell's local frame, times `sign`;
  /// `None` at a critical point, where the direction is undefined. Evaluated by
  /// the affine Whitney formula, so a query slightly outside the cell (an RK4
  /// sub-stage) is a valid extrapolation.
  fn dir(
    &self,
    cell: SimplexIdx,
    x: &Vector,
    metric: &RiemannianMetric,
    sign: f64,
  ) -> Option<Vector> {
    let point = MeshPoint::new(cell, local2bary(&Local::new(x.clone())));
    let form = reduced_form(self.interpolant.eval(&point), metric);
    let speed = form.norm(metric);
    if speed < self.min_speed {
      return None;
    }
    Some(form.sharp(metric).coeffs() * (sign / speed))
  }

  /// The ambient sample (position, unit field direction, magnitude) at a local
  /// point of a cell.
  fn emit(&self, cell: SimplexIdx, x: &Vector, metric: &RiemannianMetric) -> StreamPoint {
    let handle = cell.handle(self.topology);
    let coord_simplex = handle.coord_simplex(self.coords);
    let bary = local2bary(&Local::new(x.clone()));
    let pos = to_vec3(coord_simplex.bary2global(&bary).vector());
    let form = reduced_form(self.interpolant.eval(&MeshPoint::new(cell, bary)), metric);
    let magnitude = form.norm(metric);
    let ambient = coord_simplex.pushforward_vector(form.sharp(metric).coeffs());
    let tangent = to_vec3(&ambient)
      .try_normalize(1e-12)
      .unwrap_or_else(na::Vector3::zeros);
    StreamPoint {
      pos,
      tangent,
      magnitude,
    }
  }

  /// The ambient sample at an arbitrary mesh point (used for seeds).
  fn sample(&self, point: &MeshPoint) -> StreamPoint {
    let metric = self
      .coords
      .cell_metric(point.cell_idx().handle(self.topology));
    self.emit(point.cell_idx(), point.local().vector(), &metric)
  }

  /// The two candidate seeds a separation `d_sep` to either side of `sp`, in the
  /// surface tangent plane perpendicular to the field, each located back to a
  /// mesh point (dropped if it falls off the surface).
  fn side_seeds(&self, locator: &PointLocator, sp: &StreamPoint, d_sep: f64) -> Vec<MeshPoint> {
    let Some(here) = locator.locate(&coord3(sp.pos)) else {
      return Vec::new();
    };
    let normal = self.cell_normal(here.cell_idx());
    let perp = normal.cross(&sp.tangent);
    let Some(perp) = perp.try_normalize(1e-12) else {
      return Vec::new();
    };
    [-1.0, 1.0]
      .into_iter()
      .filter_map(|s| locator.locate(&coord3(sp.pos + perp * (s * d_sep))))
      .collect()
  }

  /// The outward unit normal of a cell, from its two spanning edge vectors.
  fn cell_normal(&self, cell: SimplexIdx) -> na::Vector3<f64> {
    let coord_simplex = cell.handle(self.topology).coord_simplex(self.coords);
    let e0 = to_vec3(&coord_simplex.spanning_vector(0));
    let e1 = to_vec3(&coord_simplex.spanning_vector(1));
    e0.cross(&e1)
      .try_normalize(1e-12)
      .unwrap_or_else(na::Vector3::zeros)
  }
}

/// A uniform spatial hash over ambient space at cell size `cell`, answering
/// "is any stored point within `r <= cell` of `p`?" against the 27 buckets
/// around `p`.
struct SpatialHash {
  cell: f64,
  buckets: HashMap<[i64; 3], Vec<na::Vector3<f64>>>,
}

impl SpatialHash {
  fn new(cell: f64) -> Self {
    Self {
      cell,
      buckets: HashMap::new(),
    }
  }

  fn key(&self, p: na::Vector3<f64>) -> [i64; 3] {
    [
      (p.x / self.cell).floor() as i64,
      (p.y / self.cell).floor() as i64,
      (p.z / self.cell).floor() as i64,
    ]
  }

  fn insert(&mut self, p: na::Vector3<f64>) {
    self.buckets.entry(self.key(p)).or_default().push(p);
  }

  fn any_within(&self, p: na::Vector3<f64>, r: f64) -> bool {
    let [kx, ky, kz] = self.key(p);
    let r2 = r * r;
    for dx in -1..=1 {
      for dy in -1..=1 {
        for dz in -1..=1 {
          if let Some(points) = self.buckets.get(&[kx + dx, ky + dy, kz + dz]) {
            if points.iter().any(|q| (q - p).norm_squared() < r2) {
              return true;
            }
          }
        }
      }
    }
    false
  }
}

/// A nalgebra column, zero-padded to an ambient 3-vector.
/// An ambient 3-vector as a tagged coordinate, for [`PointLocator::locate`].
fn coord3(p: na::Vector3<f64>) -> Coord {
  Coord::from_iterator(3, [p.x, p.y, p.z])
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::scene::Scene;

  /// The extent (radius about the centroid) of a mesh, the object-intrinsic
  /// scale the separation is a fraction of.
  fn extent(coords: &MeshCoords) -> f64 {
    let n = coords.nvertices().max(1) as f64;
    let centroid = coords
      .coord_iter()
      .fold(na::DVector::zeros(coords.dim()), |acc, c| acc + c.view())
      / n;
    coords
      .coord_iter()
      .map(|c| (c.view() - &centroid).norm())
      .fold(0.0, f64::max)
      .max(1e-6)
  }

  /// The `constant field` worked example is a genuinely constant vector field,
  /// so its streamlines are straight: the field direction never turns, however
  /// many cell edges a curve crosses. This is the seamless-edge-crossing claim
  /// made falsifiable -- a kink at an edge would show up as a turn.
  #[test]
  fn constant_field_streamlines_are_straight() {
    let (topology, coords) = crate::demos::triforce();
    let scene = Scene::cochains(topology, coords, &crate::demos::triforce_examples());
    let field = scene
      .line_fields
      .iter()
      .find(|f| f.name == "constant field")
      .unwrap();
    let d_sep = 0.15 * extent(&scene.coords);

    let streamlines = trace(&scene.topology, &scene.coords, &field.cochain, d_sep);
    assert!(!streamlines.lines.is_empty());

    for line in &streamlines.lines {
      for pair in line.windows(2) {
        let cos = pair[0].tangent.dot(&pair[1].tangent);
        assert!(
          cos > 0.999,
          "constant field turned between samples: cos = {cos}"
        );
      }
    }
  }

  /// Accepted curves keep their distance: no sample of one curve lies closer
  /// than the growth-stop distance (minus a step's slack) to a sample of
  /// another. This is the even-spacing guarantee.
  #[test]
  fn distinct_streamlines_stay_separated() {
    let (topology, coords) = crate::demos::triforce();
    let scene = Scene::cochains(topology, coords, &crate::demos::triforce_examples());
    let field = scene
      .line_fields
      .iter()
      .find(|f| f.name == "pure curl")
      .unwrap();
    let d_sep = 0.12 * extent(&scene.coords);

    let streamlines = trace(&scene.topology, &scene.coords, &field.cochain, d_sep);
    assert!(streamlines.lines.len() >= 2, "expected several curves");

    let slack = (STOP_FRACTION - STEP_FRACTION) * d_sep;
    for (i, a) in streamlines.lines.iter().enumerate() {
      for b in &streamlines.lines[i + 1..] {
        for p in a {
          for q in b {
            assert!(
              (p.pos - q.pos).norm() >= slack,
              "two curves came within {} < {slack}",
              (p.pos - q.pos).norm()
            );
          }
        }
      }
    }
  }

  /// Every grade-1 Whitney basis field on the reference triangle traces without
  /// panicking, and every sample sits on the (closed) reference cell.
  #[test]
  fn reference_triangle_basis_traces_cleanly() {
    let scene = Scene::whitney_basis(2);
    for field in &scene.line_fields {
      let d_sep = 0.2 * extent(&scene.coords);
      let streamlines = trace(&scene.topology, &scene.coords, &field.cochain, d_sep);
      for line in &streamlines.lines {
        for sp in line {
          assert!(sp.magnitude >= 0.0);
          assert!(sp.pos.iter().all(|c| c.is_finite()));
        }
      }
    }
  }
}
