//! Render a Maxwell cavity solution as a self-contained interactive HTML page.
//!
//! A 2D perfect-electric-conductor cavity $[0, pi]^2$ is kicked by a localized
//! rotational (curl-carrying) electric pulse. In the 3+1 split the magnetic
//! flux $B$ is a *scalar* 2-form -- the out-of-plane field of the iconic FDTD
//! picture -- evolved by the structure-preserving leapfrog scheme and sampled
//! on a regular grid at each frame via the Whitney reconstruction (the FEEC
//! representation formula), accelerated by the BVH point locator.
//!
//! The frames are quantized, base64-encoded and injected into an HTML template
//! ([`maxwell_cavity_template.html`]), producing `out/maxwell/cavity.html`: a
//! single self-contained file that animates the field in any browser.

extern crate nalgebra as na;

use common::linalg::nalgebra::Vector;
use ddf::{cochain::Cochain, derham::derham_map, whitney::form::WhitneyForm};
use exterior::field::{DiffFormClosure, ExteriorField};
use formoniq::{
  problems::maxwell::{solve_maxwell_leapfrog, MaxwellState, Medium},
  whitney_complex::WhitneyComplex,
};
use manifold::{gen::cartesian::CartesianMeshInfo, geometry::coord::locate::PointLocator};

use rayon::prelude::*;
use std::{f64::consts::PI, fs};

const TEMPLATE: &str = include_str!("maxwell_cavity_template.html");

fn main() {
  let dim = 2;
  let nboxes = 64;
  let grid = 96; // sampling resolution per axis
  let nframes = 80;
  let keep_every = 6;
  let medium = Medium::vacuum();
  let length = PI;

  // The PEC cavity mesh.
  let box_mesh = CartesianMeshInfo::new_unit_scaled(dim, nboxes, length);
  let (topology, coords) = box_mesh.compute_coord_complex();
  let metric = coords.to_edge_lengths(&topology);
  let fes = WhitneyComplex::new(&topology, &metric);

  // Initial electric field: a localized rotational pulse, off-center to break
  // the cavity symmetry. Its curl seeds the magnetic field, which then
  // radiates and reflects off the walls.
  let center = na::dvector![0.42 * length, 0.5 * length];
  let sigma = 0.16 * length;
  let e_field = DiffFormClosure::one_form(
    move |p| {
      let d = na::dvector![p[0] - center[0], p[1] - center[1]];
      let envelope = (-d.norm_squared() / (2.0 * sigma * sigma)).exp();
      na::dvector![-d[1], d[0]] * envelope
    },
    dim,
  );
  let e0 = derham_map(&e_field, &topology, &coords, 3);
  let b0 = Cochain::new(2, Vector::zeros(fes.ndofs(2)));
  let initial = MaxwellState::new(e0, b0);

  // Time stepping within the CFL limit; keep every k-th step as a frame.
  let dt = 0.2 * metric.mesh_width_min() / medium.wave_speed();
  let nsteps = nframes * keep_every;
  let times: Vec<f64> = (0..=nsteps).map(|i| dt * i as f64).collect();
  println!(
    "Cavity {nboxes}^{dim}: {} edges, {} faces. dt={dt:.4}, {nsteps} steps.",
    fes.ndofs(1),
    fes.ndofs(2)
  );

  let current = Vector::zeros(fes.ndofs(1));
  let solution = solve_maxwell_leapfrog(fes, medium, &times, initial, &current);

  // One point locator, reused across every frame and grid point.
  let locator = PointLocator::new(&topology, &coords);

  // Sample the reconstructed scalar B on a cell-centered grid, frames in
  // parallel. Row-major [ny][nx], physical +y up.
  println!("Sampling {nframes} frames on a {grid}x{grid} grid...");
  let cell_center = |i: usize| (i as f64 + 0.5) / grid as f64 * length;
  let frames: Vec<Vec<f64>> = solution
    .par_iter()
    .step_by(keep_every)
    .take(nframes)
    .map(|state| {
      let b_form = WhitneyForm::new(state.b.clone(), &topology, &coords).with_locator(&locator);
      let mut frame = vec![0.0; grid * grid];
      for jy in 0..grid {
        let y = cell_center(jy);
        for ix in 0..grid {
          let point = na::dvector![cell_center(ix), y];
          frame[jy * grid + ix] = b_form.at_point(&point).coeffs()[0];
        }
      }
      frame
    })
    .collect();

  // Quantize to signed bytes against the global peak amplitude.
  let vmax = frames
    .iter()
    .flat_map(|f| f.iter())
    .fold(0.0f64, |m, &v| m.max(v.abs()))
    .max(1e-30);
  let bytes: Vec<u8> = frames
    .iter()
    .flat_map(|f| f.iter())
    .map(|&v| (v / vmax * 127.0).round().clamp(-127.0, 127.0) as i8 as u8)
    .collect();

  let meta = format!(
    "{{ nx: {grid}, ny: {grid}, nframes: {}, dtFrame: {}, vmax: {vmax} }}",
    frames.len(),
    dt * keep_every as f64
  );
  let html = TEMPLATE
    .replace("__META_JSON__", &meta)
    .replace("__DATA_B64__", &base64_encode(&bytes));

  let dir = "out/maxwell";
  fs::create_dir_all(dir).unwrap();
  let path = format!("{dir}/cavity.html");
  fs::write(&path, html).unwrap();
  println!(
    "Wrote {path} ({} frames, {:.0} KB, peak |B| = {vmax:.4}).",
    frames.len(),
    fs::metadata(&path).unwrap().len() as f64 / 1024.0
  );
}

/// Minimal standard base64 encoder (no external dependency).
fn base64_encode(bytes: &[u8]) -> String {
  const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
  for chunk in bytes.chunks(3) {
    let b = |i: usize| *chunk.get(i).unwrap_or(&0) as u32;
    let n = (b(0) << 16) | (b(1) << 8) | b(2);
    out.push(T[(n >> 18 & 63) as usize] as char);
    out.push(T[(n >> 12 & 63) as usize] as char);
    out.push(if chunk.len() > 1 {
      T[(n >> 6 & 63) as usize] as char
    } else {
      '='
    });
    out.push(if chunk.len() > 2 {
      T[(n & 63) as usize] as char
    } else {
      '='
    });
  }
  out
}
