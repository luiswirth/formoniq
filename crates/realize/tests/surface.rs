//! Laws for [`realize::surface::Surface`]: the reduction is the identity at
//! and below the render primitive's own dimension, a solid reduces to a
//! proper boundary manifold of one fewer dimension, the trace restricts each
//! coefficient to its parent's, and the top grade has no trace at all.

use derham::Cochain;
use realize::surface::Surface;
use regge::mesher::{cartesian::CartesianGrid, sphere::mesh_sphere_surface};
use std::borrow::Cow;

/// The reduction is the identity at and below the render primitive's own
/// dimension: a surface is its own render surface, and nothing is copied or
/// traced. This is the base case invariant "total on the degenerate boundary"
/// asks for, the same code, returning the trivial answer.
#[test]
fn a_surface_is_its_own_render_surface() {
  let (topology, coords) = mesh_sphere_surface(2);
  let surface = Surface::of(&topology, &coords);

  assert!(std::ptr::eq(surface.complex(&topology), &topology));
  assert!(std::ptr::eq(surface.coords(&coords), &coords));
  assert_eq!(surface.dim(&topology), topology.dim());
  assert!(surface.vertex_to_parent().is_none());

  let cochain = Cochain::constant(1.0, topology.skeleton(1));
  let traced = surface.trace(&topology, &cochain).expect("a 1-form traces");
  assert!(
    matches!(traced, Cow::Borrowed(_)),
    "the identity must borrow"
  );
}

/// A solid reduces to its boundary, and the boundary is a proper manifold
/// one dimension down: it has its own complex of the right dimension, its
/// own vertices, and strictly fewer of them than the solid.
#[test]
fn a_solid_reduces_to_its_boundary_manifold() {
  let (topology, coords) = CartesianGrid::new_unit(3, 2).triangulate();
  let surface = Surface::of(&topology, &coords);

  assert_eq!(topology.dim(), 3);
  assert_eq!(surface.dim(&topology), 2, "the boundary is a 2-manifold");

  let to_parent = surface.vertex_to_parent().expect("a cube has a boundary");
  assert_eq!(surface.coords(&coords).nvertices(), to_parent.len());
  assert!(
    to_parent.len() < topology.skeleton(0).len(),
    "a cube has interior vertices the boundary does not"
  );
}

/// The trace is the restriction of coefficients: each boundary simplex
/// carries exactly its parent's value. Stated on a field that distinguishes
/// every simplex, so an index permutation cannot pass.
#[test]
fn the_trace_restricts_each_coefficient_to_its_parent() {
  let (topology, coords) = CartesianGrid::new_unit(3, 2).triangulate();
  let surface = Surface::of(&topology, &coords);

  for grade in surface.dim(&topology).range_inclusive() {
    let cochain = Cochain::from_function(|s| s.kidx() as f64, grade, &topology);
    let traced = surface.trace(&topology, &cochain).expect("grade traces");

    let boundary = topology.boundary_complex().expect("a cube has a boundary");
    let to_parent = boundary.parent_kidxs(grade);
    assert_eq!(traced.len(), to_parent.len());
    for (boundary_kidx, &parent_kidx) in to_parent.iter().enumerate() {
      assert_eq!(traced.coeffs()[boundary_kidx], parent_kidx as f64);
    }
  }
}

/// The top grade does not trace: $C^n (diff M) = 0$, so an $n$-form has no
/// surface representative at all. The predicate must say so rather than hand
/// back a zero cochain that a mark would draw as a vanishing field.
#[test]
fn the_top_grade_has_no_trace() {
  let (topology, coords) = CartesianGrid::new_unit(3, 2).triangulate();
  let surface = Surface::of(&topology, &coords);

  for grade in topology.dim().range() {
    assert!(surface.traces(&topology, grade), "grade {grade} traces");
  }
  assert!(!surface.traces(&topology, topology.dim()));

  let volume = Cochain::constant(1.0, topology.skeleton(3));
  assert!(surface.trace(&topology, &volume).is_none());
}
