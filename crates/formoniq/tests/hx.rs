//! Hiptmair-Xu auxiliary-space preconditioning: the vector-nodal
//! prolongation matches the tested de Rham map column by column, and
//! HX-preconditioned CG (direct and multigrid auxiliary solves alike)
//! reaches the same fixed point as the direct solve.

use derham::Cochain;
use derham::interpolate::interpolant::WhitneyInterpolant;
use derham::project::derham_map;
use derham::section::{CoordFieldExt, Wedge};
use formoniq::hx::{GradeKHodgeHx, vector_nodal_prolongation};
use formoniq::linalg::DirectInverse;
use formoniq::multigrid::RefinementTower;
use formoniq::whitney_complex::WhitneyComplex;
use glatt::field::DiffFormClosure;
use iterative::ApproxInverse;
use multialgebra::{Tensor, exterior_dim};
use regge::mesher::cartesian::CartesianGrid;
use simplicial::linalg::Vector;

fn unit(n: usize, i: usize) -> Vector {
  let mut v = Vector::zeros(n);
  v[i] = 1.0;
  v
}

/// Column $(a, I)$ of $Pi_"vec"$ is the de Rham map of $phi_a e_I$: the
/// assembly is validated against the tested [`derham_map`], the primitive it
/// stands in for. Swept over dimension and grade so it is one statement, not a
/// fixed case.
#[test]
fn vector_nodal_prolongation_is_the_derham_map() {
  for dim in 1..=3 {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let nvertices = coords.nvertices();
    for grade in 1..=dim {
      let pi = vector_nodal_prolongation(&topology, &coords, grade);
      let ncovectors = exterior_dim(dim, grade);
      for covector in 0..ncovectors {
        // The constant ambient basis k-covector e_I, pulled onto the mesh.
        let e_i = DiffFormClosure::new(
          move |_| Tensor::multiform(unit(ncovectors, covector), dim, grade),
          dim,
          grade,
        );
        for a in 0..nvertices {
          let pulled = e_i.pullback_on(&topology, &coords);
          let hat = WhitneyInterpolant::new(Cochain::new(0, unit(nvertices, a)), &topology);
          let reference = derham_map(&Wedge::new(hat, pulled), &topology, 2);
          let assembled = &pi * unit(pi.ncols(), covector * nvertices + a);
          let err = (reference.coeffs() - &assembled).norm();
          assert!(
            err < 1e-9,
            "dim {dim} grade {grade} covector {covector} vertex {a}: \
             Pi_vec column disagrees with derham map, err {err}"
          );
        }
      }
    }
  }
}

/// The multigrid-block preconditioner reaches the same fixed point as the
/// direct-block one: swapping a direct auxiliary solve for a V-cycle changes the
/// solve path and its cost, never the operator or its solution. Swept over the
/// grades of a 2D and a 3D refinement tower.
#[test]
fn hx_multigrid_matches_the_direct_solve() {
  use iterative::StopCriterion;
  for dim in 2..=3 {
    let (base_topology, base_coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let base_geometry = base_coords.to_edge_lengths_sq(&base_topology);
    let tower = RefinementTower::new(base_topology, base_geometry, 2);
    let coords = tower
      .subdivisions()
      .iter()
      .fold(base_coords, |c, sub| c.refine(sub));
    for grade in 1..=dim {
      let hx = GradeKHodgeHx::with_multigrid(&tower, &coords, grade, 2);
      let n = hx.operator().nrows();
      let rhs = Vector::from_fn(n, |i, _| ((i * i + 1) as f64).cos());
      let (x, report) = hx.solve(&rhs, StopCriterion::rtol(1e-10));
      assert!(
        report.converged,
        "dim {dim} grade {grade}: HX-MG-CG did not converge"
      );
      let direct = DirectInverse::try_new(hx.operator().clone()).unwrap();
      let err = (&x - direct.apply(&rhs)).norm();
      assert!(
        err < 1e-7,
        "dim {dim} grade {grade}: HX-MG-CG disagrees with direct, err {err}"
      );
    }
  }
}

/// HX-preconditioned CG reaches the same solution as the direct solve of the
/// same grade-$k$ system: the preconditioner changes the path, not the fixed
/// point. Swept over the grades of a 2D and a 3D mesh.
#[test]
fn hx_cg_matches_the_direct_solve() {
  use iterative::StopCriterion;
  for dim in 2..=3 {
    let (topology, coords) = CartesianGrid::new_unit(dim, 3).triangulate();
    let geometry = coords.to_edge_lengths_sq(&topology);
    let complex = WhitneyComplex::new(&topology, &geometry);
    for grade in 1..=dim {
      let hx = GradeKHodgeHx::new(&complex, &coords, grade);
      let n = hx.operator().nrows();
      let rhs = Vector::from_fn(n, |i, _| ((i * i + 1) as f64).cos());
      let (x, report) = hx.solve(&rhs, StopCriterion::rtol(1e-10));
      assert!(
        report.converged,
        "dim {dim} grade {grade}: HX-CG did not converge"
      );
      let direct = DirectInverse::try_new(hx.operator().clone()).unwrap();
      let err = (&x - direct.apply(&rhs)).norm();
      assert!(
        err < 1e-7,
        "dim {dim} grade {grade}: HX-CG disagrees with direct, err {err}"
      );
    }
  }
}
