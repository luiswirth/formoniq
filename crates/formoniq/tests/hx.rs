//! Hiptmair-Xu auxiliary-space preconditioning: the vector-nodal
//! prolongation is the de Rham map of $phi_a e_I$, column by column, which
//! is what makes the auxiliary space the vector-nodal one.

use derham::Cochain;
use derham::interpolate::interpolant::WhitneyInterpolant;
use derham::project::derham_map;
use derham::section::{CoordFieldExt, Wedge};
use formoniq::hx::vector_nodal_prolongation;
use glatt::field::DiffFormClosure;
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
