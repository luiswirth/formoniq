//! Assembly, end to end.
//!
//! The only benchmark: what the engine spends its setup in, measured where a
//! solve actually pays it rather than on an operation in isolation.

use divan::Bencher;

use derham::interpolate::form::WhitneyLsf;
use formoniq::{
  assemble::assemble_galmat,
  operators::{HodgeMassElmat, ScalarLumpedMassElmat},
};
use multiindex::Dim;
use simplicial::{
  atlas::Bary, geometry::metric::mesh::MeshLengthsSq, mesher::cartesian::CartesianGrid,
  topology::complex::Complex,
};

fn main() {
  divan::main();
}

/// A refined Kuhn grid: the mesh the examples actually run on.
fn mesh(dim: usize, refinements: u32) -> (Complex, MeshLengthsSq) {
  let (mut topology, coords) = CartesianGrid::new_unit(Dim::from(dim), 1).triangulate();
  let mut metric = coords.to_edge_lengths_sq(&topology);
  let mut ordering = simplicial::topology::ordering::CellOrdering::colex(&topology);
  for _ in 0..refinements {
    let sub = topology.refine_with(&ordering, 2);
    metric = metric.refine(&sub, &topology);
    ordering = sub.ordering().clone();
    topology = sub.into_complex();
  }
  (topology, metric)
}

/// The Hodge mass matrix at every grade: the assembly the solver spends its
/// setup in, and the one that reads the exterior algebra per cell.
#[divan::bench(args = [2, 3])]
fn hodge_mass_assembly(bencher: Bencher, dim: usize) {
  let (topology, metric) = mesh(dim, if dim == 2 { 5 } else { 3 });
  bencher.bench_local(|| {
    let mut total = 0;
    for grade in 0..=dim {
      let galmat = assemble_galmat(&topology, &metric, HodgeMassElmat::new(dim, grade));
      total += divan::black_box(&galmat).nrows();
    }
    total
  });
}

/// The scalar lumped mass: the same assembly machinery with an element matrix
/// touching no exterior algebra. The difference against the Hodge mass is what
/// the algebra costs assembly.
#[divan::bench(args = [2, 3])]
fn lumped_mass_assembly(bencher: Bencher, dim: usize) {
  let (topology, metric) = mesh(dim, if dim == 2 { 5 } else { 3 });
  bencher.bench_local(|| {
    let galmat = assemble_galmat(&topology, &metric, ScalarLumpedMassElmat);
    divan::black_box(&galmat).nrows()
  });
}

/// `WhitneyLsf::at_bary`: one element built per degree of freedom per
/// quadrature point, where the element-level cost lands.
#[divan::bench(args = [2, 3, 4])]
fn whitney_at_bary(bencher: Bencher, dim: usize) {
  let nodes: Vec<Bary> = (0..8)
    .map(|i| {
      let mut weights = vec![1.0 / (dim + 1) as f64; dim + 1];
      weights[i % (dim + 1)] += 0.1;
      Bary::new(nalgebra::DVector::from_vec(weights))
    })
    .collect();
  bencher.bench_local(|| {
    let mut total = 0.0;
    for grade in 0..=dim {
      for dof in multiindex::combinations(dim + 1, grade + 1) {
        let lsf = WhitneyLsf::unit(Dim::from(dim), dof);
        for node in &nodes {
          total += divan::black_box(lsf.at_bary(node)).components().sum();
        }
      }
    }
    total
  });
}

/// The trimmed complex at each polynomial degree, assembling the mass matrix
/// and the exterior derivative over every grade.
///
/// The mesh is held fixed, so the growth is in the dof count and in the density
/// of the element matrices, not in the mesh.
#[divan::bench(args = [1, 2, 3])]
fn trimmed_assembly(bencher: Bencher, degree: usize) {
  use formoniq::{trimmed_complex::TrimmedComplex, whitney_complex::HilbertComplex};
  let (topology, metric) = mesh(3, 2);
  bencher.bench_local(|| {
    let complex = TrimmedComplex::new(&topology, &metric, degree);
    let mut total = 0;
    for grade in 0..=3 {
      total += HilbertComplex::mass(&complex, grade).nrows();
      total += HilbertComplex::dif(&complex, grade).nrows();
    }
    total
  });
}

/// Building the space itself: the geometric decomposition, the reference
/// exterior derivative and the local-to-global map.
///
/// Separate from assembly because it is per-space rather than per-matrix, and
/// because the reference exterior derivative costs a dense solve.
#[divan::bench(args = [1, 2, 3])]
fn trimmed_setup(bencher: Bencher, degree: usize) {
  use formoniq::trimmed_complex::TrimmedComplex;
  let (topology, metric) = mesh(3, 2);
  bencher.bench_local(|| divan::black_box(TrimmedComplex::new(&topology, &metric, degree)).dim());
}
