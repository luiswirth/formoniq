use crate::dense::{DifferentialMultiForm, MultiForm, MultiVector};

use geometry::coord::{manifold::SimplexCoords, CoordRef, MeshVertexCoords};
use topology::{
  complex::{attribute::Cochain, TopologyComplex},
  Dim,
};

pub trait CoordSimplexExt {
  fn difbary(&self, i: usize) -> MultiForm;
  fn difbarys(&self) -> Vec<MultiForm>;
  fn spanning_multivector(&self) -> MultiVector;
}
impl CoordSimplexExt for SimplexCoords {
  fn spanning_multivector(&self) -> MultiVector {
    let vectors = self.spanning_vectors();
    let vectors = vectors
      .column_iter()
      .map(|v| MultiVector::from_grade1(v.into_owned()));
    MultiVector::wedge_big(vectors).unwrap_or(MultiVector::one(self.dim_embedded()))
  }

  fn difbary(&self, i: usize) -> MultiForm {
    let gradbary = self.gradbary(i);
    MultiForm::from_grade1(gradbary)
  }

  fn difbarys(&self) -> Vec<MultiForm> {
    let gradbarys = self.gradbarys();
    gradbarys
      .column_iter()
      .map(|g| MultiForm::from_grade1(g.into_owned()))
      .collect()
  }
}

/// Discretize continuous coordinate-based differential k-form into
/// discrete k-cochain on CoordComplex via de Rham map (integration over k-simplex).
pub fn discretize_form_on_mesh(
  form: &impl DifferentialMultiForm,
  topology: &TopologyComplex,
  coords: &MeshVertexCoords,
) -> Cochain<Dim> {
  let cochain = topology
    .skeleton(form.grade())
    .handle_iter()
    .map(|simp| SimplexCoords::from_simplex_and_coords(simp.simplex_set(), coords))
    .map(|simp| discretize_form_on_simplex(form, &simp))
    .collect::<Vec<_>>()
    .into();
  Cochain::new(form.grade(), cochain)
}

/// Approximates the integral of a differential k-form over a k-simplex,
/// by means of a vertex based (trapezoidal?) quadrature rule.
pub fn discretize_form_on_simplex(
  differential_form: &impl DifferentialMultiForm,
  simplex: &SimplexCoords,
) -> f64 {
  let multivector = simplex.spanning_multivector();
  let f = |coord: CoordRef| {
    differential_form
      .at_point(simplex.local_to_global_coord(coord).as_view())
      .on_multivector(&multivector)
  };
  let std_simp = SimplexCoords::standard(simplex.dim_intrinsic());
  barycentric_quadrature(&f, &std_simp)
}

/// Integrates affine linear functions exactly.
pub fn barycentric_quadrature<F>(f: &F, simplex: &SimplexCoords) -> f64
where
  F: Fn(CoordRef) -> f64,
{
  simplex.vol() * f(simplex.barycenter().as_view())
}

pub fn vertex_quadature<F>(f: &F, simplex: &SimplexCoords) -> f64
where
  F: Fn(CoordRef) -> f64,
{
  let sum: f64 = simplex.vertices.coord_iter().map(f).sum();
  simplex.vol() * sum / simplex.nvertices() as f64
}

pub fn edge_midpoint_quadrature<F>(f: &F, simplex: &SimplexCoords) -> f64
where
  F: Fn(CoordRef) -> f64,
{
  if simplex.dim_intrinsic() == 0 {
    return f(simplex.vertices.coord(0).as_view());
  }

  let sum: f64 = simplex
    .edges()
    .map(|edge| edge.barycenter())
    .map(|c| f(c.as_view()))
    .sum();
  simplex.vol() * sum / simplex.nvertices() as f64
}
