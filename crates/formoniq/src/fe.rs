use crate::{operators::FeFunction, whitney::WhitneyForm};

use exterior::dense::{ExteriorField, MultiForm};
use geometry::{
  coord::{manifold::SimplexHandleExt, CoordRef, MeshVertexCoords},
  metric::MeshEdgeLengths,
};
use topology::complex::{dim::DimInfoProvider, TopologyComplex};

#[allow(unused_variables)]
pub fn l2_norm(fe: &FeFunction, topology: &TopologyComplex, geometry: &MeshEdgeLengths) -> f64 {
  todo!()
}

#[allow(unused_variables)]
pub fn hlambda_norm(
  fe: &FeFunction,
  topology: &TopologyComplex,
  geometry: &MeshEdgeLengths,
) -> f64 {
  todo!()
}

pub fn evaluate_fe_function_at_coord<'a>(
  coord: impl Into<CoordRef<'a>>,
  fe: &FeFunction,
  topology: &TopologyComplex,
  coords: &MeshVertexCoords,
) -> MultiForm {
  let coord = coord.into();

  let dim = coords.dim();
  assert_eq!(coord.len(), dim);
  let grade = fe.dim.dim(topology.dim());

  // Find facet that contains coord.
  // WARN: very slow and inefficent
  let Some(facet) = topology
    .facets()
    .handle_iter()
    .find(|facet| facet.coord_simplex(coords).is_coord_inside(coord))
  else {
    return MultiForm::zero(dim, grade);
  };

  let mut fe_value = MultiForm::zero(topology.dim(), grade);
  for dof_simp in facet.subsimps(grade) {
    let local_dof_simp = facet
      .simplex_set()
      .global_to_local_subsimp(dof_simp.simplex_set());

    let dof_value =
      fe[dof_simp] * WhitneyForm::new(facet.coord_simplex(coords), local_dof_simp).at_point(coord);
    fe_value += dof_value;
  }

  fe_value
}

pub fn evaluate_fe_function_at_facet_barycenters(
  fe: &FeFunction,
  topology: &TopologyComplex,
  coords: &MeshVertexCoords,
) -> Vec<MultiForm> {
  let grade = fe.dim.dim(topology.dim());

  topology
    .facets()
    .handle_iter()
    .map(|facet| {
      let mut value = MultiForm::zero(topology.dim(), grade);
      for dof_simp in facet.subsimps(grade) {
        let local_dof_simp = facet
          .simplex_set()
          .global_to_local_subsimp(dof_simp.simplex_set());

        let barycenter = facet.coord_simplex(coords).barycenter();

        let dof_value = fe[dof_simp]
          * WhitneyForm::new(facet.coord_simplex(coords), local_dof_simp).at_point(&barycenter);
        value += dof_value;
      }
      value
    })
    .collect()
}

pub fn evaluate_fe_function_facets_vertices(
  fe: &FeFunction,
  topology: &TopologyComplex,
  coords: &MeshVertexCoords,
) -> Vec<Vec<MultiForm>> {
  let grade = fe.dim.dim(topology.dim());

  topology
    .facets()
    .handle_iter()
    .map(|facet| {
      facet
        .vertices()
        .map(|vertex| {
          let mut value = MultiForm::zero(topology.dim(), grade);
          for dof_simp in facet.subsimps(grade) {
            let local_dof_simp = facet
              .simplex_set()
              .global_to_local_subsimp(dof_simp.simplex_set());

            let dof_value = fe[dof_simp]
              * WhitneyForm::new(facet.coord_simplex(coords), local_dof_simp)
                .at_point(&coords.coord(vertex.kidx()));
            value += dof_value;
          }
          value
        })
        .collect()
    })
    .collect()
}

pub fn write_evaluations<W: std::io::Write>(
  mut writer: W,
  facet_evals: &[Vec<MultiForm>],
) -> std::io::Result<()> {
  for facet_eval in facet_evals {
    writeln!(writer, "facet")?;
    for vertex_eval in facet_eval {
      for comp in vertex_eval.coeffs() {
        write!(writer, "{comp:.6} ")?;
      }
      writeln!(writer)?;
    }
    writeln!(writer)?;
  }
  Ok(())
}
