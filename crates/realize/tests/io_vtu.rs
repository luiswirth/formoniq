//! Laws for [`realize::io::vtu`]: the document is the mesh it declares
//! itself to be, every point is a 3-tuple with the embedding padded rather
//! than reinterpreted, a 0-cochain is written verbatim as point data, every
//! grade reduces to a scalar or a vector, a mesh above three dimensions or a
//! foreign cochain is refused, the document is balanced XML with escaped
//! field names, and the cell-vector reduction agrees with the viewer's own.

use derham::{Cochain, interpolate::interpolant::WhitneyInterpolant};
use metric::tensor::TensorExt;
use realize::{
  io::vtu::{MAX_DIM, NamedCochain, VtuError, cell_type, cell_vectors, to_string},
  reduce::{admitted_reduction_sign, reduced_form},
};
use regge::{coord::simplex::SimplexRefExt, mesher::cartesian::CartesianGrid};
use simplicial::{atlas::MeshPoint, linalg::Vector};

/// The named `DataArray`'s numbers, in document order.
fn data_array(xml: &str, name: &str) -> Vec<f64> {
  let opening = format!("Name=\"{name}\"");
  let start = xml.find(&opening).expect("no such DataArray");
  let body_start = start + xml[start..].find(">\n").expect("unterminated tag") + 2;
  let body_end = body_start + xml[body_start..].find("</DataArray>").expect("unclosed");
  xml[body_start..body_end]
    .split_whitespace()
    .map(|token| token.parse().expect("non-numeric data"))
    .collect()
}

fn attribute(xml: &str, name: &str) -> String {
  let key = format!("{name}=\"");
  let start = xml.find(&key).expect("no such attribute") + key.len();
  let end = start + xml[start..].find('"').unwrap();
  xml[start..end].to_string()
}

/// The document is a mesh: as many points and cells as it declares, every
/// cell of the declared type, its corners in range, and the offsets the
/// running ends of a uniform simplicial connectivity.
#[test]
fn the_grid_is_written_as_the_mesh_it_is() {
  for dim in 1..=MAX_DIM {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let xml = to_string(&topology, &coords, &[]).unwrap();

    let npoints: usize = attribute(&xml, "NumberOfPoints").parse().unwrap();
    let ncells: usize = attribute(&xml, "NumberOfCells").parse().unwrap();
    assert_eq!(npoints, coords.nvertices());
    assert_eq!(ncells, topology.cells().len());

    let connectivity = data_array(&xml, "connectivity");
    let offsets = data_array(&xml, "offsets");
    let types = data_array(&xml, "types");
    assert_eq!(connectivity.len(), ncells * (dim + 1));
    assert!(connectivity.iter().all(|&v| (v as usize) < npoints));
    assert_eq!(offsets.len(), ncells);
    assert_eq!(types.len(), ncells);
    let expected = f64::from(cell_type(topology.dim()).unwrap());
    assert!(types.iter().all(|&t| t == expected));
    for (icell, &offset) in offsets.iter().enumerate() {
      assert_eq!(offset as usize, (icell + 1) * (dim + 1));
    }
  }
}

/// Every point is a 3-tuple whatever the embedding's own dimension, the
/// coordinates padded out rather than reinterpreted: the leading components
/// are the embedding's and the rest are the zero slice it sits in.
#[test]
fn the_points_are_the_embedding_padded_to_three() {
  for dim in 1..=MAX_DIM {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let xml = to_string(&topology, &coords, &[]).unwrap();
    let values = data_array(&xml, "Points");
    assert_eq!(values.len(), 3 * coords.nvertices());
    for (ivertex, chunk) in values.chunks(3).enumerate() {
      for (icomponent, &value) in chunk.iter().enumerate() {
        let expected = if icomponent < dim {
          coords.coord(ivertex)[icomponent]
        } else {
          0.0
        };
        assert_eq!(value, expected);
      }
    }
  }
}

/// A 0-cochain is a function on the vertices, so it is written verbatim as
/// point data: the one grade the file loses nothing of.
#[test]
fn a_zero_cochain_is_point_data_verbatim() {
  for dim in 1..=MAX_DIM {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let nvertices = coords.nvertices();
    let coeffs = Vector::from_iterator(
      nvertices,
      (0..nvertices).map(|i| coords.coord(i).iter().sum()),
    );
    let cochain = Cochain::new(0, coeffs);
    let xml = to_string(
      &topology,
      &coords,
      &[NamedCochain::new("potential", &cochain)],
    )
    .unwrap();

    assert!(xml.contains("<PointData"));
    assert!(!xml.contains("<CellData"));
    let written = data_array(&xml, "potential");
    assert_eq!(written.len(), coords.nvertices());
    for (written, expected) in written.iter().zip(cochain.coeffs().iter()) {
      assert_eq!(written, expected);
    }
  }
}

/// The reduced grade $min(k, n-k)$ decides the mark, exactly as it does in the
/// viewer: $0$ writes one number per cell and $1$ writes three. Under the
/// dimensional cap those exhaust every grade, so the sweep leaves no grade
/// unwritten.
#[test]
fn every_grade_reduces_to_a_scalar_or_a_vector() {
  for dim in 1..=MAX_DIM {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let ncells = topology.cells().len();
    for grade in 1..=dim {
      let skeleton = topology.skeleton(grade);
      let cochain = Cochain::constant(1.0, skeleton);
      let xml = to_string(&topology, &coords, &[NamedCochain::new("field", &cochain)]).unwrap();
      let written = data_array(&xml, "field");
      let components = if grade.min(dim - grade) == 0 { 1 } else { 3 };
      assert_eq!(
        written.len(),
        components * ncells,
        "dim {dim}, grade {grade}"
      );
      assert!(xml.contains("<CellData"), "dim {dim}, grade {grade}");
    }
  }
}

/// The cap is the format's and the writer refuses rather than projecting: a
/// 4-simplex has no VTU cell, and a 4-dimensional embedding no VTU point.
#[test]
fn a_mesh_above_three_dimensions_is_refused() {
  let (topology, coords) = CartesianGrid::new_unit(4, 1).triangulate();
  assert!(matches!(
    to_string(&topology, &coords, &[]),
    Err(VtuError::CellDimTooHigh(4))
  ));

  let (topology, coords) = CartesianGrid::new_unit(2, 1).triangulate();
  let lifted = coords.embed_euclidean(4);
  assert!(matches!(
    to_string(&topology, &lifted, &[]),
    Err(VtuError::AmbientDimTooHigh(4))
  ));
}

/// A cochain that is not this complex's is refused rather than silently
/// written against a mismatched index.
#[test]
fn a_foreign_cochain_is_refused() {
  let (topology, coords) = CartesianGrid::new_unit(2, 2).triangulate();
  let (other, _) = CartesianGrid::new_unit(2, 3).triangulate();
  let foreign = Cochain::constant(1.0, other.skeleton(0));
  assert!(matches!(
    to_string(
      &topology,
      &coords,
      &[NamedCochain::new("foreign", &foreign)]
    ),
    Err(VtuError::Incompatible(_))
  ));
}

/// Every tag the document opens it closes, in order: the file a reader gets
/// is well-formed XML rather than one that happens to parse today.
#[test]
fn the_document_is_balanced_xml() {
  let (topology, coords) = CartesianGrid::new_unit(3, 2).triangulate();
  let scalar = Cochain::constant(1.0, topology.skeleton(0));
  let vector = Cochain::constant(1.0, topology.skeleton(1));
  let xml = to_string(
    &topology,
    &coords,
    &[
      NamedCochain::new("scalar", &scalar),
      NamedCochain::new("vector", &vector),
    ],
  )
  .unwrap();

  let mut stack: Vec<&str> = Vec::new();
  let mut rest = xml.as_str();
  while let Some(open) = rest.find('<') {
    rest = &rest[open + 1..];
    let close = rest.find('>').expect("unterminated tag");
    let tag = &rest[..close];
    rest = &rest[close + 1..];
    if tag.starts_with('?') || tag.ends_with('/') {
      continue;
    }
    if let Some(name) = tag.strip_prefix('/') {
      assert_eq!(stack.pop(), Some(name), "mismatched closing tag");
    } else {
      stack.push(tag.split_whitespace().next().unwrap());
    }
  }
  assert!(stack.is_empty(), "unclosed tags: {stack:?}");
}

/// A field name is not trusted to be XML: the escape is what keeps a stray
/// quote or angle bracket from ending the attribute it sits in.
#[test]
fn a_field_name_is_escaped() {
  let (topology, coords) = CartesianGrid::new_unit(2, 1).triangulate();
  let cochain = Cochain::constant(1.0, topology.skeleton(0));
  let xml = to_string(
    &topology,
    &coords,
    &[NamedCochain::new("a\"<&>b", &cochain)],
  )
  .unwrap();
  assert!(xml.contains("Name=\"a&quot;&lt;&amp;&gt;b\""));
  assert!(!xml.contains("a\"<&>b"));
}

/// The one number a top-grade cochain means: a constant density $c \/ vol_g$
/// on each cell, up to the coherent orientation the star is read against.
/// Nothing about the writer's sampling is in play here, which is what makes
/// it a check on the reduction rather than on the plumbing.
#[test]
fn a_top_cochain_writes_its_density() {
  for dim in 1..=MAX_DIM {
    let (topology, coords) = CartesianGrid::new_unit(dim, 2).triangulate();
    let lengths = coords.to_edge_lengths_sq(&topology);
    let cochain = Cochain::constant(1.0, topology.skeleton(dim));
    let xml = to_string(
      &topology,
      &coords,
      &[NamedCochain::new("density", &cochain)],
    )
    .unwrap();
    let written = data_array(&xml, "density");

    let orientation = topology.orientation().expect("a grid is orientable");
    for (cell, &value) in topology.cells().handle_iter().zip(written.iter()) {
      let volume = lengths.simplex_volume(*cell);
      let expected = orientation.sign(cell).as_f64() / volume;
      assert!(
        (value - expected).abs() < 1e-9,
        "dim {dim}: {value} vs {expected}"
      );
    }
  }
}

/// The vector reduction is the glyph mark's, not a second one: the same
/// composition (reduce, sharp, push forward) evaluated at the same point has
/// to give the same ambient vector, because a discrepancy between the viewer
/// and ParaView is exactly what the exporter exists to expose.
#[test]
fn the_vector_reduction_agrees_with_the_viewer() {
  let (topology, coords) = CartesianGrid::new_unit(3, 2).triangulate();
  let nedges = topology.skeleton(1).len();
  let coeffs = Vector::from_iterator(nedges, (0..nedges).map(|i| (i as f64).sin()));
  let cochain = Cochain::new(1, coeffs);
  let written = cell_vectors(&topology, &coords, &cochain);

  let interpolant = WhitneyInterpolant::new(cochain.clone(), &topology);
  for (cell, ambient) in topology.cells().handle_iter().zip(written) {
    let metric = coords.cell_metric(cell);
    let sign = admitted_reduction_sign(&topology, cell, cochain.grade());
    let form = interpolant.eval(&MeshPoint::barycenter(cell.idx()));
    let expected = cell.coord_simplex(&coords).pushforward_vector(
      reduced_form(form, &metric, sign)
        .musical(&metric)
        .components(),
    );
    for (icomponent, &value) in ambient.iter().enumerate() {
      let expected = expected.get(icomponent).copied().unwrap_or(0.0);
      assert!((value - expected).abs() < 1e-12);
    }
  }
}
