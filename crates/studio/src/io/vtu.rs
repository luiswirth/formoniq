//! Writing a simplicial manifold and its cochains as VTK's XML unstructured
//! grid (`.vtu`), the interchange ParaView and PyVista read.
//!
//! The format is a leaf of the extrinsic side: it wants an embedding, a
//! rendered-out vertex list and a field already reduced to a scalar or a
//! vector, so everything the engine keeps intrinsic has to be spent before a
//! file can be written. What it buys is a second, independent renderer for the
//! same data, which makes a disagreement between the viewer and ParaView a
//! visible bug rather than a silent one.
//!
//! **VTU is hard-capped at three dimensions, and the cap is the format's, not
//! this writer's.** Its points are always 3-tuples and its cell zoo stops at
//! the tetrahedron ([`cell_type`]), so a 4-simplex has no faithful encoding.
//! Reducing a manifold or an embedding above three dimensions is therefore a
//! separate, reusable stage upstream, and this module refuses rather than
//! projecting behind the caller's back: a choice of projection is a modelling
//! decision and belongs where it can be stated.
//!
//! **The reduction is shared with the viewer, not reimplemented.** A field goes
//! through the same `reduced_form`/`scalarize` rule the marks draw, so the
//! two consumers cannot drift: $min(k, n-k)$ is the reduced grade, $0$ writes a
//! scalar and $1$ a vector. Under the dimensional cap those two exhaust every
//! grade, which is the same low-dimensional accident that lets classical vector
//! calculus close, so no grade is left without a mark here.
//!
//! A grade-0 cochain is written as point data, where its coefficients already
//! live and where the encoding is exact. Every other grade is cell data,
//! sampled at the cell barycenter: a Whitney $k$-form is not constant on a cell,
//! so one sample per cell is a genuine reduction and the file is a picture of
//! the field rather than the field itself.

use std::io;
use std::path::Path;

use derham::{cochain::Cochain, interpolate::interpolant::WhitneyInterpolant};
use exterior::{ExteriorGrade, MultiForm};
use gramian::Metric;
use simplicial::{
  Dim, Sign,
  atlas::MeshPoint,
  geometry::coord::{mesh::MeshCoords, simplex::SimplexRefExt},
  topology::{complex::Complex, role::Cell},
};

use crate::scene::{reduced_form, scalarize};

/// The largest dimension VTU can encode, intrinsic and ambient alike: its cell
/// zoo stops at the tetrahedron and its points are 3-tuples.
pub const MAX_DIM: usize = 3;

/// A cochain to write, under the name it appears by in ParaView.
pub struct NamedCochain<'a> {
  pub name: &'a str,
  pub cochain: &'a Cochain,
}

impl<'a> NamedCochain<'a> {
  pub fn new(name: &'a str, cochain: &'a Cochain) -> Self {
    Self { name, cochain }
  }
}

/// Why a mesh or a field could not be written as VTU.
#[derive(Debug)]
pub enum VtuError {
  /// The manifold's own dimension exceeds the format's cell zoo. Take a
  /// cross-section first.
  CellDimTooHigh(usize),
  /// The embedding's dimension exceeds the format's 3-tuple points. Project
  /// first.
  AmbientDimTooHigh(usize),
  /// The coordinates or a cochain do not belong to this complex.
  Incompatible(String),
  Io(io::Error),
}

impl std::fmt::Display for VtuError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::CellDimTooHigh(dim) => write!(
        f,
        "cells of dimension {dim} do not fit VTU, whose cell zoo stops at the tetrahedron ({MAX_DIM})"
      ),
      Self::AmbientDimTooHigh(dim) => write!(
        f,
        "an embedding in {dim} dimensions does not fit VTU, whose points are 3-tuples"
      ),
      Self::Incompatible(what) => write!(f, "{what} does not belong to this complex"),
      Self::Io(err) => write!(f, "{err}"),
    }
  }
}

impl std::error::Error for VtuError {}

impl From<io::Error> for VtuError {
  fn from(err: io::Error) -> Self {
    Self::Io(err)
  }
}

/// Writes the mesh and its fields to `path` as VTU.
pub fn write(
  path: impl AsRef<Path>,
  topology: &Complex,
  coords: &MeshCoords,
  fields: &[NamedCochain],
) -> Result<(), VtuError> {
  let xml = to_string(topology, coords, fields)?;
  std::fs::write(path, xml)?;
  Ok(())
}

/// The VTU document for the mesh and its fields, as a string.
///
/// ASCII throughout: a `.vtu` is read by a person as often as by ParaView while
/// a mesh is being debugged, and the format's binary encodings buy size at the
/// cost of that.
pub fn to_string(
  topology: &Complex,
  coords: &MeshCoords,
  fields: &[NamedCochain],
) -> Result<String, VtuError> {
  let n = topology.dim();
  if n.index() > MAX_DIM {
    return Err(VtuError::CellDimTooHigh(n.index()));
  }
  if coords.dim().index() > MAX_DIM {
    return Err(VtuError::AmbientDimTooHigh(coords.dim().index()));
  }
  if !coords.is_compatible_with(topology) {
    return Err(VtuError::Incompatible("the coordinates".into()));
  }
  if let Some(bad) = fields
    .iter()
    .find(|f| !f.cochain.is_compatible_with(topology))
  {
    return Err(VtuError::Incompatible(format!(
      "the cochain `{}`",
      bad.name
    )));
  }

  let nvertices = coords.nvertices();
  let ncells = topology.cells().len();

  let mut xml = String::new();
  xml.push_str("<?xml version=\"1.0\"?>\n");
  xml.push_str("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
  xml.push_str("  <UnstructuredGrid>\n");
  xml.push_str(&format!(
    "    <Piece NumberOfPoints=\"{nvertices}\" NumberOfCells=\"{ncells}\">\n"
  ));

  push_points(&mut xml, coords);
  push_cells(&mut xml, topology);
  push_point_data(&mut xml, fields);
  push_cell_data(&mut xml, topology, coords, fields);

  xml.push_str("    </Piece>\n");
  xml.push_str("  </UnstructuredGrid>\n");
  xml.push_str("</VTKFile>\n");
  Ok(xml)
}

/// The VTK cell type of a $d$-simplex: `VTK_VERTEX`, `VTK_LINE`,
/// `VTK_TRIANGLE`, `VTK_TETRA`, the format's whole simplicial zoo. `None` above
/// three dimensions, where VTU has no simplex at all.
///
/// The corners are written in the [`Skeleton`]'s stored colex order. VTK reads
/// a tetrahedron's order as a winding, so a cell may land mirrored against the
/// manifold's coherent orientation; nothing the format computes from a
/// simplicial mesh depends on it, and reconciling the two would mean pushing
/// the gauge of invariant 6 into an interchange file.
///
/// [`Skeleton`]: simplicial::topology::skeleton::Skeleton
pub fn cell_type(dim: Dim) -> Option<u8> {
  match dim.index() {
    0 => Some(1),
    1 => Some(3),
    2 => Some(5),
    3 => Some(10),
    _ => None,
  }
}

/// The points, each padded from the embedding's own dimension out to the
/// 3-tuple the format fixes. A surface in the plane is the $z = 0$ slice of
/// space, which is the padding read as geometry rather than as a filler.
fn push_points(xml: &mut String, coords: &MeshCoords) {
  xml.push_str("      <Points>\n");
  xml.push_str(
    "        <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n",
  );
  for coord in coords.coord_iter() {
    let mut padded = [0.0; 3];
    for (slot, value) in padded.iter_mut().zip(coord.iter()) {
      *slot = *value;
    }
    xml.push_str(&format!(
      "          {} {} {}\n",
      padded[0], padded[1], padded[2]
    ));
  }
  xml.push_str("        </DataArray>\n");
  xml.push_str("      </Points>\n");
}

fn push_cells(xml: &mut String, topology: &Complex) {
  let vtk_type = cell_type(topology.dim()).expect("the dimensional cap is checked on entry");
  let nvertices_per_cell = topology.dim().index() + 1;

  xml.push_str("      <Cells>\n");
  xml.push_str("        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
  for cell in topology.cells().handle_iter() {
    let corners: Vec<String> = cell
      .simplex()
      .vertices
      .iter()
      .map(ToString::to_string)
      .collect();
    xml.push_str(&format!("          {}\n", corners.join(" ")));
  }
  xml.push_str("        </DataArray>\n");

  xml.push_str("        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
  for icell in 1..=topology.cells().len() {
    xml.push_str(&format!("          {}\n", icell * nvertices_per_cell));
  }
  xml.push_str("        </DataArray>\n");

  xml.push_str("        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for _ in 0..topology.cells().len() {
    xml.push_str(&format!("          {vtk_type}\n"));
  }
  xml.push_str("        </DataArray>\n");
  xml.push_str("      </Cells>\n");
}

/// The grade-0 fields, written where their coefficients already live. A
/// 0-cochain is a function on the vertices and the encoding loses nothing, so
/// this is the one grade that is not sampled.
fn push_point_data(xml: &mut String, fields: &[NamedCochain]) {
  let scalars: Vec<&NamedCochain> = fields
    .iter()
    .filter(|f| f.cochain.grade().index() == 0)
    .collect();
  if scalars.is_empty() {
    return;
  }
  xml.push_str(&format!(
    "      <PointData Scalars=\"{}\">\n",
    escape(scalars[0].name)
  ));
  for field in scalars {
    push_scalar_array(xml, field.name, field.cochain.coeffs().iter().copied());
  }
  xml.push_str("      </PointData>\n");
}

/// Every other grade, sampled once per cell at its barycenter and reduced by
/// the viewer's own rule: reduced grade 0 writes a scalar, reduced grade 1 a
/// vector pushed forward into the ambient frame.
fn push_cell_data(
  xml: &mut String,
  topology: &Complex,
  coords: &MeshCoords,
  fields: &[NamedCochain],
) {
  let n = topology.dim().index();
  let (scalars, vectors): (Vec<_>, Vec<_>) = fields
    .iter()
    .filter(|f| f.cochain.grade().index() != 0)
    .partition(|f| {
      let k = f.cochain.grade().index();
      k.min(n - k) == 0
    });
  if scalars.is_empty() && vectors.is_empty() {
    return;
  }

  let mut attrs = String::new();
  if let Some(first) = scalars.first() {
    attrs.push_str(&format!(" Scalars=\"{}\"", escape(first.name)));
  }
  if let Some(first) = vectors.first() {
    attrs.push_str(&format!(" Vectors=\"{}\"", escape(first.name)));
  }
  xml.push_str(&format!("      <CellData{attrs}>\n"));

  for field in scalars {
    let values = sample_cells(topology, coords, field.cochain, |_, form, metric, sign| {
      scalarize(form, metric, sign)
    });
    push_scalar_array(xml, field.name, values.into_iter());
  }
  for field in vectors {
    push_vector_array(
      xml,
      field.name,
      cell_vectors(topology, coords, field.cochain),
    );
  }

  xml.push_str("      </CellData>\n");
}

/// The sign the top-grade reduction is read with: the cell's coherent
/// orientation where the manifold has one, and `None` otherwise.
///
/// `None` is not a fallback that hides a failure. A signed density is a reading
/// against a global volume form, a non-orientable manifold has none, and the
/// magnitude is what is left that is true (invariant 6): the star is refused
/// rather than taken per cell against each cell's own colex frame, which would
/// paint the indexing convention onto the file.
fn orientation_sign(topology: &Complex, cell: Cell, grade: ExteriorGrade) -> Option<Sign> {
  let n = topology.dim().index();
  let k = grade.index();
  if k <= n - k {
    return None;
  }
  topology
    .orientation()
    .map(|orientation| orientation.sign(cell))
}

/// One sample of the field per cell, at the cell's barycenter, handed to the
/// caller's reduction together with the cell's metric and the sign the star is
/// read against. The one place a cochain is evaluated, so the scalar and the
/// vector mark cannot sample at different points.
fn sample_cells<T>(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  reduce: impl Fn(Cell, MultiForm, &Metric, Option<Sign>) -> T,
) -> Vec<T> {
  let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);
  topology
    .cells()
    .handle_iter()
    .map(|cell| {
      let metric = coords.cell_metric(cell);
      let sign = orientation_sign(topology, cell, cochain.grade());
      let form = interpolant.eval(&MeshPoint::barycenter(cell.idx()));
      reduce(cell, form, &metric, sign)
    })
    .collect()
}

/// The reduced grade-1 field per cell, sharped to a vector and pushed forward
/// into ambient coordinates, padded to the format's 3-tuple. The same
/// composition the glyph mark draws.
fn cell_vectors(topology: &Complex, coords: &MeshCoords, cochain: &Cochain) -> Vec<[f64; 3]> {
  sample_cells(topology, coords, cochain, |cell, form, metric, sign| {
    let field = reduced_form(form, metric, sign.unwrap_or(Sign::Pos)).sharp(metric);
    let ambient = cell
      .coord_simplex(coords)
      .pushforward_vector(field.coeffs());
    let mut padded = [0.0; 3];
    for (slot, value) in padded.iter_mut().zip(ambient.iter()) {
      *slot = *value;
    }
    padded
  })
}

fn push_scalar_array(xml: &mut String, name: &str, values: impl Iterator<Item = f64>) {
  xml.push_str(&format!(
    "        <DataArray type=\"Float64\" Name=\"{}\" NumberOfComponents=\"1\" format=\"ascii\">\n",
    escape(name)
  ));
  for value in values {
    xml.push_str(&format!("          {value}\n"));
  }
  xml.push_str("        </DataArray>\n");
}

fn push_vector_array(xml: &mut String, name: &str, values: Vec<[f64; 3]>) {
  xml.push_str(&format!(
    "        <DataArray type=\"Float64\" Name=\"{}\" NumberOfComponents=\"3\" format=\"ascii\">\n",
    escape(name)
  ));
  for [x, y, z] in values {
    xml.push_str(&format!("          {x} {y} {z}\n"));
  }
  xml.push_str("        </DataArray>\n");
}

/// XML escaping for the one place user text reaches the document, a field's
/// name.
fn escape(text: &str) -> String {
  text
    .replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
  use super::*;
  use simplicial::linalg::Vector;
  use simplicial::mesher::cartesian::CartesianGrid;

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
        let skeleton = topology.skeleton_raw(grade);
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
    let foreign = Cochain::constant(1.0, other.skeleton_raw(0));
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
    let scalar = Cochain::constant(1.0, topology.skeleton_raw(0));
    let vector = Cochain::constant(1.0, topology.skeleton_raw(1));
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
    let cochain = Cochain::constant(1.0, topology.skeleton_raw(0));
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
      let cochain = Cochain::constant(1.0, topology.skeleton_raw(dim));
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
    let nedges = topology.skeleton_raw(1).len();
    let coeffs = Vector::from_iterator(nedges, (0..nedges).map(|i| (i as f64).sin()));
    let cochain = Cochain::new(1, coeffs);
    let written = cell_vectors(&topology, &coords, &cochain);

    let interpolant = WhitneyInterpolant::new(cochain.clone(), &topology);
    for (cell, ambient) in topology.cells().handle_iter().zip(written) {
      let metric = coords.cell_metric(cell);
      let sign = crate::scene::reduction_sign(&topology, cell, cochain.grade());
      let form = interpolant.eval(&MeshPoint::barycenter(cell.idx()));
      let expected = cell
        .coord_simplex(&coords)
        .pushforward_vector(reduced_form(form, &metric, sign).sharp(&metric).coeffs());
      for (icomponent, &value) in ambient.iter().enumerate() {
        let expected = expected.get(icomponent).copied().unwrap_or(0.0);
        assert!((value - expected).abs() < 1e-12);
      }
    }
  }
}
