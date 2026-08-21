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
//! VTU is hard-capped at three dimensions, and the cap is the format's, not
//! this writer's. Its points are always 3-tuples and its cell zoo stops at
//! the tetrahedron ([`cell_type`]), so a 4-simplex has no faithful encoding.
//! Reducing a manifold or an embedding above three dimensions is therefore a
//! separate, reusable stage upstream, and this module refuses rather than
//! projecting behind the caller's back: a choice of projection is a modeling
//! decision and belongs where it can be stated.
//!
//! The reduction is shared with the viewer, not reimplemented. A field goes
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

use metric::tensor::TensorExt;
use std::io;
use std::path::Path;

use derham::{Cochain, interpolate::interpolant::WhitneyInterpolant};
use metric::Metric;
use multialgebra::Tensor;
use regge::coord::{mesh::MeshCoords, simplex::SimplexRefExt};
use simplicial::{
  Dim, Sign,
  atlas::MeshPoint,
  topology::{complex::Complex, role::Cell},
};

use crate::reduce::{reduced_form, reduction_sign, scalarize};

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
  /// A field whose reduction is a direction, on a mesh carrying no coherent
  /// orientation for the star to fire against.
  NoOrientation(String),
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
      Self::NoOrientation(what) => write!(
        f,
        "{what} reduces to a direction, which a mesh with no coherent orientation does not have"
      ),
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
  // A signed density falls back to its magnitude where no orientation fixes the
  // volume form (invariant 6), but a direction has no such reading, so a field
  // reducing to one is refused rather than starred against each cell's own
  // colex frame, which would paint the indexing convention onto the file.
  if topology.orientation().is_none()
    && let Some(bad) = fields.iter().find(|f| {
      let k = f.cochain.grade().index();
      k > n.index() - k && k.min(n.index() - k) == 1
    })
  {
    return Err(VtuError::NoOrientation(format!(
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

/// One sample of the field per cell, at the cell's barycenter, handed to the
/// caller's reduction together with the cell's metric and the sign the star is
/// read against ([`reduction_sign`], `None` where the star has no coherent
/// orientation to fire against). The one place a cochain is evaluated, so the
/// scalar and the vector mark cannot sample at different points.
fn sample_cells<T>(
  topology: &Complex,
  coords: &MeshCoords,
  cochain: &Cochain,
  reduce: impl Fn(Cell, Tensor, &Metric, Option<Sign>) -> T,
) -> Vec<T> {
  let interpolant = WhitneyInterpolant::new(cochain.clone(), topology);
  topology
    .cells()
    .handle_iter()
    .map(|cell| {
      let metric = coords.cell_metric(cell);
      let sign = reduction_sign(topology, cell, cochain.grade());
      let form = interpolant.eval(&MeshPoint::barycenter(cell.idx()));
      reduce(cell, form, &metric, sign)
    })
    .collect()
}

/// The reduced grade-1 field per cell, sharped to a vector and pushed forward
/// into ambient coordinates, padded to the format's 3-tuple. The same
/// composition the glyph mark draws.
///
/// The sign is required rather than defaulted: a direction has no
/// orientation-free reading the way a magnitude does, so a starred vector field
/// on a non-orientable mesh is refused in [`to_string`] rather than written
/// against each cell's own colex frame.
pub fn cell_vectors(topology: &Complex, coords: &MeshCoords, cochain: &Cochain) -> Vec<[f64; 3]> {
  sample_cells(topology, coords, cochain, |cell, form, metric, sign| {
    let sign = sign.expect("a starred vector field is refused on a non-orientable mesh");
    let field = reduced_form(form, metric, sign).musical(metric);
    let ambient = field
      .pushforward(&cell.coord_simplex(coords).linear_transform())
      .components()
      .clone();
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
