pub mod mesh;
pub mod quadrature;
pub mod simplex;

use common::linalg::nalgebra::{RowVector, RowVectorView, Vector, VectorView};

pub type Coord = Vector;
pub type CoordRef<'a> = VectorView<'a>;

pub type TangentVector = Vector;
pub type TangentVectorRef<'a> = VectorView<'a>;

pub type CoTangentVector = RowVector;
pub type CoTangentVectorRef<'a> = RowVectorView<'a>;
