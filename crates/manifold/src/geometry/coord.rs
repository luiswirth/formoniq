pub mod mesh;
pub mod quadrature;
pub mod simplex;

use common::linalg::nalgebra::{Vector, VectorView};

pub type Coord = Vector;
pub type CoordRef<'a> = VectorView<'a>;

pub type LocalCoord = Coord;
pub type LocalCoordRef<'a> = CoordRef<'a>;

pub type BaryCoord = Coord;
pub type BaryCoordRef<'a> = CoordRef<'a>;

pub type AmbientCoord = Coord;
pub type AmbientCoordRef<'a> = CoordRef<'a>;
