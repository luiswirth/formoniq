pub mod connection;
pub mod geometry;
pub mod mesh;
pub mod simplex;

pub use connection::Transport;
pub use geometry::CellGramians;

pub type EdgeIdx = usize;
