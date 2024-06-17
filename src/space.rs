use crate::mesh::{EntityId, Triangulation};

use std::rc::Rc;

pub type DofId = usize;

pub struct DofHandler {
  mesh: Rc<Triangulation>,
}

impl DofHandler {
  pub fn new(mesh: Rc<Triangulation>) -> Self {
    Self { mesh }
  }

  pub fn dof2simplex(&self, mut idof: DofId) -> EntityId {
    for k in 0..self.mesh.dim_intrinsic() {
      let simps = self.mesh.skeletons()[k].simplicies();
      if idof < simps.len() {
        return (k, idof);
      } else {
        idof -= simps.len();
      }
    }
    panic!("no simplex found for this dof");
  }
  pub fn simplex2dof(&self, simplex: EntityId) -> DofId {
    let mut idof = 0;
    for k in 0..simplex.0 {
      let simps = self.mesh.skeletons()[k].simplicies();
      idof += simps.len();
    }
    idof
  }
}
