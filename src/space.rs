use crate::{
  mesh::{EntityId, Triangulation},
  Dim,
};

use std::rc::Rc;

pub type DofId = usize;

pub struct FeSpace {
  mesh: Rc<Triangulation>,
}

impl FeSpace {
  pub fn new(mesh: Rc<Triangulation>) -> Self {
    Self { mesh }
  }

  pub fn mesh(&self) -> &Rc<Triangulation> {
    &self.mesh
  }

  /// The number of degrees of freedoms _associated_ with simplicies of the given dimension.
  pub fn ndofs(&self, dim: Dim) -> usize {
    self.mesh.skeletons()[dim].nsimplicies()
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

  ///// Returns the dofs, relevant to the given differential form rank, covering the supplied simplex.
  //pub fn dof_indices_global(&self, simplex: EntityId, form_rank: usize) -> Vec<DofId> {
  //  assert_eq!(form_rank, 0, "Only Lagrangian (0-form) is supported");
  //  let vertex_ids = simplex.subentites_ids(0);
  //  let vertices = simplex.subentities(0);
  //  for vertex in vertices {}
  //}
}
