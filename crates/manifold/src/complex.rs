use crate::{
  simplicial::{OrientedVertplex, SimplexExt, SortedVertplex},
  Dim,
};

use indexmap::IndexMap;

/// A simplicial complex.
#[derive(Debug, Clone)]
pub struct Complex {
  skeletons: Vec<Skeleton>,
}

impl Complex {
  pub fn dim(&self) -> Dim {
    self.skeletons.len() - 1
  }
  pub fn skeletons(&self) -> &[Skeleton] {
    &self.skeletons
  }
  pub fn skeleton(&self, dim: Dim) -> &Skeleton {
    &self.skeletons[dim]
  }
  pub fn vertices(&self) -> &Skeleton {
    self.skeleton(0)
  }
  pub fn edges(&self) -> &Skeleton {
    self.skeleton(1)
  }
  pub fn faces(&self) -> &Skeleton {
    self.skeleton(self.dim() - 1)
  }
  pub fn facets(&self) -> &Skeleton {
    self.skeleton(self.dim())
  }
}

impl Complex {
  pub fn new(facets: Vec<OrientedVertplex>) -> Self {
    let dim = facets[0].dim();

    if cfg!(debug_assertions) {
      for facet in &facets {
        assert!(facet.dim() == dim, "Inconsistent facet dimension.");
      }
    }

    let mut skeletons = vec![Skeleton::new(); dim + 1];
    for (ifacet, facet) in facets.iter().enumerate() {
      let facet = facet.clone().sort();

      for (dim_sub, subs) in skeletons.iter_mut().enumerate() {
        let nvertices_sub = dim_sub + 1;
        for sub in facet.subs(nvertices_sub) {
          let sub = subs.entry(sub.clone()).or_insert(SimplexData::stub());
          sub.parent_facets.push(ifacet);
        }
      }
    }

    // TODO: verify correct!
    // sort vertices
    skeletons[0].sort_by(|v0, _, v1, _| v0[0].cmp(&v1[0]));

    // Topology checks.
    let faces = &skeletons[dim - 1];
    for (face, SimplexData { parent_facets }) in faces {
      let nparents = parent_facets.len();

      let is_manifold = nparents == 2 || nparents == 1;
      assert!(is_manifold, "Topology must be manifold.");

      let is_boundary = nparents == 1;
      if is_boundary {
        continue;
      }

      // The same face, but as seen from the perspective of the adjacent facets.
      let relative_faces = parent_facets
        .iter()
        .map(|&ifacet| {
          facets[ifacet]
            .boundary()
            .find(|b| b.clone().sort() == *face)
            .unwrap()
        })
        .collect::<Vec<_>>();

      // Two facets are consistently oriented, if they have differing orientation as seen
      // from their shared face.
      let consistent_orientation = !relative_faces[0].orientation_eq(&relative_faces[1]);
      assert!(
        consistent_orientation,
        "Facets must be consistently oriented."
      );
    }

    Self { skeletons }
  }
}

/// A container for vertplicies of common dimension.
pub type Skeleton = IndexMap<SortedVertplex, SimplexData>;

/// Topological information of the simplex.
#[derive(Debug, Clone)]
pub struct SimplexData {
  pub parent_facets: Vec<FacetIdx>,
}

impl SimplexData {
  pub fn stub() -> Self {
    let parent_facets = Vec::new();
    Self { parent_facets }
  }
}

pub type KSimplexIdx = usize;
pub type VertexIdx = KSimplexIdx;
pub type EdgeIdx = KSimplexIdx;
pub type FaceIdx = KSimplexIdx;
pub type FacetIdx = KSimplexIdx;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimplexIdx {
  pub dim: Dim,
  pub kidx: KSimplexIdx,
}
impl From<(Dim, KSimplexIdx)> for SimplexIdx {
  fn from((dim, kidx): (Dim, KSimplexIdx)) -> Self {
    Self { dim, kidx }
  }
}
impl SimplexIdx {
  pub fn is_valid(self, mesh: &Complex) -> bool {
    self.dim <= mesh.dim() && self.kidx < mesh.skeleton(self.dim).len()
  }
  pub fn assert_valid(self, mesh: &Complex) {
    assert!(self.is_valid(mesh), "Not a valid simplex index.");
  }
}
