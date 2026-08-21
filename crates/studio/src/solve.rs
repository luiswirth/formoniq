//! The one expensive thing the viewer does, expressed as a value.
//!
//! A build is a request (a mesh and a study) and an outcome (the fields
//! that study produced on it), not a closure. That shape is forced by the web:
//! the work has to cross a `postMessage` boundary into a worker, and a closure
//! cannot cross one. Native pays nothing for it: a thread runs
//! [`SolveRequest::run`] and sends the outcome back.
//!
//! The request carries the mesh itself, not a descriptor of it. A descriptor
//! would be smaller, and would work for every mesh the gallery can regenerate,
//! and would silently exclude the one case that matters most, a mesh the
//! reader loaded themselves, which exists nowhere but in memory and is exactly
//! the mesh whose size nobody has bounded. Sending the data is what makes this
//! general.
//!
//! The outcome carries only the fields. The caller already has the mesh (it
//! sent it) and reassembles the [`Scene`] around them, so the topology and the
//! coordinates never make the return trip.
//!
//! Nothing here is browser-specific. The transport is one small trait's
//! worth of behavior, and its two implementations are a thread ([`native`])
//! and a worker ([`crate::web::worker`]), the latter living with the rest of
//! the browser layer, not here. This module knows only that an outcome arrives
//! later.

use crate::gallery::{Mesh, Study};
use crate::scene::{LineField, ScalarField, Scene};

use simplicial::topology::{complex::Complex, skeleton::Skeleton};

use regge::coord::mesh::MeshCoords;

/// A build to run: a mesh, and the study to run on it.
///
/// The mesh travels as its top skeleton and its coordinates, which is exactly
/// what [`Complex::save`] writes, every other skeleton and every cached
/// operator is `from_cells`'s job to rederive, and a `Complex` is deliberately
/// not serializable for that reason. Rebuilding gives back the same canonical
/// colex order it left in, so the cochain indices in the outcome line up with
/// the caller's own mesh.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SolveRequest {
  cells: Skeleton,
  coords: MeshCoords,
  study: Study,
}

/// What a build produces: the fields, split by the render mark their reduced
/// grade chose. Not a [`Scene`], that would send the mesh back to the caller
/// that supplied it.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct SolveOutcome {
  pub fields: Vec<ScalarField>,
  pub line_fields: Vec<LineField>,
}

impl SolveRequest {
  pub fn new(mesh: &Mesh, study: Study) -> Self {
    Self {
      cells: mesh.0.skeleton_raw(mesh.0.dim()).clone(),
      coords: mesh.1.clone(),
      study,
    }
  }

  /// Runs the study. The expensive call, wherever it happens to be running:
  /// this is the same function on a native thread and inside a web worker.
  pub fn run(&self) -> SolveOutcome {
    let mesh: Mesh = (Complex::from_cells(self.cells.clone()), self.coords.clone());
    let scene = self.study.build(&mesh);
    SolveOutcome {
      fields: scene.fields,
      line_fields: scene.line_fields,
    }
  }
}

impl SolveOutcome {
  /// The scene this outcome and the mesh it was solved on make together.
  pub fn into_scene(self, mesh: &Mesh) -> Scene {
    let mut scene = Scene::on(mesh.0.clone(), mesh.1.clone());
    scene.fields = self.fields;
    scene.line_fields = self.line_fields;
    scene
  }
}

/// CBOR, matching what the engine's own `save`/`load` use, so a payload here is
/// the same encoding as a mesh on disk rather than a second format to reason
/// about. A failure is a bug in this crate (the types are all plain data), not
/// a condition a caller can act on, so it panics rather than widening every
/// signature with an error nobody can handle.
///
/// Public rather than `test`-gated: the web worker crosses it, and the
/// round-trip tests, living outside the crate as ordinary integration tests,
/// reach it the same way any other consumer would.
pub fn encode<T: serde::Serialize>(value: &T) -> Vec<u8> {
  let mut bytes = Vec::new();
  ciborium::into_writer(value, &mut bytes).expect("a solve payload is plain data");
  bytes
}

pub fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> T {
  ciborium::from_reader(bytes).expect("a solve payload round-trips")
}

/// A build in flight. Constructed with the request and polled until the outcome
/// lands; dropping it abandons the build.
pub(crate) struct Pending {
  inner: backend::Handle,
}

impl Pending {
  pub(crate) fn spawn(request: SolveRequest) -> Self {
    Self {
      inner: backend::spawn(request),
    }
  }

  /// The outcome, once. `None` while the build is still running.
  pub(crate) fn poll(&self) -> Option<SolveOutcome> {
    self.inner.poll()
  }
}

#[cfg(target_arch = "wasm32")]
use crate::web::worker as backend;
#[cfg(not(target_arch = "wasm32"))]
use native as backend;

/// The native transport: a thread and a channel. The request moves into the
/// thread whole, so there is nothing to serialize, the encoding above exists
/// for the web's sake and native never pays for it.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod native {
  use super::{SolveOutcome, SolveRequest};

  pub(crate) struct Handle {
    rx: std::sync::mpsc::Receiver<SolveOutcome>,
  }

  pub(crate) fn spawn(request: SolveRequest) -> Handle {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
      let _ = tx.send(request.run());
    });
    Handle { rx }
  }

  impl Handle {
    pub(crate) fn poll(&self) -> Option<SolveOutcome> {
      self.rx.try_recv().ok()
    }
  }
}
