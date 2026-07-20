//! The one-time welcome shown on a reader's first launch, and the small bit of
//! persistence that keeps it to the first.
//!
//! The greeting differs by platform because the two viewers genuinely do: what
//! a native reader can reach (local files, the keyboard fly) a browser one
//! cannot, and the browser's own terms (WebGPU, the solve in a worker) are not
//! the desktop's. The persistence is likewise split at the one real seam -- a
//! marker file natively, `localStorage` on the web -- with the web half living
//! in `web.rs`, so nothing browser-specific leaks out of it.

/// The modal's title, shared across platforms.
pub(crate) const TITLE: &str = "Welcome to formoniq-studio";

/// The greeting body, in the reader's own platform's terms.
pub(crate) fn message() -> &'static str {
  #[cfg(not(target_arch = "wasm32"))]
  {
    "You are running the native viewer.\n\n\
     • Pick a preset in the Browser on the left, or compose a mesh and a study yourself.\n\
     • Orbit with the left mouse button, look with the right, pan with the middle; scroll to zoom, and fly with WASD.\n\
     • Load your own OBJ meshes and export stills from the File menu.\n\n\
     The full controls are under Help › Navigation."
  }
  #[cfg(target_arch = "wasm32")]
  {
    "You are running in the browser, on WebGPU. Studies solve in a background worker, so the view stays responsive while they run.\n\n\
     • Pick a preset in the Browser on the left, or compose a mesh and a study yourself.\n\
     • Drag to orbit and look, scroll or pinch to zoom; one finger looks, two fingers pan.\n\
     • Local files and image export are native-only, so they are absent here.\n\n\
     The full controls are under Help › Navigation."
  }
}

/// Whether the welcome should be shown -- true until the reader has dismissed it
/// once, remembered across launches per platform. A failure to read the marker
/// errs toward showing it: greeting a returning reader once more is a smaller
/// harm than hiding it from a new one.
pub(crate) fn should_show() -> bool {
  !seen()
}

#[cfg(not(target_arch = "wasm32"))]
fn marker_path() -> Option<std::path::PathBuf> {
  directories::ProjectDirs::from("", "", "formoniq-studio")
    .map(|dirs| dirs.config_dir().join("welcome_seen"))
}

#[cfg(not(target_arch = "wasm32"))]
fn seen() -> bool {
  marker_path().is_some_and(|path| path.exists())
}

/// Records that the welcome has been dismissed, so the next launch skips it.
/// Best-effort: a write failure (an unwritable config dir) leaves the greeting
/// to reappear next time rather than taking the viewer down for it.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn mark_seen() {
  if let Some(path) = marker_path() {
    if let Some(parent) = path.parent() {
      let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, b"");
  }
}

#[cfg(target_arch = "wasm32")]
fn seen() -> bool {
  crate::web::welcome_seen()
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn mark_seen() {
  crate::web::mark_welcome_seen();
}
