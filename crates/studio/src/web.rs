//! The web platform layer: everything that is specific to running the viewer in
//! a browser, kept out of the shared code so that layer stays free of `wasm`
//! concerns.
//!
//! Three things live here and nowhere else:
//!
//! - the `wasm-bindgen` entry point ([`start`]), which sets up the browser
//!   console for panics and logs and hands the winit loop to the browser's own
//!   scheduler rather than a blocking `run_app`;
//! - mounting the winit-created `<canvas>` into the document, since winit builds
//!   it but does not attach it;
//! - bridging the *async* GPU bootstrap back into the event loop. Adapter,
//!   device and surface creation are async and must yield to the browser instead
//!   of blocking it, so [`State::new`] runs on the microtask queue via
//!   [`init_state`] and the finished [`State`] is parked in a slot the loop
//!   drains with [`take_ready_state`]. The slot is a plain `RefCell`, sound
//!   because `wasm32` is single-threaded: the future and the event loop never
//!   run concurrently.

use std::cell::RefCell;
use std::sync::Arc;

use wasm_bindgen::prelude::*;
use winit::{
  event_loop::EventLoop,
  platform::web::{EventLoopExtWebSys, WindowExtWebSys},
  window::Window,
};

use crate::app::{App, State};

thread_local! {
  /// The finished [`State`] awaiting installation into the running loop. Written
  /// once by [`init_state`]'s future, read once by [`take_ready_state`].
  static READY_STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// The browser entry point: routes panics and logs to the devtools console and
/// starts the winit event loop on the browser's scheduler. Named `start` and
/// invoked from the page's JS glue after the module initializes.
#[wasm_bindgen]
pub fn start() {
  console_error_panic_hook::set_once();
  // `Info` and above; the render loop's per-frame chatter stays at `debug` and
  // is filtered out by default.
  let _ = console_log::init_with_level(log::Level::Info);

  let event_loop = EventLoop::new().expect("event loop builds");
  // `spawn_app`, not `run_app`: the web event loop cannot block the browser, so
  // it drives the app off the JS event queue and returns immediately.
  event_loop.spawn_app(App::default());
}

/// Mounts the window's canvas into the document and kicks off the async GPU
/// bootstrap, parking the finished [`State`] for the loop to pick up. Called
/// from `App::resumed` in place of the native blocking build.
pub(crate) fn init_state(window: Arc<Window>) {
  mount_canvas(&window);
  wasm_bindgen_futures::spawn_local(async move {
    let state = State::new(window).await;
    READY_STATE.with(|slot| *slot.borrow_mut() = Some(state));
  });
}

/// Hands back the bootstrapped [`State`] once its async build has finished,
/// exactly once. `None` until then.
pub(crate) fn take_ready_state() -> Option<State> {
  READY_STATE.with(|slot| slot.borrow_mut().take())
}

/// Appends the winit-created `<canvas>` to the document body and sizes it to
/// fill the viewport. winit builds the canvas but leaves attaching it to the
/// page to the embedder.
fn mount_canvas(window: &Window) {
  let canvas = window.canvas().expect("winit created a canvas on the web");
  let style = canvas.style();
  let _ = style.set_property("width", "100%");
  let _ = style.set_property("height", "100%");
  let _ = style.set_property("display", "block");

  let document = web_sys::window()
    .and_then(|w| w.document())
    .expect("a document");
  let body = document.body().expect("a document body");
  body
    .append_child(&canvas)
    .expect("the canvas mounts into the body");
}
