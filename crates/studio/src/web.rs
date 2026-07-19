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

/// The browser's answer to "run this off the main thread": a dedicated worker.
///
/// This is the whole of the web's solve transport, and it is deliberately the
/// *only* place any of it appears. [`crate::solve`] knows a build is a value
/// that produces an outcome later; it does not know what a `Worker` is, and the
/// native build never compiles a line of this.
///
/// A worker, not `SharedArrayBuffer` threads. Sharing memory with the main
/// thread would need cross-origin isolation (`COOP`/`COEP` response headers)
/// and a nightly toolchain built with the `atomics` target feature -- neither
/// available on a static host, and both a large cost for what is wanted here.
/// Message passing needs nothing: the request is CBOR in, the outcome CBOR out,
/// and the tab keeps rendering throughout.
///
/// One worker per build, torn down when the handle drops. A build abandoned
/// mid-flight (the reader picks another pair) therefore stops occupying a core
/// instead of running to completion for an answer nobody will read -- which is
/// also why the handle owns the worker rather than sharing a long-lived one.
#[cfg(target_arch = "wasm32")]
pub(crate) mod worker {
  use std::cell::RefCell;
  use std::rc::Rc;

  use wasm_bindgen::prelude::*;

  use crate::solve::{decode, encode, SolveOutcome, SolveRequest};

  /// A build running in its own worker. The outcome lands in `slot` from the
  /// message callback; [`Self::poll`] takes it. A plain `RefCell` is sound
  /// because `wasm32` is single-threaded -- the callback and the event loop
  /// never run at once, the same reasoning [`super::init_state`]'s slot rests
  /// on.
  pub(crate) struct Handle {
    worker: web_sys::Worker,
    slot: Rc<RefCell<Option<SolveOutcome>>>,
    /// Kept alive for as long as the worker can still call it.
    _onmessage: Closure<dyn FnMut(web_sys::MessageEvent)>,
  }

  impl Drop for Handle {
    fn drop(&mut self) {
      // An abandoned build is stopped, not waited for.
      self.worker.terminate();
    }
  }

  pub(crate) fn spawn(request: SolveRequest) -> Handle {
    let options = web_sys::WorkerOptions::new();
    // The worker `import`s the same wasm-bindgen glue the page does, so it must
    // be a module worker.
    options.set_type(web_sys::WorkerType::Module);
    let worker = web_sys::Worker::new_with_options("./solve_worker.js", &options)
      .expect("the solve worker script is served beside the page");

    let slot: Rc<RefCell<Option<SolveOutcome>>> = Rc::new(RefCell::new(None));
    let onmessage = {
      let slot = slot.clone();
      Closure::<dyn FnMut(web_sys::MessageEvent)>::new(move |event: web_sys::MessageEvent| {
        let bytes = js_sys::Uint8Array::new(&event.data()).to_vec();
        *slot.borrow_mut() = Some(decode(&bytes));
      })
    };
    worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

    let payload = js_sys::Uint8Array::from(encode(&request).as_slice());
    worker
      .post_message(&payload)
      .expect("the worker accepts its request");

    Handle {
      worker,
      slot,
      _onmessage: onmessage,
    }
  }

  impl Handle {
    pub(crate) fn poll(&self) -> Option<SolveOutcome> {
      self.slot.borrow_mut().take()
    }
  }
}

/// The worker's entry point: decode a request, run it, encode the outcome.
///
/// Exported for `solve_worker.js` to call. It runs inside the worker's own
/// instance of this same module, so the solver here is the identical code the
/// native build runs on a thread -- the boundary is the transport, never a
/// second implementation.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn solve_in_worker(request: &[u8]) -> Vec<u8> {
  let request: crate::solve::SolveRequest = crate::solve::decode(request);
  crate::solve::encode(&request.run())
}
