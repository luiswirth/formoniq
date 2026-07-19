// The solve worker: the viewer's one expensive call, off the main thread.
//
// It loads the *same* wasm module the page does and calls one export, so the
// solver running here is the identical code the native build runs on a thread.
// Nothing about the study is decided in JavaScript -- this file is a transport
// and nothing else, which is why it stays this short.
//
// A module worker, because the wasm-bindgen `--target web` glue is an ES
// module. Instantiation is started at load and awaited per message, so a
// request arriving before the module is ready waits rather than being dropped.
import init, { solve_in_worker } from "./formoniq_studio.js";

const ready = init();

self.onmessage = async (event) => {
  await ready;
  // The request is CBOR in, the outcome CBOR out. The reply's buffer is
  // transferred rather than copied: it is the solved fields, and the worker has
  // no further use for them.
  const outcome = solve_in_worker(new Uint8Array(event.data));
  self.postMessage(outcome, [outcome.buffer]);
};
