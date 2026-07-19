// The solve worker: the viewer's one expensive call, off the main thread.
//
// It loads the *same* wasm module the page does and calls one export, so the
// solver running here is the identical code the native build runs on a thread.
// Nothing about the study is decided in JavaScript -- this file is a transport
// and nothing else, which is why it stays this short.
//
// A module worker, because the wasm-bindgen `--target web` glue is an ES
// module. The import is dynamic so this can forward its own build stamp
// (`?v=...`, set on the URL by the code that spawns it) to the glue and the
// wasm: the page and the worker must run the *same* build, and stamping is what
// keeps a browser from pairing one from cache with another fetched fresh.
const stamp = self.location.search;

const ready = (async () => {
  const wasm = await import(`./formoniq_studio.js${stamp}`);
  await wasm.default({
    module_or_path: `./formoniq_studio_bg.wasm${stamp}`,
  });
  return wasm;
})();

// Instantiation is started at load and awaited per message, so a request
// arriving before the module is ready waits rather than being dropped.
self.onmessage = async (event) => {
  const wasm = await ready;
  // The request is CBOR in, the outcome CBOR out. The reply's buffer is
  // transferred rather than copied: it is the solved fields, and the worker has
  // no further use for them.
  const outcome = wasm.solve_in_worker(new Uint8Array(event.data));
  self.postMessage(outcome, [outcome.buffer]);
};
