# Renderer performance audit

Date: 2026-10-13. Scope: `gpui_ce_wgpu` in the working `artisan/performance` fork. This ledger records inspected code, implemented changes, and remaining proposals. It does not claim that GPU allocation totals equal process RSS or that every possible optimization is exhausted. Immutable allocator experiments use separately frozen sources.

## Implemented and checked

- Atlas pages begin at 256 pixels per side, then grow through 512 to 1024, subject to the tile dimensions and device limit. Owned raster output moves into upload staging; borrowed output is copied. Page removal drops pending uploads before its index can be reused. Sprite sorting includes the texture page before tile ID, preserving draw order and avoiding unnecessary page switches.
- Instance storage starts at 256 KiB and grows on demand. Optional path, blur, capture-copy, surface, and mesh pipelines are created when needed. The shared shader module is retained for subsequent pipeline creation; wgpu-core 29 already retains shader modules in each render pipeline, so this does not retain another independent Naga module.
- Path raster targets cover the union of the current batch, clipped to the viewport with an integer origin and one-pixel guard. Raster vertices subtract the origin; gradient, pattern, and dither evaluation adds it back. The composite samples against the actual target dimensions. Storage-buffer and integer-texture transport layouts include the origin.
- Path capacity grows componentwise during continuous use. Alternating tall and wide bounds no longer repeatedly shrink and grow allocations. A successfully acquired frame with no paths drops both MSAA and resolve textures and their views. Resize and device recovery also release them. Skipped acquisitions do not change target lifetime. Switching directly from a large path scene to a smaller path scene retains capacity until paths disappear or the window resizes.
- GPU error recovery preserves consecutive failures across cache resets and skipped frames. Error six clears resources once; error eleven reaches the existing terminal panic. Complete submission and presentation reset the count only when no error or device loss is pending. Renderer replacement preserves the count until successful presentation. Late asynchronous errors still depend on callback timing; no blocking fences or polling were added.
- Mesh geometry and targets are cleared when the scene has no meshes. Fewer visible meshes truncate the target list. Pipeline objects remain reusable.
- Allocation diagnostics are opt-in. Ordinary frames do not scan resources, allocate statistics vectors, or schedule diagnostic wakeups.

The coordinating round-five run reports 375 GPUI core tests, 54 Linux tests, and 73 GPU/text tests passing. GPU coverage includes path pixel parity at a non-power-of-two 173×117 viewport, gradients/patterns/translucent overlap and clipping, both instance transports, atlas lifetime/growth, lazy pipeline creation, target reuse/release policy, and bounded error recovery. The source review found no additional pixel or ordering defect in those changes. Actual driver-loss recovery, precise touchpad behavior, and physical compositor restoration still require native integration checks.

## Offscreen cleanup implemented after round five

Scene-color, blur ping/pong, and group textures now release on a successfully acquired frame that has neither capture nor filters. These pools remain available during continuous offscreen use. Compiled pipelines remain available. This follows the same activity boundary as path release and introduces no timer. The change was made after the passing round-five tests; its coordinating build and capture validation remain pending.

At 1600×1125, an RGBA8 scene target, two half-size blur targets, and two full-size group targets request approximately 25.2 MB of texel storage in total, before backend allocation padding. Previously, an application that closed its filters retained those textures until resize. Capture must keep its scene target through command encoding and readback; dropping the application's handles on a later normal frame is safe because submitted commands retain their resources.

Validation should cover filter → ordinary scene → filter and capture → ordinary scene transitions, including a skipped acquisition between them. Assert that all texture and view handles are cleared together, pipelines remain available, and the later filter frame re-creates correctly sized targets. A nested-filter pixel check must still preserve its parent target and compositing order.

## Mesh scratch sharing implemented

Each mesh retains its resolved RGBA8 image through main-scene compositing. Meshes with identical dimensions now share the four-sample color and Depth32Float scratch textures. Each pass clears both and resolves before the next pass uses them.

The pool uses exact width and height. Resolve textures, uniforms and composite bind groups remain separate for each mesh. The sample count and formats are fixed, so dimensions suffice as the key. The pool keeps only sizes used by the current scene and clears when no meshes remain. GPU commands retain attachments after their handles leave the pool.

This saves 32 bytes per pixel for each additional mesh of the same size. Resolution, edge thickness, projection, clipping and shader coordinates stay unchanged. Two 96×64 meshes request 245,760 texture bytes instead of 442,368, saving one 196,608-byte scratch pair. This reduces model-view storage; the idle conversation fixture has no meshes.

Pixel tests compare overlapping meshes with different cameras, colors and opacity against separate renderers with independent scratch targets. Both draw orders produce byte-identical output to their independent references. Lifetime tests cover mixed dimensions, removal, resizing, reappearance and submission after pool entries have been released.

Geometry uploads also write vertex data directly into the mapped destination buffer, eliminating an intermediate vector containing the entire model. Empty geometry leaves its resolved target transparent.

## Inactive geometry budget design, not implemented

The current geometry cache expires entries after 120 calls to `prepare`. Removing one large model while another remains static can retain the removed model's vertex buffer indefinitely, because idle frames stop. Deleting all invisible geometry immediately would cause avoidable uploads while scrolling.

Track each cached vertex buffer's byte size and last-use sequence. During normal mesh preparation, mark the current scene's mesh IDs, preserve all visible geometry, and evict least-recently-used invisible entries until their total fits a fixed inactive-byte budget. Run eviction when visibility or allocations change, with no timer. The budget should be selected from model fixtures and upload costs rather than assumed to achieve the process-memory target. Clearing all meshes can continue to release everything immediately.

Tests should cover one oversized hidden model, multiple hidden entries evicted in LRU order, reuse of a small recently hidden model without upload, shared geometry used by several visible instances, and a live working set larger than the inactive budget. Visible geometry must never be evicted to satisfy that budget.

## File coverage

| File | Review depth/status | Result |
| --- | --- | --- |
| [Cargo.toml](../crates/gpui_wgpu/Cargo.toml) | Inspected | Windows surface support has an explicit feature; wasm texture transport remains enabled. |
| [gpui_wgpu.rs](../crates/gpui_wgpu/src/gpui_wgpu.rs) | Inspected | Module wiring and re-exports only. |
| [cosmic_text_system.rs](../crates/gpui_wgpu/src/cosmic_text_system.rs) | Changed; shaping reviewed | ASCII avoids unused fallback/emoji discovery while preserving full shaping. Glyph and font cache ownership was inspected separately from renderer buffers. |
| [wgpu_context.rs](../crates/gpui_wgpu/src/wgpu_context.rs) | Inspected | Device-memory growth hints and supported-backend filtering; actual allocator/RSS effects belong to the controlled experiments. |
| [wgpu_atlas.rs](../crates/gpui_wgpu/src/wgpu_atlas.rs) | Inspected; tests reviewed | Page growth, device bounds, owned staging uploads, page deletion, index reuse, and accounting. |
| [wgpu_renderer.rs](../crates/gpui_wgpu/src/wgpu_renderer.rs) | Changed; allocation/recovery paths reviewed | Lazy resources, path rebasing, capacity growth, surface failures, capture, filter target stack, and mesh lifecycle. Large file has focused path coverage rather than a claim that every branch was executed. |
| [wgpu_recovery.rs](../crates/gpui_wgpu/src/wgpu_recovery.rs) | Changed; tests pass | Failure count survives skipped frames and resets; successful presentation starts a fresh retry budget. |
| [wgpu_paths.rs](../crates/gpui_wgpu/src/wgpu_paths.rs) | Changed; tests pass | Batch bounds, outward rounding, viewport clamp, tall/wide retention, and no-path release/reset. |
| [wgpu_pipelines.rs](../crates/gpui_wgpu/src/wgpu_pipelines.rs) | Inspected; tests reviewed | Lazy groups retain explicit layouts, blend states, formats, and shader module reuse. Transparency changes re-create the groups on demand. |
| [wgpu_diagnostics.rs](../crates/gpui_wgpu/src/wgpu_diagnostics.rs) | Inspected | Opt-in allocation scans; excludes swapchain, driver/compiler memory, and allocator padding from texel totals. |
| [wgpu_mesh.rs](../crates/gpui_wgpu/src/wgpu_mesh.rs) | Changed; pixel/lifetime tests pass | Exact-size color/depth scratch sharing and direct mapped vertex uploads remove duplicate storage. Frame-count geometry expiration remains a candidate. |
| [shaders.wgsl](../crates/gpui_wgpu/src/shaders.wgsl) | Path/gradient/blur paths reviewed | Path origin restores screen coordinates for color evaluation; clipping remains in screen space. Blur region transforms remain unchanged. |
| [shaders_storage.wgsl](../crates/gpui_wgpu/src/shaders_storage.wgsl) | Transport screened | Uses typed records shared with the path declarations. |
| [shaders_webgl.wgsl](../crates/gpui_wgpu/src/shaders_webgl.wgsl) | Changed records reviewed | Path vertex/sprite strides match Rust and include origin fields, including records crossing texture rows. |
| [shaders_subpixel.wgsl](../crates/gpui_wgpu/src/shaders_subpixel.wgsl) | Feature/transport screened | Dual-source blending remains optional; no shader behavior change in this work. |
| [shaders_mesh.wgsl](../crates/gpui_wgpu/src/shaders_mesh.wgsl) | Depth/edge semantics screened | Scratch pooling must preserve target resolution and pixel-based edge treatment. |
| [shaders_mesh_composite.wgsl](../crates/gpui_wgpu/src/shaders_mesh_composite.wgsl) | Inspected | Samples the complete resolved target; changing allocation dimensions would require UV/viewport care. |
| [path_tests.rs](../crates/gpui_wgpu/src/path_tests.rs) | Changed; tests pass | Production pipelines compare full-window and compact targets, including odd viewport dimensions and both transports. |
| [mesh_tests.rs](../crates/gpui_wgpu/src/mesh_tests.rs) | Changed; tests pass | Independent-target pixel references cover overlap, depth, opacity and both draw orders. Pool lifetime checks cover mixed sizes and commands surviving resource removal. |
| [gradient_tests.rs](../crates/gpui_wgpu/src/gradient_tests.rs) | Test helper/coverage screened | Existing offscreen pixel comparison pattern informed path tests. |
| [shadow_tests.rs](../crates/gpui_wgpu/src/shadow_tests.rs) | Test helper/coverage screened | Existing native readback/device setup reviewed; shadow behavior was not changed in this work. |

## Vulkan command-buffer retention audit

The clean round-seven Vulkan/llvmpipe heaptrack snapshot at 31.005 seconds attributes 208 live 65,616-byte allocations to raw command-buffer linear arenas, totaling 13,648,128 requested bytes. There are 73 initialization-origin allocations and 135 reset-origin allocations. This does not establish the number of Vulkan pools or the resident-memory cost.

Resolved wgpu-hal 29.0.4 allocates raw command buffers in batches of sixteen when an encoder's free list empties. Resolved wgpu-core uses multiple raw buffers per public encoder: render-pass work and its resource transitions are recorded separately, with extra submission/presentation transitions. Completed encoders reset their pools and return to an unbounded device-level free encoder cache. Queue submission already performs nonblocking device maintenance. Additional polling cannot trim that cache, and the public Device API exposes no trim or allocation-granularity setting. MemoryHints configures GPU allocation blocks, not command buffers.

GPUI already submits one public main encoder per normal frame; its meshes, paths, filters and final blit share that encoder. Queue writes use wgpu's separate internal upload encoder. There is no redundant GPUI submission per render pass to remove. The public maximum-frame-latency hint could reduce concurrent frames, but it changes the concurrency/throughput policy and cannot release cached pools; it is outside this experiment.

Two dependency copies under the Editor workspace's `.dist/performance/command-pool-audit/` differ only in Vulkan allocation granularity, sixteen versus one. Shared Cargo registry sources and production GPUI are untouched. Matched release diagnostic binaries enable wgpu's existing counters to expose actual HAL pool count; heaptrack supplies raw-buffer counts. Both builds passed, and their GPUI/Editor source manifests and profiles match.

The paired llvmpipe runs report five pools in both variants. Granularity one reduces raw command arenas from 208 to 135, removing 4,789,968 requested bytes. Unprofiled 60-second samples show RSS of 210,640,896 and 210,300,928 bytes, a difference of only 339,968 bytes. CPU totals are zero and 0.01 seconds respectively. Both windows stay focused/visible at the same 1898×1125 rectangle, and all four captures show the same completed conversation with an empty composer. Numeric pixel comparison matches outside the fixture-specific avatar and last-frame timing overlay.

This single pair establishes lower requested command capacity on the tested Mesa driver, but not a repeatable resident-memory improvement. Smaller batches can increase native allocation calls during growth; active rendering performance and other Vulkan drivers remain untested. The isolated dependency patch is not recommended for production from this evidence alone. Full qualification, attribution and raw samples are recorded in the experiment's `results.json` and per-run directories.

## Surface validation probe and synchronization

`WgpuContext::try_adapter_with_surface` configures a 64×64 FIFO surface and checks a validation error scope. It does not acquire or present an image. This is true in the pinned original fork and the current source, so an acquire without submission cannot explain the observed AMD worker spin through this probe.

Resolved wgpu-core 29.0.4 `device/resource.rs::configure_surface` waits for existing queue work, rejects reconfiguration while an acquired texture exists, calls the backend configuration method, then records `acquired_texture: None`. DX12 configuration waits for its separate present queue with an event and an infinite timeout. Reconfiguration releases the old latency handle and resizes the swapchain buffers. Vulkan reconfiguration waits for device idle, releases old acquisition/presentation synchronization objects, and replaces the swapchain. No probe frame is pending in either backend.

Removing the probe in an isolated diagnostic tests whether an extra swapchain configuration triggers the driver issue. Source inspection does not establish that causal link or justify removing the adapter fallback check in production. The real frame submission path supplies texture transitions and presentation synchronization; adding an acquire/present pair to the probe would introduce work it currently does not perform.

## EGL library lifetime after discovery

Resolved khronos-egl loads libEGL on Linux with `RTLD_NOW | RTLD_NODELETE`. Its source explicitly identifies this as a fix for a segmentation fault, with [the upstream issue](https://github.com/timothee-haudebourg/khronos-egl/issues/14) pointing to the driver-unloading discussion. wgpu-hal's EGL `Inner::drop` destroys the native context and terminates the display, but that does not unload the library marked NODELETE. Persistent code mappings after context cleanup therefore do not alone establish a retained GPUI context. This accounts for libEGL's lifetime; the exact libgallium dependency chain still needs loader attribution. Preventing unused backend discovery avoids these loads. Forcing library unmapping is not a safe cleanup policy.

## Native Windows atlas and blur allocation

The native D3D11 atlas now uses the same 256/512/1024 page growth policy as
WGPU. The helper lives on `AtlasTextureList`; moving it does not change WGPU's
policy. Removed page slots do not inflate the live page count. Individual large
images still receive a sufficient page, and invalid or oversized dimensions
are rejected before texture creation.

Native scene/ping/pong blur textures are allocated on the first frame that
needs filters and released after a frame without filters. Content-group targets
grow to the actual nesting depth, up to the existing two-level limit. Sibling
filters share that capacity. Resize and device-resource recreation reset the
optional allocation. The render loop retains group capacity during continuous
filter use to avoid reallocating during animation.

Four native Windows WARP tests pass for atlas hole reuse, filter nesting,
texture/view reuse and resource recreation. The core/Linux/GPU suites pass
385/58/80 tests, and strict library/test Clippy passes. An independent source
review found no ownership, dimension or resource-release blocker. Native
display/device-loss integration remains separate from these headless tests.

An isolated D3D11 idle comparison measured 83.81 MB mean resident memory and
196.06 MB private commit before this change, versus 84.45 MB and 171.25 MB after.
Both recorded zero CPU time over 60 seconds. The 24.81 MB commit reduction is
not a resident-memory reduction. Both binaries retain known appearance
differences and do not qualify as a production replacement. The atlas/blur
allocation changes preserve those binaries' existing rendering behavior.
Artifacts are in Editor `.dist/performance/native-allocation/`, including
`native-tests/`, `idle/`, the immutable executable hash and source manifest.
