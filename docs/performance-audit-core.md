# Core allocation audit

Date: 2026-10-13. This is a focused review of frame, element, and layout allocation patterns in the working performance fork. Byte counts below describe Rust layouts or allocator requests; they are not process RSS measurements or counts from the Editor fixture.

## Confirmed lifetime defect, fixed

The resolved Taffy 0.13.0 `TaffyTree::clear` clears its node, child, and parent slot maps but leaves `node_context_data` populated. Its measurement contexts can therefore keep captured data alive after GPUI finishes the frame. An ordinary, unmeasured node reusing that slot does not release the previous context. GPUI text measurement closures capture text, runs, styles, and shared layout state.

A standalone program linked against the resolved Taffy library observed zero captured destructors after `clear`, zero after ordinary-node reuse, and one only when the tree dropped. GPUI now records measured node IDs and removes their contexts before clearing the tree. The existing frame-clear boundary is unchanged. This releases payloads while preserving Taffy's reusable backing capacity.

The latest coordinating run reports **382 core, 54 Linux, and 75 GPU/text tests passing**, including new tests for same-frame lifetime, exact destructor counts, ordinary-node reuse, repeated clearing, and independent layout engines. The measured-ID vector costs eight bytes per measured node at capacity, excluding its fixed header; it prevents retention of the much larger closure payloads. Clearing a context marks its ancestors dirty, but Taffy's traversal stops at already-dirty nodes.

## Focused input ownership during window teardown

Typing-only captures on Windows and Linux exposed a pre-existing shutdown leak. The original pinned fork also retained the focused TextEditor entity. The editor has no background task responsible for the reference: `ElementInputHandler` owns a strong entity handle, and both native platforms defer window-state destruction to foreground tasks. That state can outlive GPUI's Window and its entity teardown.

Window destruction now takes and drops its installed platform input handler synchronously, before native cleanup can be deferred. It preserves normal active-window input behavior and also covers ordinary window closure. New tests hold native TestWindow state alive across close and shutdown; both fail before the fix because the focused entity remains alive. Both pass after the fix, along with the existing input configuration/focus-state test. A native Linux capture subsequently rendered multilingual typed text, opened and closed the command palette, and exited normally with no leaked-entity report. This complements the focused lifecycle tests; it is not a Windows physical-input test. Before/after evidence is saved in the Editor workspace as `.dist/performance/input-handler-lifetime-before.log` and `.dist/performance/input-opacity-regressions-after.log`. No Editor-specific ownership workaround was added.

## Measured layouts

These values came from `size_of` and compiler layout output against the already-built release libraries, with no production source instrumentation. Linux x86-64 and Windows x86-64 gave the same element layouts.

| Type/storage | Bytes | Implication |
| --- | ---: | --- |
| `Div` | 1,344 | Includes its `Interactivity` and inline child handles. |
| `Drawable<Div>` | 1,480 | Actual arena object includes the largest draw-phase state. |
| `Interactivity` | 1,240 | Most of a plain Div's inline storage. |
| Nineteen listener/style vector headers | 456 | Present even when every vector is empty. |
| `AriaProperties` inside Interactivity | 328 | Present even when no explicit accessibility properties were set. |
| `StyleRefinement` | 776 | Every ordinary Div owns a separately boxed base style. |
| `Style` | 760 | Materialized during layout/style computation. |
| `TextStyleRefinement` | 216 | Embedded in both style types, including layout-only elements. |
| `AnyElement` | 24 | Includes the arena handle's lifetime-generation ownership. |
| `Drawable<UniformList>` | 2,256 | Includes a 944-byte draw-phase enum. |
| `UniformListFrameState` | 832 | Its 32-item inline SmallVec contributes 776 bytes. |

Thus a `Drawable<Div>` and its base-style allocation account for at least **2,256 bytes** before listener buffers, heap child storage, text, or layout-engine nodes. These transient objects also determine the retained high-water capacity of the element arena. The number of objects built in a frame matters more than the number of rows in the underlying model; list virtualization and cached subtree reuse must be preserved.

## Taffy backing storage

A separate global-allocator counter around the resolved Taffy library measured the following requested live bytes for disconnected default-style leaf nodes. The context type in this probe was `usize`; no text, closures, child-vector contents, or GPUI wrapper maps were included.

| Nodes | Unmeasured, active | Unmeasured, after clear | Measured, active/after clear |
| ---: | ---: | ---: | ---: |
| 0 | 20,672 | 20,672 | 20,672 |
| 100 | 159,664 | 159,664 | 162,064 |
| 1,000 | 1,271,600 | 1,271,600 | 1,322,000 |
| 10,000 | 20,333,360 | 20,333,360 | 20,742,160 |

Dropping each tree returned all measured bytes. Capacity growth makes cost nonlinear across these sample counts. These results establish substantial backing-storage retention, but do not establish how many nodes the normal Editor conversation builds.

Taffy exposes no capacity-shrink method. A future reset can replace an already-cleared tree with a smaller one and restore disabled rounding. It should use recent node counts and a substantial size ratio/minimum reclaim threshold, and run at an existing draw boundary. Replacing it after every small frame would cause allocation churn when complex and simple UI alternate.

## Listener storage options and fixture counts

All nineteen Interactivity vector fields are crate-private and their accesses are contained within `elements/div.rs`. Public builder methods and public base-style access can remain unchanged.

| Representation | Inline header bytes per element | Additional cost when used | Tradeoff |
| --- | ---: | --- | --- |
| Existing nineteen Vecs | 456 | Existing element buffers | Direct length/capacity access. |
| One optional box holding all nineteen Vecs | 8 | One 456-byte box for each element using any group | Best when most elements have no listeners; an interactive element retains every empty header. |
| Nineteen `Option<Box<Vec<T>>>` wrappers | 152 | One 24-byte header box and extra allocation per used group | Small source migration, but increases allocation count for controls and listener clones. |
| Nineteen optional boxes holding one-item SmallVecs | 152 | One allocation per used group; the first item fits inside the box | Uses the existing SmallVec dependency; additional items can spill into a second allocation. |
| Nineteen thin vectors | 152 | Length/capacity stored with each existing element buffer | Avoids a second allocation; adds indirection and a dependency with internal unsafe allocation code. |

Mozilla's [thin-vec implementation](https://github.com/mozilla/thin-vec) provides the one-pointer representation; its [API](https://docs.rs/thin-vec/latest/thin_vec/struct.ThinVec.html) includes allocation-free empty construction, draining, cloning, and iteration. No dependency or representation change has been made.

For `D` elements, `E` elements with any nonempty group, and `N` total nonempty groups, approximate header storage is `456D` today, `8D + 456E` for the grouped box, or `152D + 24N` for boxed Vec headers. This excludes allocator size-class overhead and existing listener payload buffers. A thin vector moves its length/capacity into the payload allocation, so it needs separate measurement rather than being treated as a free pointer substitution.

A separate diagnostic GPUI snapshot is available in the Editor workspace at `.dist/performance/interactivity-audit/gpui`. `GPUI_INTERACTIVITY_DIAGNOSTICS=1` emits per-draw JSON with Div construction/layout counts, unique Interactivity addresses, group occupancy, listener lengths/capacity bytes, and explicit accessibility-property use before prepaint drains listeners. Repeated layout visits to the same address are deduplicated; nested draws have independent counters. The diagnostic-only patch and source hashes are retained beside the snapshot. A standalone harness checks duplicate visits, nested scopes, listener/ARIA counts, and disabled output. The independent Editor snapshot compiled with capture support and Linux test-support forwarding. Its saved diagnostic binary has SHA-256 `9d556829f0f3ee9459a6f06f2e127c98e4b0cd22928e9ae215f2869b49674509`; the fixture results follow.

The first scripted Linux fixture produced 61 draw records. Peak draw 24 constructed 407 Divs and visited 435 distinct Interactivity addresses. Of those, 394 had no listener groups, 37 had one, and four had two. The 45 nonempty groups held 83 listeners, including 40 action listeners concentrated on two elements. No explicit ARIA properties or roles appeared in any draw. A grouped listener box would save 176,184 requested header bytes, about 172 KiB, at that peak; per-field boxed Vec headers would save 131,160 bytes, about 128 KiB. These small totals do not explain the much larger process-memory gap. The final draws used 142 Interactivity objects and 131 Divs. Raw records are in `.dist/performance/linux-visual-validation/interactivity-basic/interactivity-counts.json` in the Editor workspace.

Counts describe elements laid out during that draw. Cached subtrees can skip those visits, and elements constructed outside the draw are absent from its constructor count. Use initial/full-draw records and transitions alongside maxima; do not treat a small cached draw as the whole live scene. Diagnostic allocations and logging make its RSS/CPU unsuitable for release comparisons.

## Removed duplicate tab-order allocation

`TabStopMap::clear` replaced the whole value with `Self::default()`, cleared the newly empty collections, and then constructed another empty SumTree. `SumTree::new` allocates an Arc leaf, so the second construction immediately discarded the first allocation. Clear now retains only the original default replacement. This preserves the existing drop/reset behavior of every tab-order field and removes one unnecessary leaf allocation per frame clear. Capacity reuse is a separate choice and has not changed.

## Allocation-free layout invalidation traversal

Recomputing an already-computed layout root invalidates the cached bounds of its descendants. The old walk called Taffy's `children`, which clones its child vector for each visited parent. It now uses the existing `TraversePartialTree::child_ids` borrowed iterator. The resolved implementation preserves the same child order. A counted-allocator probe over a 1,365-node tree observed 341 temporary allocations with the old walk and zero with the borrowed iterator, with identical traversal order. This addresses transient work during repeated layout, not idle retained memory. All 13 selected Taffy and tab-order tests pass; the probe and test log are retained in the Editor workspace.

## Capacity diagnostic snapshot

The fixed, pre-ARIA-boxing source is frozen under `.dist/performance/core-capacity-audit/` in the Editor workspace. Its capture-enabled Linux binary has SHA-256 `210a47b48e616ac5d7aac62d876812b155bf2dc56324b87879d20a1974e894dd`. `GPUI_CORE_CAPACITY_DIAGNOSTICS=1` emits records only during existing draws. They report rendered and spare frame storage, scene primitives and replay buffers, dispatch listeners, tab order, Taffy node counts before clear, and shared arena reserved/used bytes before the final scope exits. Taffy does not expose its backing capacity; that limitation is explicit in the records. Map byte totals are entry-storage lower bounds, not allocator totals. The core package disables debug assertions to avoid including inspector bounds in the normal frame footprint, while capture test-support allocations remain identified separately. Source hashes, diagnostic patch, build command, and immutable binary are saved together.

## Measured fixture capacities

The frozen binary completed two isolated Linux/X11 capture sequences at 1.25 scale and a 1,898 × 1,125 physical viewport. The basic initial/scroll sequence recorded 35 draws; typing multilingual text and opening/closing the command palette recorded 70. Both processes exited normally. Initial conversation content, code, table, typed text, and palette content are visible in the saved captures. The typing sequence saved its test draft on the synthetic Forge server; copying local Editor state alone does not isolate drafts. That server and its artifacts are retained for provenance, and subsequent baseline runs require a fresh fixture server. These are capacity and correctness diagnostics, not idle CPU/RSS samples.

| Storage | Observed peak or final retained storage | Assessment |
| --- | ---: | --- |
| Layout nodes | 544 total, 86 measured | Taffy private backing capacity remains unreported. |
| Element arena | 809,584 bytes used; one 1,048,576-byte chunk retained | Normal full layout already uses most of the chunk; shrinking it is unsupported by these counts. |
| Event arena | 113 bytes peak used; one 1,048,576-byte chunk retained | A 4 KiB first event chunk is now implemented; 1 MiB overflow chunks preserve large event support. |
| Fallback element arena | No allocated chunks | No storage to reclaim in this fixture. |
| Rendered frame | About 578–584 KB after interactions | Includes live primitive/listener buffers and their spare capacity. |
| Cleared spare frame | About 518–527 KB retained | Supports next draw without allocation; dropping it on every clear would churn. |
| Layout wrapper buffers | 19,056 bytes retained | Too small to justify a complicated shrink policy here. |
| Test-support bounds maps | 8,960 bytes across both frames at the end | Excluded from the frame totals above. |

The largest repeated frame capacities are bounds-tree nodes (139,264 bytes each), subpixel sprites (114,688 bytes each), quads (81,920 bytes each), dispatch nodes (65,536 bytes each), and replay operations (49,152 bytes each). The known total ends near 3.24 MB, excluding private Taffy storage, allocator metadata, captured/style/text payloads, GPU resources, and driver memory. Raw records and `capacity-analysis.json` are under `.dist/performance/linux-visual-validation/capacity-basic/` and `capacity-typing/` in the Editor workspace.

The event arena now uses a private constructor with a 4 KiB initial chunk and the existing 1 MiB overflow size. Ordinary Arena::new behavior is unchanged. Capacity accounting sums actual chunk extents, and clear retains/reuses both chunk sizes. All 23 selected arena/app tests pass, including a regression covering an 8 KiB payload with 256-byte alignment, exact destructor counts, stale-handle invalidation, and same-address reuse after clear. The event arena's requested allocation was not necessarily fully resident: chunks use raw allocation and only used pages need become resident. This reduces requested storage for the measured workload by 1,044,480 bytes; RSS impact requires measurement.

## Remaining capacity candidates

- `Frame::clear` empties most frame-pool maps/vectors while retaining capacity; tab-order storage is an exception because its whole value is replaced. Element-state payloads migrate to the new frame, and scene replay relies on the completed frame. Capacity decisions belong after that migration and swap, using both live and spare sizes. The opt-in measurement reports lengths/capacities separately before choosing thresholds.
- The element/event arenas retain every allocated 1 MiB chunk. Shrinking is only valid in `force_clear` after all scopes end and destructor processing finishes; nested draw scopes deliberately defer clearing. The element arena is shared across windows, so the last small window must not immediately evict capacity needed by another window. Prefer a recent-usage envelope plus a byte threshold over immediate truncation.
- Taffy's GPUI-side absolute-bounds/origin maps, computed-layout set, and traversal scratch vector also retain capacity. Resetting the underlying tree alone does not reclaim those maps.
- `DispatchTree` drops listener payloads when its node vector clears, but retains node/map backing capacity in both frame pools. Each node also embeds three Vec headers even when it carries no key listeners. Fixture node counts are needed before changing this representation.
- UniformList's large inline frame vector trades arena footprint for fewer allocations when showing up to 32 rows. Reducing it is not automatically a net win; measure visible item counts and overflow allocation frequency first.

## Source coverage

| File | Review performed | Status |
| --- | --- | --- |
| `arena.rs` | Chunk/generation/destructor/scope lifetime, high-water storage, initial/overflow sizes | Small first event chunk implemented with large-payload/reuse regression; no shrink policy. |
| `app.rs` | Event arena construction, emit/flush lifetime | Small first chunk preserves the existing event-flush clear boundary. |
| `window.rs` | Frame construction, finish/swap/clear, element arena scopes, layout-engine lifetime, input-handler teardown | Input-handler lifetime fix tested; no frame representation changes. |
| `input.rs` | Focused entity ownership through ElementInputHandler and platform state | Close/shutdown regressions reproduce pre-fix retention and pass after cleanup. |
| `taffy.rs` | Node/context creation, frame clear, wrapper maps, measurement entry points, descendant invalidation | Context lifetime fix tested; borrowed child iteration removes verified transient allocations. |
| Resolved `taffy_tree.rs` | Clear/remove/context ownership and dirty propagation | Reproducer confirms retained context payload and backing storage. |
| `element.rs` | Arena boxing and largest draw-phase storage | Release type sizes measured. |
| `elements/div.rs` | Interactivity fields, listener mutation/drain/take/clone paths, style and accessibility payload | Release layouts measured; representation change deferred. |
| `key_dispatch.rs` | Node payload movement, clear, listener storage | Source-only capacity candidate. |
| `tab_stop.rs` | Full reset, replay, focus ownership, SumTree allocation | Removed redundant second empty-tree allocation; existing capacity-release semantics retained. |
| `elements/uniform_list.rs` | Virtualization, render return buffer, frame inline buffers | Layout costs measured; no virtualization changes. |
| `elements/list.rs` | Item summary storage and visible/overdraw layout elements | Existing virtualization preserved. |
| `elements/text.rs` | Measurement closure captures and shared layout state | Identifies concrete retained payload fixed by Taffy context cleanup. |
