# Memory evidence and cache audit

This records resource ownership and measured byte counts from the Artisan
Editor audit on 2026-10-13. It complements
[`performance-audit-renderer.md`](performance-audit-renderer.md) and
[`performance-audit-platforms.md`](performance-audit-platforms.md).

## Renderer allocations

Windows measurements use the same completed 40-item conversation at 1600×1125
physical pixels, with an AMD Radeon RX 9070 XT through DX12, driver
32.0.31041.1004. Logs are in the Editor checkout under
`.dist/performance/windows-conversation-release-round2/` and
`.dist/performance/windows-conversation-release-round4/`.

| Resource | Round2 bytes | Round4 bytes |
| --- | ---: | ---: |
| Path resolve and four-sample MSAA textures | 36,000,000 | 5,242,880 |
| Atlas textures | 9,437,184 | 1,900,544 |
| Instance buffer | 2,097,152 | 524,288 |
| Offscreen scene and blur textures | 10,796,800 | 10,796,800 |
| All counted renderer GPU requests | 58,397,248 | 18,530,624 |
| Backend live suballocations | 73,334,784 | 28,508,160 |
| Backend reserved heaps | 159,383,552 | 54,525,952 |

These are GPU requests and backend heaps, not process working set or RSS.
They exclude swapchain images, driver and pipeline/compiler storage, general
application allocations, and alignment outside the counted resource extents.
The scene uses four backdrop filters, so the offscreen textures serve current
content. Its five path batches correspond to five paths with individual
bounds around 952×142 or 962×197. The retained path target is 1024×256. This
does not establish that packing separate paths would save further space.

The changes producing these allocations are geometric atlas pages
(256² → 512² → 1024²), an initial 256 KiB instance buffer with existing growth,
batch-sized path targets, and manual allocator block hints of 4–16 MiB for
device memory. Host allocation sizes are derived separately by wgpu-hal.
Lazy pipeline groups preserve all formats and features. The shared shader
module remains cached because wgpu-core also retains it through existing
pipelines; recreating modules for lazy groups would duplicate shader IR.

Set `GPUI_RENDERER_DIAGNOSTICS=1` and enable the tracing target, for example
`RUST_LOG=info,gpui::renderer_memory=info`. It reports changed allocation state
after existing successful draws, with no timer. Disabled diagnostics perform
no resource scan. DX12 and Vulkan expose backend allocator reports; another
backend may return none. Scene counters cover typed primitive vectors and
path vertex capacities, not every scene or frame structure.

## Process measurements and allocator experiment

The original pinned fork's Windows release build used about 224.09 MB of
working set for this fixture; round2 used about 206.24 MB. Approximately
8.3 MB of that difference was AMD image-page residency. The earlier debug
comparison included a roughly 152 MB anonymous allocation and DX12 debug
layers absent from both release builds; those differences are not savings
from this patch. Some round4 processes have a spinning AMD native thread and
therefore do not demonstrate the idle CPU target.

The frozen allocator comparison uses identical GPUI and Editor source except
for the two-line global allocator declaration. Dependency features remain
the same. `.dist/performance/allocator-ab/` retains both Editor source archives,
the GPUI source, source manifests, the exact diff, build commands/logs, and
SHA256 hashes for all four release binaries. Forge was not stopped.

Windows System allocation reduced mean working set by 3,364,025 bytes and
private commit by 32,625,849 bytes relative to mimalloc 0.1.52 with `no_thp`.
GPU live/reserved counts and AMD image residency matched. Both samples had
the same AMD-thread spin. Linux System measured 221,937,664 bytes RSS versus
225,148,928 for mimalloc, both with zero sampled idle CPU at 1518×900. The
Linux launches retained different atlas tile counts (401 versus 365), so
the roughly 3.2 MB RSS difference is directional rather than an isolated
allocator effect. Active-workload costs also matter to an allocator choice.

Allocator JSON instrumentation is confined to a separate diagnostic Editor
snapshot. The pinned mimalloc wrapper exposes safe `MiMalloc::stats_json()`
with its `extended` feature. Release builds use `MI_STAT=0`, which omits normal
live-allocation counters. The attribution build uses `-DMI_STAT=1`; it is not
the benchmark baseline. A single opt-in foreground task waits for the chosen
settling delay, writes the raw JSON, and ends. The stats API merges the
calling thread's counters with shared heap totals; other live threads can
still have unmerged local counters. Committed/reserved memory and reported
live bytes must retain those scope qualifications.

The Linux stat1 attribution launch, recorded under
`.dist/performance/linux-allocator-attribution/`, used the completed conversation
at scale 1.25 and a 1898×1125 X11 window. At roughly 30 seconds, mimalloc 3.3.2
reported these counters:

| Counter | Bytes | Scope |
| --- | ---: | --- |
| `malloc_normal.current + malloc_huge.current` | 18,065,040 | Reported live allocation block sizes, including size-class rounding and subject to unmerged thread counters |
| `page_committed.current` | 28,519,520 | Allocator bookkeeping for committed space within its pages |
| `committed.current` | 67,502,080 | Mimalloc's committed-memory accounting |
| `reserved.current` | 1,076,166,656 | Reserved virtual address space, not resident memory |
| External `/proc` process RSS | 238,780,416 | Whole-process resident mappings during the later 40–50 second sample |

The JSON's `process.rss_current` also says 67,502,080, but it is not an OS RSS
measurement on Linux. The pinned `stats.c::mi_process_info` initializes it from
mimalloc's committed counter. `prim/unix/prim.c::_mi_prim_process_info` replaces
peak RSS using `getrusage`, while leaving current RSS at that fallback value.
The external `/proc` sample is the whole-process measurement. Its ten-second
idle interval recorded zero CPU ticks; the one-shot stats write had already
completed.

`malloc_requested`, normal allocation counts, and per-size malloc-bin counters
require `MI_STAT>1`; their zeros in this stat1 build mean unavailable data.
The reported live block total excludes native allocations served by another
allocator and may omit unmerged live-thread changes. It is not an exact Rust
payload census. Neither the committed-minus-live difference nor the
RSS-minus-mimalloc difference directly measures reclaimable memory.

The same run's `smaps` gives a narrower resident attribution: named mimalloc
mappings total 22,167,552 bytes, LLVM mappings 60,018,688, allocation memfds
49,872,896, the native process heap 31,875,072, and the executable 31,096,832.
The complete pathname grouping is saved in `attribution-summary.json` beside
the raw samples and reconciles to the measured process RSS. These mapping
groups do not overlap one another, but they overlap the allocator counters
above. The difference between mimalloc's 67.50 MB committed counter and its
22.17 MB resident mappings is not free resident memory awaiting a purge.

A separate glibc attribution preload is prepared under
`.dist/performance/glibc-diagnostics/`. It changes no allocation symbols or
options and performs no trimming. One opt-in delayed worker writes `mallinfo2`
JSON and `malloc_info` XML, then exits before external idle/RSS sampling. Its
standalone smoke checks pass. This diagnostic separates glibc arena bookkeeping
from mimalloc and direct mappings; it does not identify individual native
allocation call sites or turn usable-block counts into exact payload bytes.

The subsequent same-process dual snapshot is in
`.dist/performance/linux-dual-allocator-attribution/`. Mimalloc reported
17,877,472 live block bytes and occupied 23,359,488 bytes in its named resident
mappings. Glibc reported 100,356,096 arena bytes, including 95,560,784 classified
in use and 4,795,312 free, plus 3,579,904 bytes in its large mmap allocations.
The later XML report includes one additional 135,168-byte reporting arena.
These glibc counts include allocator overhead and thread-cache accounting;
they are not resident payload. The actual process RSS was 237,125,632 bytes
with zero CPU ticks in the ten-second sample after both workers completed.
The fixture's loaded conversation data was verified, but this attribution run
did not capture pixels to verify visible transcript content. It predates the
inherited-opacity cache correction and is not an acceptance measurement.
LLVM still accounted for 60,002,304 resident bytes, allocation memfds for
49,856,512, and the native process heap for 31,715,328. Native allocation
call-site profiling is the next attribution step; another blanket purge
experiment is not supported by these counters alone.

## Text, font and asset ownership

| Owner | Established retention and limits |
| --- | --- |
| `text_system.rs`, `text_system/line.rs` | Before this change, x86_64 `DecorationRun` was 92 bytes, `ShapedLine` was 2,984 bytes, and every wrapped logical line reserved 32×92 = 2,944 heap bytes for decorations. Both line forms now keep one run inline and spill for arbitrary rich text. Plain-style merging/cache reuse and 40-style split/spill cases are tested. The per-line reservation is known; whole-process savings require a live-line count. |
| `text_system/line_layout.rs` | Current/previous-frame maps retain shared glyph layouts and text keys. `finish_frame` removes stale entries while retaining map/vector capacity. Content-hash APIs exist but are not used by Editor. |
| `text_system/line_wrapper.rs` | Each pooled wrapper retains 128 optional ASCII widths plus a map for other characters. Pools are keyed by font and size. This is session growth across distinct font/size choices, not a large fixed preallocation. |
| `text_system.rs` | Font-resolution, metrics and raster-bound maps retain entries for the text-system lifetime. Glyph raster-bound keys include font, size, scale and subpixel variant. Element scale transforms reuse unscaled glyph rasters, so those animations do not create a new size each frame. |
| `gpui_wgpu/src/cosmic_text_system.rs` | Loaded font faces, family matches, emoji-chain and glyph-color decisions persist. Grapheme-formation probes are bounded by `CLUSTER_FORMATION_CACHE_LIMIT`. The pending glyph image map is now a single-entry bounds-to-pixels handoff. Bounds-only calls or atlas hits retain at most one raster; interleaved requests rerasterize displaced images. Real-font tests prove identical monochrome/subpixel pixels, consumption, and preservation of the latest image across a miss. |
| cosmic-text 0.19 | The font cache is keyed by face and weight and owns HarfRust shaping data. GPUI loads family faces with NORMAL weight before actual shaping can request other weights; duplicate weight-specific shaping state is a hypothesis to measure. The enabled default features do not include the optional shape-run cache or monospace fallback tables. Font data is shared rather than copied into every face. |
| Swash scale context | Its default font cache is bounded to eight entries. Raster buffers have explicit ownership and move into atlas uploads; owned atlas upload data is no longer cloned merely to swizzle channels. |
| `svg_renderer.rs` | The system font database is a lazy process singleton; an enriched database is initialized only when SVG text requests font resolution. Path-only icons do not trigger it. Parsed SVG trees and rendered pixel buffers are owned by callers, not an unbounded renderer cache. `ParsedSvg::size()` avoids rasterizing a natural-size image just to read aspect ratio. |
| Editor `full_color_svg.rs` | Parsed catalog trees are limited to 64 entries. Rasters are limited to 128 entries and 8 MiB, with a 1024-pixel dimension cap. Eviction calls `drop_image` to remove corresponding atlas images. These are upper bounds, not observed use. |
| `app.rs`, `elements/img.rs`, `elements/image_cache.rs` | The default generic asset cache retains completed shared tasks until explicit removal. `RetainAllImageCache` also retains images until remove/clear/entity release; its old LRU doc comment does not match its implementation. Decoded `RenderImage` holds all animation frames. Image ownership and explicit eviction must be considered together with atlas ownership. |
| Editor rich-media `picture.rs` | Image loads use owner-specific asset keys and the generic asset cache. Supported sniffed formats explicitly include PNG, JPEG, GIF, WebP, BMP, ICO, TIFF and SVG; stripping them would remove used product capabilities. Owner-specific keys intentionally distinguish rewritten files used by different blocks, so cross-block deduplication requires content identity. |

The fixture has hundreds of atlas entries, and embedded Editor font bytes total
only a few megabytes. Neither fact explains the remaining process memory by
itself. Image-heavy conversations, repeated scale changes and font switching
need their own capacity measurements before choosing eviction budgets.

## Atlas eviction lifetime analysis

This section describes a possible policy; automatic eviction is not enabled.

Cached views keep paint ranges, not independent scene copies. In `view.rs`,
`ViewElementState` stores ranges into the preceding frame. A reused prepaint
or paint copies its range into the next frame and updates those indices.
`Window::reuse_paint` calls `next_frame.scene.replay` with `rendered_frame.scene`.
Deferred draws follow the same transfer. `Frame::finish` moves only accessed
element state into the next frame. `Window::draw` then swaps the frames and
clears the old scene, old deferred draws, and remaining old element state.

At the renderer's draw boundary, the supplied scene therefore includes every
atlas tile that GPUI's built-in cached views can replay into the following
frame. A current-scene pin set is sufficient for those views. Retaining the
preceding presented scene's pins would add reuse tolerance for alternating
content, rather than protect an older hidden view's independent scene.
The Editor does not directly use `AtlasTile`, `AtlasKey`, or `PlatformAtlas`.
These are public GPUI types, however, and `AtlasTile` is copyable. A general
eviction policy must define the lifetime of externally retained tiles or
remain an explicit framework-owned/opt-in policy. This compatibility question
prevents enabling eviction as a transparent change.

A possible implementation must also satisfy these conditions:

- Read usage from all three scene sprite vectors. A cache-hit view can replay
  without calling `get_or_insert_with`, so lookup timestamps alone are wrong.
- Identify a tile by texture kind, texture index and tile ID together. Tile
  IDs can repeat across atlas pages.
- Flush pending uploads before removing tiles or reusing their regions. The
  existing explicit removal filters uploads only when an entire page dies.
- Advance frame pin sets only with a healthy acquired frame. A failed surface
  acquisition must not age the only retained drawable scene out of the cache.
- Respect in-flight GPU resource ownership and queue ordering. Releasing a
  host handle does not justify explicitly destroying a texture still used by
  submitted work.
- Treat whole-page bytes separately from occupied texel bytes. Evicting an
  inactive key frees allocator space, but a page containing one pinned tile
  remains allocated. Moving tiles would require rewriting every retained
  scene reference and is a separate change.

The measured round4 Windows atlas uses 1,900,544 bytes in total. Its monochrome
256² and 512² pages plus its subpixel 256² page use 589,824 bytes. The remainder
is polychrome storage and can contain images or color glyphs. This single
fixture does not justify a universal glyph budget. The next useful counters
are key class, current-scene referenced tile bytes, and pages with no pinned
tiles, collected under the existing opt-in diagnostic flag while changing
font sizes or DPI. Those counters would distinguish avoidable history from
current content without adding an idle timer.

Raster-bound eviction needs its own budget and relationship to atlas entries.
Previously, evicting bounds while keeping an atlas tile could recreate a bitmap
that an atlas hit would never consume. The one-entry handoff now bounds that
bitmap retention, but raster-bound metadata and atlas storage still have
independent owners and lifetimes.

## Cached opacity correctness

Windows captures of both the original pinned fork and the optimized fork
showed an initially blank transcript that appeared after scrolling. Cached
children replayed colors already multiplied by an ancestor's old zero
opacity. The view cache key did not include inherited opacity, and div
prepaint did not expose the opacity scope used in paint. The optimized scene
storage did not introduce this bug.

Div prepaint now carries the same inherited opacity as paint, and a cached
view rebuilds when that value changes. Tests cover zero-to-one reveal, nested
nonzero opacity changes, and repeated parent invalidation with unchanged
opacity retaining the child cache. The combined run passed 381 core, 54 Linux,
and 75 wgpu/text tests; the log is
`.dist/performance/opacity-input-full-tests.log`. Fresh platform captures are
required before treating a loaded-conversation memory sample as a verified
visible-conversation acceptance result. The later round7 Linux sample includes
an external 1898×1125 image showing the conversation and an empty composer.

## Native allocation attribution and backend discovery

Heaptrack traces of the same immutable round7 release binary distinguish
native retained allocations from Rust allocation counters. The binary SHA-256
is `d2ee8a04878c8f81f1a04842081a3fdc67c1a15984ab9073b1267c2c295bc908`.
Both runs used X11 at scale 1.25, the same 40-item completed-conversation
scenario, and external images verified the visible content and empty composer.
The Vulkan repeat used a new fixture server; its avatar identity differs.
Trace snapshots were cut about 30 seconds after content settled, before quit.

| Counter | Default discovery | Explicit Vulkan |
| --- | ---: | ---: |
| Native live requested bytes intercepted by heaptrack | 97,977,165 | 68,170,582 |
| Allocations still live at the snapshot | 35,896 | 13,240 |
| Live bytes attributed to the Vulkan driver as recorded leaf module | 66,780,781 | 66,780,813 |
| Live bytes attributed to Gallium as recorded leaf module | 29,500,640 | 0 |
| Separate uninstrumented 60-second process RSS | 236,142,592 | 210,796,544 |
| CPU time during that uninstrumented sample | 0.00 s | 0.01 s |

Both selected Vulkan llvmpipe, Mesa 26.2.2 / LLVM 21.1.8. Default discovery
also created an unused EGL context. Its blitter shader cache alone retained
26,906,344 requested bytes. Excluding GL discovery removed 29,806,583 native
live requested bytes, while the selected Vulkan driver's live total changed
by only 32 bytes. The separate process samples differed by 25,346,048 RSS
bytes. Intercepted allocation size, process RSS and GPU resource size are
different measurements; the profiler's own process RSS is not an acceptance
measurement. Statically linked mimalloc allocations and direct mappings are
not fully covered by this native trace.

The remaining Vulkan stacks include a 26,906,216-byte blitter cache,
19,817,876 bytes from graphics-pipeline creation, and 13,648,128 bytes in 208
command-buffer linear arenas of 65,616 bytes each. These stacks identify
owners for further experiments; they do not establish that the allocations
can be removed without changing rendering or command submission behavior.

The working fork now discovers native backends in separate instances and
keeps only the selected backend's instance, adapters and surface alive.
GL discovery can be deferred only when the known adapter outranks a
conservative best possible GL adapter. Otherwise GL is discovered before
selection, preserving explicit GPU choices, compositor affinity, hardware
over software, and failure fallback. Existing public constructors and the
web path remain available. Tests cover priority bounds and backend filtering.

Round8 then validated the default path with the same untouched fixture and
an external image. Its native live requested bytes were 68,384,745, down
29,592,420 from round7 default. Gallium live bytes fell from 29,500,640 to 176;
the selected Vulkan driver retained 66,780,813 bytes, exactly matching the
explicit-Vulkan experiment. This establishes that the losing GL context is
released in practice. A separate visible 60-second sample reported constant
RSS of 228,118,528 bytes and 0.01 seconds CPU time (0.0167% of one core).
Round8 also includes the event-arena, ARIA and mesh changes, so its 8,024,064-byte
RSS reduction is a combined checkpoint result.

Dropping the context does not unload all native driver code or return every
free allocator page. Round8 still had 11,739,136 resident bytes in
`libgallium`, absent from explicit Vulkan. Its main `[heap]` mapping used
27,484,160 resident bytes versus 22,372,352 in the explicit-Vulkan sample.
These maps explain why freeing about 29.6 MB of native requested allocations
does not imply an equal RSS reduction. The resolved `khronos-egl` 6.0 loader
deliberately uses `RTLD_NODELETE` on Linux. Its upstream discussion describes
a TLS-destructor failure when unloading libraries; removing that workaround
requires a separate compatibility investigation, not just dropping another
Rust handle. [Upstream loader issue](https://github.com/timothee-haudebourg/khronos-egl/issues/14).

A separate child-only preload called `malloc_trim(0)` once after 30 seconds
and captured glibc counters and `/proc/self/smaps` immediately before and
after. RSS fell 974,848 bytes (229,171,200 to 228,196,352); live `uordblks`
stayed 67,254,880 bytes. The call took 0.299 ms. This fixture does not justify
adding production trimming. The helper thread exited before the subsequent
10-second diagnostic sample, which recorded no CPU ticks. This was not an
acceptance run: the window manager moved the window after initial placement,
so its final external image clipped the bottom of the client. The same
1898×1125 renderer viewport, completed conversation and GPU allocations were
recorded. Before/after allocator evidence and the capture limitation are
preserved in `.dist/performance/linux-glibc-trim-round8`; no production trim
policy was added.

Raw traces, captures, exact snapshot timestamps and the combined comparison
are under `.dist/performance/linux-native-heaptrack-round7`,
`linux-native-heaptrack-round7-vulkan-clean`, and
`linux-native-heaptrack-backend-comparison.json` in the Editor workspace.
Round8 attribution and process results are in
`linux-native-heaptrack-round8-default`,
`linux-conversation-release-round8-default`, and
`linux-native-heaptrack-round8-comparison.json`.
Earlier Vulkan runs that inherited a typed draft are preserved with
`excluded-from-comparison.json`; their results are excluded from this table.

The 25–30 MB whole-application target is not established. GPU reservations,
Rust allocator storage, native allocations and mapped driver pages overlap
in process accounting and must not be summed as independent RSS savings.
