// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Mesh, MeshStyle,
    Pixels, Point, Radians, ScaledFilter, ScaledPixels, Size, bounds_tree::BoundsTree, point,
};
use smallvec::SmallVec;
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
    sync::Arc,
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    sort_scratch: Vec<SortEntry>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    pub meshes: Vec<PaintMesh>,
    pub backdrop_filters: Vec<BackdropFilter>,
    pub filter_boundaries: Vec<FilterBoundary>,
    /// The [`pulse_clock_ms`] reading renderers evaluate [`QuadPulse`]s at.
    /// The window stamps it on every present, so a frame that only advances
    /// pulses re-presents this scene without rebuilding it.
    pub pulse_clock_ms: u32,
    /// Counts drawn scenes. The window bumps it each time it finishes a new
    /// scene; a present that only advances pulses keeps it, so a renderer can
    /// tell "same content, later clock" from "new content".
    pub generation: u64,
    /// Device-pixel regions whose pixels may differ from scene
    /// `generation - 1`. `None` means anything may have changed and the
    /// renderer must repaint the whole target. Pulsing quads are not listed:
    /// they change on every present, and renderers add their bounds themselves.
    pub damage: Option<Vec<Bounds<ScaledPixels>>>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.sort_scratch.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.meshes.clear();
        self.backdrop_filters.clear();
        self.filter_boundaries.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    /// Raise the draw-order floor so every primitive inserted afterwards sorts above everything
    /// inserted before. Called before painting deferred draws so overlays (tooltips, popovers,
    /// drag images) sort above the main scene — and a deferred backdrop's order can't fall inside
    /// a content-filter (`filter`) order range left behind by the main scene.
    pub fn raise_order_floor(&mut self) {
        let floor = self.primitive_bounds.max_order() + 1;
        self.primitive_bounds.set_order_floor(floor);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let primitive = primitive.into();
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        // Content-filter boundaries must always be inserted as matched pairs — dropping one
        // (e.g. for an empty clipped region) would orphan its partner and corrupt the renderer's
        // target stack. Each marker takes an order strictly above ALL prior content, so the start
        // sorts after everything painted before it and the element's own children (which overlap
        // the marker bounds) sort strictly above the start. This keeps a marker's order range from
        // colliding with unrelated non-overlapping content that reuses low orderings (e.g. a
        // background grid), which would otherwise sweep that content into the group. Content
        // painted *after* the group is held above it by raising the order floor when the end
        // marker is inserted (see below) — otherwise a later non-overlapping sibling could reuse a
        // low order that lands inside the start..end range and be swept into the group.
        let is_filter_boundary = matches!(primitive, Primitive::FilterBoundary(_));

        if clipped_bounds.is_empty() && !is_filter_boundary {
            return;
        }

        let order = if is_filter_boundary {
            let order_bounds = if clipped_bounds.is_empty() {
                *primitive.bounds()
            } else {
                clipped_bounds
            };
            self.primitive_bounds.insert_above_all(order_bounds)
        } else {
            self.layer_stack
                .last()
                .copied()
                .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds))
        };
        let (kind, index) = match primitive {
            Primitive::Shadow(mut shadow) => {
                shadow.order = order;
                let index = self.shadows.len();
                self.shadows.push(shadow);
                (PrimitiveKind::Shadow, index)
            }
            Primitive::Quad(mut quad) => {
                quad.order = order;
                let index = self.quads.len();
                self.quads.push(quad);
                (PrimitiveKind::Quad, index)
            }
            Primitive::Path(mut path) => {
                path.order = order;
                let index = self.paths.len();
                path.id = PathId(index);
                self.paths.push(path);
                (PrimitiveKind::Path, index)
            }
            Primitive::Underline(mut underline) => {
                underline.order = order;
                let index = self.underlines.len();
                self.underlines.push(underline);
                (PrimitiveKind::Underline, index)
            }
            Primitive::MonochromeSprite(mut sprite) => {
                sprite.order = order;
                let index = self.monochrome_sprites.len();
                self.monochrome_sprites.push(sprite);
                (PrimitiveKind::MonochromeSprite, index)
            }
            Primitive::SubpixelSprite(mut sprite) => {
                sprite.order = order;
                let index = self.subpixel_sprites.len();
                self.subpixel_sprites.push(sprite);
                (PrimitiveKind::SubpixelSprite, index)
            }
            Primitive::PolychromeSprite(mut sprite) => {
                sprite.order = order;
                let index = self.polychrome_sprites.len();
                self.polychrome_sprites.push(sprite);
                (PrimitiveKind::PolychromeSprite, index)
            }
            Primitive::Surface(mut surface) => {
                surface.order = order;
                let index = self.surfaces.len();
                self.surfaces.push(surface);
                (PrimitiveKind::Surface, index)
            }
            Primitive::Mesh(mut mesh) => {
                mesh.order = order;
                let index = self.meshes.len();
                self.meshes.push(mesh);
                (PrimitiveKind::Mesh, index)
            }
            Primitive::BackdropFilter(mut filter) => {
                filter.order = order;
                let index = self.backdrop_filters.len();
                self.backdrop_filters.push(filter);
                (PrimitiveKind::BackdropFilter, index)
            }
            Primitive::FilterBoundary(mut boundary) => {
                boundary.order = order;
                if !boundary.is_start {
                    // A closed content-filter group is a draw-order barrier: everything painted
                    // afterwards must sort above the group's end marker so it can't fall back
                    // inside the group's order range (subsequent non-overlapping content otherwise
                    // reuses a low order). Mirrors the floor raised before deferred draws in
                    // `raise_order_floor`.
                    self.primitive_bounds.set_order_floor(order + 1);
                }
                let index = self.filter_boundaries.len();
                let kind = if boundary.is_start {
                    PrimitiveKind::FilterBoundaryStart
                } else {
                    PrimitiveKind::FilterBoundaryEnd
                };
                self.filter_boundaries.push(boundary);
                (kind, index)
            }
        };
        self.paint_operations
            .push(PaintOperation::Primitive { kind, index });
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive { kind, index } => {
                    self.insert_primitive(prev_scene.primitive_at(*kind, *index));
                }
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    fn primitive_at(&self, kind: PrimitiveKind, index: usize) -> Primitive {
        match kind {
            PrimitiveKind::Shadow => Primitive::Shadow(self.shadows[index]),
            PrimitiveKind::Quad => Primitive::Quad(self.quads[index]),
            PrimitiveKind::Path => Primitive::Path(self.paths[index].clone()),
            PrimitiveKind::Underline => Primitive::Underline(self.underlines[index]),
            PrimitiveKind::MonochromeSprite => {
                Primitive::MonochromeSprite(self.monochrome_sprites[index])
            }
            PrimitiveKind::SubpixelSprite => {
                Primitive::SubpixelSprite(self.subpixel_sprites[index])
            }
            PrimitiveKind::PolychromeSprite => {
                Primitive::PolychromeSprite(self.polychrome_sprites[index])
            }
            PrimitiveKind::Surface => Primitive::Surface(self.surfaces[index].clone()),
            PrimitiveKind::Mesh => Primitive::Mesh(self.meshes[index].clone()),
            PrimitiveKind::BackdropFilter => {
                Primitive::BackdropFilter(self.backdrop_filters[index].clone())
            }
            PrimitiveKind::FilterBoundaryStart | PrimitiveKind::FilterBoundaryEnd => {
                Primitive::FilterBoundary(self.filter_boundaries[index].clone())
            }
        }
    }

    pub fn finish(&mut self) {
        // Group scratch by typed storage in one pass over the replay log.
        // Each entry records its current vector position and owning operation.
        let counts = [
            self.shadows.len(),
            self.quads.len(),
            self.paths.len(),
            self.underlines.len(),
            self.monochrome_sprites.len(),
            self.subpixel_sprites.len(),
            self.polychrome_sprites.len(),
            self.surfaces.len(),
            self.meshes.len(),
            self.backdrop_filters.len(),
            self.filter_boundaries.len(),
        ];
        let mut offsets = [0; 12];
        self.sort_scratch.clear();
        for (kind, count) in counts.into_iter().enumerate() {
            offsets[kind + 1] = offsets[kind] + count;
            self.sort_scratch.extend((0..count).map(|index| SortEntry {
                index,
                operation: usize::MAX,
            }));
        }
        for (operation, op) in self.paint_operations.iter().enumerate() {
            if let PaintOperation::Primitive { kind, index } = op {
                self.sort_scratch[offsets[kind.storage_index()] + index].operation = operation;
            }
        }

        macro_rules! sort {
            ($field:ident, $kind:expr, $key:expr) => {
                sort_primitives(
                    &mut self.$field,
                    &mut self.paint_operations,
                    &mut self.sort_scratch[offsets[$kind]..offsets[$kind + 1]],
                    $key,
                );
            };
        }
        sort!(shadows, 0, |shadow| shadow.order);
        sort!(quads, 1, |quad| quad.order);
        sort!(paths, 2, |path| path.order);
        sort!(underlines, 3, |underline| underline.order);
        sort!(monochrome_sprites, 4, |sprite| (
            sprite.order,
            sprite.tile.texture_id.index,
            sprite.tile.tile_id
        ));
        sort!(subpixel_sprites, 5, |sprite| (
            sprite.order,
            sprite.tile.texture_id.index,
            sprite.tile.tile_id
        ));
        sort!(polychrome_sprites, 6, |sprite| (
            sprite.order,
            sprite.tile.texture_id.index,
            sprite.tile.tile_id
        ));
        sort!(surfaces, 7, |surface| surface.order);
        sort!(meshes, 8, |mesh| mesh.order);
        sort!(backdrop_filters, 9, |filter| filter.order);
        // A degenerate group whose markers tie must still open before closing.
        sort!(filter_boundaries, 10, |boundary| (
            boundary.order,
            !boundary.is_start
        ));
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
            meshes_start: 0,
            meshes_iter: self.meshes.iter().peekable(),
            backdrop_filters_start: 0,
            backdrop_filters_iter: self.backdrop_filters.iter().peekable(),
            filter_boundaries_start: 0,
            filter_boundaries_iter: self.filter_boundaries.iter().peekable(),
        }
    }
}

/// Internal representation of [`palette::Hsla`] which is layout sensitive, as its provided to the renderer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub struct SceneHsla {
    /// Hue, in a range from 0 to 1
    pub(crate) h: f32,
    /// Saturation, in a range from 0 to 1
    pub(crate) s: f32,
    /// Lightness, in a range from 0 to 1
    pub(crate) l: f32,
    /// Alpha, in a range from 0 to 1
    pub(crate) a: f32,
}
impl Into<palette::Hsla> for SceneHsla {
    fn into(self) -> palette::Hsla {
        palette::Hsla::new(self.h * 360.0, self.s, self.l, self.a)
    }
}
impl From<palette::Hsla> for SceneHsla {
    fn from(hsla: palette::Hsla) -> Self {
        Self {
            h: hsla.hue.into_positive_degrees() / 360.0,
            s: hsla.saturation,
            l: hsla.lightness,
            a: hsla.alpha,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    // Lowest discriminant: at an equal order, a content-filter group-start is emitted before
    // the group's own content so the renderer redirects rendering before any child draws.
    FilterBoundaryStart,
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
    Mesh,
    BackdropFilter,
    // Highest discriminant: at an equal order, a group-end is emitted after the group's content
    // so the renderer composites the filtered group only once every child has been drawn.
    FilterBoundaryEnd,
}

pub(crate) enum PaintOperation {
    Primitive { kind: PrimitiveKind, index: usize },
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

impl PrimitiveKind {
    fn storage_index(self) -> usize {
        match self {
            Self::Shadow => 0,
            Self::Quad => 1,
            Self::Path => 2,
            Self::Underline => 3,
            Self::MonochromeSprite => 4,
            Self::SubpixelSprite => 5,
            Self::PolychromeSprite => 6,
            Self::Surface => 7,
            Self::Mesh => 8,
            Self::BackdropFilter => 9,
            Self::FilterBoundaryStart | Self::FilterBoundaryEnd => 10,
        }
    }
}

struct SortEntry {
    index: usize,
    operation: usize,
}

fn sort_primitives<T, K: Ord>(
    primitives: &mut [T],
    operations: &mut [PaintOperation],
    scratch: &mut [SortEntry],
    key: impl Fn(&T) -> K,
) {
    // The original index breaks ties, matching stable sorting while avoiding
    // the stable sort's temporary allocation for every primitive kind.
    scratch.sort_unstable_by(|a, b| {
        key(&primitives[a.index])
            .cmp(&key(&primitives[b.index]))
            .then_with(|| a.index.cmp(&b.index))
    });
    for (index, entry) in scratch.iter().enumerate() {
        if entry.operation != usize::MAX
            && let PaintOperation::Primitive {
                index: recorded, ..
            } = &mut operations[entry.operation]
        {
            *recorded = index;
        }
    }
    // The operation mapping is no longer needed. Reuse that column for the
    // inverse permutation and move the typed values into place with swaps.
    for index in 0..scratch.len() {
        let original = scratch[index].index;
        scratch[original].operation = index;
    }
    for index in 0..scratch.len() {
        while scratch[index].operation != index {
            let destination = scratch[index].operation;
            primitives.swap(index, destination);
            let next = scratch[destination].operation;
            scratch[destination].operation = destination;
            scratch[index].operation = next;
        }
    }
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
    Mesh(PaintMesh),
    BackdropFilter(BackdropFilter),
    FilterBoundary(FilterBoundary),
}

#[expect(missing_docs)]
impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
            Primitive::Mesh(mesh) => &mesh.bounds,
            Primitive::BackdropFilter(filter) => &filter.bounds,
            Primitive::FilterBoundary(boundary) => &boundary.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
            Primitive::Mesh(mesh) => &mesh.content_mask,
            Primitive::BackdropFilter(filter) => &filter.content_mask,
            Primitive::FilterBoundary(boundary) => &boundary.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
    meshes_start: usize,
    meshes_iter: Peekable<slice::Iter<'a, PaintMesh>>,
    backdrop_filters_start: usize,
    backdrop_filters_iter: Peekable<slice::Iter<'a, BackdropFilter>>,
    filter_boundaries_start: usize,
    filter_boundaries_iter: Peekable<slice::Iter<'a, FilterBoundary>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
            (
                self.meshes_iter.peek().map(|m| m.order),
                PrimitiveKind::Mesh,
            ),
            (
                self.backdrop_filters_iter.peek().map(|f| f.order),
                PrimitiveKind::BackdropFilter,
            ),
            (
                self.filter_boundaries_iter.peek().map(|b| b.order),
                // The same vec yields both start and end markers; the discriminant decides
                // where the next marker sorts relative to draw batches at an equal order
                // (start before content, end after).
                match self.filter_boundaries_iter.peek() {
                    Some(boundary) if boundary.is_start => PrimitiveKind::FilterBoundaryStart,
                    _ => PrimitiveKind::FilterBoundaryEnd,
                },
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
            PrimitiveKind::Mesh => {
                let meshes_start = self.meshes_start;
                let mut meshes_end = meshes_start + 1;
                self.meshes_iter.next();
                while self
                    .meshes_iter
                    .next_if(|mesh| (mesh.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    meshes_end += 1;
                }
                self.meshes_start = meshes_end;
                Some(PrimitiveBatch::Meshes(meshes_start..meshes_end))
            }
            PrimitiveKind::BackdropFilter => {
                let backdrop_filters_start = self.backdrop_filters_start;
                let mut backdrop_filters_end = backdrop_filters_start + 1;
                self.backdrop_filters_iter.next();
                while self
                    .backdrop_filters_iter
                    .next_if(|filter| (filter.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    backdrop_filters_end += 1;
                }
                self.backdrop_filters_start = backdrop_filters_end;
                Some(PrimitiveBatch::BackdropFilters(
                    backdrop_filters_start..backdrop_filters_end,
                ))
            }
            // Boundaries are emitted one at a time (never merged) so the renderer can switch
            // render targets at exactly the right point in the batch stream.
            PrimitiveKind::FilterBoundaryStart | PrimitiveKind::FilterBoundaryEnd => {
                let index = self.filter_boundaries_start;
                self.filter_boundaries_iter.next();
                self.filter_boundaries_start = index + 1;
                Some(PrimitiveBatch::FilterBoundary(index))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
    /// Indices into [`Scene::meshes`].
    Meshes(Range<usize>),
    BackdropFilters(Range<usize>),
    /// A single content-filter group boundary; index into [`Scene::filter_boundaries`]. Read
    /// `is_start` to tell whether this opens the group (switch render target) or closes it
    /// (filter the offscreen target and composite it back).
    FilterBoundary(usize),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
            Self::Meshes(range) => format!("meshes ({})", range.len()),
            Self::BackdropFilters(range) => format!("backdrop filters ({})", range.len()),
            Self::FilterBoundary(ix) => format!("filter boundary ({ix})"),
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: SceneHsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
    pub pulse: QuadPulse,
}

/// A repeating opacity breath the renderer evaluates on the GPU every frame:
/// `min_opacity + (1 - min_opacity) * (cos(progress * TAU) * 0.5 + 0.5)`, where progress
/// runs from 0 to 1 over `period_ms` starting at `start_ms`. A pulsing quad
/// costs a present per frame, not a render, layout and paint.
#[derive(Default, Debug, Copy, Clone, PartialEq)]
#[repr(C)]
pub struct QuadPulse {
    /// The [`pulse_clock_ms`] reading the breath started at, so it opens at full opacity.
    pub start_ms: u32,
    /// One full breath. Zero means the quad does not pulse.
    pub period_ms: u32,
    /// The opacity factor at the bottom of the breath.
    pub min_opacity: f32,
    /// Keeps the struct 16 bytes, matching the shader declarations.
    pub pad: u32,
}

impl QuadPulse {
    /// Whether this quad breathes at all.
    pub fn is_active(&self) -> bool {
        self.period_ms != 0
    }
}

/// Milliseconds on the clock pulses share, wrapping every ~49 days. Shaders
/// subtract `start_ms` with wrapping arithmetic, so the wrap is invisible.
pub fn pulse_clock_ms() -> u32 {
    static EPOCH: std::sync::LazyLock<std::time::Instant> =
        std::sync::LazyLock::new(std::time::Instant::now);
    EPOCH.elapsed().as_millis() as u32
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: SceneHsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: SceneHsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    pub pad: u32, // align to 8 bytes
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// A backdrop filter blurs (and may otherwise filter) the content already rendered behind
/// `bounds`, compositing the result into a rounded rectangle — the frosted-glass effect.
/// Emitted by [`crate::Window::paint_backdrop_filter`]; produces the CSS `backdrop-filter` effect.
#[derive(Default, Debug, Clone)]
#[expect(missing_docs)]
pub struct BackdropFilter {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    /// The filter chain applied to the backdrop, in scene (device-pixel) space. Identity filters
    /// are dropped at paint time, so a `BackdropFilter` is only emitted when this is non-empty.
    ///
    /// Inline capacity is 4: a `SmallVec<[ScaledFilter; 4]>` is the same size as capacity 1 here
    /// (the heap repr already occupies that space), so chains up to 4 filters avoid allocating
    /// at no extra struct size.
    pub filters: SmallVec<[ScaledFilter; 4]>,
    /// Element opacity captured at paint time, multiplied into the composited result.
    pub opacity: f32,
}

impl From<BackdropFilter> for Primitive {
    fn from(filter: BackdropFilter) -> Self {
        Primitive::BackdropFilter(filter)
    }
}

/// The start or end marker of a content-filter (`filter`) isolation group. The element's
/// subtree is painted between a matched start/end pair; the renderer redirects that span into
/// an offscreen target, filters it, and composites it back at `bounds`. Produces the CSS
/// `filter` effect (e.g. blurring the element and its children as a single group).
#[derive(Debug, Clone)]
#[expect(missing_docs)]
pub struct FilterBoundary {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    /// The filter chain applied to the isolated group, in scene (device-pixel) space. Identity
    /// filters are dropped at paint time, so a `FilterBoundary` is only emitted when non-empty.
    /// Inline capacity 4 (same struct size as 1 here — see [`BackdropFilter::filters`]).
    pub filters: SmallVec<[ScaledFilter; 4]>,
    pub opacity: f32,
    /// `true` for the start marker (opens the group), `false` for the end marker (closes it).
    pub is_start: bool,
}

impl From<FilterBoundary> for Primitive {
    fn from(boundary: FilterBoundary) -> Self {
        Primitive::FilterBoundary(boundary)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: SceneHsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: SceneHsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(target_os = "macos")]
    pub image_buffer: core_video::pixel_buffer::CVPixelBuffer,
    #[cfg(any(
        target_os = "linux",
        target_os = "freebsd",
        all(target_os = "windows", feature = "wgpu-surfaces")
    ))]
    pub texture: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    #[cfg(any(
        target_os = "linux",
        target_os = "freebsd",
        all(target_os = "windows", feature = "wgpu-surfaces")
    ))]
    pub texture_size: Size<crate::DevicePixels>,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

/// A [`Mesh`] drawn into `bounds`; emitted by [`crate::Window::paint_mesh`].
#[derive(Clone, Debug)]
pub struct PaintMesh {
    /// Draw order, assigned when the primitive is inserted.
    pub order: DrawOrder,
    /// Where the mesh's viewport lies, in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    /// The clip in effect when the mesh was painted.
    pub content_mask: ContentMask<ScaledPixels>,
    /// Corners the composited viewport is rounded to.
    pub corner_radii: Corners<ScaledPixels>,
    /// Element opacity multiplied into the composite.
    pub opacity: f32,
    /// The geometry, cached on the GPU by its id.
    pub mesh: Arc<Mesh>,
    /// Camera, projection, and colors for this paint.
    pub style: MeshStyle,
}

impl From<PaintMesh> for Primitive {
    fn from(mesh: PaintMesh) -> Self {
        Primitive::Mesh(mesh)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Point, Size};

    fn sp(value: f32) -> ScaledPixels {
        ScaledPixels(value)
    }

    /// All test primitives cover the same region so the bounds tree assigns strictly
    /// increasing orders in insertion order — making the expected batch order deterministic.
    fn full_bounds() -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: sp(0.0),
                y: sp(0.0),
            },
            size: Size {
                width: sp(100.0),
                height: sp(100.0),
            },
        }
    }

    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: full_bounds(),
        }
    }

    fn quad() -> Quad {
        Quad {
            bounds: full_bounds(),
            content_mask: mask(),
            ..Default::default()
        }
    }

    /// A 100x100 quad whose bounds don't overlap `full_bounds()` (used to exercise the
    /// order-reuse path: non-overlapping content reuses low draw-orders).
    fn detached_quad() -> Quad {
        let bounds = Bounds {
            origin: Point {
                x: sp(200.0),
                y: sp(200.0),
            },
            size: Size {
                width: sp(100.0),
                height: sp(100.0),
            },
        };
        Quad {
            bounds,
            content_mask: ContentMask { bounds },
            ..Default::default()
        }
    }

    fn boundary(is_start: bool) -> FilterBoundary {
        FilterBoundary {
            order: 0,
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            filters: smallvec::smallvec![ScaledFilter::Blur(sp(8.0))],
            opacity: 1.0,
            is_start,
        }
    }

    fn backdrop() -> BackdropFilter {
        BackdropFilter {
            bounds: full_bounds(),
            content_mask: mask(),
            corner_radii: Corners::default(),
            filters: smallvec::smallvec![ScaledFilter::Blur(sp(20.0))],
            opacity: 1.0,
            ..Default::default()
        }
    }

    fn batch_kinds(scene: &mut Scene) -> Vec<&'static str> {
        scene.finish();
        scene
            .batches()
            .map(|batch| match batch {
                PrimitiveBatch::Quads(_) => "quad",
                PrimitiveBatch::BackdropFilters(_) => "backdrop",
                PrimitiveBatch::FilterBoundary(ix) => {
                    if scene.filter_boundaries[ix].is_start {
                        "start"
                    } else {
                        "end"
                    }
                }
                _ => "other",
            })
            .collect()
    }

    #[test]
    fn content_filter_group_brackets_its_children() {
        let mut scene = Scene::default();
        // Background painted before the filtered element.
        scene.insert_primitive(quad());
        // A content-filtered element: start marker, its child, end marker.
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(false));

        // The start must precede the group's child and the end must follow it, so the
        // renderer can redirect rendering for exactly the group's span.
        assert_eq!(
            batch_kinds(&mut scene),
            vec!["quad", "start", "quad", "end"]
        );
    }

    // Note: this validates only the *scene ordering* of nested filter boundaries (start/child/
    // end interleaving), not that a renderer actually isolates both levels — that depends on the
    // backend's group-texture pool (see MAX_FILTER_DEPTH) and is exercised by the `blur` example.
    #[test]
    fn nested_content_filters_emit_well_nested_ordering() {
        let mut scene = Scene::default();
        scene.insert_primitive(boundary(true)); // outer start
        scene.insert_primitive(quad()); // outer child
        scene.insert_primitive(boundary(true)); // inner start
        scene.insert_primitive(quad()); // inner child
        scene.insert_primitive(boundary(false)); // inner end
        scene.insert_primitive(boundary(false)); // outer end

        assert_eq!(
            batch_kinds(&mut scene),
            vec!["start", "quad", "start", "quad", "end", "end"]
        );
    }

    #[test]
    fn content_after_a_filter_group_sorts_above_it() {
        let mut scene = Scene::default();
        // A content-filtered element: start marker, its child, end marker.
        scene.insert_primitive(boundary(true));
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(false));
        // A sibling painted after the group that does NOT overlap it. Without the close-time
        // order-floor it would reuse the lowest order, tie with the start marker, and be swept
        // into the group (start, quad, quad, end); it must instead sort after the end marker.
        scene.insert_primitive(detached_quad());

        assert_eq!(
            batch_kinds(&mut scene),
            vec!["start", "quad", "end", "quad"]
        );
    }

    #[test]
    fn backdrop_filter_sorts_before_a_later_overlapping_quad() {
        let mut scene = Scene::default();
        // Content behind the frosted panel.
        scene.insert_primitive(quad());
        // The panel: its backdrop snapshot, then its (translucent) background quad on top.
        scene.insert_primitive(backdrop());
        scene.insert_primitive(quad());

        assert_eq!(batch_kinds(&mut scene), vec!["quad", "backdrop", "quad"]);
    }

    #[test]
    fn compact_sort_preserves_stable_ties_and_every_replay_reference() {
        for count in 0..80 {
            for seed in 0..12 {
                let mut values = (0..count)
                    .map(|index| ((index * 37 + seed * 13) % 11, index))
                    .collect::<Vec<_>>();
                let mut expected = values.clone();
                expected.sort_by_key(|value| value.0);
                let mut operations = (0..count)
                    .map(|index| PaintOperation::Primitive {
                        kind: PrimitiveKind::Quad,
                        index,
                    })
                    .collect::<Vec<_>>();
                let mut scratch = (0..count)
                    .map(|index| SortEntry {
                        index,
                        operation: index,
                    })
                    .collect::<Vec<_>>();

                sort_primitives(&mut values, &mut operations, &mut scratch, |value| value.0);
                assert_eq!(values, expected);
                for (original, op) in operations.iter().enumerate() {
                    let PaintOperation::Primitive { index, .. } = op else {
                        unreachable!()
                    };
                    assert_eq!(values[*index].1, original);
                }
            }
        }
    }

    fn sprite(tile: u32, marker: f32) -> MonochromeSprite {
        MonochromeSprite {
            order: 0,
            pad: 0,
            bounds: full_bounds(),
            content_mask: mask(),
            color: SceneHsla {
                l: marker,
                ..Default::default()
            },
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: 0,
                    kind: crate::AtlasTextureKind::Monochrome,
                },
                tile_id: crate::TileId(tile),
                padding: 0,
                bounds: Bounds::default(),
            },
            transformation: TransformationMatrix::default(),
        }
    }

    fn recorded_sprites(scene: &Scene) -> Vec<(u32, f32)> {
        scene
            .paint_operations
            .iter()
            .filter_map(|op| {
                let PaintOperation::Primitive {
                    kind: PrimitiveKind::MonochromeSprite,
                    index,
                } = op
                else {
                    return None;
                };
                let sprite = &scene.monochrome_sprites[*index];
                Some((sprite.tile.tile_id.0, sprite.color.l))
            })
            .collect()
    }

    #[test]
    fn sprite_pages_batch_together_without_changing_cached_replay() {
        use crate::AtlasTextureKind;

        fn recorded(scene: &Scene) -> Vec<(u32, u32, f32)> {
            scene
                .paint_operations
                .iter()
                .map(|operation| {
                    let PaintOperation::Primitive { kind, index } = operation else {
                        unreachable!();
                    };
                    let (tile, bounds) = match scene.primitive_at(*kind, *index) {
                        Primitive::MonochromeSprite(sprite) => (sprite.tile, sprite.bounds),
                        Primitive::SubpixelSprite(sprite) => (sprite.tile, sprite.bounds),
                        Primitive::PolychromeSprite(sprite) => (sprite.tile, sprite.bounds),
                        _ => unreachable!(),
                    };
                    (tile.texture_id.index, tile.tile_id.0, bounds.origin.x.0)
                })
                .collect()
        }

        for kind in [
            AtlasTextureKind::Monochrome,
            AtlasTextureKind::Subpixel,
            AtlasTextureKind::Polychrome,
        ] {
            let mut scene = Scene::default();
            for (i, (page, tile)) in [(1, 0), (0, 0), (1, 1), (0, 1), (1, 0)]
                .into_iter()
                .enumerate()
            {
                let mut sprite = sprite(tile, i as f32);
                sprite.bounds.origin.x = sp(i as f32 * 20.0);
                sprite.bounds.size.width = sp(10.0);
                sprite.tile.texture_id = AtlasTextureId { index: page, kind };
                match kind {
                    AtlasTextureKind::Monochrome => scene.insert_primitive(sprite),
                    AtlasTextureKind::Subpixel => scene.insert_primitive(SubpixelSprite {
                        order: 0,
                        pad: 0,
                        bounds: sprite.bounds,
                        content_mask: sprite.content_mask,
                        color: sprite.color,
                        tile: sprite.tile,
                        transformation: sprite.transformation,
                    }),
                    AtlasTextureKind::Polychrome => scene.insert_primitive(PolychromeSprite {
                        order: 0,
                        pad: 0,
                        grayscale: false.into(),
                        opacity: 1.0,
                        bounds: sprite.bounds,
                        content_mask: sprite.content_mask,
                        corner_radii: Corners::default(),
                        tile: sprite.tile,
                    }),
                }
            }
            let expected = recorded(&scene);
            scene.finish();
            scene.finish();
            assert_eq!(recorded(&scene), expected);
            let batches = scene
                .batches()
                .map(|batch| match batch {
                    PrimitiveBatch::MonochromeSprites { texture_id, range }
                    | PrimitiveBatch::SubpixelSprites { texture_id, range }
                    | PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                        (texture_id.index, range.len())
                    }
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            assert_eq!(batches, vec![(0, 2), (1, 3)]);

            let mut replay = Scene::default();
            replay.replay(1..4, &scene);
            replay.finish();
            assert_eq!(recorded(&replay), expected[1..4]);
        }
    }

    #[test]
    fn sprite_page_sort_keeps_overlapping_draws_in_paint_order() {
        let mut scene = Scene::default();
        let mut back = sprite(0, 0.0);
        back.tile.texture_id.index = 1;
        scene.insert_primitive(back);
        scene.insert_primitive(sprite(0, 1.0));
        scene.finish();

        assert_eq!(scene.monochrome_sprites[0].tile.texture_id.index, 1);
        assert_eq!(scene.monochrome_sprites[1].tile.texture_id.index, 0);
        assert!(scene.monochrome_sprites[0].order < scene.monochrome_sprites[1].order);
        assert_eq!(scene.batches().count(), 2);
    }

    #[test]
    fn cached_replay_survives_tile_sort_repeated_finish_and_append() {
        let mut scene = Scene::default();
        scene.push_layer(full_bounds());
        for (tile, marker) in [(3, 0.), (1, 1.), (1, 2.), (2, 3.)] {
            scene.insert_primitive(sprite(tile, marker));
        }
        scene.pop_layer();
        let expected = recorded_sprites(&scene);
        scene.finish();
        scene.finish();
        assert_eq!(recorded_sprites(&scene), expected);
        assert_eq!(
            scene
                .monochrome_sprites
                .iter()
                .map(|sprite| sprite.color.l)
                .collect::<Vec<_>>(),
            vec![1., 2., 3., 0.]
        );

        // Replay a cached subrange, not only the entire sorted scene.
        let mut partial = Scene::default();
        partial.push_layer(full_bounds());
        partial.replay(2..4, &scene);
        partial.pop_layer();
        partial.finish();
        assert_eq!(recorded_sprites(&partial), expected[1..3]);

        scene.insert_primitive(sprite(0, 4.));
        scene.finish();
        let mut replay = Scene::default();
        replay.replay(0..scene.len(), &scene);
        replay.finish();
        assert_eq!(
            recorded_sprites(&replay),
            vec![(3, 0.), (1, 1.), (1, 2.), (2, 3.), (0, 4.)]
        );
        assert!(replay.layer_stack.is_empty());
        let capacity = replay.sort_scratch.capacity();
        replay.clear();
        replay.insert_primitive(sprite(9, 5.));
        replay.finish();
        assert_eq!(recorded_sprites(&replay), vec![(9, 5.)]);
        assert_eq!(replay.sort_scratch.capacity(), capacity);
    }

    #[test]
    fn path_replay_assigns_fresh_ids_after_order_sort() {
        let mut scene = Scene::default();
        for marker in 0..3 {
            let mut path = Path::new(point(crate::px(marker as f32), crate::px(0.))).scale(1.);
            path.bounds = if marker == 2 {
                detached_quad().bounds
            } else {
                full_bounds()
            };
            path.content_mask = ContentMask {
                bounds: path.bounds,
            };
            scene.insert_primitive(path);
        }
        scene.finish();
        let mut replay = Scene::default();
        replay.replay(1..3, &scene);
        assert_eq!(
            replay
                .paths
                .iter()
                .map(|path| path.id.0)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            replay
                .paths
                .iter()
                .map(|path| path.start.x.0)
                .collect::<Vec<_>>(),
            vec![1., 2.]
        );
    }

    #[test]
    fn filter_and_layer_replay_preserves_batch_order() {
        let mut scene = Scene::default();
        scene.insert_primitive(quad());
        scene.insert_primitive(boundary(true));
        scene.push_layer(full_bounds());
        scene.insert_primitive(quad());
        scene.insert_primitive(backdrop());
        scene.pop_layer();
        scene.insert_primitive(boundary(false));
        scene.insert_primitive(detached_quad());
        let expected = batch_kinds(&mut scene);
        let mut replay = Scene::default();
        replay.replay(0..scene.len(), &scene);
        assert_eq!(batch_kinds(&mut replay), expected);
    }
}
