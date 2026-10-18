// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTextureKind, AtlasTile, Background, Bounds, ContentMask, Corners, Edges,
    Hsla, Mesh, MeshStyle, Pixels, Point, Radians, ScaledFilter, ScaledPixels, Size, TextColorMap,
    TileId, bounds_tree::BoundsTree, point,
};
use collections::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::{
    fmt::Debug,
    iter::Peekable,
    mem,
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
    /// The frame of a [`crate::SnapshotTransition`] to composite over this
    /// scene, stamped by the window on every present like
    /// [`Self::pulse_clock_ms`]. `None` when no transition plays; a renderer
    /// holding a snapshot drops it then and repaints the whole target.
    pub snapshot_transition: Option<SnapshotTransitionFrame>,
    /// Counts drawn scenes. The window bumps it each time it finishes a new
    /// scene; a present that only advances pulses keeps it, so a renderer can
    /// tell "same content, later clock" from "new content".
    pub generation: u64,
    /// Device-pixel regions whose pixels may differ from scene
    /// `generation - 1`, and from every earlier scene drawn since the window
    /// last presented one, so damage also holds against the last scene a
    /// renderer presented. `None` means anything may have changed and the
    /// renderer must repaint the whole target. Pulsing quads are not listed:
    /// they change on every present, and renderers add their bounds themselves.
    /// Each region is rounded out to whole device pixels plus one pixel of
    /// margin, and covers the full reach of what changed: shadow blur, sprite
    /// transforms, and every backdrop or content filter whose blur samples it.
    pub damage: Option<Vec<Bounds<ScaledPixels>>>,
    /// Per typed vector, the indices whose contents may differ from scene
    /// `generation - 1`, and from every scene drawn since the window last
    /// presented one: a renderer that keeps the previous instance buffers can
    /// upload just these. Empty ranges are unchanged. `None` means any index
    /// may have changed, and so may every vector's length.
    pub changed_primitives: Option<PrimitiveRanges>,
    /// The regions repainted in place: what [`crate::Window::paint_animated`]
    /// painted, and cached views promoted to live layers. Each takes one
    /// operation where it was painted. Regions replaced by a repaint of the
    /// region around them stay, marked dead, so indices hold.
    pub(crate) repaint_regions: Vec<RepaintRegion>,
    /// The live regions, by id.
    pub(crate) region_indices: FxHashMap<RepaintId, usize>,
    /// The regions being painted, innermost last.
    open_regions: Vec<OpenRegion>,
    /// Kept between regions so opening one doesn't allocate.
    spare_region_bounds: Vec<BoundsTree<ScaledPixels>>,
    /// The vector ranges region replacements changed since the window last
    /// took them.
    pub(crate) replaced_primitives: PrimitiveRanges,
    /// While [`crate::Window::record_paint`] runs, paints land here instead
    /// of in the scene.
    pub(crate) recording: Option<Vec<RecordedPaint>>,
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
        self.repaint_regions.clear();
        self.region_indices.clear();
        self.open_regions.clear();
        self.replaced_primitives = PrimitiveRanges::default();
        self.recording = None;
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        // A recording keeps layers, so a replayed line of text still stacks
        // its glyphs at one order.
        if let Some(recording) = &mut self.recording {
            recording.push(RecordedPaint::StartLayer(bounds));
            return;
        }
        let order = match self.open_regions.last_mut() {
            Some(region) => {
                self.repaint_regions[region.index]
                    .operations
                    .push(RegionOperation::StartLayer(bounds));
                region.bounds.insert(bounds)
            }
            None => {
                self.paint_operations
                    .push(PaintOperation::StartLayer(bounds));
                self.primitive_bounds.insert(bounds)
            }
        };
        self.layer_stack.push(order);
    }

    pub fn pop_layer(&mut self) {
        if let Some(recording) = &mut self.recording {
            recording.push(RecordedPaint::EndLayer);
            return;
        }
        self.layer_stack.pop();
        match self.open_regions.last() {
            Some(region) => self.repaint_regions[region.index]
                .operations
                .push(RegionOperation::EndLayer),
            None => self.paint_operations.push(PaintOperation::EndLayer),
        }
    }

    /// Raise the draw-order floor so every primitive inserted afterwards sorts above everything
    /// inserted before. Called before painting deferred draws so overlays (tooltips, popovers,
    /// drag images) sort above the main scene — and a deferred backdrop's order can't fall inside
    /// a content-filter (`filter`) order range left behind by the main scene.
    pub fn raise_order_floor(&mut self) {
        debug_assert!(self.open_regions.is_empty());
        let floor = self.primitive_bounds.max_order() + 1;
        self.primitive_bounds.set_order_floor(floor);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let primitive = primitive.into();
        if let Some(recording) = &mut self.recording {
            recording.push(RecordedPaint::Primitive(primitive));
            return;
        }
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

        // A repaint region stacks its primitives in its own orders.
        let bounds = match self.open_regions.last_mut() {
            Some(region) => &mut region.bounds,
            None => &mut self.primitive_bounds,
        };
        let order = if is_filter_boundary {
            let order_bounds = if clipped_bounds.is_empty() {
                *primitive.bounds()
            } else {
                clipped_bounds
            };
            bounds.insert_above_all(order_bounds)
        } else {
            self.layer_stack
                .last()
                .copied()
                .unwrap_or_else(|| bounds.insert(clipped_bounds))
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
                    bounds.set_order_floor(order + 1);
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
        match self.open_regions.last() {
            Some(region) => {
                let region = region.index;
                let primitive = self.primitive_at(kind, index);
                self.repaint_regions[region]
                    .operations
                    .push(RegionOperation::Primitive(primitive));
            }
            None => self
                .paint_operations
                .push(PaintOperation::Primitive { kind, index }),
        }
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive { kind, index } => {
                    self.insert_primitive(prev_scene.primitive_at(*kind, *index));
                }
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
                PaintOperation::Repaint(index) => self.replay_region(*index, prev_scene),
            }
        }
    }

    fn replay_region(&mut self, index: usize, prev_scene: &Scene) {
        let region = &prev_scene.repaint_regions[index];
        // A recording keeps paint output, not regions: the region's current
        // primitives are recorded as they are.
        let recording = self.recording.is_some();
        if !recording {
            self.begin_repaint_region(region.id);
        }
        for operation in &region.operations {
            match operation {
                RegionOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                RegionOperation::StartLayer(bounds) => self.push_layer(*bounds),
                RegionOperation::EndLayer => self.pop_layer(),
                RegionOperation::Region(index) => self.replay_region(*index, prev_scene),
            }
        }
        if !recording {
            self.end_repaint_region();
        }
    }

    /// Opens a repaint region stacking above everything painted so far, in
    /// the region around it if any. Until [`Self::end_repaint_region`],
    /// primitives join the region.
    pub(crate) fn begin_repaint_region(&mut self, id: RepaintId) {
        let bounds = match self.open_regions.last() {
            Some(region) => &region.bounds,
            None => &self.primitive_bounds,
        };
        let base = (bounds.max_order() + 1).max(bounds.order_floor());
        self.open_region(id, base, true);
    }

    /// Opens a region stacking from `base`, to paint a region's replacement
    /// into an empty scene at the region's own base. The replacement keeps
    /// the slack the region already has rather than reserving its own.
    pub(crate) fn begin_repaint_region_at(&mut self, id: RepaintId, base: DrawOrder) {
        self.open_region(id, base, false);
    }

    fn open_region(&mut self, id: RepaintId, base: DrawOrder, slack: bool) {
        let mut bounds = self.spare_region_bounds.pop().unwrap_or_default();
        bounds.clear();
        bounds.set_order_floor(base);
        let index = self.repaint_regions.len();
        match self.open_regions.last() {
            Some(parent) => self.repaint_regions[parent.index]
                .operations
                .push(RegionOperation::Region(index)),
            None => self.paint_operations.push(PaintOperation::Repaint(index)),
        }
        self.repaint_regions.push(RepaintRegion {
            id,
            orders: base..base,
            painted_end: base,
            operations: Vec::new(),
            dead: false,
        });
        self.region_indices.insert(id, index);
        let painted = self.slack_lengths();
        self.open_regions.push(OpenRegion {
            index,
            bounds,
            outer_layers: mem::take(&mut self.layer_stack),
            slack: slack.then_some(painted),
        });
    }

    /// Closes the innermost open region, reserving its orders: everything
    /// painted afterwards stacks above them. A region painted in place leaves
    /// slack behind its primitives, so repainting it can usually overwrite its
    /// own slots instead of moving the rest of the scene.
    pub(crate) fn end_repaint_region(&mut self) {
        let OpenRegion {
            index,
            bounds,
            outer_layers,
            slack,
        } = self.open_regions.pop().expect("no repaint region is open");
        debug_assert!(self.layer_stack.is_empty(), "unbalanced layer in region");
        self.layer_stack = outer_layers;
        let region = &mut self.repaint_regions[index];
        // The floor covers the orders reserved by regions nested in it.
        region.painted_end = (bounds.max_order() + 1)
            .max(bounds.order_floor())
            .max(region.orders.start);
        region.orders.end = region
            .painted_end
            .max(region.orders.start.saturating_add(REPAINT_REGION_ORDERS));
        let end = region.orders.end;
        match self.open_regions.last_mut() {
            Some(parent) => parent.bounds.set_order_floor(end),
            None => self.primitive_bounds.set_order_floor(end),
        }
        self.spare_region_bounds.push(bounds);
        if let Some(painted) = slack {
            self.reserve_slack(painted, end - 1);
        }
    }

    /// The lengths of the vectors regions keep slack in.
    fn slack_lengths(&self) -> [usize; 6] {
        [
            self.shadows.len(),
            self.quads.len(),
            self.underlines.len(),
            self.monochrome_sprites.len(),
            self.subpixel_sprites.len(),
            self.polychrome_sprites.len(),
        ]
    }

    /// Appends invisible primitives at `order` after what a region painted
    /// since the vectors had lengths `painted`: half as many again plus two
    /// in each vector the region painted into.
    fn reserve_slack(&mut self, painted: [usize; 6], order: DrawOrder) {
        macro_rules! reserve {
            ($field:ident, $slot:expr) => {{
                let count = self.$field.len() - painted[$slot];
                if let Some(template) = self.$field.last() {
                    if count > 0 {
                        let tombstone = template.tombstone(order);
                        self.$field
                            .extend(std::iter::repeat_n(tombstone, count / 2 + 2));
                    }
                }
            }};
        }
        reserve!(shadows, 0);
        reserve!(quads, 1);
        reserve!(underlines, 2);
        reserve!(monochrome_sprites, 3);
        reserve!(subpixel_sprites, 4);
        reserve!(polychrome_sprites, 5);
    }

    /// Whether the region at `index`, or one nested in it, holds a pulsing
    /// quad.
    pub(crate) fn region_pulses(&self, index: usize) -> bool {
        self.repaint_regions[index]
            .operations
            .iter()
            .any(|operation| match operation {
                RegionOperation::Primitive(Primitive::Quad(quad)) => quad.pulse.is_active(),
                RegionOperation::Region(index) => self.region_pulses(*index),
                _ => false,
            })
    }

    /// Replaces the primitives of the finished scene's region at `index` with
    /// what `repainted` painted for its first region (and the regions nested
    /// in it), from the same base order, and adds both to `damage`. The
    /// region's orders are its own, so its primitives form one run in each
    /// sorted vector: the replacement overwrites the run and its slack, and
    /// only moves the rest of a vector when it outgrows the run. Returns
    /// `false`, changing nothing, when the replacement stacks beyond the
    /// orders reserved for the region.
    pub(crate) fn replace_repaint_region(
        &mut self,
        index: usize,
        repainted: &mut Scene,
        damage: &mut Damage,
    ) -> bool {
        let orders = self.repaint_regions[index].orders.clone();
        debug_assert_eq!(repainted.repaint_regions[0].orders.start, orders.start);
        if repainted.repaint_regions[0].painted_end > orders.end {
            return false;
        }
        self.add_repaint_damage(index, repainted, damage);

        // Per storage: where the region's run ended, and how far it grew.
        let mut shifts = [(usize::MAX, 0isize); 11];
        let mut changed = PrimitiveRanges::default();
        macro_rules! replace {
            ($field:ident, $storage:expr, $key:expr, $tombstones:expr) => {{
                let start = self.$field.partition_point(|p| p.order < orders.start);
                let end = self.$field.partition_point(|p| p.order < orders.end);
                let mut run = mem::take(&mut repainted.$field);
                let old_len = self.$field.len();
                if run.len() <= end - start && ($tombstones || run.len() == end - start) {
                    if run.len() < end - start {
                        let template = run.last().unwrap_or(&self.$field[start]).clone();
                        let tombstone = template.tombstone(orders.end - 1);
                        run.resize(end - start, tombstone);
                    }
                    run.sort_by_key($key);
                    for (slot, primitive) in self.$field[start..end].iter_mut().zip(run.drain(..)) {
                        *slot = primitive;
                    }
                    changed.$field = start..end;
                } else {
                    run.sort_by_key($key);
                    shifts[$storage] = (end, run.len() as isize - (end - start) as isize);
                    self.$field.splice(start..end, run.drain(..));
                    changed.$field = start..old_len.max(self.$field.len());
                }
                repainted.$field = run;
            }};
        }
        replace!(shadows, 0, |shadow| shadow.order, true);
        replace!(quads, 1, |quad| quad.order, true);
        replace!(paths, 2, |path| path.order, false);
        replace!(underlines, 3, |underline| underline.order, true);
        replace!(
            monochrome_sprites,
            4,
            |sprite| (
                sprite.order,
                sprite.tile.texture_id.index,
                sprite.tile.tile_id
            ),
            true
        );
        replace!(
            subpixel_sprites,
            5,
            |sprite| (
                sprite.order,
                sprite.tile.texture_id.index,
                sprite.tile.tile_id
            ),
            true
        );
        replace!(
            polychrome_sprites,
            6,
            |sprite| (
                sprite.order,
                sprite.tile.texture_id.index,
                sprite.tile.tile_id
            ),
            true
        );
        replace!(surfaces, 7, |surface| surface.order, false);
        replace!(meshes, 8, |mesh| mesh.order, false);
        replace!(backdrop_filters, 9, |filter| filter.order, false);
        replace!(
            filter_boundaries,
            10,
            |boundary| (boundary.order, !boundary.is_start),
            false
        );
        // Operations outside the region index past runs that moved.
        if shifts.iter().any(|(_, growth)| *growth != 0) {
            for operation in &mut self.paint_operations {
                if let PaintOperation::Primitive { kind, index } = operation {
                    let (end, growth) = shifts[kind.storage_index()];
                    if *index >= end {
                        *index = (*index as isize + growth) as usize;
                    }
                }
            }
        }
        self.replaced_primitives.extend(&changed);

        // The regions nested in the old contents are gone; those nested in
        // the new ones move in after the existing regions.
        self.retire_nested_regions(index);
        let offset = self.repaint_regions.len() - 1;
        let remap = |operations: &mut Vec<RegionOperation>| {
            for operation in operations {
                if let RegionOperation::Region(nested) = operation {
                    *nested += offset;
                }
            }
        };
        let mut regions = repainted.repaint_regions.drain(..);
        let mut replacement = regions.next().unwrap();
        remap(&mut replacement.operations);
        for (position, mut nested) in regions.enumerate() {
            remap(&mut nested.operations);
            self.region_indices.insert(nested.id, offset + 1 + position);
            self.repaint_regions.push(nested);
        }
        let region = &mut self.repaint_regions[index];
        region.painted_end = replacement.painted_end;
        region.operations = replacement.operations;
        true
    }

    /// Whether `repainted`'s first region stacks within the orders reserved
    /// for the region at `index`, so [`Self::replace_repaint_region`] can
    /// take it.
    pub(crate) fn fits_repaint(&self, index: usize, repainted: &Scene) -> bool {
        repainted.repaint_regions[0].painted_end <= self.repaint_regions[index].orders.end
    }

    /// Marks every region nested in the region at `index` dead.
    fn retire_nested_regions(&mut self, index: usize) {
        let nested = self.repaint_regions[index]
            .operations
            .iter()
            .filter_map(|operation| match operation {
                RegionOperation::Region(nested) => Some(*nested),
                _ => None,
            })
            .collect::<SmallVec<[usize; 4]>>();
        for index in nested {
            self.retire_nested_regions(index);
            let region = &mut self.repaint_regions[index];
            region.dead = true;
            if self.region_indices.get(&region.id) == Some(&index) {
                self.region_indices.remove(&region.id);
            }
        }
    }

    /// The ids of the regions nested in the region at `index`, at any depth.
    pub(crate) fn nested_region_ids(&self, index: usize, ids: &mut Vec<RepaintId>) {
        for operation in &self.repaint_regions[index].operations {
            if let RegionOperation::Region(nested) = operation {
                ids.push(self.repaint_regions[*nested].id);
                self.nested_region_ids(*nested, ids);
            }
        }
    }

    /// Adds to `damage` where `repainted`'s first region paints differently
    /// from the region at `index`: the primitives between their common prefix
    /// and suffix, as [`Self::damage_since`] does for whole scenes, so a
    /// repaint that changes one primitive damages only that one.
    fn add_repaint_damage(&self, index: usize, repainted: &Scene, damage: &mut Damage) {
        let mut old = Vec::new();
        self.region_primitives(index, &mut old);
        let mut new = Vec::new();
        repainted.region_primitives(0, &mut new);
        let common = old.len().min(new.len());
        let prefix = (0..common)
            .find(|&ix| !old[ix].paints_same_as(new[ix]))
            .unwrap_or(common);
        let suffix = (0..common - prefix)
            .find(|&ix| !old[old.len() - 1 - ix].paints_same_as(new[new.len() - 1 - ix]))
            .unwrap_or(common - prefix);
        for primitive in old[prefix..old.len() - suffix]
            .iter()
            .chain(&new[prefix..new.len() - suffix])
        {
            damage.add(primitive.visible_bounds());
        }
    }

    /// The primitives of the region at `index` and those nested in it, in
    /// paint order.
    fn region_primitives<'a>(&'a self, index: usize, primitives: &mut Vec<&'a Primitive>) {
        for operation in &self.repaint_regions[index].operations {
            match operation {
                RegionOperation::Primitive(primitive) => primitives.push(primitive),
                RegionOperation::Region(nested) => self.region_primitives(*nested, primitives),
                RegionOperation::StartLayer(_) | RegionOperation::EndLayer => {}
            }
        }
    }

    fn add_region_damage(&self, index: usize, damage: &mut Damage) {
        for operation in &self.repaint_regions[index].operations {
            match operation {
                RegionOperation::Primitive(primitive) => damage.add(primitive.visible_bounds()),
                RegionOperation::Region(nested) => self.add_region_damage(*nested, damage),
                RegionOperation::StartLayer(_) | RegionOperation::EndLayer => {}
            }
        }
    }

    fn region_paints_same(&self, index: usize, other: &Scene, other_index: usize) -> bool {
        let (region, other_region) = (
            &self.repaint_regions[index],
            &other.repaint_regions[other_index],
        );
        region.orders == other_region.orders
            && region.operations.len() == other_region.operations.len()
            && region
                .operations
                .iter()
                .zip(&other_region.operations)
                .all(|pair| match pair {
                    (RegionOperation::Primitive(a), RegionOperation::Primitive(b)) => {
                        a.paints_same_as(b)
                    }
                    (RegionOperation::StartLayer(a), RegionOperation::StartLayer(b)) => a == b,
                    (RegionOperation::EndLayer, RegionOperation::EndLayer) => true,
                    (RegionOperation::Region(a), RegionOperation::Region(b)) => {
                        self.region_paints_same(*a, other, *b)
                    }
                    _ => false,
                })
    }

    /// The paint of the operations in `range`, in paint order, with orders
    /// zeroed: what a redraw must reproduce, whatever orders and slack it
    /// ends up with. The regions in `layers` stand in for their contents by
    /// id, and those in `animated` anonymously, since their ids change each
    /// time they're painted.
    pub(crate) fn flatten(
        &self,
        range: Range<usize>,
        layers: &FxHashSet<RepaintId>,
        animated: &FxHashSet<RepaintId>,
    ) -> Vec<FlatPaint> {
        let mut paints = Vec::new();
        for operation in &self.paint_operations[range] {
            match operation {
                PaintOperation::Primitive { kind, index } => {
                    paints.push(FlatPaint::Primitive(flat_primitive(
                        self.primitive_at(*kind, *index),
                    )));
                }
                PaintOperation::StartLayer(bounds) => paints.push(FlatPaint::StartLayer(*bounds)),
                PaintOperation::EndLayer => paints.push(FlatPaint::EndLayer),
                PaintOperation::Repaint(index) => {
                    self.flatten_region(*index, layers, animated, &mut paints)
                }
            }
        }
        paints
    }

    fn flatten_region(
        &self,
        index: usize,
        layers: &FxHashSet<RepaintId>,
        animated: &FxHashSet<RepaintId>,
        paints: &mut Vec<FlatPaint>,
    ) {
        let region = &self.repaint_regions[index];
        if layers.contains(&region.id) {
            paints.push(FlatPaint::Region(Some(region.id)));
            return;
        }
        if animated.contains(&region.id) {
            paints.push(FlatPaint::Region(None));
            return;
        }
        for operation in &region.operations {
            match operation {
                RegionOperation::Primitive(primitive) => {
                    paints.push(FlatPaint::Primitive(flat_primitive(primitive.clone())));
                }
                RegionOperation::StartLayer(bounds) => paints.push(FlatPaint::StartLayer(*bounds)),
                RegionOperation::EndLayer => paints.push(FlatPaint::EndLayer),
                RegionOperation::Region(nested) => {
                    self.flatten_region(*nested, layers, animated, paints)
                }
            }
        }
    }

    /// Adds to `damage` everywhere this scene may paint differently from
    /// `previous`, the scene drawn before it.
    ///
    /// Operations match while they paint equal primitives at equal orders, so
    /// a common prefix and suffix of matching operations paints identically
    /// wherever nothing between them reaches. Every primitive between them
    /// counts, in both scenes: what disappeared must be repainted too. A
    /// repaint region in the same orders in both matches as a whole and counts
    /// its own primitives if they changed.
    pub(crate) fn damage_since(&self, previous: &Scene, damage: &mut Damage) {
        let (operations, previous_operations) =
            (&self.paint_operations, &previous.paint_operations);
        let common = operations.len().min(previous_operations.len());
        let mut prefix = 0;
        while prefix < common && self.operations_match(prefix, previous, prefix, damage) {
            prefix += 1;
        }
        let mut suffix = 0;
        while prefix + suffix < common
            && self.operations_match(
                operations.len() - 1 - suffix,
                previous,
                previous_operations.len() - 1 - suffix,
                damage,
            )
        {
            suffix += 1;
        }
        for operation in &previous_operations[prefix..previous_operations.len() - suffix] {
            previous.add_operation_damage(operation, damage);
        }
        for operation in &operations[prefix..operations.len() - suffix] {
            self.add_operation_damage(operation, damage);
        }
    }

    fn operations_match(
        &self,
        index: usize,
        previous: &Scene,
        previous_index: usize,
        damage: &mut Damage,
    ) -> bool {
        match (
            &self.paint_operations[index],
            &previous.paint_operations[previous_index],
        ) {
            (
                PaintOperation::Primitive { kind, index },
                PaintOperation::Primitive {
                    kind: previous_kind,
                    index: previous_index,
                },
            ) => {
                kind == previous_kind && self.paints_same(*kind, *index, previous, *previous_index)
            }
            (PaintOperation::StartLayer(bounds), PaintOperation::StartLayer(previous_bounds)) => {
                bounds == previous_bounds
            }
            (PaintOperation::EndLayer, PaintOperation::EndLayer) => true,
            (PaintOperation::Repaint(index), PaintOperation::Repaint(previous_index)) => {
                // Regions own their orders, so one in the same orders matches
                // whichever closure painted it.
                if self.repaint_regions[*index].orders
                    != previous.repaint_regions[*previous_index].orders
                {
                    return false;
                }
                if !self.region_paints_same(*index, previous, *previous_index) {
                    self.add_region_damage(*index, damage);
                    previous.add_region_damage(*previous_index, damage);
                }
                true
            }
            _ => false,
        }
    }

    fn paints_same(
        &self,
        kind: PrimitiveKind,
        index: usize,
        other: &Scene,
        other_index: usize,
    ) -> bool {
        match kind {
            PrimitiveKind::Shadow => self.shadows[index] == other.shadows[other_index],
            PrimitiveKind::Quad => self.quads[index] == other.quads[other_index],
            PrimitiveKind::Path => self.paths[index].paints_same_as(&other.paths[other_index]),
            PrimitiveKind::Underline => self.underlines[index] == other.underlines[other_index],
            PrimitiveKind::MonochromeSprite => {
                self.monochrome_sprites[index] == other.monochrome_sprites[other_index]
            }
            PrimitiveKind::SubpixelSprite => {
                self.subpixel_sprites[index] == other.subpixel_sprites[other_index]
            }
            PrimitiveKind::PolychromeSprite => {
                self.polychrome_sprites[index] == other.polychrome_sprites[other_index]
            }
            // A surface's texture changes without the primitive changing.
            PrimitiveKind::Surface => false,
            PrimitiveKind::Mesh => self.meshes[index].paints_same_as(&other.meshes[other_index]),
            PrimitiveKind::BackdropFilter => {
                self.backdrop_filters[index] == other.backdrop_filters[other_index]
            }
            PrimitiveKind::FilterBoundaryStart | PrimitiveKind::FilterBoundaryEnd => {
                self.filter_boundaries[index] == other.filter_boundaries[other_index]
            }
        }
    }

    fn add_operation_damage(&self, operation: &PaintOperation, damage: &mut Damage) {
        match operation {
            PaintOperation::Primitive { kind, index } => {
                damage.add(self.visible_bounds(*kind, *index));
            }
            PaintOperation::Repaint(index) => self.add_region_damage(*index, damage),
            PaintOperation::StartLayer(_) | PaintOperation::EndLayer => {}
        }
    }

    fn visible_bounds(&self, kind: PrimitiveKind, index: usize) -> Bounds<ScaledPixels> {
        match kind {
            PrimitiveKind::Shadow => self.shadows[index].visible_bounds(),
            PrimitiveKind::Quad => self.quads[index].visible_bounds(),
            PrimitiveKind::Path => self.paths[index].visible_bounds(),
            PrimitiveKind::Underline => self.underlines[index].visible_bounds(),
            PrimitiveKind::MonochromeSprite => self.monochrome_sprites[index].visible_bounds(),
            PrimitiveKind::SubpixelSprite => self.subpixel_sprites[index].visible_bounds(),
            PrimitiveKind::PolychromeSprite => self.polychrome_sprites[index].visible_bounds(),
            PrimitiveKind::Surface => self.surfaces[index].visible_bounds(),
            PrimitiveKind::Mesh => self.meshes[index].visible_bounds(),
            PrimitiveKind::BackdropFilter => self.backdrop_filters[index].visible_bounds(),
            PrimitiveKind::FilterBoundaryStart | PrimitiveKind::FilterBoundaryEnd => {
                self.filter_boundaries[index].visible_bounds()
            }
        }
    }

    /// Backdrop filters and content-filter groups blur what lies within their
    /// reach, so damage anywhere in that reach changes all they paint. Adds
    /// each such filter's painted bounds to `damage`, until no more apply.
    pub(crate) fn add_filter_damage(&self, damage: &mut Damage) {
        let filters = self
            .backdrop_filters
            .iter()
            .map(|filter| (filter.bounds, &filter.filters, filter.visible_bounds()))
            .chain(
                self.filter_boundaries
                    .iter()
                    .filter(|boundary| boundary.is_start)
                    .map(|boundary| {
                        (
                            boundary.bounds,
                            &boundary.filters,
                            boundary.visible_bounds(),
                        )
                    }),
            )
            .collect::<SmallVec<[_; 8]>>();
        let mut added = SmallVec::<[bool; 8]>::from_elem(false, filters.len());
        loop {
            let mut grew = false;
            for ((bounds, filters, visible), added) in filters.iter().zip(&mut added) {
                if !*added && damage.intersects(&bounds.dilate(blur_reach(filters))) {
                    damage.add(*visible);
                    *added = true;
                    grew = true;
                }
            }
            if !grew {
                break;
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
    Primitive {
        kind: PrimitiveKind,
        index: usize,
    },
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
    /// The repaint region at this index of [`Scene::repaint_regions`].
    Repaint(usize),
}

/// Identifies a repaint region across frames. The window keys the region's
/// retained paint by it, and replaying the region keeps it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RepaintId(pub(crate) u64);

impl RepaintId {
    /// The frame-rate overlay's region, repainted by the window itself.
    /// Retained paints number theirs from here.
    pub(crate) const FRAME_OVERLAY: Self = Self(0);
}

/// Draw orders reserved for each repaint region. A region stacks its
/// primitives in its own orders, above everything painted before it and below
/// everything painted after, so a repaint restacks the region without
/// reordering the rest of the scene, however its primitives move.
const REPAINT_REGION_ORDERS: DrawOrder = 256;

/// What a [`crate::Window::paint_animated`] closure painted. Its primitives
/// sit in the typed vectors like any other; its operations live here instead
/// of in [`Scene::paint_operations`], where the region takes one slot, so a
/// repaint can change how much it paints without moving the operation indices
/// that cached views and deferred draws hold into the scene.
pub(crate) struct RepaintRegion {
    pub(crate) id: RepaintId,
    /// The orders reserved for the region. Its primitives use a prefix.
    pub(crate) orders: Range<DrawOrder>,
    /// One past the highest order the region's primitives use.
    painted_end: DrawOrder,
    operations: Vec<RegionOperation>,
    /// Replaced along with the region around it.
    pub(crate) dead: bool,
}

enum RegionOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
    /// A region nested in this one.
    Region(usize),
}

struct OpenRegion {
    index: usize,
    /// Assigns the region's orders, starting at its base.
    bounds: BoundsTree<ScaledPixels>,
    /// The layers open around the region, which its primitives don't join.
    outer_layers: Vec<DrawOrder>,
    /// To reserve slack when it closes: the slack vectors' lengths when it
    /// opened.
    slack: Option<[usize; 6]>,
}

/// A paint in paint order, as [`Scene::flatten`] lists them.
#[derive(Debug)]
pub(crate) enum FlatPaint {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
    /// A region whose contents aren't compared: a layer by id, or a retained
    /// paint.
    Region(Option<RepaintId>),
}

impl PartialEq for FlatPaint {
    /// Surfaces always differ (see [`Primitive::paints_same_as`]), so they
    /// compare by geometry here.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                FlatPaint::Primitive(Primitive::Surface(a)),
                FlatPaint::Primitive(Primitive::Surface(b)),
            ) => a.bounds == b.bounds && a.content_mask == b.content_mask,
            (FlatPaint::Primitive(a), FlatPaint::Primitive(b)) => a.paints_same_as(b),
            (FlatPaint::StartLayer(a), FlatPaint::StartLayer(b)) => a == b,
            (FlatPaint::EndLayer, FlatPaint::EndLayer) => true,
            (FlatPaint::Region(a), FlatPaint::Region(b)) => a == b,
            _ => false,
        }
    }
}

/// A primitive as compared across redraws: its order is the redraw's own.
fn flat_primitive(mut primitive: Primitive) -> Primitive {
    match &mut primitive {
        Primitive::Shadow(shadow) => shadow.order = 0,
        Primitive::Quad(quad) => quad.order = 0,
        Primitive::Path(path) => {
            path.order = 0;
            path.id = PathId(0);
        }
        Primitive::Underline(underline) => underline.order = 0,
        Primitive::MonochromeSprite(sprite) => sprite.order = 0,
        Primitive::SubpixelSprite(sprite) => sprite.order = 0,
        Primitive::PolychromeSprite(sprite) => sprite.order = 0,
        Primitive::Surface(surface) => surface.order = 0,
        Primitive::Mesh(mesh) => mesh.order = 0,
        Primitive::BackdropFilter(filter) => filter.order = 0,
        Primitive::FilterBoundary(boundary) => boundary.order = 0,
    }
    primitive
}

/// For each typed primitive vector of a [`Scene`], a range of indices.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[expect(missing_docs)]
pub struct PrimitiveRanges {
    pub shadows: Range<usize>,
    pub quads: Range<usize>,
    pub paths: Range<usize>,
    pub underlines: Range<usize>,
    pub monochrome_sprites: Range<usize>,
    pub subpixel_sprites: Range<usize>,
    pub polychrome_sprites: Range<usize>,
    pub surfaces: Range<usize>,
    pub meshes: Range<usize>,
    pub backdrop_filters: Range<usize>,
    pub filter_boundaries: Range<usize>,
}

impl PrimitiveRanges {
    /// Widens each range to cover `other`'s too.
    pub(crate) fn extend(&mut self, other: &Self) {
        fn cover(range: &mut Range<usize>, other: &Range<usize>) {
            if other.start >= other.end {
            } else if range.start >= range.end {
                *range = other.clone();
            } else {
                *range = range.start.min(other.start)..range.end.max(other.end);
            }
        }
        cover(&mut self.shadows, &other.shadows);
        cover(&mut self.quads, &other.quads);
        cover(&mut self.paths, &other.paths);
        cover(&mut self.underlines, &other.underlines);
        cover(&mut self.monochrome_sprites, &other.monochrome_sprites);
        cover(&mut self.subpixel_sprites, &other.subpixel_sprites);
        cover(&mut self.polychrome_sprites, &other.polychrome_sprites);
        cover(&mut self.surfaces, &other.surfaces);
        cover(&mut self.meshes, &other.meshes);
        cover(&mut self.backdrop_filters, &other.backdrop_filters);
        cover(&mut self.filter_boundaries, &other.filter_boundaries);
    }
}

/// A primitive that paints nothing, holding a slot in a region's run.
trait Tombstone: Clone {
    fn tombstone(&self, order: DrawOrder) -> Self;
}

macro_rules! tombstone {
    // A slack sprite names no texture.
    (sprite $type:ty $(, $field:ident = $value:expr)*) => {
        impl Tombstone for $type {
            fn tombstone(&self, order: DrawOrder) -> Self {
                Self {
                    order,
                    bounds: Bounds::default(),
                    content_mask: ContentMask::default(),
                    tile: slack_tile(self.tile.texture_id.kind),
                    $($field: $value,)*
                    ..self.clone()
                }
            }
        }
    };
    ($type:ty $(, $field:ident = $value:expr)*) => {
        impl Tombstone for $type {
            fn tombstone(&self, order: DrawOrder) -> Self {
                Self {
                    order,
                    bounds: Bounds::default(),
                    content_mask: ContentMask::default(),
                    $($field: $value,)*
                    ..self.clone()
                }
            }
        }
    };
}
tombstone!(
    Shadow,
    blur_radius = ScaledPixels(0.),
    element_bounds = Bounds::default()
);
tombstone!(Quad, pulse = QuadPulse::default());
tombstone!(Underline);
tombstone!(sprite MonochromeSprite, transformation = TransformationMatrix::unit());
tombstone!(sprite SubpixelSprite, transformation = TransformationMatrix::unit());
tombstone!(sprite PolychromeSprite);

/// The texture index of slack sprites, which no atlas texture has: slack
/// names no texture, so it can't name one that's gone.
const SLACK_TEXTURE_INDEX: u32 = u32::MAX;

fn slack_tile(kind: AtlasTextureKind) -> AtlasTile {
    AtlasTile::slack(kind)
}

fn is_slack(tile: &AtlasTile) -> bool {
    tile.is_slack()
}

impl AtlasTile {
    /// The tile of a slack sprite: an empty slot a live layer keeps in a
    /// sprite vector. It names no atlas texture and sorts after every real
    /// tile at its order. [`Scene::batches`] never starts a batch with one;
    /// inside a batch, slack draws as nothing with the batch's texture.
    #[doc(hidden)]
    pub fn slack(kind: AtlasTextureKind) -> Self {
        AtlasTile {
            texture_id: AtlasTextureId {
                index: SLACK_TEXTURE_INDEX,
                kind,
            },
            tile_id: TileId(u32::MAX),
            padding: 0,
            bounds: Bounds::default(),
        }
    }

    /// Whether this is the tile of a slack sprite (see [`Self::slack`]).
    #[doc(hidden)]
    pub fn is_slack(&self) -> bool {
        self.texture_id.index == SLACK_TEXTURE_INDEX
    }
}

/// Kinds without slack only ever replace their run with as many primitives.
macro_rules! no_tombstone {
    ($($type:ty),*) => {$(
        impl Tombstone for $type {
            fn tombstone(&self, _: DrawOrder) -> Self {
                unreachable!("{} keeps no slack", stringify!($type))
            }
        }
    )*};
}
no_tombstone!(
    Path<ScaledPixels>,
    PaintSurface,
    PaintMesh,
    BackdropFilter,
    FilterBoundary
);

/// A paint captured by [`crate::Window::record_paint`].
#[derive(Clone)]
pub(crate) enum RecordedPaint {
    Primitive(Primitive),
    /// A glyph keeps its color unmapped, so replaying it under a
    /// [`TextColorMap`] maps it afresh.
    Glyph(RecordedGlyph),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
pub(crate) struct RecordedGlyph {
    pub(crate) bounds: Bounds<ScaledPixels>,
    pub(crate) content_mask: ContentMask<ScaledPixels>,
    pub(crate) color: Hsla,
    pub(crate) opacity: f32,
    pub(crate) tile: AtlasTile,
    pub(crate) subpixel: bool,
    /// The map in effect when recorded, used when none is at replay.
    pub(crate) color_map: Option<TextColorMap>,
}

/// Device-pixel regions that may paint differently, merged into a few
/// rectangles. Merging only ever grows the regions.
#[derive(Default)]
pub(crate) struct Damage {
    regions: SmallVec<[Bounds<ScaledPixels>; MAX_DAMAGE_REGIONS]>,
}

/// Beyond this many regions, the closest ones merge. Renderers redraw each
/// region separately, so a few larger ones beat many slivers.
const MAX_DAMAGE_REGIONS: usize = 16;

impl Damage {
    pub(crate) fn add(&mut self, bounds: Bounds<ScaledPixels>) {
        let covers = |outer: &Bounds<ScaledPixels>, inner: &Bounds<ScaledPixels>| {
            outer.left() <= inner.left()
                && outer.top() <= inner.top()
                && outer.right() >= inner.right()
                && outer.bottom() >= inner.bottom()
        };
        if bounds.is_empty() || self.regions.iter().any(|region| covers(region, &bounds)) {
            return;
        }
        self.regions.retain(|region| !covers(&bounds, region));
        if self.regions.len() < MAX_DAMAGE_REGIONS {
            self.regions.push(bounds);
            return;
        }
        let area = |bounds: &Bounds<ScaledPixels>| bounds.size.width.0 * bounds.size.height.0;
        if let Some(closest) = self.regions.iter_mut().min_by(|a, b| {
            let growth =
                |region: &Bounds<ScaledPixels>| area(&region.union(&bounds)) - area(region);
            growth(a).total_cmp(&growth(b))
        }) {
            *closest = closest.union(&bounds);
        }
    }

    pub(crate) fn intersects(&self, bounds: &Bounds<ScaledPixels>) -> bool {
        self.regions.iter().any(|region| region.intersects(bounds))
    }

    pub(crate) fn merge(&mut self, other: Damage) {
        for region in other.regions {
            self.add(region);
        }
    }

    /// The regions, rounded out to whole device pixels with a pixel of margin
    /// for antialiasing.
    pub(crate) fn regions(&self) -> Vec<Bounds<ScaledPixels>> {
        self.regions
            .iter()
            .map(|region| {
                Bounds::from_corners(
                    point(
                        ScaledPixels(region.left().0.floor() - 1.0),
                        ScaledPixels(region.top().0.floor() - 1.0),
                    ),
                    point(
                        ScaledPixels(region.right().0.ceil() + 1.0),
                        ScaledPixels(region.bottom().0.ceil() + 1.0),
                    ),
                )
            })
            .collect()
    }
}

/// How far a filter chain's blur reaches beyond the content it samples: three
/// standard deviations of each blur, the extent renderers sample.
fn blur_reach(filters: &[ScaledFilter]) -> ScaledPixels {
    ScaledPixels(filters.iter().fold(0.0, |reach, filter| match filter {
        ScaledFilter::Blur(radius) => reach + 3.0 * radius.0,
    }))
}

fn clip(bounds: Bounds<ScaledPixels>, mask: &ContentMask<ScaledPixels>) -> Bounds<ScaledPixels> {
    bounds.intersect(&mask.bounds)
}

impl Shadow {
    fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        // Outer shadows leave room for the gaussian tail; inset ones paint
        // inside the element.
        let reach = self
            .bounds
            .dilate(ScaledPixels(3.0 * self.blur_radius.0))
            .union(&self.element_bounds);
        clip(reach, &self.content_mask)
    }
}

impl Quad {
    fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        clip(self.bounds, &self.content_mask)
    }
}

impl Path<ScaledPixels> {
    fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        clip(self.bounds, &self.content_mask)
    }

    /// Like `==`, without the id, which only numbers paths within a scene.
    fn paints_same_as(&self, other: &Self) -> bool {
        self.order == other.order
            && self.bounds == other.bounds
            && self.content_mask == other.content_mask
            && self.color == other.color
            && self.vertices == other.vertices
    }
}

impl Underline {
    fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        // A wave may swing past its box by up to its thickness.
        let reach = if self.wavy == PaddedBool32::from(true) {
            self.bounds.dilate(self.thickness)
        } else {
            self.bounds
        };
        clip(reach, &self.content_mask)
    }
}

impl MonochromeSprite {
    fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        clip(
            transformed_bounds(self.bounds, &self.transformation),
            &self.content_mask,
        )
    }
}

impl SubpixelSprite {
    fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        clip(
            transformed_bounds(self.bounds, &self.transformation),
            &self.content_mask,
        )
    }
}

impl PolychromeSprite {
    fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        clip(self.bounds, &self.content_mask)
    }
}

impl PaintSurface {
    fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        clip(self.bounds, &self.content_mask)
    }
}

impl PaintMesh {
    fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        clip(self.bounds, &self.content_mask)
    }

    /// Meshes are immutable once shared, so the same mesh paints the same.
    fn paints_same_as(&self, other: &Self) -> bool {
        self.order == other.order
            && self.bounds == other.bounds
            && self.content_mask == other.content_mask
            && self.corner_radii == other.corner_radii
            && self.opacity == other.opacity
            && Arc::ptr_eq(&self.mesh, &other.mesh)
            && self.style == other.style
    }
}

impl BackdropFilter {
    /// The filter paints only inside its bounds, whatever its blur samples.
    fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        clip(self.bounds, &self.content_mask)
    }
}

impl FilterBoundary {
    /// The filtered group's blur spreads past the group's bounds, as far as
    /// its blur reaches.
    fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        clip(
            self.bounds.dilate(blur_reach(&self.filters)),
            &self.content_mask,
        )
    }
}

/// Where a primitive drawn at `bounds` through `transformation` lands.
fn transformed_bounds(
    bounds: Bounds<ScaledPixels>,
    transformation: &TransformationMatrix,
) -> Bounds<ScaledPixels> {
    if *transformation == TransformationMatrix::unit() {
        return bounds;
    }
    let [[a, b], [c, d]] = transformation.rotation_scale;
    let [tx, ty] = transformation.translation;
    let corners = [
        bounds.origin,
        bounds.top_right(),
        bounds.bottom_left(),
        bounds.bottom_right(),
    ]
    .map(|corner| {
        (
            a * corner.x.0 + b * corner.y.0 + tx,
            c * corner.x.0 + d * corner.y.0 + ty,
        )
    });
    let (mut left, mut top, mut right, mut bottom) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (x, y) in corners {
        left = left.min(x);
        top = top.min(y);
        right = right.max(x);
        bottom = bottom.max(y);
    }
    Bounds::from_corners(
        point(ScaledPixels(left), ScaledPixels(top)),
        point(ScaledPixels(right), ScaledPixels(bottom)),
    )
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

#[derive(Clone, Debug)]
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
    /// Where the primitive can paint pixels: its bounds, widened by blur,
    /// shadow and transform reach, and clipped to its content mask.
    pub(crate) fn visible_bounds(&self) -> Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => shadow.visible_bounds(),
            Primitive::Quad(quad) => quad.visible_bounds(),
            Primitive::Path(path) => path.visible_bounds(),
            Primitive::Underline(underline) => underline.visible_bounds(),
            Primitive::MonochromeSprite(sprite) => sprite.visible_bounds(),
            Primitive::SubpixelSprite(sprite) => sprite.visible_bounds(),
            Primitive::PolychromeSprite(sprite) => sprite.visible_bounds(),
            Primitive::Surface(surface) => surface.visible_bounds(),
            Primitive::Mesh(mesh) => mesh.visible_bounds(),
            Primitive::BackdropFilter(filter) => filter.visible_bounds(),
            Primitive::FilterBoundary(boundary) => boundary.visible_bounds(),
        }
    }

    /// Whether drawing `other` instead would paint the same pixels at the
    /// same order. Surfaces never do: their texture changes underneath.
    pub(crate) fn paints_same_as(&self, other: &Primitive) -> bool {
        match (self, other) {
            (Primitive::Shadow(a), Primitive::Shadow(b)) => a == b,
            (Primitive::Quad(a), Primitive::Quad(b)) => a == b,
            (Primitive::Path(a), Primitive::Path(b)) => a.paints_same_as(b),
            (Primitive::Underline(a), Primitive::Underline(b)) => a == b,
            (Primitive::MonochromeSprite(a), Primitive::MonochromeSprite(b)) => a == b,
            (Primitive::SubpixelSprite(a), Primitive::SubpixelSprite(b)) => a == b,
            (Primitive::PolychromeSprite(a), Primitive::PolychromeSprite(b)) => a == b,
            (Primitive::Mesh(a), Primitive::Mesh(b)) => a.paints_same_as(b),
            (Primitive::BackdropFilter(a), Primitive::BackdropFilter(b)) => a == b,
            (Primitive::FilterBoundary(a), Primitive::FilterBoundary(b)) => a == b,
            _ => false,
        }
    }

    pub(crate) fn content_mask_mut(&mut self) -> &mut ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &mut shadow.content_mask,
            Primitive::Quad(quad) => &mut quad.content_mask,
            Primitive::Path(path) => &mut path.content_mask,
            Primitive::Underline(underline) => &mut underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &mut sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &mut sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &mut sprite.content_mask,
            Primitive::Surface(surface) => &mut surface.content_mask,
            Primitive::Mesh(mesh) => &mut mesh.content_mask,
            Primitive::BackdropFilter(filter) => &mut filter.content_mask,
            Primitive::FilterBoundary(boundary) => &mut boundary.content_mask,
        }
    }

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
        // No batch starts with slack, so every batch names a real texture.
        while self
            .monochrome_sprites_iter
            .next_if(|sprite| is_slack(&sprite.tile))
            .is_some()
        {
            self.monochrome_sprites_start += 1;
        }
        while self
            .subpixel_sprites_iter
            .next_if(|sprite| is_slack(&sprite.tile))
            .is_some()
        {
            self.subpixel_sprites_start += 1;
        }
        while self
            .polychrome_sprites_iter
            .next_if(|sprite| is_slack(&sprite.tile))
            .is_some()
        {
            self.polychrome_sprites_start += 1;
        }
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
                // Slack inside a run draws with it, as nothing.
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && (sprite.tile.texture_id == texture_id || is_slack(&sprite.tile))
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
                // Slack inside a run draws with it, as nothing.
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && (sprite.tile.texture_id == texture_id || is_slack(&sprite.tile))
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
                // Slack inside a run draws with it, as nothing.
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && (sprite.tile.texture_id == texture_id || is_slack(&sprite.tile))
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

#[derive(Default, Debug, Copy, Clone, PartialEq)]
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

/// One present of a [`crate::SnapshotTransition`]: how far the diagonal edge
/// has swept, and how the band along it looks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnapshotTransitionFrame {
    /// Eased progress in `[0, 1]`. At 0 the snapshot covers the window; at 1
    /// the band has left past the bottom-right corner.
    pub progress: f32,
    /// The band's width as a fraction of the window diagonal.
    pub band: f32,
    /// The band's highlight as straight sRGB `[r, g, b, a]`; `a` is the peak
    /// strength.
    pub shimmer: [f32; 4],
}

impl SnapshotTransitionFrame {
    /// Where the edge sits on the diagonal coordinate `(x / W + y / H) / 2`:
    /// it starts a band before 0 and ends a band past 1, so the band enters
    /// and leaves whole. Pixels past the edge still show the snapshot.
    pub fn edge(&self) -> f32 {
        -self.band + self.progress * (1.0 + 2.0 * self.band)
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

#[derive(Debug, Copy, Clone, PartialEq)]
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

#[derive(Debug, Copy, Clone, PartialEq)]
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
#[derive(Default, Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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

#[derive(Copy, Clone, Debug, PartialEq)]
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

#[derive(Copy, Clone, Debug, PartialEq)]
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

#[derive(Copy, Clone, Debug, PartialEq)]
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

#[derive(Clone, Debug, PartialEq)]
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

    fn colored(bounds: Bounds<ScaledPixels>, lightness: f32) -> Quad {
        Quad {
            bounds,
            content_mask: ContentMask { bounds },
            background: crate::hsla(0.0, 0.0, lightness, 1.0).into(),
            ..Default::default()
        }
    }

    /// Paints `quads` stacked, as one region, into an empty scene at `base`.
    fn repainted(base: DrawOrder, quads: &[Quad]) -> Scene {
        let mut scene = Scene::default();
        scene.begin_repaint_region_at(RepaintId(1), base);
        for quad in quads {
            scene.insert_primitive(*quad);
        }
        scene.end_repaint_region();
        scene
    }

    fn lightness(quad: &Quad) -> f32 {
        quad.background.solid.l
    }

    /// The lightness of each quad that paints, skipping a region's slack.
    fn painted(quads: &[Quad]) -> Vec<f32> {
        quads
            .iter()
            .filter(|quad| !quad.bounds.is_empty())
            .map(lightness)
            .collect()
    }

    #[test]
    fn a_repainted_region_keeps_the_scene_around_it_in_paint_order() {
        let mut scene = Scene::default();
        scene.insert_primitive(colored(full_bounds(), 0.1));
        scene.begin_repaint_region(RepaintId(1));
        scene.insert_primitive(colored(full_bounds(), 0.2));
        scene.end_repaint_region();
        scene.insert_primitive(colored(full_bounds(), 0.9));
        scene.finish();
        let base = scene.repaint_regions[0].orders.start;

        // Three quads where one was fill the region's slack in place: the
        // quad painted after it stays where it is.
        let quads = scene.quads.len();
        let after = scene.quads.len() - 1;
        let stacked = [0.3, 0.4, 0.5].map(|l| colored(full_bounds(), l));
        let mut damage = Damage::default();
        assert!(scene.replace_repaint_region(0, &mut repainted(base, &stacked), &mut damage));
        assert_eq!(painted(&scene.quads), vec![0.1, 0.3, 0.4, 0.5, 0.9]);
        assert_eq!(scene.quads.len(), quads);
        assert_eq!(lightness(&scene.quads[after]), 0.9);
        assert_eq!(scene.replaced_primitives.quads, 1..after);

        // Outgrowing the slack moves what follows, and its operation too.
        let five = [0.3, 0.4, 0.5, 0.6, 0.7].map(|l| colored(full_bounds(), l));
        assert!(scene.replace_repaint_region(0, &mut repainted(base, &five), &mut damage));
        assert_eq!(
            painted(&scene.quads),
            vec![0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
        );
        let stacked = [0.3, 0.4, 0.5].map(|l| colored(full_bounds(), l));
        assert!(scene.replace_repaint_region(0, &mut repainted(base, &stacked), &mut damage));
        assert!(
            scene
                .quads
                .windows(2)
                .all(|pair| pair[0].order <= pair[1].order)
        );
        assert_eq!(damage.regions().len(), 1);

        // Replaying the patched scene, as a reused cached view does, paints
        // the same thing.
        let mut replay = Scene::default();
        replay.replay(0..scene.len(), &scene);
        replay.finish();
        assert_eq!(painted(&replay.quads), vec![0.1, 0.3, 0.4, 0.5, 0.9]);

        // Shrinking back moves the later quad's operation back too.
        let single = [colored(full_bounds(), 0.6)];
        assert!(scene.replace_repaint_region(0, &mut repainted(base, &single), &mut damage));
        let mut replay = Scene::default();
        replay.replay(0..scene.len(), &scene);
        replay.finish();
        assert_eq!(painted(&replay.quads), vec![0.1, 0.6, 0.9]);
    }

    #[test]
    fn a_region_too_deep_for_its_orders_is_left_alone() {
        let mut scene = Scene::default();
        scene.begin_repaint_region(RepaintId(1));
        scene.insert_primitive(quad());
        scene.end_repaint_region();
        scene.finish();
        let base = scene.repaint_regions[0].orders.start;

        let deep = vec![quad(); REPAINT_REGION_ORDERS as usize + 1];
        let mut damage = Damage::default();
        assert!(!scene.replace_repaint_region(0, &mut repainted(base, &deep), &mut damage));
        assert_eq!(painted(&scene.quads).len(), 1);
        assert!(damage.regions().is_empty());
    }

    #[test]
    fn damage_covers_what_changed_between_scenes() {
        let left = Bounds::new(point(sp(0.), sp(0.)), Size::new(sp(10.), sp(10.)));
        let right = Bounds::new(point(sp(50.), sp(0.)), Size::new(sp(10.), sp(10.)));
        let draw = |right_lightness: f32| {
            let mut scene = Scene::default();
            scene.insert_primitive(colored(left, 0.1));
            scene.insert_primitive(colored(right, right_lightness));
            scene.finish();
            scene
        };
        let damage_between = |previous: &Scene, next: &Scene| {
            let mut damage = Damage::default();
            next.damage_since(previous, &mut damage);
            next.add_filter_damage(&mut damage);
            damage.regions()
        };
        let rounded = |bounds: Bounds<ScaledPixels>| {
            Bounds::from_corners(
                point(bounds.left() - sp(1.), bounds.top() - sp(1.)),
                point(bounds.right() + sp(1.), bounds.bottom() + sp(1.)),
            )
        };

        let previous = draw(0.5);
        assert!(damage_between(&previous, &draw(0.5)).is_empty());
        assert_eq!(damage_between(&previous, &draw(0.7)), vec![rounded(right)]);

        // A backdrop blurring near the change repaints all it covers.
        let mut blurred = draw(0.7);
        let glass = Bounds::new(point(sp(65.), sp(0.)), Size::new(sp(30.), sp(30.)));
        blurred.insert_primitive(BackdropFilter {
            bounds: glass,
            content_mask: ContentMask { bounds: glass },
            filters: smallvec::smallvec![ScaledFilter::Blur(sp(4.0))],
            opacity: 1.0,
            ..Default::default()
        });
        blurred.finish();
        let mut glass_before = draw(0.5);
        glass_before.insert_primitive(blurred.backdrop_filters[0].clone());
        glass_before.finish();
        let damage = damage_between(&glass_before, &blurred);
        assert!(damage.contains(&rounded(right)));
        assert!(damage.contains(&rounded(glass)));

        // A shadow's blur reaches past its box.
        let mut shadowed = draw(0.5);
        shadowed.insert_primitive(Shadow {
            order: 0,
            blur_radius: sp(2.),
            bounds: left,
            corner_radii: Corners::default(),
            content_mask: ContentMask {
                bounds: full_bounds(),
            },
            color: crate::hsla(0., 0., 0., 0.5).into(),
            element_bounds: left,
            element_corner_radii: Corners::default(),
            inset: 0,
            pad: 0,
        });
        shadowed.finish();
        let damage = damage_between(&previous, &shadowed);
        let reach = left.dilate(sp(6.)).intersect(&full_bounds());
        assert!(
            damage.iter().any(|region| region.left() <= reach.left()
                && region.right() >= reach.right()
                && region.top() <= reach.top()
                && region.bottom() >= reach.bottom()),
            "{damage:?}"
        );
    }

    #[test]
    fn removing_a_blurred_group_damages_its_blur_ring() {
        let group = Bounds::new(point(sp(100.), sp(100.)), Size::new(sp(20.), sp(20.)));
        let window = Bounds::new(point(sp(0.), sp(0.)), Size::new(sp(400.), sp(400.)));
        let marker = |is_start| FilterBoundary {
            order: 0,
            bounds: group,
            content_mask: ContentMask { bounds: window },
            corner_radii: Corners::default(),
            filters: smallvec::smallvec![ScaledFilter::Blur(sp(8.0))],
            opacity: 1.0,
            is_start,
        };
        let mut blurred = Scene::default();
        blurred.insert_primitive(marker(true));
        blurred.insert_primitive(colored(group, 0.5));
        blurred.insert_primitive(marker(false));
        blurred.finish();
        let mut empty = Scene::default();
        empty.finish();

        let mut damage = Damage::default();
        empty.damage_since(&blurred, &mut damage);
        empty.add_filter_damage(&mut damage);
        // The blur reaches three radii past the group, clipped to its mask.
        let ring = group.dilate(sp(24.));
        assert!(
            damage
                .regions()
                .iter()
                .any(|region| region.left() <= ring.left()
                    && region.top() <= ring.top()
                    && region.right() >= ring.right()
                    && region.bottom() >= ring.bottom()),
            "{:?}",
            damage.regions()
        );
    }

    fn image(page: u32, left: f32) -> PolychromeSprite {
        let bounds = Bounds::new(point(sp(left), sp(0.)), Size::new(sp(10.), sp(10.)));
        PolychromeSprite {
            order: 0,
            pad: 0,
            grayscale: false.into(),
            opacity: 1.,
            bounds,
            content_mask: ContentMask { bounds },
            corner_radii: Corners::default(),
            tile: AtlasTile {
                texture_id: AtlasTextureId {
                    index: page,
                    kind: AtlasTextureKind::Polychrome,
                },
                tile_id: TileId(page),
                padding: 0,
                bounds: Bounds::default(),
            },
        }
    }

    fn sprite_batches(scene: &Scene) -> Vec<(u32, Range<usize>)> {
        scene
            .batches()
            .filter_map(|batch| match batch {
                PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                    Some((texture_id.index, range))
                }
                _ => None,
            })
            .collect()
    }

    #[test]
    fn slack_after_a_released_image_names_no_texture_and_starts_no_batch() {
        // A region shows an image on atlas page 3, keeping slack behind it.
        let mut scene = Scene::default();
        scene.begin_repaint_region(RepaintId(1));
        scene.insert_primitive(image(3, 0.));
        scene.end_repaint_region();
        scene.finish();
        let base = scene.repaint_regions[0].orders.start;
        assert!(
            scene.polychrome_sprites[1..]
                .iter()
                .all(|sprite| is_slack(&sprite.tile)),
            "slack names no atlas page"
        );

        // The image is released and the region shows nothing now: every
        // slot is slack, and no batch draws any of it.
        let mut empty = Scene::default();
        empty.begin_repaint_region_at(RepaintId(1), base);
        empty.end_repaint_region();
        let mut damage = Damage::default();
        assert!(scene.replace_repaint_region(0, &mut empty, &mut damage));
        assert!(
            scene
                .polychrome_sprites
                .iter()
                .all(|sprite| is_slack(&sprite.tile))
        );
        assert_eq!(sprite_batches(&scene), vec![]);

        // An image on page 5 fills a slot; the slack around it draws in its
        // batch, which names page 5 only.
        let mut repainted = Scene::default();
        repainted.begin_repaint_region_at(RepaintId(1), base);
        repainted.insert_primitive(image(5, 20.));
        repainted.end_repaint_region();
        assert!(scene.replace_repaint_region(0, &mut repainted, &mut damage));
        let batches = sprite_batches(&scene);
        assert_eq!(batches.len(), 1);
        let (page, range) = batches[0].clone();
        assert_eq!(page, 5);
        assert!(!is_slack(&scene.polychrome_sprites[range.start].tile));
    }

    #[test]
    fn slack_inside_a_run_does_not_split_its_batch() {
        // Images on page 1 before and after a region whose slack sorts
        // between them: one batch, slack drawn with it as nothing.
        let mut scene = Scene::default();
        scene.begin_repaint_region(RepaintId(1));
        scene.insert_primitive(image(1, 0.));
        scene.end_repaint_region();
        scene.insert_primitive(image(1, 40.));
        scene.finish();
        let batches = sprite_batches(&scene);
        assert_eq!(batches, vec![(1, 0..scene.polychrome_sprites.len())]);
    }
}
