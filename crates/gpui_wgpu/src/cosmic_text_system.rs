use anyhow::{Context as _, Ok, Result};
use collections::HashMap;
use cosmic_text::{
    Attrs, AttrsList, Ellipsize, Family, Font as CosmicTextFont,
    FontFeatures as CosmicFontFeatures, FontSystem, ShapeBuffer, ShapeLine, Stretch, Style, Weight,
};
use gpui::{
    Bounds, DevicePixels, Font, FontFallbacks, FontFeatures, FontId, FontMetrics, FontRun, GlyphId,
    IsZero as _, LineLayout, Pixels, PlatformTextSystem, RenderGlyphParams, SUBPIXEL_VARIANTS_X,
    SUBPIXEL_VARIANTS_Y, ShapedGlyph, ShapedRun, SharedString, Size, TextRenderingMode, point,
    size,
};

use itertools::Itertools;
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::{borrow::Cow, ops::Range, sync::Arc};
use swash::{
    scale::{Render, ScaleContext, Source, StrikeWith},
    zeno::{Format, Vector},
};
use unicode_segmentation::UnicodeSegmentation;

pub struct CosmicTextSystem(RwLock<CosmicTextSystemState>);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FontKey {
    family: SharedString,
    features: FontFeatures,
    fallbacks: Option<FontFallbacks>,
}

impl FontKey {
    fn new(family: SharedString, features: FontFeatures, fallbacks: Option<FontFallbacks>) -> Self {
        Self {
            family,
            features,
            fallbacks,
        }
    }
}

struct CosmicTextSystemState {
    font_system: FontSystem,
    scratch: ShapeBuffer,
    swash_scale_context: ScaleContext,
    pending_glyph_images: HashMap<RenderGlyphParams, swash::scale::image::Image>,
    /// Contains all already loaded fonts, including all faces. Indexed by `FontId`.
    loaded_fonts: Vec<LoadedFont>,
    /// Caches the `FontId`s associated with a specific family to avoid iterating the font database
    /// for every font face in a family.
    font_ids_by_family_cache: HashMap<FontKey, SmallVec<[FontId; 4]>>,
    system_font_fallback: String,
    /// Caches whether shaping a grapheme with a family yields one
    /// combined glyph. Keyed by family name and grapheme text; shaping
    /// topology never depends on size, so one entry serves all frames.
    /// Capped: model output can stream unbounded unique graphemes, so at
    /// the cap new probes run uncached instead of growing memory.
    cluster_formation_cache: HashMap<(SharedString, String), bool>,
    /// Lazily built native-first emoji chain shared by every run in the
    /// line (families are run-independent by construction). Mirrors the
    /// staleness policy of `font_ids_by_family_cache`: built once, not
    /// refreshed by later `add_fonts`.
    emoji_chain_cache: Option<Arc<[(FontId, SharedString)]>>,
}

struct LoadedFont {
    font: Arc<CosmicTextFont>,
    features: CosmicFontFeatures,
    is_known_emoji_font: bool,
    /// resolved at load time so `layout_line` shares one chain across faces.
    /// `Arc` keeps clone cheap on the per-run hot path.
    user_fallback_chain: Arc<[(FontId, SharedString)]>,
}

struct FontMatchProperties {
    primary_family_name: SharedString,
    stretch: Stretch,
    style: Style,
    weight: Weight,
    features: CosmicFontFeatures,
    fallback_chain: Arc<[(FontId, SharedString)]>,
}

impl FontMatchProperties {
    fn attributes<'a>(&'a self, font_id: FontId, family_name: &'a str) -> Attrs<'a> {
        Attrs::new()
            .metadata(font_id.0)
            .family(Family::Name(family_name))
            .stretch(self.stretch)
            .style(self.style)
            .weight(self.weight)
            .font_features(self.features.clone())
    }
}

impl CosmicTextSystem {
    pub fn new(system_font_fallback: &str) -> Self {
        let font_system = FontSystem::new();

        Self(RwLock::new(CosmicTextSystemState {
            font_system,
            scratch: ShapeBuffer::default(),
            swash_scale_context: ScaleContext::new(),
            pending_glyph_images: HashMap::default(),
            loaded_fonts: Vec::new(),
            font_ids_by_family_cache: HashMap::default(),
            system_font_fallback: system_font_fallback.to_string(),
            cluster_formation_cache: HashMap::default(),
            emoji_chain_cache: None,
        }))
    }

    pub fn new_without_system_fonts(system_font_fallback: &str) -> Self {
        let font_system = FontSystem::new_with_locale_and_db(
            "en-US".to_string(),
            cosmic_text::fontdb::Database::new(),
        );

        Self(RwLock::new(CosmicTextSystemState {
            font_system,
            scratch: ShapeBuffer::default(),
            swash_scale_context: ScaleContext::new(),
            pending_glyph_images: HashMap::default(),
            loaded_fonts: Vec::new(),
            font_ids_by_family_cache: HashMap::default(),
            system_font_fallback: system_font_fallback.to_string(),
            cluster_formation_cache: HashMap::default(),
            emoji_chain_cache: None,
        }))
    }
}

impl PlatformTextSystem for CosmicTextSystem {
    fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        self.0.write().add_fonts(fonts)
    }

    fn all_font_names(&self) -> Vec<String> {
        let mut result = self
            .0
            .read()
            .font_system
            .db()
            .faces()
            .filter_map(|face| face.families.first().map(|family| family.0.clone()))
            .collect_vec();
        result.sort_unstable();
        result.dedup();
        result
    }

    fn font_id(&self, font: &Font) -> Result<FontId> {
        let mut state = self.0.write();
        let key = FontKey::new(
            font.family.clone(),
            font.features.clone(),
            font.fallbacks.clone(),
        );
        let candidates = if let Some(font_ids) = state.font_ids_by_family_cache.get(&key) {
            font_ids.as_slice()
        } else {
            let font_ids =
                state.load_family(&font.family, &font.features, font.fallbacks.as_ref())?;
            state.font_ids_by_family_cache.insert(key.clone(), font_ids);
            state.font_ids_by_family_cache[&key].as_ref()
        };

        let ix = find_best_match(font, candidates, &state)?;

        Ok(candidates[ix])
    }

    fn prewarm_fonts(&self, font_ids: &[FontId]) {
        self.0.write().prewarm_fonts(font_ids);
    }

    fn font_metrics(&self, font_id: FontId) -> FontMetrics {
        let metrics = self
            .0
            .read()
            .loaded_font(font_id)
            .font
            .as_swash()
            .metrics(&[]);

        FontMetrics {
            units_per_em: metrics.units_per_em as u32,
            ascent: metrics.ascent,
            descent: -metrics.descent,
            line_gap: metrics.leading,
            underline_position: metrics.underline_offset,
            underline_thickness: metrics.stroke_size,
            cap_height: metrics.cap_height,
            x_height: metrics.x_height,
            bounding_box: Bounds {
                origin: point(0.0, 0.0),
                size: size(metrics.max_width, metrics.ascent + metrics.descent),
            },
        }
    }

    fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
        let lock = self.0.read();
        let glyph_metrics = lock.loaded_font(font_id).font.as_swash().glyph_metrics(&[]);
        let glyph_id = glyph_id.0 as u16;
        Ok(Bounds {
            origin: point(0.0, 0.0),
            size: size(
                glyph_metrics.advance_width(glyph_id),
                glyph_metrics.advance_height(glyph_id),
            ),
        })
    }

    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        self.0.read().advance(font_id, glyph_id)
    }

    fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        self.0.read().glyph_for_char(font_id, ch)
    }

    fn glyph_raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        self.0.write().raster_bounds(params)
    }

    fn rasterize_glyph(
        &self,
        params: &RenderGlyphParams,
        raster_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        self.0.write().rasterize_glyph(params, raster_bounds)
    }

    fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout {
        self.0.write().layout_line(text, font_size, runs)
    }

    fn recommended_rendering_mode(
        &self,
        _font_id: FontId,
        _font_size: Pixels,
    ) -> TextRenderingMode {
        TextRenderingMode::Subpixel
    }
}

/// Emoji fallback families in native-first order with the Twemoji
/// fallback last. Families absent from the database are dropped by
/// `load_family`, so every platform resolves its own subset; the order
/// is what keeps platform faces ahead of Twemoji wherever they cover.
const EMOJI_FALLBACK_FAMILIES: &[&str] = &[
    "Segoe UI Emoji",
    "Noto Color Emoji",
    "AppleColorEmoji",
    "Twemoji Mozilla",
];

/// Maximum `cluster_formation_cache` entries. Past this, probes run
/// uncached rather than retaining unbounded model output.
const CLUSTER_FORMATION_CACHE_LIMIT: usize = 512;

impl CosmicTextSystemState {
    fn loaded_font(&self, font_id: FontId) -> &LoadedFont {
        &self.loaded_fonts[font_id.0]
    }

    fn font_match_properties(&self, font_id: FontId) -> Option<FontMatchProperties> {
        let loaded_font = self.loaded_font(font_id);
        let Some(face) = self.font_system.db().face(loaded_font.font.id()) else {
            log::warn!("font face not found in database for font_id {:?}", font_id);
            return None;
        };
        let Some(first_family) = face.families.first() else {
            log::warn!("font face has no family names for font_id {:?}", font_id);
            return None;
        };

        Some(FontMatchProperties {
            primary_family_name: first_family.0.clone().into(),
            stretch: face.stretch,
            style: face.style,
            weight: face.weight,
            features: loaded_font.features.clone(),
            fallback_chain: Arc::clone(&loaded_font.user_fallback_chain),
        })
    }

    fn prewarm_fonts(&mut self, font_ids: &[FontId]) {
        for &font_id in font_ids {
            let Some(properties) = self.font_match_properties(font_id) else {
                continue;
            };
            let primary_attributes =
                properties.attributes(font_id, &properties.primary_family_name);
            self.font_system.get_font_matches(&primary_attributes);

            for (fallback_id, fallback_name) in &*properties.fallback_chain {
                let fallback_attributes = properties.attributes(*fallback_id, fallback_name);
                self.font_system.get_font_matches(&fallback_attributes);
            }
        }
    }

    #[profiling::function]
    fn add_fonts(&mut self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        let db = self.font_system.db_mut();
        for bytes in fonts {
            db.load_font_source(cosmic_text::fontdb::Source::Binary(Arc::new(bytes)));
        }
        Ok(())
    }

    #[profiling::function]
    fn load_family(
        &mut self,
        name: &str,
        features: &FontFeatures,
        fallbacks: Option<&FontFallbacks>,
    ) -> Result<SmallVec<[FontId; 4]>> {
        // recurse with `fallbacks = None` so a fallback family cannot pull in
        // another chain. missing fallback families are dropped so a typo in
        // settings still lets the primary family load.
        let user_fallback_chain: Arc<[(FontId, SharedString)]> = match fallbacks {
            Some(fallbacks) if !fallbacks.fallback_list().is_empty() => {
                let mut chain: Vec<(FontId, SharedString)> = Vec::new();
                for fallback_name in fallbacks.fallback_list() {
                    let fb_key = FontKey::new(
                        SharedString::from(fallback_name.clone()),
                        features.clone(),
                        None,
                    );
                    let fb_ids = if let Some(cached) = self.font_ids_by_family_cache.get(&fb_key) {
                        cached.clone()
                    } else {
                        let loaded = self.load_family(fallback_name, features, None)?;
                        self.font_ids_by_family_cache
                            .insert(fb_key.clone(), loaded.clone());
                        loaded
                    };
                    let Some(&fb_id) = fb_ids.first() else {
                        continue;
                    };
                    let db_id = self.loaded_fonts[fb_id.0].font.id();
                    if let Some(face) = self.font_system.db().face(db_id)
                        && let Some(family) = face.families.first()
                    {
                        chain.push((fb_id, SharedString::from(family.0.clone())));
                    }
                }
                Arc::from(chain)
            }
            _ => Arc::from(Vec::new()),
        };

        let name = gpui::font_name_with_fallbacks(name, &self.system_font_fallback);

        let families = self
            .font_system
            .db()
            .faces()
            .filter(|face| face.families.iter().any(|family| *name == family.0))
            .map(|face| (face.id, face.post_script_name.clone()))
            .collect::<SmallVec<[_; 4]>>();

        let cosmic_features = cosmic_font_features(features)?;

        let mut loaded_font_ids = SmallVec::new();
        for (font_id, postscript_name) in families {
            let font = self
                .font_system
                .get_font(font_id, cosmic_text::Weight::NORMAL)
                .context("Could not load font")?;

            // HACK: To let the storybook run and render Windows caption icons. We should actually do better font fallback.
            // Color emoji faces carry no Latin coverage by design; every
            // known color-emoji face must survive loading so fallback
            // resolution can reach it. The set mirrors
            // `check_is_known_emoji_font`.
            let allowed_bad_font_names = [
                "SegoeFluentIcons", // NOTE: Segoe fluent icons postscript name is inconsistent
                "Segoe Fluent Icons",
                "NotoColorEmoji",
                "SegoeUIEmoji",
                "AppleColorEmoji",
                ".AppleColorEmojiUI",
                "TwemojiMozilla",
            ];

            if font.as_swash().charmap().map('m') == 0
                && !allowed_bad_font_names.contains(&postscript_name.as_str())
            {
                self.font_system.db_mut().remove_face(font.id());
                continue;
            };

            let font_id = FontId(self.loaded_fonts.len());
            loaded_font_ids.push(font_id);
            self.loaded_fonts.push(LoadedFont {
                font,
                features: cosmic_features.clone(),
                is_known_emoji_font: check_is_known_emoji_font(&postscript_name),
                user_fallback_chain: Arc::clone(&user_fallback_chain),
            });
        }

        Ok(loaded_font_ids)
    }

    /// Resolves the shared native-first emoji fallback chain for runs
    /// that carry no user chain. Emoji faces take default features:
    /// stylistic sets must not leak into fallback glyphs, and shaping
    /// topology (ligatures) never depends on them. Results ride the
    /// existing per-family cache, so repeated layouts pay one lookup.
    fn emoji_fallback_chain(&mut self) -> Arc<[(FontId, SharedString)]> {
        if let Some(chain) = self.emoji_chain_cache.clone() {
            return chain;
        }
        let mut chain: Vec<(FontId, SharedString)> = Vec::new();
        for family in EMOJI_FALLBACK_FAMILIES {
            let key = FontKey::new(
                SharedString::from(*family),
                FontFeatures::default(),
                None,
            );
            let ids = if let Some(cached) = self.font_ids_by_family_cache.get(&key) {
                cached.clone()
            } else {
                let std::result::Result::Ok(loaded) =
                    self.load_family(family, &FontFeatures::default(), None)
                else {
                    continue;
                };
                self.font_ids_by_family_cache
                    .insert(key, loaded.clone());
                loaded
            };
            let Some(&id) = ids.first() else {
                continue;
            };
            let db_id = self.loaded_fonts[id.0].font.id();
            if let Some(face) = self.font_system.db().face(db_id)
                && let Some(family) = face.families.first()
            {
                chain.push((id, SharedString::from(family.0.clone())));
            }
        }
        let chain: Arc<[(FontId, SharedString)]> = Arc::from(chain);
        self.emoji_chain_cache = Some(chain.clone());
        chain
    }

    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        let glyph_metrics = self.loaded_font(font_id).font.as_swash().glyph_metrics(&[]);
        Ok(Size {
            width: glyph_metrics.advance_width(glyph_id.0 as u16),
            height: glyph_metrics.advance_height(glyph_id.0 as u16),
        })
    }

    fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        let glyph_id = self.loaded_font(font_id).font.as_swash().charmap().map(ch);
        if glyph_id == 0 {
            None
        } else {
            Some(GlyphId(glyph_id.into()))
        }
    }

    fn raster_bounds(&mut self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        let image = self.render_glyph_image(params)?;
        let bounds = Bounds {
            origin: point(image.placement.left.into(), (-image.placement.top).into()),
            size: size(image.placement.width.into(), image.placement.height.into()),
        };
        if !bounds.is_zero() {
            self.pending_glyph_images.insert(params.clone(), image);
        }
        Ok(bounds)
    }

    #[profiling::function]
    fn rasterize_glyph(
        &mut self,
        params: &RenderGlyphParams,
        glyph_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        if glyph_bounds.size.width.0 == 0 || glyph_bounds.size.height.0 == 0 {
            anyhow::bail!("glyph bounds are empty");
        }

        let mut image = match self.pending_glyph_images.remove(params) {
            Some(image) => image,
            None => self.render_glyph_image(params)?,
        };
        let bitmap_size = glyph_bounds.size;
        match image.content {
            swash::scale::image::Content::Color | swash::scale::image::Content::SubpixelMask => {
                // Convert from RGBA to BGRA.
                for pixel in image.data.chunks_exact_mut(4) {
                    pixel.swap(0, 2);
                }
                Ok((bitmap_size, image.data))
            }
            swash::scale::image::Content::Mask => {
                if params.subpixel_rendering {
                    // We must always return RGBA data when subpixel rendering is requested.
                    let expanded = image.data.iter().flat_map(|&a| [a, a, a, a]).collect();
                    Ok((bitmap_size, expanded))
                } else {
                    Ok((bitmap_size, image.data))
                }
            }
        }
    }

    fn render_glyph_image(
        &mut self,
        params: &RenderGlyphParams,
    ) -> Result<swash::scale::image::Image> {
        let loaded_font = &self.loaded_fonts[params.font_id.0];
        let font_ref = loaded_font.font.as_swash();
        let pixel_size = f32::from(params.font_size);

        let subpixel_offset = Vector::new(
            params.subpixel_variant.x as f32 / SUBPIXEL_VARIANTS_X as f32 / params.scale_factor,
            params.subpixel_variant.y as f32 / SUBPIXEL_VARIANTS_Y as f32 / params.scale_factor,
        );

        let mut scaler = self
            .swash_scale_context
            .builder(font_ref)
            .size(pixel_size * params.scale_factor)
            .hint(true)
            .build();

        let sources: &[Source] = if params.is_emoji {
            &[
                Source::ColorOutline(0),
                Source::ColorBitmap(StrikeWith::BestFit),
                Source::Outline,
            ]
        } else {
            &[Source::Bitmap(StrikeWith::ExactSize), Source::Outline]
        };

        let mut renderer = Render::new(sources);
        if params.subpixel_rendering {
            // There seems to be a bug in Swash where the B and R values are swapped.
            renderer
                .format(Format::subpixel_bgra())
                .offset(subpixel_offset);
        } else {
            renderer.format(Format::Alpha).offset(subpixel_offset);
        }

        let glyph_id: u16 = params.glyph_id.0.try_into()?;
        renderer
            .render(&mut scaler, glyph_id)
            .with_context(|| format!("unable to render glyph via swash for {params:?}"))
    }

    /// This is used when cosmic_text has chosen a fallback font instead of using the requested
    /// font, typically to handle some unicode characters. When this happens, `loaded_fonts` may not
    /// yet have an entry for this fallback font, and so one is added.
    ///
    /// Note that callers shouldn't use this `FontId` somewhere that will retrieve the corresponding
    /// `LoadedFont.features`, as it will have an arbitrarily chosen or empty value. The only
    /// current use of this field is for the *input* of `layout_line`, and so it's fine to use
    /// `font_id_for_cosmic_id` when computing the *output* of `layout_line`.
    fn font_id_for_cosmic_id(&mut self, id: cosmic_text::fontdb::ID) -> Result<FontId> {
        if let Some(ix) = self
            .loaded_fonts
            .iter()
            .position(|loaded_font| loaded_font.font.id() == id)
        {
            Ok(FontId(ix))
        } else {
            let font = self
                .font_system
                .get_font(id, cosmic_text::Weight::NORMAL)
                .context("failed to get fallback font from cosmic-text font system")?;
            let face = self
                .font_system
                .db()
                .face(id)
                .context("fallback font face not found in cosmic-text database")?;

            let font_id = FontId(self.loaded_fonts.len());
            self.loaded_fonts.push(LoadedFont {
                font,
                features: CosmicFontFeatures::new(),
                is_known_emoji_font: check_is_known_emoji_font(&face.post_script_name),
                user_fallback_chain: Arc::from(Vec::new()),
            });

            Ok(font_id)
        }
    }

    #[profiling::function]
    fn layout_line(&mut self, text: &str, font_size: Pixels, font_runs: &[FontRun]) -> LineLayout {
        if contains_paragraph_separator(text) {
            self.layout_line_with_separators(text, font_size, font_runs)
        } else {
            self.layout_line_no_separators(text, font_size, font_runs)
        }
    }

    fn layout_line_with_separators(
        &mut self,
        text: &str,
        font_size: Pixels,
        font_runs: &[FontRun],
    ) -> LineLayout {
        let mut layout = LineLayout {
            font_size,
            len: text.len(),
            ..Default::default()
        };
        let mut paragraph_start = 0;

        for (separator_start, separator) in text
            .char_indices()
            .filter(|(_, character)| is_paragraph_separator(*character))
        {
            let separator_end = separator_start + separator.len_utf8();
            self.shape_segment(
                text,
                paragraph_start..separator_start,
                font_size,
                font_runs,
                &mut layout,
            );
            self.shape_segment(
                text,
                separator_start..separator_end,
                font_size,
                font_runs,
                &mut layout,
            );
            paragraph_start = separator_end;
        }

        self.shape_segment(
            text,
            paragraph_start..text.len(),
            font_size,
            font_runs,
            &mut layout,
        );

        layout
    }

    fn shape_segment(
        &mut self,
        text: &str,
        range: Range<usize>,
        font_size: Pixels,
        font_runs: &[FontRun],
        layout: &mut LineLayout,
    ) {
        if range.is_empty() {
            return;
        }

        let segment_font_runs = clip_font_runs(font_runs, range.clone());
        let segment =
            self.layout_line_no_separators(&text[range.clone()], font_size, &segment_font_runs);

        let mut segment_runs = segment.runs;
        for run in &mut segment_runs {
            for glyph in &mut run.glyphs {
                glyph.index += range.start;
                glyph.position.x += layout.width;
            }
        }

        for mut run in segment_runs {
            if let Some(same_run) = layout
                .runs
                .last_mut()
                .filter(|last| last.font_id == run.font_id)
            {
                same_run.glyphs.append(&mut run.glyphs);
            } else {
                layout.runs.push(run);
            }
        }

        layout.width += segment.width;
        layout.ascent = layout.ascent.max(segment.ascent);
        layout.descent = layout.descent.max(segment.descent);
    }

    fn layout_line_no_separators(
        &mut self,
        text: &str,
        font_size: Pixels,
        font_runs: &[FontRun],
    ) -> LineLayout {
        let mut attrs_list = AttrsList::new(&Attrs::new());
        let mut offs = 0;
        for run in font_runs {
            let run_end = offs + run.len;

            let Some(properties) = self.font_match_properties(run.font_id) else {
                offs = run_end;
                continue;
            };

            let letter_spacing = run
                .letter_spacing
                .map(|spacing| spacing.as_f32() / font_size.as_f32());

            // Resolve the effective chain: the run's explicit user chain
            // when set, otherwise the synthesized native-first emoji
            // chain. Span slots index into this chain, so all attribute
            // and span building below uses it — never the raw user chain.
            let emoji_chain;
            let chain: &[(FontId, SharedString)] = if properties.fallback_chain.is_empty() {
                emoji_chain = self.emoji_fallback_chain();
                &emoji_chain
            } else {
                &properties.fallback_chain
            };

            // build one `Attrs` per slot up front. each clone of span attrs
            // would otherwise re-allocate the `font_features` Vec.
            let mut primary_attrs =
                properties.attributes(run.font_id, &properties.primary_family_name);
            if let Some(letter_spacing) = letter_spacing {
                primary_attrs = primary_attrs.letter_spacing(letter_spacing);
            }
            let fallback_attrs: SmallVec<[Attrs<'_>; 4]> = chain
                .iter()
                .map(|(font_id, family_name)| {
                    let mut attrs = properties.attributes(*font_id, family_name);
                    if let Some(letter_spacing) = letter_spacing {
                        attrs = attrs.letter_spacing(letter_spacing);
                    }
                    attrs
                })
                .collect();

            let spans = if chain.is_empty() {
                let loaded_fonts = &self.loaded_fonts;
                let covers = |id: FontId, ch: char| charmap_covers(loaded_fonts, id, ch);
                compute_run_spans(
                    text,
                    offs,
                    run.len,
                    run.font_id,
                    chain,
                    &covers,
                )
            } else {
                let CosmicTextSystemState {
                    loaded_fonts,
                    font_system,
                    scratch,
                    cluster_formation_cache,
                    ..
                } = &mut *self;
                let covers = |id: FontId, ch: char| charmap_covers(loaded_fonts, id, ch);
                let probe_size = f32::from(font_size);
                let mut forms_single = |id: FontId, grapheme: &str| {
                    face_forms_single_cluster(
                        loaded_fonts,
                        font_system,
                        scratch,
                        cluster_formation_cache,
                        id,
                        probe_size,
                        grapheme,
                    )
                };
                compute_cluster_spans(
                    text,
                    offs,
                    run.len,
                    run.font_id,
                    chain,
                    &covers,
                    &mut forms_single,
                )
            };

            for span in spans {
                let attrs = match span.slot {
                    None => &primary_attrs,
                    Some(ix) => &fallback_attrs[ix],
                };
                attrs_list.add_span(span.start..span.end, attrs);
            }
            offs = run_end;
        }

        let line = ShapeLine::new(
            &mut self.font_system,
            text,
            &attrs_list,
            cosmic_text::Shaping::Advanced,
            4,
        );
        let mut layout_lines = Vec::with_capacity(1);
        line.layout_to_buffer(
            &mut self.scratch,
            f32::from(font_size),
            None, // We do our own wrapping
            cosmic_text::Wrap::None,
            Ellipsize::None,
            None,
            &mut layout_lines,
            None,
            cosmic_text::Hinting::Disabled,
        );

        let Some(layout) = layout_lines.first() else {
            return LineLayout {
                font_size,
                width: Pixels::ZERO,
                ascent: Pixels::ZERO,
                descent: Pixels::ZERO,
                runs: Vec::new(),
                len: text.len(),
            };
        };

        let mut runs: Vec<ShapedRun> = Vec::new();
        for glyph in &layout.glyphs {
            let mut font_id = FontId(glyph.metadata);
            let mut loaded_font = self.loaded_font(font_id);
            if loaded_font.font.id() != glyph.font_id {
                match self.font_id_for_cosmic_id(glyph.font_id) {
                    std::result::Result::Ok(resolved_id) => {
                        font_id = resolved_id;
                        loaded_font = self.loaded_font(font_id);
                    }
                    Err(error) => {
                        log::warn!(
                            "failed to resolve cosmic font id {:?}: {error:#}",
                            glyph.font_id
                        );
                        continue;
                    }
                }
            }
            let is_emoji = loaded_font.is_known_emoji_font;

            // HACK: Prevent crash caused by variation selectors.
            if glyph.glyph_id == 3 && is_emoji {
                continue;
            }

            let shaped_glyph = ShapedGlyph {
                id: GlyphId(glyph.glyph_id as u32),
                position: point(glyph.x.into(), glyph.y.into()),
                index: glyph.start,
                is_emoji,
            };

            if let Some(last_run) = runs
                .last_mut()
                .filter(|last_run| last_run.font_id == font_id)
            {
                last_run.glyphs.push(shaped_glyph);
            } else {
                runs.push(ShapedRun {
                    font_id,
                    glyphs: vec![shaped_glyph],
                });
            }
        }

        LineLayout {
            font_size,
            width: layout.w.into(),
            ascent: layout.max_ascent.into(),
            descent: layout.max_descent.into(),
            runs,
            len: text.len(),
        }
    }
}

#[inline(always)]
fn is_paragraph_separator(character: char) -> bool {
    unicode_bidi::bidi_class(character) == unicode_bidi::BidiClass::B
}

fn contains_paragraph_separator(text: &str) -> bool {
    if text
        .bytes()
        .any(|byte| matches!(byte, b'\n' | b'\r' | 0x1c | 0x1d | 0x1e))
    {
        return true;
    }

    !text.is_ascii() && text.chars().any(is_paragraph_separator)
}

fn clip_font_runs(font_runs: &[FontRun], range: Range<usize>) -> SmallVec<[FontRun; 4]> {
    let mut clipped = SmallVec::new();
    let mut offs = 0;
    for run in font_runs {
        let run_start = offs;
        offs += run.len;
        if offs <= range.start {
            continue;
        }
        if run_start >= range.end {
            break;
        }
        let start = run_start.max(range.start);
        let end = offs.min(range.end);
        if start < end {
            clipped.push(FontRun {
                len: end - start,
                font_id: run.font_id,
                letter_spacing: run.letter_spacing,
            });
        }
    }
    clipped
}

#[cfg(feature = "font-kit")]
fn find_best_match(
    font: &Font,
    candidates: &[FontId],
    state: &CosmicTextSystemState,
) -> Result<usize> {
    let candidate_properties = candidates
        .iter()
        .map(|font_id| {
            let database_id = state.loaded_font(*font_id).font.id();
            let face_info = state
                .font_system
                .db()
                .face(database_id)
                .context("font face not found in database")?;
            Ok(face_info_into_properties(face_info))
        })
        .collect::<Result<SmallVec<[_; 4]>>>()?;

    let ix =
        font_kit::matching::find_best_match(&candidate_properties, &font_into_properties(font))
            .context("requested font family contains no font matching the other parameters")?;

    Ok(ix)
}

#[cfg(not(feature = "font-kit"))]
fn find_best_match(
    font: &Font,
    candidates: &[FontId],
    state: &CosmicTextSystemState,
) -> Result<usize> {
    if candidates.is_empty() {
        anyhow::bail!("requested font family contains no font matching the other parameters");
    }
    if candidates.len() == 1 {
        return Ok(0);
    }

    let target_weight = font.weight.0;
    let target_italic = matches!(
        font.style,
        gpui::FontStyle::Italic | gpui::FontStyle::Oblique
    );

    let mut best_index = 0;
    let mut best_score = u32::MAX;

    for (index, font_id) in candidates.iter().enumerate() {
        let database_id = state.loaded_font(*font_id).font.id();
        let face_info = state
            .font_system
            .db()
            .face(database_id)
            .context("font face not found in database")?;

        let is_italic = matches!(
            face_info.style,
            cosmic_text::Style::Italic | cosmic_text::Style::Oblique
        );
        let style_penalty: u32 = if is_italic == target_italic { 0 } else { 1000 };
        let weight_diff = (face_info.weight.0 as i32 - target_weight as i32).unsigned_abs();
        let score = style_penalty + weight_diff;

        if score < best_score {
            best_score = score;
            best_index = index;
        }
    }

    Ok(best_index)
}

/// one contiguous slice of a `FontRun` that maps to a single slot. `slot` is
/// `None` for the primary font and `Some(ix)` for `fallback_chain[ix]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunSpan {
    start: usize,
    end: usize,
    slot: Option<usize>,
    font_id: FontId,
}

/// Groups a run into whole-grapheme spans like [`compute_run_spans`],
/// but selects each span's face by full-cluster coverage in native-first
/// chain order, preferring the first face shaping the grapheme into one
/// combined glyph. ASCII and primary-covered clusters never leave the
/// primary face; clusters nothing covers keep the legacy first-scalar
/// rule so cosmic resolves them exactly as today.
fn compute_cluster_spans(
    text: &str,
    run_offset: usize,
    run_len: usize,
    primary: FontId,
    fallback_chain: &[(FontId, SharedString)],
    covers: &impl Fn(FontId, char) -> bool,
    forms_single_cluster: &mut impl FnMut(FontId, &str) -> bool,
) -> SmallVec<[RunSpan; 4]> {
    let mut spans = SmallVec::new();
    let run_end = run_offset + run_len;
    if run_end <= run_offset {
        return spans;
    }
    if fallback_chain.is_empty() {
        spans.push(RunSpan {
            start: run_offset,
            end: run_end,
            slot: None,
            font_id: primary,
        });
        return spans;
    }
    let run_text = &text[run_offset..run_end];
    let mut span_start = run_offset;
    let mut span_slot: Option<usize> = None;
    let mut span_font_id = primary;
    for (grapheme_idx, grapheme) in run_text.grapheme_indices(true) {
        let abs = run_offset + grapheme_idx;
        let next_slot = pick_cluster_slot(
            grapheme,
            primary,
            fallback_chain,
            covers,
            forms_single_cluster,
        );
        if next_slot == span_slot {
            continue;
        }
        if abs > span_start {
            spans.push(RunSpan {
                start: span_start,
                end: abs,
                slot: span_slot,
                font_id: span_font_id,
            });
        }
        span_start = abs;
        span_slot = next_slot;
        span_font_id = slot_font_id(next_slot, primary, fallback_chain);
    }
    if span_start < run_end {
        spans.push(RunSpan {
            start: span_start,
            end: run_end,
            slot: span_slot,
            font_id: span_font_id,
        });
    }
    spans
}

/// walks `text[run_offset..run_offset + run_len]` and groups codepoints into
/// spans. inheriting codepoints stay in the current span so shaping clusters
/// like emoji zwj sequences and combining marks are not torn apart.
fn compute_run_spans(
    text: &str,
    run_offset: usize,
    run_len: usize,
    primary: FontId,
    fallback_chain: &[(FontId, SharedString)],
    covers: &impl Fn(FontId, char) -> bool,
) -> SmallVec<[RunSpan; 4]> {
    let mut spans = SmallVec::new();
    let run_end = run_offset + run_len;
    if run_end <= run_offset {
        return spans;
    }
    if fallback_chain.is_empty() {
        spans.push(RunSpan {
            start: run_offset,
            end: run_end,
            slot: None,
            font_id: primary,
        });
        return spans;
    }
    let run_text = &text[run_offset..run_end];
    let mut span_start = run_offset;
    let mut span_slot: Option<usize> = None;
    let mut span_font_id = primary;
    for (grapheme_idx, grapheme) in run_text.grapheme_indices(true) {
        let abs = run_offset + grapheme_idx;
        let ch = grapheme.chars().next().unwrap_or('\0');
        let next_slot = pick_covering_slot(ch, span_slot, primary, fallback_chain, covers);
        if next_slot == span_slot {
            continue;
        }
        if abs > span_start {
            spans.push(RunSpan {
                start: span_start,
                end: abs,
                slot: span_slot,
                font_id: span_font_id,
            });
        }
        span_start = abs;
        span_slot = next_slot;
        span_font_id = slot_font_id(next_slot, primary, fallback_chain);
    }
    if span_start < run_end {
        spans.push(RunSpan {
            start: span_start,
            end: run_end,
            slot: span_slot,
            font_id: span_font_id,
        });
    }
    spans
}

fn slot_font_id(
    slot: Option<usize>,
    primary: FontId,
    fallback_chain: &[(FontId, SharedString)],
) -> FontId {
    match slot {
        None => primary,
        Some(ix) => fallback_chain[ix].0,
    }
}

fn pick_covering_slot(
    ch: char,
    current: Option<usize>,
    primary: FontId,
    fallback_chain: &[(FontId, SharedString)],
    covers: &impl Fn(FontId, char) -> bool,
) -> Option<usize> {
    if (ch as u32) <= 0x7F {
        return None;
    }
    if covers(primary, ch) {
        return None;
    }
    let current_id = slot_font_id(current, primary, fallback_chain);
    if covers(current_id, ch) {
        return current;
    }

    fallback_chain
        .iter()
        .position(|(fb_id, _)| covers(*fb_id, ch))
}

/// Selects the fallback slot for one whole grapheme cluster.
///
/// ASCII graphemes (including CRLF pairs) stay primary exactly as
/// before. Otherwise the first face covering the FULL cluster wins in
/// native-first chain order; a face covering only the first scalar never
/// splits the cluster. Multi-scalar clusters with several full-covering
/// faces go to the first face shaping them into one combined glyph
/// (ligated flag/ZWJ/keycap/skin sequences); ties and single-scalar
/// clusters keep chain order, and clusters nothing covers fall back to
/// the legacy first-scalar rule so cosmic resolves them as today.
fn pick_cluster_slot(
    grapheme: &str,
    primary: FontId,
    fallback_chain: &[(FontId, SharedString)],
    covers: &impl Fn(FontId, char) -> bool,
    forms_single_cluster: &mut impl FnMut(FontId, &str) -> bool,
) -> Option<usize> {
    if grapheme.chars().all(|ch| (ch as u32) <= 0x7F) {
        return None;
    }
    let covers_all = |id: FontId| grapheme.chars().all(|ch| covers(id, ch));
    if covers_all(primary) {
        return None;
    }
    let mut full = SmallVec::<[usize; 4]>::new();
    for (ix, (fb_id, _)) in fallback_chain.iter().enumerate() {
        if covers_all(*fb_id) {
            full.push(ix);
        }
    }
    let &[first, ..] = full.as_slice() else {
        return pick_covering_slot(
            grapheme.chars().next().unwrap_or('\0'),
            None,
            primary,
            fallback_chain,
            covers,
        );
    };
    if grapheme.chars().count() == 1 {
        return Some(first);
    }
    for ix in full {
        if forms_single_cluster(fallback_chain[ix].0, grapheme) {
            return Some(ix);
        }
    }
    Some(first)
}

/// Probes whether shaping `grapheme` with the candidate face yields one
/// combined glyph FROM THAT FACE. Results are cached per family+grapheme;
/// shaping topology never depends on size, so one probe size serves all
/// frames. Only glyphs the requested face itself shaped count: fallback
/// substitution inside the probe would credit another face's work, and
/// `.notdef` (glyph 0) never proves coverage. The glyph-3 skip mirrors
/// production exactly — it applies only when the probed face is a known
/// color-emoji face. Errors shape to `false` (no reroute).
fn face_forms_single_cluster(
    loaded_fonts: &[LoadedFont],
    font_system: &mut FontSystem,
    scratch: &mut ShapeBuffer,
    cache: &mut HashMap<(SharedString, String), bool>,
    font_id: FontId,
    font_size: f32,
    grapheme: &str,
) -> bool {
    if grapheme.chars().count() <= 1 {
        return true;
    }
    let Some(loaded) = loaded_fonts.get(font_id.0) else {
        return false;
    };
    let expected = loaded.font.id();
    let is_emoji_face = loaded.is_known_emoji_font;
    let Some(family) = font_system
        .db()
        .face(expected)
        .and_then(|face| face.families.first())
        .map(|(name, _)| SharedString::from(name.clone()))
    else {
        return false;
    };
    if let Some(hit) = cache.get(&(family.clone(), grapheme.to_owned())) {
        return *hit;
    }
    let attrs = Attrs::new()
        .metadata(0)
        .family(Family::Name(family.as_ref()));
    let attrs_list = AttrsList::new(&attrs);
    let line = ShapeLine::new(
        font_system,
        grapheme,
        &attrs_list,
        cosmic_text::Shaping::Advanced,
        4,
    );
    let mut layouts = Vec::with_capacity(1);
    line.layout_to_buffer(
        scratch,
        font_size,
        None, // We do our own wrapping
        cosmic_text::Wrap::None,
        Ellipsize::None,
        None,
        &mut layouts,
        None,
        cosmic_text::Hinting::Disabled,
    );
    let single = layouts.first().is_some_and(|layout| {
        is_single_combined_cluster(
            layout
                .glyphs
                .iter()
                .map(|glyph| (glyph.font_id == expected, glyph.glyph_id)),
            is_emoji_face,
        )
    });
    if cache.len() < CLUSTER_FORMATION_CACHE_LIMIT {
        cache.insert((family, grapheme.to_owned()), single);
    }
    single
}

/// Decides whether shaped output counts as one combined cluster from
/// the expected face. Each item is `(is_expected_face, glyph_id)`.
/// Ignored strays (`.notdef`, and glyph 3 in known color-emoji faces)
/// prove nothing either way; every remaining visible glyph must come
/// from the expected face, and exactly one must remain. In particular
/// one good glyph plus another face's output is a fragmented cluster,
/// not a combination.
fn is_single_combined_cluster(
    glyphs: impl Iterator<Item = (bool, u16)>,
    is_emoji_face: bool,
) -> bool {
    let mut visible = 0;
    for (expected, glyph_id) in glyphs {
        // `.notdef` never proves coverage: a missing glyph rejects the
        // whole probe immediately. It must not be blessed as stray.
        if glyph_id == 0 {
            return false;
        }
        // Glyph 3 is ignorable only as the expected color face's own
        // remnant. A non-color face's own gid 3 is an ordinary valid
        // glyph and counts normally below; any other face's gid 3 takes
        // the normal path and fails the expected check.
        if glyph_id == 3 && expected && is_emoji_face {
            continue;
        }
        if !expected {
            return false;
        }
        visible += 1;
    }
    visible == 1
}

fn charmap_covers(loaded_fonts: &[LoadedFont], id: FontId, ch: char) -> bool {
    loaded_fonts
        .get(id.0)
        .is_some_and(|loaded| loaded.font.as_swash().charmap().map(ch) != 0)
}

fn cosmic_font_features(features: &FontFeatures) -> Result<CosmicFontFeatures> {
    let mut result = CosmicFontFeatures::new();
    for feature in features.0.iter() {
        let name_bytes: [u8; 4] = feature
            .0
            .as_bytes()
            .try_into()
            .context("Incorrect feature flag format")?;

        let tag = cosmic_text::FeatureTag::new(&name_bytes);

        result.set(tag, feature.1);
    }
    Ok(result)
}

#[cfg(feature = "font-kit")]
fn font_into_properties(font: &gpui::Font) -> font_kit::properties::Properties {
    font_kit::properties::Properties {
        style: match font.style {
            gpui::FontStyle::Normal => font_kit::properties::Style::Normal,
            gpui::FontStyle::Italic => font_kit::properties::Style::Italic,
            gpui::FontStyle::Oblique => font_kit::properties::Style::Oblique,
        },
        weight: font_kit::properties::Weight(font.weight.0),
        stretch: Default::default(),
    }
}

#[cfg(feature = "font-kit")]
fn face_info_into_properties(
    face_info: &cosmic_text::fontdb::FaceInfo,
) -> font_kit::properties::Properties {
    font_kit::properties::Properties {
        style: match face_info.style {
            cosmic_text::Style::Normal => font_kit::properties::Style::Normal,
            cosmic_text::Style::Italic => font_kit::properties::Style::Italic,
            cosmic_text::Style::Oblique => font_kit::properties::Style::Oblique,
        },
        weight: font_kit::properties::Weight(face_info.weight.0.into()),
        stretch: match face_info.stretch {
            cosmic_text::Stretch::Condensed => font_kit::properties::Stretch::CONDENSED,
            cosmic_text::Stretch::Expanded => font_kit::properties::Stretch::EXPANDED,
            cosmic_text::Stretch::ExtraCondensed => font_kit::properties::Stretch::EXTRA_CONDENSED,
            cosmic_text::Stretch::ExtraExpanded => font_kit::properties::Stretch::EXTRA_EXPANDED,
            cosmic_text::Stretch::Normal => font_kit::properties::Stretch::NORMAL,
            cosmic_text::Stretch::SemiCondensed => font_kit::properties::Stretch::SEMI_CONDENSED,
            cosmic_text::Stretch::SemiExpanded => font_kit::properties::Stretch::SEMI_EXPANDED,
            cosmic_text::Stretch::UltraCondensed => font_kit::properties::Stretch::ULTRA_CONDENSED,
            cosmic_text::Stretch::UltraExpanded => font_kit::properties::Stretch::ULTRA_EXPANDED,
        },
    }
}

fn check_is_known_emoji_font(postscript_name: &str) -> bool {
    // Glyphs from these faces rasterize through the color sources
    // (`ColorOutline`/`ColorBitmap`); every other face stays on the
    // monochrome outline path so ordinary text and symbol fonts never
    // change appearance. `TwemojiMozilla` is fallback-only: it is never
    // requested as a primary family, so platform faces keep precedence
    // wherever they cover a cluster.
    matches!(
        postscript_name,
        "NotoColorEmoji" | "SegoeUIEmoji" | "AppleColorEmoji" | ".AppleColorEmojiUI" | "TwemojiMozilla"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fid(i: usize) -> FontId {
        FontId(i)
    }

    fn chain(ids: &[usize]) -> SmallVec<[(FontId, SharedString); 4]> {
        ids.iter()
            .map(|&i| (fid(i), SharedString::from(format!("fb{i}"))))
            .collect()
    }

    fn span(start: usize, end: usize, slot: Option<usize>, font_id: FontId) -> RunSpan {
        RunSpan {
            start,
            end,
            slot,
            font_id,
        }
    }

    const IBM_PLEX: &[u8] =
        include_bytes!("../../../assets/fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf");

    /// Every code point of `Bidi_Class=B`, each of which starts a new bidi
    /// paragraph and so can split one line into mixed-direction paragraphs.
    const SEPARATORS: &[char] = &[
        '\u{000a}', '\u{000d}', '\u{001c}', '\u{001d}', '\u{001e}', '\u{0085}', '\u{2029}',
    ];

    fn text_system() -> Result<CosmicTextSystem> {
        let text_system = CosmicTextSystem::new_without_system_fonts("IBM Plex Sans");
        text_system.add_fonts(vec![Cow::Borrowed(IBM_PLEX)])?;
        Ok(text_system)
    }

    fn layout_text(text_system: &CosmicTextSystem, text: &str) -> Result<LineLayout> {
        let font_id = text_system.font_id(&gpui::font("IBM Plex Sans"))?;
        let runs = [FontRun {
            len: text.len(),
            font_id,
            letter_spacing: None,
        }];
        Ok(text_system.layout_line(text, gpui::px(14.0), &runs))
    }

    /// Mirrors the original crash: mixed-direction text reaching the shaper
    /// through `shape_text`, which only splits lines on `\n`.
    #[test]
    fn shape_text_with_mixed_direction_paragraphs() -> Result<()> {
        let platform_text_system = Arc::new(text_system()?);
        let text_system = Arc::new(gpui::TextSystem::new(platform_text_system));
        let window_text_system = gpui::WindowTextSystem::new(text_system);

        let text: SharedString = "first line\n\u{05d0}\u{001c}A".into();
        let runs = [gpui::TextRun {
            len: text.len(),
            font: gpui::font("IBM Plex Sans"),
            ..Default::default()
        }];

        let lines = window_text_system.shape_text(text, gpui::px(14.0), &runs, None, None)?;

        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1].len(), "\u{05d0}\u{001c}A".len());
        assert!(lines[1].width() > Pixels::ZERO);
        Ok(())
    }

    #[test]
    fn layout_line_with_mixed_direction_paragraphs() -> Result<()> {
        let text_system = text_system()?;

        for separator in SEPARATORS {
            for text in [
                format!("\u{05d0}{separator}A"),
                format!("A{separator}\u{05d0}"),
            ] {
                let layout = layout_text(&text_system, &text)?;

                assert_eq!(layout.len, text.len(), "{text:?}");
                assert!(layout.width > Pixels::ZERO, "{text:?}");
                assert!(
                    layout.runs.iter().any(|run| !run.glyphs.is_empty()),
                    "{text:?}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn layout_line_with_separators_at_line_edges() -> Result<()> {
        let text_system = text_system()?;

        for text in [
            "\u{001c}",
            "\u{001c}\u{001c}",
            "\u{001c}\u{05d0}",
            "\u{05d0}\u{001c}",
            "\u{05d0}\u{001c}\u{001c}A",
            "\u{001c}\u{05d0}\u{001c}A\u{001c}",
        ] {
            let layout = layout_text(&text_system, text)?;
            assert_eq!(layout.len, text.len(), "{text:?}");
        }

        Ok(())
    }

    /// Glyph indices must stay absolute and positions ordered across segment
    /// boundaries, otherwise cursor placement and hit testing desync. Uses
    /// single-direction text so visual order matches logical order.
    #[test]
    fn layout_line_keeps_indices_and_positions_ordered_across_paragraphs() -> Result<()> {
        let text_system = text_system()?;
        let text = "ab\u{001c}cd\u{2029}ef";
        let layout = layout_text(&text_system, text)?;

        let glyphs: Vec<_> = layout.runs.iter().flat_map(|run| &run.glyphs).collect();
        assert!(!glyphs.is_empty());

        for glyph in &glyphs {
            assert!(glyph.index < text.len(), "{:?}", glyph.index);
            assert!(text.is_char_boundary(glyph.index), "{:?}", glyph.index);
        }
        for pair in glyphs.windows(2) {
            assert!(pair[0].index < pair[1].index);
            assert!(pair[0].position.x <= pair[1].position.x);
        }

        // Every segment contributes width, so the whole line is wider than its
        // leading paragraph alone.
        assert!(layout.width > layout_text(&text_system, "ab")?.width);
        Ok(())
    }

    /// A font run boundary that does not line up with a paragraph boundary must
    /// still be clipped to the right segments.
    #[test]
    fn layout_line_with_font_run_straddling_a_separator() -> Result<()> {
        let text_system = text_system()?;
        let font_id = text_system.font_id(&gpui::font("IBM Plex Sans"))?;
        let text = "ab\u{001c}\u{05d0}\u{05d1}";

        // The run boundary falls inside the trailing RTL paragraph.
        let runs = [
            FontRun {
                len: "ab\u{001c}\u{05d0}".len(),
                font_id,
                letter_spacing: None,
            },
            FontRun {
                len: "\u{05d1}".len(),
                font_id,
                letter_spacing: None,
            },
        ];
        let layout = text_system.layout_line(text, gpui::px(14.0), &runs);

        assert_eq!(layout.len, text.len());
        assert!(layout.width > Pixels::ZERO);
        Ok(())
    }

    /// Lines with no separator take the fast path and must be shaped exactly as
    /// they were before paragraph splitting existed.
    #[test]
    fn layout_line_without_separators_takes_fast_path() -> Result<()> {
        let text_system = text_system()?;

        for text in [
            "hello world",
            "\u{05d0}\u{05d1}\u{05d2}",
            "mixed \u{05d0}\u{05d1}",
        ] {
            assert!(!contains_paragraph_separator(text), "{text:?}");
            let layout = layout_text(&text_system, text)?;
            assert_eq!(layout.len, text.len(), "{text:?}");
            assert!(layout.width > Pixels::ZERO, "{text:?}");
        }

        Ok(())
    }

    #[test]
    fn paragraph_separator_detection() {
        for separator in SEPARATORS {
            assert!(is_paragraph_separator(*separator), "{separator:?}");
            assert!(contains_paragraph_separator(&format!("a{separator}b")));
        }

        for text in [
            "",
            "plain ascii",
            "\u{05d0}",
            "tab\there",
            "emoji \u{1f600}",
        ] {
            assert!(!contains_paragraph_separator(text), "{text:?}");
        }
    }

    #[test]
    fn font_runs_are_clipped_to_segment() {
        let runs = [
            FontRun {
                len: 3,
                font_id: fid(1),
                letter_spacing: None,
            },
            FontRun {
                len: 4,
                font_id: fid(2),
                letter_spacing: None,
            },
        ];

        assert_eq!(clip_font_runs(&runs, 0..7).as_slice(), &runs);
        assert_eq!(
            clip_font_runs(&runs, 2..5).as_slice(),
            &[
                FontRun {
                    len: 1,
                    font_id: fid(1),
                    letter_spacing: None,
                },
                FontRun {
                    len: 2,
                    font_id: fid(2),
                    letter_spacing: None,
                },
            ]
        );
        assert_eq!(
            clip_font_runs(&runs, 3..7).as_slice(),
            &[FontRun {
                len: 4,
                font_id: fid(2),
                letter_spacing: None,
            }]
        );
        assert!(clip_font_runs(&runs, 5..5).is_empty());
    }

    /// Color-emoji faces per shipped platform take the color raster path;
    /// body and symbol faces stay monochrome. `Segoe UI Symbol` is the
    /// deliberate negative: text-presentation symbols must not flip.
    #[test]
    fn known_emoji_fonts_cover_shipped_platforms() {
        for name in [
            "NotoColorEmoji",
            "SegoeUIEmoji",
            "AppleColorEmoji",
            ".AppleColorEmojiUI",
            "TwemojiMozilla",
        ] {
            assert!(
                check_is_known_emoji_font(name),
                "{name} must take the color raster path"
            );
        }
        for name in [
            "Spline Sans",
            "Segoe UI",
            "Segoe UI Symbol",
            "Arial",
            "",
        ] {
            assert!(
                !check_is_known_emoji_font(name),
                "{name} must stay on the outline path"
            );
        }
    }

    /// End to end on Windows: the inbox `Segoe UI Emoji` face shapes
    /// U+1F389 with the emoji flag set and rasterizes BGRA bytes holding
    /// real color variance — not a monochrome mask. Skips (passes) where
    /// the platform does not provide the face.
    #[test]
    fn system_emoji_rasterizes_color_where_platform_provides_it() -> Result<()> {
        if !cfg!(target_os = "windows") {
            return Ok(());
        }
        let text_system = CosmicTextSystem::new("Segoe UI");
        if !text_system
            .all_font_names()
            .iter()
            .any(|name| name == "Segoe UI Emoji")
        {
            return Ok(());
        }

        let font_id = text_system.font_id(&gpui::font("Segoe UI Emoji"))?;
        let size = gpui::px(32.0);
        let text = "\u{1F389}";
        let runs = [FontRun {
            len: text.len(),
            font_id,
            letter_spacing: None,
        }];
        let layout = text_system.layout_line(text, size, &runs);

        let (run_font_id, glyph_id) = layout
            .runs
            .iter()
            .flat_map(|run| run.glyphs.iter().map(|glyph| (run.font_id, glyph)))
            .find(|(_, glyph)| glyph.is_emoji)
            .map(|(font_id, glyph)| (font_id, glyph.id))
            .expect("party popper must shape with the emoji flag set");

        let params = RenderGlyphParams {
            font_id: run_font_id,
            glyph_id,
            font_size: size,
            subpixel_variant: gpui::point(0u8, 0u8),
            scale_factor: 1.0,
            is_emoji: true,
            subpixel_rendering: false,
            dilation: 0,
        };
        let bounds = text_system.glyph_raster_bounds(&params)?;
        assert!(!bounds.is_zero(), "emoji raster bounds must be non-empty");
        let (_, data) = text_system.rasterize_glyph(&params, bounds)?;
        assert!(
            !data.is_empty() && data.len() % 4 == 0,
            "emoji raster must be BGRA bytes, not a 1-byte mask"
        );
        let vivid = data.chunks_exact(4).any(|pixel| {
            let opaque = pixel[3] > 0;
            opaque && (pixel[0] != pixel[1] || pixel[1] != pixel[2])
        });
        assert!(
            vivid,
            "rasterized emoji must hold non-grayscale color pixels"
        );
        Ok(())
    }

    /// Implicit routing: an ordinary body-font request with NO explicit
    /// emoji family must still reach the platform color face through
    /// cosmic fallback, with the emoji flag set. This distinguishes the
    /// color-raster fix from font selection — the previous test requests
    /// the emoji face directly and cannot prove routing. Skips (passes)
    /// where the platform does not provide the face.
    #[test]
    fn implicit_fallback_routes_emoji_to_system_color_face() -> Result<()> {
        if !cfg!(target_os = "windows") {
            return Ok(());
        }
        let text_system = CosmicTextSystem::new("Segoe UI");
        text_system.add_fonts(vec![Cow::Borrowed(IBM_PLEX)])?;
        if !text_system
            .all_font_names()
            .iter()
            .any(|name| name == "Segoe UI Emoji")
        {
            return Ok(());
        }

        let font_id = text_system.font_id(&gpui::font("IBM Plex Sans"))?;
        let text = "Whoopty \u{1F389}";
        let runs = [FontRun {
            len: text.len(),
            font_id,
            letter_spacing: None,
        }];
        let layout = text_system.layout_line(text, gpui::px(32.0), &runs);

        let emoji_start = text.len() - "\u{1F389}".len();
        let emoji_font_id = layout
            .runs
            .iter()
            .find(|run| {
                run.glyphs
                    .iter()
                    .any(|glyph| glyph.index == emoji_start && glyph.is_emoji)
            })
            .map(|run| run.font_id)
            .expect("emoji scalar under a body-font request must shape from a color face");
        let postscript = {
            let state = text_system.0.read();
            let db_id = state.loaded_fonts[emoji_font_id.0].font.id();
            state
                .font_system
                .db()
                .face(db_id)
                .map(|face| face.post_script_name.clone())
        };
        assert_eq!(
            postscript.as_deref(),
            Some("SegoeUIEmoji"),
            "native-first precedence: the platform color face must win the fallback"
        );
        Ok(())
    }

    /// Loads the superproject Twemoji asset when the vendor tree is
    /// checked out as a submodule; `None` keeps the tree standalone-safe
    /// by skipping Twemoji-dependent proofs.
    fn twemoji_bytes() -> Option<Vec<u8>> {
        let path = format!(
            "{}/../../../../modules/assets/fonts/twemoji-mozilla.ttf",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read(path).ok()
    }

    /// Resolves the winning face's PostScript name for one shaped run.
    fn face_postscript(text_system: &CosmicTextSystem, font_id: FontId) -> Option<String> {
        let state = text_system.0.read();
        let db_id = state.loaded_fonts[font_id.0].font.id();
        state
            .font_system
            .db()
            .face(db_id)
            .map(|face| face.post_script_name.clone())
    }

    /// Rasterizes one emoji glyph and proves BGRA bytes with real color
    /// variance — not a monochrome mask.
    fn assert_color_raster(
        text_system: &CosmicTextSystem,
        font_id: FontId,
        glyph_id: GlyphId,
        size: gpui::Pixels,
    ) -> Result<()> {
        let params = RenderGlyphParams {
            font_id,
            glyph_id,
            font_size: size,
            subpixel_variant: gpui::point(0u8, 0u8),
            scale_factor: 1.0,
            is_emoji: true,
            subpixel_rendering: false,
            dilation: 0,
        };
        let bounds = text_system.glyph_raster_bounds(&params)?;
        assert!(!bounds.is_zero(), "emoji raster bounds must be non-empty");
        let (_, data) = text_system.rasterize_glyph(&params, bounds)?;
        assert!(
            !data.is_empty() && data.len() % 4 == 0,
            "emoji raster must be BGRA bytes, not a 1-byte mask"
        );
        assert!(
            data.chunks_exact(4)
                .any(|pixel| pixel[3] > 0 && (pixel[0] != pixel[1] || pixel[1] != pixel[2])),
            "rasterized emoji must hold non-grayscale color pixels"
        );
        Ok(())
    }

    /// Platform-first with Twemoji registered: the native face must still
    /// win covered clusters and body text must stay off the emoji face.
    #[test]
    fn platform_face_wins_over_registered_twemoji() -> Result<()> {
        if !cfg!(target_os = "windows") {
            return Ok(());
        }
        let twemoji = require_twemoji_bytes();
        let text_system = CosmicTextSystem::new("Segoe UI");
        text_system.add_fonts(vec![Cow::Owned(twemoji)])?;
        if !text_system
            .all_font_names()
            .iter()
            .any(|name| name == "Segoe UI Emoji")
        {
            return Ok(());
        }

        let font_id = text_system.font_id(&gpui::font("Segoe UI"))?;
        let text = "Whoopty \u{1F389}";
        let runs = [FontRun {
            len: text.len(),
            font_id,
            letter_spacing: None,
        }];
        let layout = text_system.layout_line(text, gpui::px(32.0), &runs);

        let emoji_start = text.len() - "\u{1F389}".len();
        let mut saw_emoji = false;
        for run in &layout.runs {
            let face = face_postscript(&text_system, run.font_id);
            for glyph in &run.glyphs {
                if glyph.index >= emoji_start {
                    saw_emoji = true;
                    assert!(glyph.is_emoji, "emoji scalar must take the color path");
                    assert_eq!(
                        face.as_deref(),
                        Some("SegoeUIEmoji"),
                        "platform face must win over registered Twemoji"
                    );
                } else {
                    assert_eq!(
                        face.as_deref(),
                        Some("SegoeUI"),
                        "body text must stay on the requested body font"
                    );
                }
            }
        }
        assert!(saw_emoji, "party popper must shape at least one glyph");
        Ok(())
    }

    /// Twemoji fallback without system fonts: an uncovered flag cluster
    /// resolves to one Twemoji run with the emoji flag and colored pixels.
    #[test]
    fn twemoji_serves_uncovered_flag_as_one_cluster() -> Result<()> {
        if !cfg!(target_os = "windows") {
            return Ok(());
        }
        let twemoji = require_twemoji_bytes();
        let text_system = CosmicTextSystem::new_without_system_fonts("IBM Plex Sans");
        text_system.add_fonts(vec![Cow::Borrowed(IBM_PLEX)])?;
        text_system.add_fonts(vec![Cow::Owned(twemoji)])?;

        let font_id = text_system.font_id(&gpui::font("IBM Plex Sans"))?;
        let text = "\u{1F1F3}\u{1F1F4}";
        let runs = [FontRun {
            len: text.len(),
            font_id,
            letter_spacing: None,
        }];
        let layout = text_system.layout_line(text, gpui::px(32.0), &runs);

        let mut faces = std::collections::HashSet::new();
        let mut glyphs = 0;
        let mut first: Option<(FontId, GlyphId)> = None;
        for run in &layout.runs {
            for glyph in &run.glyphs {
                glyphs += 1;
                assert!(glyph.is_emoji, "flag cluster must take the color path");
                faces.insert(run.font_id.0);
                if first.is_none() {
                    first = Some((run.font_id, glyph.id));
                }
            }
        }
        assert!(glyphs > 0, "flag cluster must shape glyphs");
        assert_eq!(glyphs, 1, "flag cluster must shape as one combined glyph");
        assert_eq!(faces.len(), 1, "flag cluster must resolve to one face");
        let (font_id, glyph_id) = first.expect("non-empty glyphs");
        assert_eq!(
            face_postscript(&text_system, font_id).as_deref(),
            Some("TwemojiMozilla"),
            "uncovered flag must resolve to Twemoji"
        );
        assert_color_raster(&text_system, font_id, glyph_id, gpui::px(32.0))
    }

    /// Loads the superproject Twemoji asset; the gate runs on Windows
    /// with the asset present, so a missing file fails loudly instead of
    /// silently skipping the proof. Other platforms skip (out of scope).
    fn require_twemoji_bytes() -> Vec<u8> {
        twemoji_bytes().expect(
            "Twemoji asset required at modules/assets/fonts/twemoji-mozilla.ttf \
             (see docs/reference-emoji-interface.md)",
        )
    }

    /// Shapes `text` with an ordinary body-font request.
    fn shape_body(
        text_system: &CosmicTextSystem,
        font_id: FontId,
        text: &str,
    ) -> LineLayout {
        let runs = [FontRun {
            len: text.len(),
            font_id,
            letter_spacing: None,
        }];
        text_system.layout_line(text, gpui::px(32.0), &runs)
    }

    /// Collects `(font, glyph, is_emoji)` for glyphs addressing
    /// `start..end`, so whole-grapheme outcomes (ZWJ, VS16, modifiers,
    /// regional and tag sequences) are asserted — never a first scalar.
    fn cluster_glyphs(
        layout: &LineLayout,
        start: usize,
        end: usize,
    ) -> Vec<(FontId, GlyphId, bool)> {
        let mut out = Vec::new();
        for run in &layout.runs {
            for glyph in &run.glyphs {
                if glyph.index >= start && glyph.index < end {
                    out.push((run.font_id, glyph.id, glyph.is_emoji));
                }
            }
        }
        out
    }

    /// Asserts every cluster glyph shares one face, names it, and proves
    /// the color raster on its first glyph.
    fn assert_single_color_cluster(
        text_system: &CosmicTextSystem,
        layout: &LineLayout,
        start: usize,
        end: usize,
        expected_face: &str,
        text: &str,
    ) -> Result<()> {
        let glyphs = cluster_glyphs(layout, start, end);
        assert!(!glyphs.is_empty(), "{text:?} must shape glyphs");
        let (font_id, _, _) = glyphs[0];
        assert!(
            glyphs.iter().all(|(id, _, _)| *id == font_id),
            "{text:?} cluster must not fragment across faces"
        );
        assert!(
            glyphs.iter().all(|(_, _, emoji)| *emoji),
            "{text:?} cluster must take the color path"
        );
        assert_eq!(
            face_postscript(text_system, font_id).as_deref(),
            Some(expected_face),
            "{text:?} cluster must resolve to {expected_face}"
        );
        let (_, glyph_id, _) = glyphs[0];
        assert_color_raster(text_system, font_id, glyph_id, gpui::px(32.0))
    }

    /// Mandatory precedence proof with BOTH system fonts and Twemoji:
    /// party popper stays native while the Norway regional flag — whose
    /// scalars Segoe covers but cannot combine (no RI composition in its
    /// GSUB, measured) — resolves to Twemoji as one combined glyph with
    /// colored pixels. First-scalar coverage alone would strand it on
    /// Segoe uncombined; whole-grapheme selection with formation probing
    /// is what routes it.
    #[test]
    fn native_popper_and_twemoji_flag_share_one_layout() -> Result<()> {
        if !cfg!(target_os = "windows") {
            return Ok(());
        }
        let twemoji = require_twemoji_bytes();
        let text_system = CosmicTextSystem::new("Segoe UI");
        text_system.add_fonts(vec![Cow::Owned(twemoji)])?;
        if !text_system
            .all_font_names()
            .iter()
            .any(|name| name == "Segoe UI Emoji")
        {
            return Ok(());
        }

        let font_id = text_system.font_id(&gpui::font("Segoe UI"))?;
        let popper = "\u{1F389}";
        let flag = "\u{1F1F3}\u{1F1F4}";
        let text = format!("{popper}{flag}");
        let layout = shape_body(&text_system, font_id, &text);

        let popper_glyphs = cluster_glyphs(&layout, 0, popper.len());
        assert!(!popper_glyphs.is_empty());
        for (id, _, emoji) in popper_glyphs {
            assert!(emoji, "party popper must take the color path");
            assert_eq!(
                face_postscript(&text_system, id).as_deref(),
                Some("SegoeUIEmoji"),
                "party popper must stay native with Twemoji registered"
            );
        }
        let flag_glyphs = cluster_glyphs(&layout, popper.len(), text.len());
        assert_eq!(
            flag_glyphs.len(),
            1,
            "Norway flag must shape as one combined glyph"
        );
        assert_single_color_cluster(
            &text_system,
            &layout,
            popper.len(),
            text.len(),
            "TwemojiMozilla",
            flag,
        )
    }

    /// England tag sequence: Segoe covers only the black flag scalar, so
    /// coverage alone routes the whole cluster to Twemoji, which ligates
    /// it into one glyph with colored pixels. Uses the valid sequence
    /// (black flag + g b e n g + cancel tag).
    #[test]
    fn england_tag_flag_routes_to_twemoji_by_coverage() -> Result<()> {
        if !cfg!(target_os = "windows") {
            return Ok(());
        }
        let twemoji = require_twemoji_bytes();
        let text_system = CosmicTextSystem::new("Segoe UI");
        text_system.add_fonts(vec![Cow::Owned(twemoji)])?;
        if !text_system
            .all_font_names()
            .iter()
            .any(|name| name == "Segoe UI Emoji")
        {
            return Ok(());
        }

        let font_id = text_system.font_id(&gpui::font("Segoe UI"))?;
        let text = "\u{1F3F4}\u{E0067}\u{E0062}\u{E0065}\u{E006E}\u{E0067}\u{E007F}";
        let layout = shape_body(&text_system, font_id, text);
        let glyphs = cluster_glyphs(&layout, 0, text.len());
        assert_eq!(
            glyphs.len(),
            1,
            "England flag must shape as one combined glyph"
        );
        assert_single_color_cluster(
            &text_system,
            &layout,
            0,
            text.len(),
            "TwemojiMozilla",
            text,
        )
    }

    /// VS16 requests emoji presentation: whole cluster, native face, color.
    #[test]
    fn vs16_heart_shapes_native_color_cluster() -> Result<()> {
        if !cfg!(target_os = "windows") {
            return Ok(());
        }
        let twemoji = require_twemoji_bytes();
        let text_system = CosmicTextSystem::new("Segoe UI");
        text_system.add_fonts(vec![Cow::Owned(twemoji)])?;
        if !text_system
            .all_font_names()
            .iter()
            .any(|name| name == "Segoe UI Emoji")
        {
            return Ok(());
        }

        let font_id = text_system.font_id(&gpui::font("Segoe UI"))?;
        let text = "\u{2764}\u{FE0F}";
        let layout = shape_body(&text_system, font_id, text);
        assert_single_color_cluster(
            &text_system,
            &layout,
            0,
            text.len(),
            "SegoeUIEmoji",
            text,
        )
    }

    /// VS15 requests text presentation: Twemoji lacks U+FE0E, so it must
    /// not serve any glyph of the cluster — native handling is preserved
    /// as-is. Presentation policy itself is a separate packet.
    #[test]
    fn vs15_heart_never_routes_to_twemoji() -> Result<()> {
        if !cfg!(target_os = "windows") {
            return Ok(());
        }
        let twemoji = require_twemoji_bytes();
        let text_system = CosmicTextSystem::new("Segoe UI");
        text_system.add_fonts(vec![Cow::Owned(twemoji)])?;
        if !text_system
            .all_font_names()
            .iter()
            .any(|name| name == "Segoe UI Emoji")
        {
            return Ok(());
        }

        let font_id = text_system.font_id(&gpui::font("Segoe UI"))?;
        let text = "\u{2764}\u{FE0E}";
        let layout = shape_body(&text_system, font_id, text);
        let glyphs = cluster_glyphs(&layout, 0, text.len());
        assert!(!glyphs.is_empty(), "VS15 cluster must shape glyphs");
        for (id, _, _) in &glyphs {
            assert_ne!(
                face_postscript(&text_system, *id).as_deref(),
                Some("TwemojiMozilla"),
                "VS15 text presentation must not route to the color fallback face"
            );
        }
        Ok(())
    }

    /// ZWJ family: Segoe covers every scalar but cannot combine the full
    /// sequence (no complete ligature in its GSUB); Twemoji ligates it
    /// into one glyph. Formation probing routes the whole cluster there.
    #[test]
    fn zwj_family_routes_to_twemoji_single_glyph() -> Result<()> {
        if !cfg!(target_os = "windows") {
            return Ok(());
        }
        let twemoji = require_twemoji_bytes();
        let text_system = CosmicTextSystem::new("Segoe UI");
        text_system.add_fonts(vec![Cow::Owned(twemoji)])?;
        if !text_system
            .all_font_names()
            .iter()
            .any(|name| name == "Segoe UI Emoji")
        {
            return Ok(());
        }

        let font_id = text_system.font_id(&gpui::font("Segoe UI"))?;
        let text = "\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}\u{200D}\u{1F466}";
        let layout = shape_body(&text_system, font_id, text);
        let glyphs = cluster_glyphs(&layout, 0, text.len());
        assert_eq!(
            glyphs.len(),
            1,
            "ZWJ family must shape as one combined glyph"
        );
        assert_single_color_cluster(
            &text_system,
            &layout,
            0,
            text.len(),
            "TwemojiMozilla",
            text,
        )
    }

    /// Skin tone: both faces ligate, so native-first order keeps Segoe
    /// with one combined glyph and colored pixels.
    #[test]
    fn skin_tone_stays_native_single_glyph() -> Result<()> {
        if !cfg!(target_os = "windows") {
            return Ok(());
        }
        let twemoji = require_twemoji_bytes();
        let text_system = CosmicTextSystem::new("Segoe UI");
        text_system.add_fonts(vec![Cow::Owned(twemoji)])?;
        if !text_system
            .all_font_names()
            .iter()
            .any(|name| name == "Segoe UI Emoji")
        {
            return Ok(());
        }

        let font_id = text_system.font_id(&gpui::font("Segoe UI"))?;
        let text = "\u{1F44D}\u{1F3FD}";
        let layout = shape_body(&text_system, font_id, text);
        let glyphs = cluster_glyphs(&layout, 0, text.len());
        assert_eq!(
            glyphs.len(),
            1,
            "skin-tone cluster must shape as one combined glyph"
        );
        assert_single_color_cluster(
            &text_system,
            &layout,
            0,
            text.len(),
            "SegoeUIEmoji",
            text,
        )
    }

    /// Keycap: whichever face forms the sequence wins it whole with
    /// colored pixels; plain `#` alone always stays on the body font
    /// (ASCII fast path), so ordinary text never changes faces.
    #[test]
    fn keycap_cluster_resolves_to_single_color_face() -> Result<()> {
        if !cfg!(target_os = "windows") {
            return Ok(());
        }
        let twemoji = require_twemoji_bytes();
        let text_system = CosmicTextSystem::new("Segoe UI");
        text_system.add_fonts(vec![Cow::Owned(twemoji)])?;
        if !text_system
            .all_font_names()
            .iter()
            .any(|name| name == "Segoe UI Emoji")
        {
            return Ok(());
        }

        let font_id = text_system.font_id(&gpui::font("Segoe UI"))?;
        let text = "#\u{FE0F}\u{20E3}";
        let layout = shape_body(&text_system, font_id, text);
        let glyphs = cluster_glyphs(&layout, 0, text.len());
        assert_eq!(
            glyphs.len(),
            1,
            "keycap cluster must shape as one combined glyph"
        );
        assert!(!glyphs.is_empty(), "keycap must shape glyphs");
        let (font_id, _, _) = glyphs[0];
        assert!(
            glyphs.iter().all(|(id, _, _)| *id == font_id),
            "keycap cluster must not fragment across faces"
        );
        let face = face_postscript(&text_system, font_id);
        assert!(
            matches!(
                face.as_deref(),
                Some("SegoeUIEmoji") | Some("TwemojiMozilla")
            ),
            "keycap cluster must resolve to a color face"
        );
        assert!(
            glyphs.iter().all(|(_, _, emoji)| *emoji),
            "keycap cluster must take the color path"
        );
        let (_, glyph_id, _) = glyphs[0];
        assert_color_raster(&text_system, font_id, glyph_id, gpui::px(32.0))
    }

    /// Whole-grapheme slot selection with a stubbed formation probe.
    /// ASCII stays primary, primary-full stays primary, single
    /// full-coverers win without probing, and formation breaks ties.
    #[test]
    fn pick_cluster_slot_prefers_forming_native_first_face() {
        let primary = fid(0);
        let fb = chain(&[1, 2]);
        // Panic stub: none of these fixtures may consult formation.
        let mut never = |_: FontId, _: &str| -> bool {
            panic!("formation probe must not run here");
        };

        // ASCII (including multi-char CRLF) never leaves primary.
        assert_eq!(
            pick_cluster_slot("\r\n", primary, &fb, &|_, _| true, &mut never),
            None
        );
        // Primary covering the whole cluster wins without probing.
        assert_eq!(
            pick_cluster_slot("é", primary, &fb, &|_, _| true, &mut never),
            None
        );
        // Single full-coverer wins without probing.
        let covers_emoji = |id: FontId, ch: char| id == fid(1) && ch != '\u{200D}';
        assert_eq!(
            pick_cluster_slot("\u{1F389}", primary, &fb, &covers_emoji, &mut never),
            Some(0)
        );
        // Nothing covering falls back to the legacy first-scalar rule.
        let covers_none = |_: FontId, _: char| false;
        assert_eq!(
            pick_cluster_slot("字", primary, &fb, &covers_none, &mut never),
            pick_covering_slot('字', None, primary, &fb, &covers_none)
        );
    }

    #[test]
    fn pick_cluster_slot_formation_breaks_full_coverage_ties() {
        let primary = fid(0);
        let fb = chain(&[1, 2]);
        let covers_all = |id: FontId, _: char| id != fid(0);
        // First face forms: native-first order holds.
        let mut first_forms = |id: FontId, _: &str| id == fid(1);
        assert_eq!(
            pick_cluster_slot("\u{1F1F3}\u{1F1F4}", primary, &fb, &covers_all, &mut first_forms),
            Some(0)
        );
        // Only the fallback forms: whole cluster reroutes.
        let mut second_forms = |id: FontId, _: &str| id == fid(2);
        assert_eq!(
            pick_cluster_slot("\u{1F1F3}\u{1F1F4}", primary, &fb, &covers_all, &mut second_forms),
            Some(1)
        );
        // Nothing forms: chain order, same as coverage-only selection.
        let mut none_forms = |_: FontId, _: &str| false;
        assert_eq!(
            pick_cluster_slot("\u{1F1F3}\u{1F1F4}", primary, &fb, &covers_all, &mut none_forms),
            Some(0)
        );
    }

    /// The counting rule behind formation probing: `.notdef` rejects
    /// immediately, glyph 3 is ignorable only as the expected color
    /// face's own remnant, every remaining visible glyph must come from
    /// the expected face, and exactly one must remain. In particular one
    /// good glyph plus `.notdef` or another face's output is NOT a
    /// combination.
    #[test]
    fn single_combined_cluster_rejects_mixed_output() {
        assert!(is_single_combined_cluster([(true, 42u16)].into_iter(), true));
        assert!(!is_single_combined_cluster(
            [(true, 42u16), (true, 0u16)].into_iter(),
            true
        ));
        assert!(!is_single_combined_cluster(
            [(true, 42u16), (false, 0u16)].into_iter(),
            true
        ));
        assert!(!is_single_combined_cluster(
            [(true, 42u16), (false, 200u16)].into_iter(),
            true
        ));
        assert!(!is_single_combined_cluster(
            [(true, 42u16), (true, 43u16)].into_iter(),
            true
        ));
        assert!(!is_single_combined_cluster([].into_iter(), true));
        assert!(!is_single_combined_cluster([(true, 3u16)].into_iter(), true));
        assert!(!is_single_combined_cluster([(false, 3u16)].into_iter(), true));
        assert!(is_single_combined_cluster([(true, 3u16)].into_iter(), false));
    }

    #[test]
    fn primary_wins_over_current_fallback_when_primary_covers() {
        let primary = fid(0);
        let fb = chain(&[1, 2]);
        let covers = |id: FontId, _: char| id == fid(0) || id == fid(1);
        assert_eq!(
            pick_covering_slot('a', Some(0), primary, &fb, &covers),
            None
        );
    }

    #[test]
    fn primary_preferred_over_fallback_when_both_cover() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |_: FontId, _: char| true;
        assert_eq!(pick_covering_slot('a', None, primary, &fb, &covers), None);
    }

    #[test]
    fn falls_through_chain_in_order() {
        let primary = fid(0);
        let fb = chain(&[1, 2, 3]);
        // only fallback 2 at index 1 covers.
        let covers = |id: FontId, _: char| id == fid(2);
        assert_eq!(
            pick_covering_slot('字', None, primary, &fb, &covers),
            Some(1)
        );
    }

    #[test]
    fn no_coverage_returns_primary() {
        let primary = fid(0);
        let fb = chain(&[1, 2]);
        let covers = |_: FontId, _: char| false;
        // nothing covers. return `None` so the `cosmic-text` built in script
        // fallback can take over during shaping.
        assert_eq!(
            pick_covering_slot('\u{1F600}', Some(1), primary, &fb, &covers),
            None
        );
    }

    #[test]
    fn empty_chain_always_returns_primary() {
        let primary = fid(0);
        let fb: SmallVec<[(FontId, SharedString); 4]> = SmallVec::new();
        let covers = |_: FontId, _: char| false;
        assert_eq!(pick_covering_slot('a', None, primary, &fb, &covers), None);
    }

    #[test]
    fn slot_font_id_resolution() {
        let primary = fid(7);
        let fb = chain(&[10, 20]);
        assert_eq!(slot_font_id(None, primary, &fb), fid(7));
        assert_eq!(slot_font_id(Some(0), primary, &fb), fid(10));
        assert_eq!(slot_font_id(Some(1), primary, &fb), fid(20));
    }

    #[test]
    fn run_spans_with_no_chain_emit_one_primary_span() {
        let primary = fid(0);
        let fb: SmallVec<[(FontId, SharedString); 4]> = SmallVec::new();
        let covers = |_: FontId, _: char| false;
        let text = "hello";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), None, primary)]);
    }

    #[test]
    fn run_spans_use_byte_offsets_for_multibyte_chars() {
        let primary = fid(0);
        let fb = chain(&[1]);
        // primary covers ascii. fallback covers cjk.
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                !ch.is_ascii()
            }
        };
        let text = "a字b";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        // '字' is 3 bytes so split is at 1 then 4.
        assert_eq!(
            spans.as_slice(),
            &[
                span(0, 1, None, primary),
                span(1, 4, Some(0), fid(1)),
                span(4, 5, None, primary),
            ]
        );
    }

    #[test]
    fn run_spans_respect_run_offset() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                !ch.is_ascii()
            }
        };
        // outer text has a prefix that is not part of this run.
        let text = "xx字y";
        let run_offset = 2;
        let run_len = text.len() - run_offset;
        let spans = compute_run_spans(text, run_offset, run_len, primary, &fb, &covers);
        assert_eq!(
            spans.as_slice(),
            &[span(2, 5, Some(0), fid(1)), span(5, 6, None, primary)]
        );
    }

    #[test]
    fn run_spans_keep_combining_marks_with_base_in_fallback() {
        let primary = fid(0);
        let fb = chain(&[1]);
        // primary covers ascii only. fallback covers the base char.
        // combining mark must stay in the fallback span even when fallback
        // does not advertise coverage of it.
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                ch == '\u{0905}'
            }
        };
        // \u{0905} devanagari short a + \u{0902} candrabindu mark.
        let text = "\u{0905}\u{0902}";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), Some(0), fid(1))]);
    }

    #[test]
    fn run_spans_keep_zwj_inside_emoji_cluster() {
        let primary = fid(0);
        let fb = chain(&[1]);
        // only fallback covers the emoji codepoints. zwj must not split.
        let covers = |id: FontId, ch: char| id == fid(1) && ch != '\u{200D}';
        // family zwj sequence woman zwj girl.
        let text = "\u{1F469}\u{200D}\u{1F467}";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), Some(0), fid(1))]);
    }

    #[test]
    fn run_spans_collapse_adjacent_same_slot() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |id: FontId, ch: char| {
            if id == primary {
                ch.is_ascii()
            } else {
                !ch.is_ascii()
            }
        };
        let text = "字字字";
        let spans = compute_run_spans(text, 0, text.len(), primary, &fb, &covers);
        assert_eq!(spans.as_slice(), &[span(0, text.len(), Some(0), fid(1))]);
    }

    #[test]
    fn run_spans_empty_run_returns_no_spans() {
        let primary = fid(0);
        let fb = chain(&[1]);
        let covers = |_: FontId, _: char| true;
        let spans = compute_run_spans("anything", 3, 0, primary, &fb, &covers);
        assert!(spans.is_empty());
    }
}
