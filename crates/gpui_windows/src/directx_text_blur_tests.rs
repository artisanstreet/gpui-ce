//! Text inside a content-filter group (`.blur()`) keeps its colour: the group renders into a
//! transparent offscreen target, so glyph blending must leave coverage in its alpha for the
//! premultiplied composite. The regression: dark subpixel text in a blurred, fading layer showed
//! the light background through (white) instead of darkening it.

use super::DirectXRenderer;
use super::path_pixel_tests::renderer;
use anyhow::{Context, Result};
use gpui::{
    AtlasKey, AtlasTile, Bounds, ContentMask, Corners, DevicePixels, FilterBoundary, FontId,
    GlyphId, MonochromeSprite, PlatformAtlas, Quad, RenderGlyphParams, ScaledFilter, ScaledPixels,
    Scene, SubpixelSprite, TransformationMatrix, WindowBackgroundAppearance, hsla, point, px, size,
};
use std::borrow::Cow;

const VIEWPORT: [u32; 2] = [64, 48];
const GLYPH: f32 = 16.0;
const GLYPH_ORIGIN: [f32; 2] = [24.0, 16.0];

fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
    Bounds::new(
        point(ScaledPixels(x), ScaledPixels(y)),
        size(ScaledPixels(width), ScaledPixels(height)),
    )
}

/// A fully covered glyph tile, standing in for the inside of a rasterised stem.
fn glyph_tile(renderer: &DirectXRenderer, subpixel: bool) -> Result<AtlasTile> {
    let side = GLYPH as i32;
    let bytes_per_pixel = if subpixel { 4 } else { 1 };
    let key = AtlasKey::Glyph(RenderGlyphParams {
        font_id: FontId(0),
        glyph_id: GlyphId(1),
        font_size: px(GLYPH),
        subpixel_variant: point(0, 0),
        scale_factor: 1.0,
        is_emoji: false,
        subpixel_rendering: subpixel,
        dilation: 0,
    });
    renderer
        .atlas
        .get_or_insert_with(&key, &mut || {
            Ok(Some((
                size(DevicePixels(side), DevicePixels(side)),
                Cow::Owned(vec![255u8; (side * side * bytes_per_pixel) as usize]),
            )))
        })?
        .context("glyph tile not inserted")
}

/// A light (or dark) window with one dark (or light) glyph, optionally inside a blurred group,
/// the glyph colour carrying the layer's opacity as gpui paints it.
fn scene(
    renderer: &DirectXRenderer,
    subpixel: bool,
    background_lightness: f32,
    text_lightness: f32,
    blur: Option<f32>,
    opacity: f32,
) -> Result<Scene> {
    let full = rect(0.0, 0.0, VIEWPORT[0] as f32, VIEWPORT[1] as f32);
    let mask = ContentMask { bounds: full };
    let glyph = rect(GLYPH_ORIGIN[0], GLYPH_ORIGIN[1], GLYPH, GLYPH);
    let tile = glyph_tile(renderer, subpixel)?;
    let color = hsla(0.6, 0.1, text_lightness, opacity).into();
    let boundary = blur.map(|radius| FilterBoundary {
        order: 1,
        bounds: glyph,
        content_mask: mask,
        corner_radii: Corners::default(),
        filters: smallvec::smallvec![ScaledFilter::Blur(ScaledPixels(radius))],
        opacity: 1.0,
        is_start: true,
    });

    let mut scene = Scene::default();
    scene.insert_primitive(Quad {
        order: 0,
        bounds: full,
        content_mask: mask,
        background: hsla(0.6, 0.1, background_lightness, 1.0).into(),
        ..Default::default()
    });
    if let Some(boundary) = &boundary {
        scene.insert_primitive(boundary.clone());
    }
    if subpixel {
        scene.insert_primitive(SubpixelSprite {
            order: 2,
            pad: 0,
            bounds: glyph,
            content_mask: mask,
            color,
            tile,
            transformation: TransformationMatrix::unit(),
        });
    } else {
        scene.insert_primitive(MonochromeSprite {
            order: 2,
            pad: 0,
            bounds: glyph,
            content_mask: mask,
            color,
            tile,
            transformation: TransformationMatrix::unit(),
        });
    }
    if let Some(boundary) = boundary {
        scene.insert_primitive(FilterBoundary {
            order: 3,
            is_start: false,
            ..boundary
        });
    }
    scene.finish();
    Ok(scene)
}

fn brightness(pixel: &image::Rgba<u8>) -> u32 {
    pixel.0[..3].iter().map(|&channel| channel as u32).sum()
}

/// The brightness of the glyph's centre and of the background far from it.
fn render(
    subpixel: bool,
    background_lightness: f32,
    text_lightness: f32,
    blur: Option<f32>,
    opacity: f32,
) -> Result<(u32, u32)> {
    let mut renderer = renderer(VIEWPORT)?;
    let scene = scene(
        &renderer,
        subpixel,
        background_lightness,
        text_lightness,
        blur,
        opacity,
    )?;
    let image = renderer.render_to_image(&scene, WindowBackgroundAppearance::Opaque)?;
    let centre = image.get_pixel(
        (GLYPH_ORIGIN[0] + GLYPH / 2.0) as u32,
        (GLYPH_ORIGIN[1] + GLYPH / 2.0) as u32,
    );
    Ok((brightness(centre), brightness(image.get_pixel(2, 2))))
}

#[test]
fn dark_text_in_a_blurred_fading_layer_darkens_a_light_background() -> Result<()> {
    for subpixel in [true, false] {
        for (blur, opacity) in [(Some(1.0), 0.5), (Some(2.0), 0.1), (Some(0.5), 0.9)] {
            let (text, background) = render(subpixel, 0.95, 0.34, blur, opacity)?;
            // Half-covered dark text sits well below the background; the bug lifted it above.
            let darkening = (background as f32 * 0.66 * opacity * 0.5) as u32;
            assert!(
                text + darkening.max(3) <= background,
                "dark text washed out: subpixel={subpixel} blur={blur:?} opacity={opacity} \
                 text={text} background={background}"
            );
        }
    }
    Ok(())
}

#[test]
fn dark_text_without_blur_matches_dark_text_in_a_blurred_opaque_layer() -> Result<()> {
    for subpixel in [true, false] {
        let (plain, background) = render(subpixel, 0.95, 0.34, None, 1.0)?;
        assert!(
            plain + 300 < background,
            "unblurred dark text not dark: subpixel={subpixel} text={plain} background={background}"
        );
        // A 1px blur leaves the middle of a 16px glyph as it was.
        let (blurred, _) = render(subpixel, 0.95, 0.34, Some(1.0), 1.0)?;
        assert!(
            blurred.abs_diff(plain) <= 9,
            "blurred opaque text changed colour: subpixel={subpixel} blurred={blurred} plain={plain}"
        );
    }
    Ok(())
}

#[test]
fn light_text_in_a_blurred_fading_layer_lightens_a_dark_background() -> Result<()> {
    for subpixel in [true, false] {
        let (text, background) = render(subpixel, 0.12, 0.9, Some(1.0), 0.5)?;
        let (plain, _) = render(subpixel, 0.12, 0.9, None, 0.5)?;
        assert!(
            text > background + 100,
            "light text lost: subpixel={subpixel} text={text} background={background}"
        );
        assert!(
            text.abs_diff(plain) <= 9,
            "blurred light text differs from unblurred: subpixel={subpixel} blurred={text} plain={plain}"
        );
    }
    Ok(())
}
