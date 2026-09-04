//! Blur Filters Example
//!
//! Demonstrates the two CSS-style blur filters:
//!
//! 1. `backdrop_blur` — frosted glass: blurs whatever is rendered behind the element.
//!    Shown as a translucent panel, and again inside a `deferred()` popover (to prove the
//!    backdrop snapshot includes everything beneath an overlay, and that the overlay sorts on top).
//! 2. `blur` — content blur: blurs the element and its own children as a group.
//!    Shown with text and again with a row of colored chips.
//!
//! A band of bright tiles animates behind the frosted panel (`live_ticker_band`), so the
//! frost is visibly re-snapshotting live content every frame.
//!
//! It also stresses two content-blur edge cases:
//!
//! 3. Nested content blur — a `blur()` element inside another `blur()` element, so the inner
//!    subtree is blurred twice (its own filter, then again as part of the outer group). This
//!    exercises the renderer's per-nesting-level group textures.
//! 4. Adjacent blocks with no gap on a dark parent, shown two ways: `blur` on each block (every
//!    block is its own group, so the parent shows through each seam — exactly like CSS `filter`
//!    on each sibling) versus `blur` on the parent (one group covering all blocks, so the blur is
//!    continuous and the seams are clean — the CSS "blur the wrapper" idiom).

use std::time::Duration;

use gpui::{
    Animation, AnimationExt as _, App, Bounds, Context, Render, Window, WindowBounds,
    WindowOptions, deferred, div, linear, point, prelude::*, px, rgb, rgba, size,
};

struct BlurExample;

/// A vivid, gap-free background so blur is obvious: a grid of saturated tiles filling the window.
fn busy_background() -> impl IntoElement {
    let palette = [
        0xef4444, 0xf97316, 0xeab308, 0x22c55e, 0x06b6d4, 0x3b82f6, 0x8b5cf6, 0xec4899,
    ];
    let mut next = 0usize;
    div()
        .absolute()
        .inset_0()
        .flex()
        .flex_col()
        .children((0..9).map(|_| {
            div().flex().flex_1().children(
                (0..8)
                    .map(|_| {
                        let hex = palette[next % palette.len()];
                        next += 1;
                        div()
                            .flex_1()
                            .h_full()
                            .bg(rgb(hex))
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_color(rgb(0xffffff))
                            .text_xl()
                            .child("◆")
                    })
                    .collect::<Vec<_>>(),
            )
        }))
}

/// A band of bright tiles sliding back and forth *behind* the frosted panel, clipped to
/// a small region around it so the motion doesn't occlude the other demos. Because the
/// tiles keep moving, the frosted panel is visibly re-snapshotting live content every
/// frame — a stale one-shot backdrop copy would show frozen tiles behind the frost.
fn live_ticker_band() -> impl IntoElement {
    const TILE_W: f32 = 100.;
    const TILE_COUNT: usize = 14;
    /// How far the strip travels between its extremes, in logical pixels.
    const TRAVEL: f32 = 340.;
    /// Left edge of the strip at the start of the loop. The 1400px strip always covers
    /// the 400px band for every position in `[LEFT_MIN, LEFT_MIN + TRAVEL]`.
    const LEFT_MIN: f32 = -660.;
    let palette = [
        0xef4444, 0xf97316, 0xeab308, 0x22c55e, 0x06b6d4, 0x3b82f6, 0x8b5cf6,
    ];
    div()
        .absolute()
        .left(px(40.))
        .top(px(100.))
        .w(px(400.))
        .h(px(240.))
        .overflow_hidden()
        .rounded_xl()
        .child(
            div()
                .absolute()
                .top(px(0.))
                .h_full()
                .w(px(TILE_W * TILE_COUNT as f32))
                .flex()
                .flex_row()
                .children((0..TILE_COUNT).map(|i| {
                    div()
                        .w(px(TILE_W))
                        .h_full()
                        .flex_shrink_0()
                        .bg(rgb(palette[i % palette.len()]))
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_color(rgb(0x111111))
                        .text_3xl()
                        .child(format!("{}", i + 1))
                }))
                .with_animation(
                    "backdrop_ticker",
                    Animation::new(Duration::from_secs(6))
                        .repeat()
                        .with_easing(linear),
                    |strip, delta: f32| {
                        // Ping-pong 0 -> 1 -> 0 so the loop has no visible jump.
                        let t = 1.0 - (delta * 2.0 - 1.0).abs();
                        strip.left(px(LEFT_MIN + TRAVEL * t))
                    },
                ),
        )
}

/// A translucent rounded panel that frosts the content behind it.
fn frosted_panel() -> impl IntoElement {
    div()
        .absolute()
        .left(px(60.))
        .top(px(120.))
        .w(px(360.))
        .h(px(200.))
        .rounded_xl()
        .bg(rgba(0xffffff30))
        .backdrop_blur(px(24.))
        .border_1()
        .border_color(rgba(0xffffff60))
        .flex()
        .items_center()
        .justify_center()
        .text_color(rgb(0x111111))
        .text_2xl()
        .child("backdrop_blur(24px)")
}

/// A popover painted via `deferred()` so it sits above everything; its backdrop blur must
/// still pick up the panel and background beneath it.
fn deferred_popover() -> impl IntoElement {
    deferred(
        div()
            .absolute()
            .left(px(260.))
            .top(px(260.))
            .w(px(300.))
            .h(px(150.))
            .rounded_lg()
            .bg(rgba(0x1e293b66))
            .backdrop_blur(px(12.))
            .border_1()
            .border_color(rgba(0xffffffaa))
            .flex()
            .items_center()
            .justify_center()
            .text_color(rgb(0xffffff))
            .text_xl()
            .child("deferred + backdrop_blur"),
    )
}

/// A self-blurred element (CSS `filter: blur`) — its own content is blurred as a group.
fn content_blurred() -> impl IntoElement {
    div()
        .absolute()
        .left(px(120.))
        .top(px(420.))
        .w(px(280.))
        .h(px(120.))
        .blur(px(5.))
        .bg(rgb(0x0f172a))
        .rounded_lg()
        .flex()
        .items_center()
        .justify_center()
        .text_color(rgb(0xfacc15))
        .text_3xl()
        .child("blur(5px) content")
}

/// A content-blurred element with richer content (a row of colored chips), so the `filter: blur`
/// effect — the element and its children blurred as one group — is clearly visible.
fn content_blurred_rich() -> impl IntoElement {
    div()
        .absolute()
        .left(px(120.))
        .top(px(580.))
        .w(px(280.))
        .h(px(120.))
        .blur(px(6.))
        .bg(rgb(0x1e293b))
        .rounded_lg()
        .flex()
        .items_center()
        .justify_center()
        .gap_3()
        .children(
            [0xef4444, 0x22c55e, 0x3b82f6]
                .into_iter()
                .map(|hex| div().w(px(48.)).h(px(48.)).rounded_md().bg(rgb(hex))),
        )
}

/// Nested content blur: a `blur()` element inside another `blur()` element. The inner block is
/// blurred by its own filter and then again as part of the outer group, exercising the renderer's
/// per-nesting-level isolated group textures (up to `MAX_FILTER_DEPTH`). The inner content should
/// read as markedly softer than the outer block's own text.
fn nested_content_blurred() -> impl IntoElement {
    div()
        .absolute()
        .left(px(740.))
        .top(px(80.))
        .w(px(290.))
        .h(px(220.))
        .blur(px(3.))
        .bg(rgb(0x1e293b))
        .rounded_xl()
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .gap_4()
        .text_color(rgb(0xe2e8f0))
        .text_xl()
        .child("outer blur(3px)")
        .child(
            div()
                .w(px(190.))
                .h(px(100.))
                .blur(px(8.))
                .bg(rgb(0xf59e0b))
                .rounded_lg()
                .flex()
                .items_center()
                .justify_center()
                .text_color(rgb(0x111111))
                .text_2xl()
                .child("inner blur(8px)"),
        )
}

const SEAM_COLORS: [u32; 4] = [0xef4444, 0x22c55e, 0x3b82f6, 0xeab308];

/// One numbered, brightly-coloured block of the seam row. With `blur_each` it becomes its own
/// content-filter group; otherwise it is a plain block (relying on a blurred parent, if any).
/// Square corners on purpose: gpui content masks are axis-aligned rectangles, so a *rounded*
/// parent would not clip the blurred children to its radius and the busy background would leak
/// through the corner triangles — a separate concern from the seam blending under test here.
fn seam_block(i: usize, hex: u32, blur_each: bool) -> impl IntoElement {
    let block = div().flex_1().h_full();
    let block = if blur_each { block.blur(px(5.)) } else { block };
    block
        .bg(rgb(hex))
        .flex()
        .items_center()
        .justify_center()
        .text_color(rgb(0xffffff))
        .text_2xl()
        .child(format!("{}", i + 1))
}

/// Adjacent blocks, each its OWN content-filter group (`blur` on every block), no gap, dark parent.
/// This is CSS `filter: blur()` on each sibling: every block fades to transparent at its edges and
/// composites independently, so the dark parent shows through each seam by roughly `α_left · α_right`
/// (peaking at ~25% right on the seam). This matches the web — the clean alternative is the next panel.
fn adjacent_per_block_blur() -> impl IntoElement {
    div()
        .absolute()
        .left(px(740.))
        .top(px(350.))
        .w(px(290.))
        .h(px(110.))
        .bg(rgb(0x050505))
        .flex()
        .children(
            SEAM_COLORS
                .into_iter()
                .enumerate()
                .map(|(i, hex)| seam_block(i, hex, true)),
        )
}

/// The same adjacent blocks, but `blur` is on the PARENT — one content-filter group covering all
/// four. The blocks are opaque and touching, so the group's interior has no transparency: the blur
/// is continuous across the seams and only the group's outer edge fades. This is the CSS "blur the
/// wrapper, not each child" idiom, and the seams come out clean.
fn adjacent_group_blur() -> impl IntoElement {
    div()
        .absolute()
        .left(px(740.))
        .top(px(510.))
        .w(px(290.))
        .h(px(110.))
        .bg(rgb(0x050505))
        .blur(px(5.))
        .flex()
        .children(
            SEAM_COLORS
                .into_iter()
                .enumerate()
                .map(|(i, hex)| seam_block(i, hex, false)),
        )
}

/// A small dark pill label so the two new test sections are identifiable over the busy background.
fn caption(text: &'static str, left: f32, top: f32) -> impl IntoElement {
    div()
        .absolute()
        .left(px(left))
        .top(px(top))
        .px_2()
        .py_1()
        .rounded_md()
        .bg(rgba(0x000000cc))
        .text_color(rgb(0xffffff))
        .text_sm()
        .child(text)
}

impl Render for BlurExample {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .relative()
            .size_full()
            .bg(rgb(0x000000))
            .child(busy_background())
            .child(live_ticker_band())
            .child(frosted_panel())
            .child(content_blurred())
            .child(content_blurred_rich())
            .child(nested_content_blurred())
            .child(adjacent_per_block_blur())
            .child(adjacent_group_blur())
            .child(caption("nested content blur", 740., 50.))
            .child(caption("live: tiles slide behind the frost", 40., 72.))
            .child(caption(
                "adjacent — blur each block (seams, = CSS)",
                740.,
                322.,
            ))
            .child(caption(
                "adjacent — blur the parent (one group, clean)",
                740.,
                482.,
            ))
            .child(deferred_popover())
    }
}

fn main() {
    // Log adapter/backend selection (`RUST_LOG=gpui_wgpu=info`): the `wgpu` cargo
    // feature selects the wgpu renderer at compile time, and the adapter log below
    // is the runtime proof of which renderer path is actually drawing.
    let _ = env_logger::try_init();
    gpui_platform::application().run(|cx: &mut App| {
        cx.activate(true);
        cx.on_window_closed(|cx, _| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        let bounds = Bounds {
            origin: point(px(100.), px(100.)),
            size: size(px(1060.), px(760.)),
        };
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| BlurExample),
        )
        .expect("failed to open window");
    });
}
