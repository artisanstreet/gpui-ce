use gpui::{Path, PrimitiveBatch, ScaledPixels, Scene};

pub(super) fn path_target_size(scene: &Scene, viewport: [u32; 2]) -> Option<[u32; 2]> {
    if scene.paths.is_empty() {
        return None;
    }
    let mut size = [1u32, 1];
    for batch in scene.batches() {
        if let PrimitiveBatch::Paths(range) = batch {
            let [_, _, width, height] = path_batch_bounds(&scene.paths[range], viewport);
            size[0] = size[0].max(width);
            size[1] = size[1].max(height);
        }
    }
    // Ignore small geometry movements within a capacity bucket.
    Some([
        size[0].next_multiple_of(64).min(viewport[0]).max(1),
        size[1].next_multiple_of(64).min(viewport[1]).max(1),
    ])
}

pub(super) fn retained_path_target_size(
    current: Option<[u32; 2]>,
    required: Option<[u32; 2]>,
) -> Option<[u32; 2]> {
    let required = required?;
    let current = current.unwrap_or([0; 2]);
    // Retain capacity throughout continuous path use. Immediate shrinking can
    // reallocate every frame when an animation alternates tall and wide bounds.
    // A scene without paths releases both targets and resets this high-water mark.
    Some([current[0].max(required[0]), current[1].max(required[1])])
}

/// Integer-aligned extent of a path batch, with a one-pixel sampling guard.
/// All paths retain their screen-space positions relative to this origin.
pub(super) fn path_batch_bounds(paths: &[Path<ScaledPixels>], viewport: [u32; 2]) -> [u32; 4] {
    let mut left = viewport[0] as f32;
    let mut top = viewport[1] as f32;
    let mut right = 0.0f32;
    let mut bottom = 0.0f32;
    for path in paths {
        let bounds = path.clipped_bounds();
        left = left.min(bounds.origin.x.0);
        top = top.min(bounds.origin.y.0);
        right = right.max(bounds.origin.x.0 + bounds.size.width.0);
        bottom = bottom.max(bounds.origin.y.0 + bounds.size.height.0);
    }
    let x = (left.floor() - 1.0)
        .max(0.0)
        .min(viewport[0].saturating_sub(1) as f32) as u32;
    let y = (top.floor() - 1.0)
        .max(0.0)
        .min(viewport[1].saturating_sub(1) as f32) as u32;
    let right = ((right.ceil() + 1.0).max(0.0) as u32).min(viewport[0]);
    let bottom = ((bottom.ceil() + 1.0).max(0.0) as u32).min(viewport[1]);
    [
        x,
        y,
        right.saturating_sub(x).max(1),
        bottom.saturating_sub(y).max(1),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{Bounds, ContentMask, point, px, size};

    fn path(x: f32, y: f32, width: f32, height: f32) -> Path<ScaledPixels> {
        let mut path = Path::new(point(px(x), px(y))).scale(1.0);
        path.bounds = Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        );
        path.content_mask = ContentMask {
            bounds: Bounds::new(
                point(ScaledPixels(-1000.0), ScaledPixels(-1000.0)),
                size(ScaledPixels(4000.0), ScaledPixels(4000.0)),
            ),
        };
        path
    }

    #[test]
    fn fractional_path_bounds_round_outward_and_keep_sampling_guard() {
        assert_eq!(
            path_batch_bounds(&[path(37.4, 18.7, 42.1, 30.2)], [256, 192]),
            [36, 17, 45, 33]
        );
    }

    #[test]
    fn batch_bounds_include_distant_paths_and_clip_to_viewport_and_mask() {
        let mut clipped = path(250.0, 140.0, 20.0, 30.0);
        clipped.content_mask.bounds.size.height = ScaledPixels(1150.0);
        assert_eq!(
            path_batch_bounds(&[path(20.0, 20.0, 10.0, 10.0), clipped], [256, 192]),
            [19, 19, 237, 132]
        );
        assert_eq!(
            path_batch_bounds(&[path(-12.8, -9.1, 35.0, 30.0)], [256, 192]),
            [0, 0, 24, 22]
        );
        assert_eq!(
            path_batch_bounds(&[path(900.0, 900.0, 10.0, 10.0)], [256, 192]),
            [255, 191, 1, 1]
        );
    }

    #[test]
    fn target_sizes_follow_batches_and_empty_scenes_need_no_target() {
        let mut scene = Scene::default();
        assert_eq!(path_target_size(&scene, [1600, 1125]), None);
        scene.insert_primitive(path(37.4, 18.7, 42.1, 30.2));
        scene.finish();
        assert_eq!(path_target_size(&scene, [1600, 1125]), Some([64, 64]));
        scene.insert_primitive(path(1500.0, 1000.0, 99.0, 124.0));
        scene.finish();
        // Nonoverlapping paths share one batch, so their relative positions
        // must remain intact even when most of its bounding rectangle is empty.
        assert_eq!(path_target_size(&scene, [1600, 1125]), Some([1600, 1125]));
    }

    #[test]
    fn alternating_tall_and_wide_paths_grow_once_then_reuse_the_target() {
        let mut current = Some([1024, 64]);
        let required_sizes = [[64, 1024], [1024, 64], [64, 1024], [1024, 64], [64, 64]];
        let mut allocations = 0;
        for required in required_sizes {
            let retained = retained_path_target_size(current, Some(required));
            allocations += usize::from(retained != current);
            current = retained;
            assert_eq!(current, Some([1024, 1024]));
        }
        assert_eq!(allocations, 1);
    }

    #[test]
    fn no_path_frame_releases_capacity_and_next_path_starts_small() {
        let current = retained_path_target_size(Some([1600, 1125]), None);
        assert_eq!(current, None);
        assert_eq!(retained_path_target_size(current, None), None);
        assert_eq!(
            retained_path_target_size(current, Some([64, 64])),
            Some([64, 64])
        );
        assert_eq!(
            retained_path_target_size(Some([128, 64]), Some([192, 64])),
            Some([192, 64])
        );
    }
}
