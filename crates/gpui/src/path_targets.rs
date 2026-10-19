//! Shared bounds and retained capacity for antialiased path render targets.

use crate::{Path, PrimitiveBatch, ScaledPixels, Scene};

/// Required target capacity for the scene, rounded to stable 64-pixel buckets.
pub fn path_target_size(scene: &Scene, viewport: [u32; 2]) -> Option<[u32; 2]> {
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

/// Paths of one batch that rasterize together: their [`path_batch_bounds`] rectangles overlap,
/// directly or through other members. Distinct clusters of a batch have disjoint `bounds`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathCluster {
    /// `[x, y, width, height]` in device pixels, as [`path_batch_bounds`] gives it.
    pub bounds: [u32; 4],
    /// Indices into the batch, ascending.
    pub paths: Vec<usize>,
}

/// Batches with more paths than this rasterize as one cluster, keeping the grouping cost bounded.
const MAX_CLUSTERED_PATHS: usize = 256;

/// Splits a batch into [`PathCluster`]s so distant paths no longer share one target spanning
/// the space between them: a tab outline at the top of a window and a meter arc at its bottom
/// used to need a window-sized multisampled target.
pub fn path_clusters(paths: &[Path<ScaledPixels>], viewport: [u32; 2]) -> Vec<PathCluster> {
    if paths.len() > MAX_CLUSTERED_PATHS {
        return vec![PathCluster {
            bounds: path_batch_bounds(paths, viewport),
            paths: (0..paths.len()).collect(),
        }];
    }
    let mut clusters: Vec<PathCluster> = Vec::new();
    for (ix, path) in paths.iter().enumerate() {
        let mut cluster = PathCluster {
            bounds: path_batch_bounds(std::slice::from_ref(path), viewport),
            paths: vec![ix],
        };
        // Absorbing a cluster grows the bounds, which can reach further clusters.
        while let Some(found) = clusters
            .iter()
            .position(|other| rects_overlap(other.bounds, cluster.bounds))
        {
            let other = clusters.swap_remove(found);
            cluster.bounds = union_rect(cluster.bounds, other.bounds);
            cluster.paths.extend(other.paths);
        }
        clusters.push(cluster);
    }
    for cluster in &mut clusters {
        cluster.paths.sort_unstable();
    }
    clusters.sort_unstable_by_key(|cluster| cluster.paths[0]);
    clusters
}

/// Required target capacity when every batch rasterizes cluster by cluster, rounded to stable
/// 64-pixel buckets.
pub fn clustered_path_target_size(scene: &Scene, viewport: [u32; 2]) -> Option<[u32; 2]> {
    if scene.paths.is_empty() {
        return None;
    }
    let mut size = [1u32, 1];
    for batch in scene.batches() {
        if let PrimitiveBatch::Paths(range) = batch {
            for cluster in path_clusters(&scene.paths[range], viewport) {
                size[0] = size[0].max(cluster.bounds[2]);
                size[1] = size[1].max(cluster.bounds[3]);
            }
        }
    }
    Some([
        size[0].next_multiple_of(64).min(viewport[0]).max(1),
        size[1].next_multiple_of(64).min(viewport[1]).max(1),
    ])
}

fn rects_overlap(a: [u32; 4], b: [u32; 4]) -> bool {
    a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]
}

fn union_rect(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    let x = a[0].min(b[0]);
    let y = a[1].min(b[1]);
    let right = (a[0] + a[2]).max(b[0] + b[2]);
    let bottom = (a[1] + a[3]).max(b[1] + b[3]);
    [x, y, right - x, bottom - y]
}

/// Grow existing targets componentwise, releasing them when the scene has no paths.
pub fn retained_path_target_size(
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
pub fn path_batch_bounds(paths: &[Path<ScaledPixels>], viewport: [u32; 2]) -> [u32; 4] {
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
    use crate::{Bounds, ContentMask, point, px, size};

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
    fn distant_paths_rasterize_apart_and_overlapping_ones_together() {
        let viewport = [1600, 1125];
        let paths = [
            path(20.0, 20.0, 40.0, 30.0),
            path(1500.0, 1000.0, 60.0, 80.0),
            path(50.0, 40.0, 30.0, 30.0),
        ];
        let clusters = path_clusters(&paths, viewport);
        assert_eq!(
            clusters,
            [
                PathCluster {
                    bounds: [19, 19, 62, 52],
                    paths: vec![0, 2],
                },
                PathCluster {
                    bounds: [1499, 999, 62, 82],
                    paths: vec![1],
                },
            ]
        );
        let mut scene = Scene::default();
        for path in paths {
            scene.insert_primitive(path);
        }
        scene.finish();
        // The batch spans the window, its clusters do not.
        assert_eq!(path_target_size(&scene, viewport), Some([1600, 1088]));
        assert_eq!(
            clustered_path_target_size(&scene, viewport),
            Some([64, 128])
        );
    }

    #[test]
    fn a_merged_cluster_absorbs_clusters_its_union_now_reaches() {
        // The last path bridges the first two, and their union then covers the third, which
        // touches none of them.
        let paths = [
            path(0.0, 0.0, 10.0, 10.0),
            path(50.0, 50.0, 10.0, 10.0),
            path(50.0, 0.0, 5.0, 3.0),
            path(5.0, 5.0, 50.0, 50.0),
        ];
        let clusters = path_clusters(&paths, [256, 256]);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].paths, [0, 1, 2, 3]);
        assert_eq!(clusters[0].bounds, path_batch_bounds(&paths, [256, 256]));
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
