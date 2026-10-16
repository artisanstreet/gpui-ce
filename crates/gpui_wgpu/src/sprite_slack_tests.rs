//! Live layers keep slack sprites in the scene's sprite vectors. Slack names
//! no atlas texture, so a slot left by a released image can't make the
//! renderer bind a texture the atlas has freed, and it draws nothing.
use super::*;
use gpui::{
    AtlasKey, AtlasTextureKind, AtlasTile, ContentMask, Corners, ImageId, PlatformAtlas as _,
    PolychromeSprite, RenderImageParams, Scene, block_on, point, size,
};
use std::{borrow::Cow, sync::mpsc, time::Duration};
use wgpu::util::DeviceExt as _;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const VIEWPORT: [u32; 2] = [32, 32];

#[test]
fn a_released_images_slack_sprite_is_batched_and_draws_nothing() -> Result<()> {
    let (device, queue) = test_device()?;
    let (device, queue) = (Arc::new(device), Arc::new(queue));
    let atlas = WgpuAtlas::new(device.clone(), queue.clone(), FORMAT);
    let image = |id| {
        AtlasKey::Image(RenderImageParams {
            image_id: ImageId(id),
            frame_index: 0,
        })
    };
    let solid = |rgba: [u8; 4]| {
        let size = Size {
            width: DevicePixels(8),
            height: DevicePixels(8),
        };
        Ok(Some((size, Cow::Owned(rgba.repeat(64)))))
    };

    // A live layer showed image 1; when it stopped, its slot became slack.
    let shown = atlas
        .get_or_insert_with(&image(1), &mut || solid([255, 0, 0, 255]))?
        .expect("tile");
    let live = |tile, left: f32| {
        let bounds = Bounds::new(
            point(ScaledPixels(left), ScaledPixels(4.)),
            size(ScaledPixels(16.), ScaledPixels(16.)),
        );
        PolychromeSprite {
            order: 1,
            pad: 0,
            grayscale: false.into(),
            opacity: 1.,
            bounds,
            content_mask: ContentMask { bounds },
            corner_radii: Corners::default(),
            tile,
        }
    };
    let slack = PolychromeSprite {
        bounds: Bounds::default(),
        content_mask: ContentMask::default(),
        tile: AtlasTile::slack(AtlasTextureKind::Polychrome),
        ..live(shown, 4.)
    };
    // The image is released, freeing its atlas page, and another one takes
    // its place in the atlas.
    atlas.remove(&image(1));
    let replacement = atlas
        .get_or_insert_with(&image(2), &mut || solid([0, 0, 255, 255]))?
        .expect("tile");
    atlas.before_frame();

    // Slack alone starts no batch, so no texture is looked up for it.
    let mut only_slack = Scene::default();
    only_slack.polychrome_sprites = vec![slack, slack];
    assert_eq!(only_slack.batches().count(), 0);

    // Slack after a live sprite draws in its batch, which names the live
    // sprite's texture, and paints exactly what the live sprite alone does.
    let mut with_slack = Scene::default();
    with_slack.polychrome_sprites = vec![live(replacement, 8.), slack, slack];
    let mut without = Scene::default();
    without.polychrome_sprites = vec![live(replacement, 8.)];
    let batches = with_slack.batches().collect::<Vec<_>>();
    assert_eq!(batches.len(), 1);
    let PrimitiveBatch::PolychromeSprites { texture_id, range } = &batches[0] else {
        panic!("a sprite batch");
    };
    assert_eq!(*texture_id, replacement.texture_id);
    assert_eq!(range.clone(), 0..3, "the slack draws inside the batch");

    let drawn = render(&device, &queue, &atlas, &with_slack)?;
    let alone = render(&device, &queue, &atlas, &without)?;
    assert_eq!(
        alone
            .chunks_exact(4)
            .filter(|pixel| pixel[3] == 255)
            .count(),
        16 * 16,
        "the image paints its bounds"
    );
    assert_eq!(drawn, alone);
    Ok(())
}

/// Draws the scene's polychrome sprite batches as the renderer does: the
/// batch's atlas texture, the batch's instances, the shipping pipeline.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: &WgpuAtlas,
    scene: &Scene,
) -> Result<Vec<u8>> {
    let layouts = WgpuRenderer::create_bind_group_layouts(device, false);
    let pipelines = WgpuPipelines::new(
        device,
        &layouts,
        FORMAT,
        wgpu::CompositeAlphaMode::Opaque,
        1,
        false,
        false,
    );
    let globals = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sprite globals"),
        contents: bytemuck::bytes_of(&GlobalParams {
            viewport_size: VIEWPORT.map(|v| v as f32),
            premultiplied_alpha: 0,
            pulse_clock_ms: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let gamma = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sprite gamma"),
        contents: bytemuck::bytes_of(&GammaParams::zeroed()),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let globals = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sprite globals"),
        layout: &layouts.globals,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gamma.as_entire_binding(),
            },
        ],
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sprite target"),
        size: wgpu::Extent3d {
            width: VIEWPORT[0],
            height: VIEWPORT[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = output.create_view(&Default::default());
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sprites"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        for batch in scene.batches() {
            let PrimitiveBatch::PolychromeSprites { texture_id, range } = batch else {
                continue;
            };
            let texture = atlas.get_texture_info(texture_id);
            let sprites = &scene.polychrome_sprites[range];
            let instances = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sprite instances"),
                contents: unsafe { WgpuRenderer::instance_bytes(sprites) },
                usage: wgpu::BufferUsages::STORAGE,
            });
            let instances = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sprite instances"),
                layout: &layouts.instances,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instances.as_entire_binding(),
                }],
            });
            let texture = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sprite texture"),
                layout: &layouts.texture,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            pass.set_pipeline(&pipelines.poly_sprites);
            pass.set_bind_group(0, &globals, &[]);
            pass.set_bind_group(1, &instances, &[]);
            pass.set_bind_group(2, &texture, &[]);
            pass.draw(0..4, 0..sprites.len() as u32);
        }
    }
    let row_bytes = VIEWPORT[0] * 4;
    let padded_row_bytes =
        row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sprite pixels"),
        size: u64::from(padded_row_bytes) * u64::from(VIEWPORT[1]),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        output.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(VIEWPORT[1]),
            },
        },
        output.size(),
    );
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: Some(Duration::from_secs(10)),
    })?;
    receiver.recv_timeout(Duration::from_secs(10))??;
    let pixels = readback
        .slice(..)
        .get_mapped_range()
        .chunks_exact(padded_row_bytes as usize)
        .flat_map(|row| row[..row_bytes as usize].iter().copied())
        .collect();
    readback.unmap();
    Ok(pixels)
}

fn test_device() -> Result<(wgpu::Device, wgpu::Queue)> {
    block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: Default::default(),
            memory_budget_thresholds: Default::default(),
            display: None,
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sprite slack regression"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits())
                    .using_alignment(adapter.limits()),
                memory_hints: crate::WgpuContext::memory_hints(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(anyhow::Error::from)
    })
}
