//! Compare the shipping path raster/composite shaders with full-window and
//! compact intermediates. Integer origins must preserve screen-space effects.
use super::*;
use gpui::{
    ContentMask, block_on, hsla, linear_color_stop, linear_gradient, pattern_slash, point, px, size,
};
use std::{sync::mpsc, time::Duration};
use wgpu::util::DeviceExt as _;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[test]
fn compact_path_pixels_match_full_target_with_storage_instances() -> Result<()> {
    compare_targets(false, [128, 128])
}

#[test]
fn compact_path_pixels_match_full_target_with_texture_instances() -> Result<()> {
    compare_targets(true, [128, 128])
}

#[test]
fn compact_paths_preserve_pixels_in_odd_rectangular_viewport_with_storage() -> Result<()> {
    compare_targets(false, [173, 117])
}

#[test]
fn compact_paths_preserve_pixels_in_odd_rectangular_viewport_with_textures() -> Result<()> {
    compare_targets(true, [173, 117])
}

fn compare_targets(webgl: bool, viewport: [u32; 2]) -> Result<()> {
    let (device, queue) = test_device()?;
    let layouts = WgpuRenderer::create_bind_group_layouts(&device, webgl);
    let mut pipelines = WgpuPipelines::new(
        &device,
        &layouts,
        FORMAT,
        wgpu::CompositeAlphaMode::PreMultiplied,
        4,
        false,
        webgl,
    );
    pipelines.ensure_paths(&device, &layouts);
    let paths = pipelines.path_pipelines.as_ref().unwrap();
    let raster = &paths.rasterization;
    let composite = &paths.composite;
    let dark_gradient = linear_gradient(
        173.0,
        linear_color_stop(hsla(0.72, 0.1, 0.13, 0.9), 0.0),
        linear_color_stop(hsla(0.72, 0.1, 0.095, 0.7), 1.0),
    );
    let patterned = pattern_slash(hsla(0.4, 0.7, 0.6, 0.8), 2.3, 7.1);
    for (name, paths) in [
        ("gradient and dither", vec![curved_path(dark_gradient, 0.0)]),
        ("slash pattern", vec![curved_path(patterned, 0.0)]),
        (
            "overlapping translucent paths",
            vec![curved_path(dark_gradient, 0.0), curved_path(patterned, 5.0)],
        ),
    ] {
        let [x, y, width, height] = path_batch_bounds(&paths, viewport);
        assert!(x > 0 && y > 0 && (x % 4 != 0 || y % 4 != 0));
        let compact_size = [width.next_power_of_two(), height.next_power_of_two()];
        assert!(compact_size[0] * compact_size[1] < viewport[0] * viewport[1]);
        let full = render_paths(
            &device,
            &queue,
            raster,
            composite,
            &paths,
            viewport,
            [0, 0],
            viewport,
            webgl,
        )?;
        let compact = render_paths(
            &device,
            &queue,
            raster,
            composite,
            &paths,
            viewport,
            [x, y],
            compact_size,
            webgl,
        )?;
        assert!(
            full.chunks_exact(4).filter(|p| p[3] > 0).count() > 200,
            "empty {name} fixture"
        );
        assert!(
            full.chunks_exact(4).any(|p| p[3] > 0 && p[3] < 150),
            "missing antialiased/transparent coverage in {name}"
        );
        assert_eq!(&compact[..4], &[0, 0, 0, 0], "{name} escaped its clip");
        let mut differing_bytes = 0;
        for (index, (&expected, &actual)) in full.iter().zip(&compact).enumerate() {
            assert!(
                expected.abs_diff(actual) <= 1,
                "{name}, texture transport={webgl}: byte {index}: full={expected}, compact={actual}"
            );
            differing_bytes += usize::from(expected != actual);
        }
        // A one-code tolerance alone would miss a displaced Bayer pattern.
        // Permit only isolated float/quantization differences, not phase shifts.
        assert!(
            differing_bytes <= 64,
            "{name}, texture transport={webgl}: {differing_bytes} bytes changed"
        );
    }
    Ok(())
}

fn curved_path(color: Background, offset: f32) -> Path<ScaledPixels> {
    let start = point(px(21.137 + offset), px(17.281));
    let mut path = Path::new(start);
    path.line_to(point(px(71.719 + offset), px(20.413)));
    path.curve_to(
        point(px(55.317 + offset), px(72.193)),
        point(px(82.719 + offset), px(67.431)),
    );
    path.line_to(point(px(24.619 + offset), px(65.137)));
    path.line_to(start);
    let mut path = path.scale(1.0);
    path.order = u32::from(offset != 0.0);
    path.color = color;
    path.content_mask = ContentMask {
        bounds: Bounds::new(
            point(ScaledPixels(27.13), ScaledPixels(23.17)),
            size(ScaledPixels(49.25), ScaledPixels(43.71)),
        ),
    };
    path
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
        anyhow::ensure!(
            adapter
                .get_texture_format_features(FORMAT)
                .flags
                .sample_count_supported(4),
            "path regression requires four-sample MSAA"
        );
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("path pixel regression"),
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

fn globals(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    dimensions: [u32; 2],
) -> wgpu::BindGroup {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("path globals"),
        contents: bytemuck::bytes_of(&GlobalParams {
            viewport_size: dimensions.map(|v| v as f32),
            premultiplied_alpha: 0,
            pad: 0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let gamma = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("unused path gamma"),
        contents: bytemuck::bytes_of(&GammaParams::zeroed()),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("path globals"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gamma.as_entire_binding(),
            },
        ],
    })
}

fn instances<T>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    values: &[T],
    webgl: bool,
) -> wgpu::BindGroup {
    // Same fully initialized repr(C) records as the shipping instance upload.
    let bytes = unsafe { WgpuRenderer::instance_bytes(values) };
    if !webgl {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("path instances"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        return device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path instances"),
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
    }
    // Odd width deliberately exercises record reads across texture rows.
    let width = 13;
    let height = (bytes.len() as u32).div_ceil(width * 16);
    let mut padded = bytes.to_vec();
    padded.resize((width * height * 16) as usize, 0);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("path texture instances"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        &padded,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: None,
        },
        texture.size(),
    );
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("path texture instances"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&texture.create_view(&Default::default())),
        }],
    })
}

fn texture(
    device: &wgpu::Device,
    dimensions: [u32; 2],
    samples: u32,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("path regression target"),
        size: wgpu::Extent3d {
            width: dimensions[0],
            height: dimensions[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage,
        view_formats: &[],
    })
}

#[allow(clippy::too_many_arguments)]
fn render_paths(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    raster: &wgpu::RenderPipeline,
    composite: &wgpu::RenderPipeline,
    paths: &[Path<ScaledPixels>],
    viewport: [u32; 2],
    origin: [u32; 2],
    dimensions: [u32; 2],
    webgl: bool,
) -> Result<Vec<u8>> {
    let texture_origin = point(
        ScaledPixels(origin[0] as f32),
        ScaledPixels(origin[1] as f32),
    );
    let vertices = paths
        .iter()
        .flat_map(|path| {
            path.vertices.iter().map(|vertex| PathRasterizationVertex {
                xy_position: vertex.xy_position,
                st_position: vertex.st_position,
                color: path.color,
                bounds: path.clipped_bounds(),
                texture_origin,
            })
        })
        .collect::<Vec<_>>();
    let bounds = paths
        .iter()
        .skip(1)
        .fold(paths[0].clipped_bounds(), |bounds, path| {
            bounds.union(&path.clipped_bounds())
        });
    // Composite the union once, as production does for an overlapping batch.
    let sprites = [PathSprite {
        bounds,
        texture_origin,
    }];
    let raster_globals = globals(device, raster, dimensions);
    let composite_globals = globals(device, composite, viewport);
    let vertices_group = instances(device, queue, raster, &vertices, webgl);
    let sprites_group = instances(device, queue, composite, &sprites, webgl);
    let intermediate = texture(
        device,
        dimensions,
        1,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let msaa = texture(
        device,
        dimensions,
        4,
        wgpu::TextureUsages::RENDER_ATTACHMENT,
    );
    let output = texture(
        device,
        viewport,
        1,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let intermediate_view = intermediate.create_view(&Default::default());
    let msaa_view = msaa.create_view(&Default::default());
    let output_view = output.create_view(&Default::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let texture_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("path resolved texture"),
        layout: &composite.get_bind_group_layout(2),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&intermediate_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for is_raster in [true, false] {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("path regression"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: if is_raster { &msaa_view } else { &output_view },
                resolve_target: is_raster.then_some(&intermediate_view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        if is_raster {
            pass.set_pipeline(raster);
            pass.set_bind_group(0, &raster_globals, &[]);
            pass.set_bind_group(1, &vertices_group, &[]);
            pass.draw(0..vertices.len() as u32, 0..1);
        } else {
            pass.set_pipeline(composite);
            pass.set_bind_group(0, &composite_globals, &[]);
            pass.set_bind_group(1, &sprites_group, &[]);
            pass.set_bind_group(2, &texture_group, &[]);
            pass.draw(0..4, 0..1);
        }
    }
    let row_bytes = viewport[0] * 4;
    let padded_row_bytes =
        row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("path pixels"),
        size: u64::from(padded_row_bytes) * u64::from(viewport[1]),
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
                rows_per_image: Some(viewport[1]),
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

/// Content-filter targets can disappear while persistent backdrop blur keeps the rest
/// of the offscreen pool alive. Encoded work must survive dropping every pool handle.
#[test]
fn content_group_targets_release_without_invalidating_queued_commands() -> Result<()> {
    let (device, queue) = test_device()?;
    let mut textures = Vec::new();
    let mut views = Vec::new();
    let allocate = || WgpuRenderer::create_path_intermediate(&device, FORMAT, 8, 8);
    for _ in 0..2 {
        let (texture, view) = allocate();
        textures.push(texture);
        views.push(view);
    }
    release_unused_group_targets(1, &mut textures, &mut views);
    assert_eq!(
        textures.len(),
        2,
        "active filters retain nesting high-water capacity"
    );
    assert_eq!(views.len(), 2);

    let row_bytes = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let target_bytes = u64::from(row_bytes) * 8;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("released content-group readback"),
        size: target_bytes * 2,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    for (index, color) in [wgpu::Color::RED, wgpu::Color::GREEN]
        .into_iter()
        .enumerate()
    {
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("old content group"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &views[index],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }
        encoder.copy_texture_to_buffer(
            textures[index].as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: index as u64 * target_bytes,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(8),
                },
            },
            textures[index].size(),
        );
    }
    let old_commands = encoder.finish();
    release_unused_group_targets(0, &mut textures, &mut views);
    assert!(textures.is_empty());
    assert!(views.is_empty());
    // Recreate the same target sizes before submitting work on the previous pool.
    let mut replacement_encoder = device.create_command_encoder(&Default::default());
    for _ in 0..2 {
        let (texture, view) = allocate();
        {
            let _pass = replacement_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("replacement content group"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }
        textures.push(texture);
        views.push(view);
    }
    let submission = queue.submit([replacement_encoder.finish(), old_commands]);
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
    {
        let pixels = readback.slice(..).get_mapped_range();
        for (index, expected) in [[255, 0, 0, 255], [0, 255, 0, 255]].iter().enumerate() {
            let begin = index * target_bytes as usize;
            for row in pixels[begin..begin + target_bytes as usize].chunks_exact(row_bytes as usize)
            {
                assert!(row[..8 * 4].chunks_exact(4).all(|pixel| pixel == expected));
            }
        }
    }
    readback.unmap();
    Ok(())
}
