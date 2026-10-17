//! Offscreen device and pixel readback shared by the pixel tests.
use anyhow::Result;
use gpui::block_on;
use std::{sync::mpsc, time::Duration};

/// The format every pixel test renders into and reads back.
pub(crate) const PIXEL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A device on any adapter, or `None` when the host has no adapter at all, so
/// pixel tests skip (with a note on stderr) instead of failing there.
pub(crate) fn test_device(label: &str) -> Result<Option<(wgpu::Device, wgpu::Queue)>> {
    block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Ok(adapter) => adapter,
            Err(error) => {
                eprintln!("skipping {label}: no wgpu adapter ({error})");
                return Ok(None);
            }
        };
        let device = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(label),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits())
                    .using_alignment(adapter.limits()),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;
        Ok(Some(device))
    })
}

/// Clears a `width`×`height` [`PIXEL_FORMAT`] texture to `clear`, records
/// `draw` into one pass on it after whatever `encoder` already holds, submits,
/// and returns the pixels as tightly packed RGBA rows.
pub(crate) fn render_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    (width, height): (u32, u32),
    clear: wgpu::Color,
    draw: impl FnOnce(&mut wgpu::RenderPass<'_>),
) -> Result<Vec<u8>> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test pixels"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: PIXEL_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let row_bytes = width * 4;
    let padded_row_bytes = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: u64::from(padded_row_bytes * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let view = texture.create_view(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test pixels"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        draw(&mut pass);
    }
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row_bytes),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::channel();
    output
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: Some(Duration::from_secs(10)),
    })?;
    receiver.recv_timeout(Duration::from_secs(10))??;
    let pixels = output
        .slice(..)
        .get_mapped_range()
        .chunks_exact(padded_row_bytes as usize)
        .flat_map(|row| &row[..row_bytes as usize])
        .copied()
        .collect();
    output.unmap();
    Ok(pixels)
}
