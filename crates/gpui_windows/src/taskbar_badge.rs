//! The count badge drawn over a window's taskbar button. Windows takes a
//! small icon, not text, so the count is rasterized here: GDI draws the
//! digits as coverage, inked white or dark against a disc in the caller's
//! colour.

use std::sync::OnceLock;

use anyhow::Result;
use windows::{
    Win32::{
        Foundation::{HWND, RECT},
        Graphics::Gdi::*,
        System::Com::{CLSCTX_INPROC_SERVER, CoCreateInstance},
        UI::{
            HiDpi::{GetDpiForWindow, GetSystemMetricsForDpi},
            Shell::{ITaskbarList3, TaskbarList},
            WindowsAndMessaging::*,
        },
    },
    core::{HSTRING, PCWSTR, w},
};

use gpui::{Hsla, Rgba, hsla_to_rgba};

/// Sent by Explorer once a window's taskbar button exists, and again after
/// Explorer restarts; overlays set before then are lost.
pub(crate) fn taskbar_button_created() -> u32 {
    static MESSAGE: OnceLock<u32> = OnceLock::new();
    *MESSAGE.get_or_init(|| unsafe { RegisterWindowMessageW(w!("TaskbarButtonCreated")) })
}

/// Shows `count` on a `color` disc over the window's taskbar button; zero clears it.
pub(crate) fn set_overlay(hwnd: HWND, count: usize, color: Hsla) -> Result<()> {
    let taskbar: ITaskbarList3 =
        unsafe { CoCreateInstance(&TaskbarList, None, CLSCTX_INPROC_SERVER)? };
    unsafe { taskbar.HrInit()? };
    if count == 0 {
        unsafe { taskbar.SetOverlayIcon(hwnd, HICON::default(), PCWSTR::null())? };
        return Ok(());
    }
    let dpi = unsafe { GetDpiForWindow(hwnd) }.max(96);
    let size = unsafe { GetSystemMetricsForDpi(SM_CXSMICON, dpi) }.max(16);
    let label = if count > 9 {
        "9+".to_owned()
    } else {
        count.to_string()
    };
    let icon = badge_icon(&label, size, hsla_to_rgba(color))?;
    let description = HSTRING::from(format!("{count} unread"));
    let result = unsafe { taskbar.SetOverlayIcon(hwnd, icon, &description) };
    // The taskbar keeps its own copy.
    unsafe { DestroyIcon(icon).ok() };
    result?;
    Ok(())
}

fn badge_icon(label: &str, size: i32, color: Rgba) -> Result<HICON> {
    let fill = [color.red, color.green, color.blue].map(|channel| (channel * 255.).round() as u8);
    // Digits take whichever ink contrasts more with the fill (WCAG ratio).
    let linear = |channel: f32| {
        if channel <= 0.04045 {
            channel / 12.92
        } else {
            ((channel + 0.055) / 1.055).powf(2.4)
        }
    };
    let luminance =
        0.2126 * linear(color.red) + 0.7152 * linear(color.green) + 0.0722 * linear(color.blue);
    let dark = linear(0x1A as f32 / 255.);
    let ink = if (luminance + 0.05) / (dark + 0.05) > 1.05 / (luminance + 0.05) {
        0x1A
    } else {
        0xFF
    } as f32;
    let header = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: size,
            // Top-down rows.
            biHeight: -size,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    };
    unsafe {
        let dc = CreateCompatibleDC(None);
        let mut bits = std::ptr::null_mut();
        let color = match CreateDIBSection(Some(dc), &header, DIB_RGB_COLORS, &mut bits, None, 0) {
            Ok(color) => color,
            Err(error) => {
                DeleteDC(dc).ok().ok();
                return Err(error.into());
            }
        };
        // Every pixel's green channel is read back as text coverage.
        std::ptr::write_bytes(bits.cast::<u8>(), 0, (size * size * 4) as usize);
        let previous_bitmap = SelectObject(dc, color.into());

        // White digits on black: any channel is the text's coverage.
        let font = CreateFontW(
            -(size * 11 / 16),
            0,
            0,
            0,
            FW_BOLD.0 as i32,
            0,
            0,
            0,
            DEFAULT_CHARSET,
            OUT_TT_PRECIS,
            CLIP_DEFAULT_PRECIS,
            ANTIALIASED_QUALITY,
            0,
            w!("Segoe UI"),
        );
        let previous_font = SelectObject(dc, font.into());
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, windows::Win32::Foundation::COLORREF(0x00FF_FFFF));
        let mut text: Vec<u16> = label.encode_utf16().collect();
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: size,
            bottom: size,
        };
        DrawTextW(
            dc,
            &mut text,
            &mut rect,
            DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX,
        );
        GdiFlush().ok().ok();

        let pixels = std::slice::from_raw_parts_mut(bits.cast::<[u8; 4]>(), (size * size) as usize);
        let radius = size as f32 / 2.;
        for (index, pixel) in pixels.iter_mut().enumerate() {
            let x = (index as i32 % size) as f32 + 0.5 - radius;
            let y = (index as i32 / size) as f32 + 0.5 - radius;
            let disc = (radius - (x * x + y * y).sqrt() + 0.5).clamp(0., 1.);
            let text = pixel[1] as f32 / 255.;
            let mix = |channel: u8| (channel as f32 + (ink - channel as f32) * text).round() as u8;
            // BGRA, straight alpha.
            *pixel = [
                mix(fill[2]),
                mix(fill[1]),
                mix(fill[0]),
                (disc * 255.).round() as u8,
            ];
        }

        SelectObject(dc, previous_font);
        SelectObject(dc, previous_bitmap);
        DeleteObject(font.into()).ok().ok();
        DeleteDC(dc).ok().ok();

        // An all-zero mask: the colour bitmap's alpha decides everything.
        let stride = ((size as usize + 15) / 16) * 2;
        let mask_bits = vec![0u8; stride * size as usize];
        let mask = CreateBitmap(size, size, 1, 1, Some(mask_bits.as_ptr().cast()));
        let info = ICONINFO {
            fIcon: true.into(),
            xHotspot: 0,
            yHotspot: 0,
            hbmMask: mask,
            hbmColor: color,
        };
        let icon = CreateIconIndirect(&info);
        DeleteObject(mask.into()).ok().ok();
        DeleteObject(color.into()).ok().ok();
        Ok(icon?)
    }
}
