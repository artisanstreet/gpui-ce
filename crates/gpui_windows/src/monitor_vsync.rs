//! One demand-driven display clock per window. Idle clocks sleep until
//! invalidation or animation requests a frame.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use windows::Win32::{
    Foundation::HWND,
    Graphics::{
        Dxgi::{CreateDXGIFactory1, IDXGIFactory1, IDXGIOutput},
        Gdi::{
            HMONITOR, MONITOR_DEFAULTTONEAREST, MonitorFromWindow, RDW_INVALIDATE, RedrawWindow,
        },
    },
    UI::WindowsAndMessaging::{IsIconic, IsWindow, IsWindowVisible},
};

use crate::{SafeHwnd, frame_signal::FrameSignal};

pub(crate) struct FrameClock {
    signal: Arc<FrameSignal>,
}

impl FrameClock {
    pub(crate) fn request(&self) {
        self.signal.request();
    }
}

impl Drop for FrameClock {
    fn drop(&mut self) {
        self.signal.close();
    }
}

pub(crate) fn start(hwnd: SafeHwnd, ready: Arc<AtomicBool>) -> std::io::Result<FrameClock> {
    let signal = Arc::new(FrameSignal::default());
    let worker_signal = signal.clone();
    std::thread::Builder::new()
        .name("WindowVSync".into())
        .spawn(move || {
            let mut clock = MonitorClock::default();
            while worker_signal.wait() {
                let hwnd = hwnd.as_raw();
                if !unsafe { IsWindow(Some(hwnd)) }.as_bool() {
                    break;
                }
                if unsafe { !IsIconic(hwnd).as_bool() && IsWindowVisible(hwnd).as_bool() } {
                    clock.wait(hwnd);
                }
                if !worker_signal.deliver(|| {
                    unsafe {
                        // Check visibility while holding the signal lock. A
                        // concurrent restore request must either join this
                        // redraw or wake the worker after this request clears.
                        if !IsIconic(hwnd).as_bool() && IsWindowVisible(hwnd).as_bool() {
                            ready.store(true, Ordering::Release);
                            let _ = RedrawWindow(Some(hwnd), None, None, RDW_INVALIDATE);
                        }
                    }
                }) {
                    break;
                }
            }
        })
        .map(|_| FrameClock { signal })
}

#[derive(Default)]
struct MonitorClock {
    monitor: HMONITOR,
    factory: Option<IDXGIFactory1>,
    output: Option<IDXGIOutput>,
    retry_after: Option<Instant>,
    early_waits: u8,
}

impl MonitorClock {
    fn wait(&mut self, hwnd: HWND) {
        let monitor = unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) };
        let now = Instant::now();
        let topology_changed = self
            .factory
            .as_ref()
            .is_some_and(|factory| !unsafe { factory.IsCurrent() }.as_bool());
        if monitor != self.monitor
            || topology_changed
            || self.retry_after.is_none_or(|retry| now >= retry) && self.output.is_none()
        {
            self.monitor = monitor;
            self.factory = unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }.ok();
            self.output = self
                .factory
                .as_ref()
                .and_then(|factory| output_for_monitor(factory, monitor));
            self.retry_after = Some(now + Duration::from_secs(1));
        }
        let started = Instant::now();
        if self
            .output
            .as_ref()
            .is_some_and(|output| unsafe { output.WaitForVBlank() }.is_ok())
        {
            // Occlusion/disconnection can turn a driver wait into an immediate
            // return. Drop that output and retry later instead of busy-spinning.
            if started.elapsed() >= Duration::from_millis(1) {
                self.early_waits = 0;
                return;
            }
            self.early_waits = self.early_waits.saturating_add(1);
            if self.early_waits < 3 {
                return;
            }
        }
        self.output = None;
        self.retry_after = Some(now + Duration::from_secs(1));
        // Remote sessions and outputs without a usable DXGI vblank retain a
        // bounded fallback. Local monitors use their actual hardware clock.
        std::thread::sleep(Duration::from_micros(16_667));
    }
}

fn output_for_monitor(factory: &IDXGIFactory1, monitor: HMONITOR) -> Option<IDXGIOutput> {
    let mut adapter_index = 0;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(adapter_index) } {
        let mut output_index = 0;
        while let Ok(output) = unsafe { adapter.EnumOutputs(output_index) } {
            if unsafe { output.GetDesc() }
                .ok()
                .is_some_and(|desc| desc.Monitor == monitor && desc.AttachedToDesktop.as_bool())
            {
                return Some(output);
            }
            output_index += 1;
        }
        adapter_index += 1;
    }
    None
}
