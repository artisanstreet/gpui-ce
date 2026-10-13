use crate::{DevicePixels, Pixels, Result, SharedString, Size, size};
use smallvec::SmallVec;

use image::{Delay, Frame};
use parking_lot::Mutex;
use std::{
    borrow::Cow,
    fmt,
    hash::Hash,
    sync::{
        OnceLock,
        atomic::{AtomicUsize, Ordering::SeqCst},
    },
};

/// A source of assets for this app to use.
pub trait AssetSource: 'static + Send + Sync {
    /// Load the given asset from the source path.
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>>;

    /// List the assets at the given path.
    fn list(&self, path: &str) -> Result<Vec<SharedString>>;
}

impl AssetSource for () {
    fn load(&self, _path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        Ok(None)
    }

    fn list(&self, _path: &str) -> Result<Vec<SharedString>> {
        Ok(vec![])
    }
}

/// A unique identifier for the image cache
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub usize);

#[derive(PartialEq, Eq, Hash, Clone)]
#[expect(missing_docs)]
pub struct RenderImageParams {
    pub image_id: ImageId,
    pub frame_index: usize,
}

/// A cached and processed image, in BGRA format
pub struct RenderImage {
    /// The ID associated with this image
    pub id: ImageId,
    /// The scale factor of this image on render.
    pub(crate) scale_factor: f32,
    data: ImageData,
}

enum ImageData {
    Decoded(SmallVec<[Frame; 1]>),
    Recoverable(Box<RecoverableImage>),
}

struct RecoverableImage {
    encoded: Box<[u8]>,
    format: image::ImageFormat,
    size: Size<DevicePixels>,
    delay: Delay,
    initial: Mutex<Option<Frame>>,
    // Public callers can borrow pixels indefinitely. Only those callers need
    // a retained CPU buffer after the atlas has taken the first upload.
    borrowed: OnceLock<Option<Frame>>,
}

impl RecoverableImage {
    fn take_frame(&self) -> Result<Frame> {
        if let Some(frame) = self.initial.lock().take() {
            return Ok(frame);
        }
        let mut frames = crate::platform::decode_static_image(&self.encoded, self.format)?;
        frames
            .pop()
            .ok_or_else(|| anyhow::anyhow!("static image has no decoded frame"))
    }

    fn borrowed_bytes(&self) -> Option<&[u8]> {
        self.borrowed
            .get_or_init(|| {
                self.take_frame()
                    .inspect_err(|error| {
                        log::error!("Restoring static image pixels: {error}");
                    })
                    .ok()
            })
            .as_ref()
            .map(|frame| frame.buffer().as_raw().as_slice())
    }

    fn upload_bytes(&self) -> Result<Cow<'_, [u8]>> {
        if let Some(Some(frame)) = self.borrowed.get() {
            return Ok(Cow::Borrowed(frame.buffer().as_raw()));
        }
        Ok(Cow::Owned(self.take_frame()?.into_buffer().into_raw()))
    }
}

impl PartialEq for RenderImage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RenderImage {}

impl RenderImage {
    /// Create a new image from the given data.
    pub fn new(data: impl Into<SmallVec<[Frame; 1]>>) -> Self {
        Self::with_data(ImageData::Decoded(data.into()))
    }

    fn with_data(data: ImageData) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            id: ImageId(NEXT_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            data,
        }
    }

    pub(crate) fn from_static_encoded(bytes: &[u8], format: image::ImageFormat) -> Result<Self> {
        let frames = crate::platform::decode_static_image(bytes, format)?;
        Ok(Self::from_static_frames(frames, bytes, format))
    }

    pub(crate) fn from_static_frames(
        mut frames: SmallVec<[Frame; 1]>,
        bytes: &[u8],
        format: image::ImageFormat,
    ) -> Self {
        // Small images stay decoded. Retain encoded bytes only when replacing
        // a substantial buffer saves at least three quarters of its storage.
        let recoverable = frames.len() == 1
            && frames.first().is_some_and(|frame| {
                let decoded_len = frame.buffer().as_raw().len();
                decoded_len >= 256 * 1024 && bytes.len() <= decoded_len / 4
            });
        if !recoverable {
            return Self::new(frames);
        }
        let frame = frames.pop().unwrap();
        let (width, height) = frame.buffer().dimensions();
        Self::with_data(ImageData::Recoverable(Box::new(RecoverableImage {
            encoded: bytes.into(),
            format,
            size: size(width.into(), height.into()),
            delay: frame.delay(),
            initial: Mutex::new(Some(frame)),
            borrowed: OnceLock::new(),
        })))
    }

    /// Convert this image into a byte slice.
    pub fn as_bytes(&self, frame_index: usize) -> Option<&[u8]> {
        match &self.data {
            ImageData::Decoded(frames) => frames
                .get(frame_index)
                .map(|frame| frame.buffer().as_raw().as_slice()),
            ImageData::Recoverable(image) => {
                (frame_index == 0).then(|| image.borrowed_bytes()).flatten()
            }
        }
    }

    pub(crate) fn upload_bytes(&self, frame_index: usize) -> Result<Option<Cow<'_, [u8]>>> {
        match &self.data {
            ImageData::Decoded(_) => Ok(self.as_bytes(frame_index).map(Cow::Borrowed)),
            ImageData::Recoverable(image) if frame_index == 0 => image.upload_bytes().map(Some),
            ImageData::Recoverable(_) => Ok(None),
        }
    }

    /// Get the size of this image, in pixels.
    pub fn size(&self, frame_index: usize) -> Size<DevicePixels> {
        match &self.data {
            ImageData::Recoverable(image) => {
                return if frame_index == 0 {
                    image.size
                } else {
                    Size::default()
                };
            }
            ImageData::Decoded(frames) => frames,
        }
        .get(frame_index)
        .map(|frame| {
            let (width, height) = frame.buffer().dimensions();
            size(width.into(), height.into())
        })
        .unwrap_or_default()
    }

    /// Get the size of this image, in pixels for display, adjusted for the scale factor.
    pub(crate) fn render_size(&self, frame_index: usize) -> Size<Pixels> {
        self.size(frame_index)
            .map(|v| (v.0 as f32 / self.scale_factor).into())
    }

    /// Get the delay of this frame from the previous
    pub fn delay(&self, frame_index: usize) -> Delay {
        match &self.data {
            ImageData::Recoverable(image) => {
                return if frame_index == 0 {
                    image.delay
                } else {
                    Delay::from_numer_denom_ms(100, 1)
                };
            }
            ImageData::Decoded(frames) => frames,
        }
        .get(frame_index)
        .map(|frame| frame.delay())
        .unwrap_or(Delay::from_numer_denom_ms(100, 1))
    }

    /// Get the number of frames for this image.
    pub fn frame_count(&self) -> usize {
        match &self.data {
            ImageData::Decoded(frames) => frames.len(),
            ImageData::Recoverable(_) => 1,
        }
    }
}

impl fmt::Debug for RenderImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageData")
            .field("id", &self.id)
            .field("size", &(self.frame_count() != 0).then(|| self.size(0)))
            .finish()
    }
}

#[cfg(test)]
#[path = "assets_image_tests.rs"]
mod recoverable_image_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    #[test]
    fn empty_render_image_does_not_panic() {
        let image = RenderImage::new(SmallVec::new());
        assert_eq!(image.frame_count(), 0);
        assert_eq!(image.size(0), Size::default());
        assert_eq!(image.as_bytes(0), None);
        assert_eq!(image.render_size(0), Size::default());
        assert_eq!(image.delay(0), Delay::from_numer_denom_ms(100, 1));
        let _ = format!("{image:?}");
    }
}
