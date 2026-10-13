use super::*;
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage};
use std::{io::Cursor, sync::Arc};

fn encoded_image(format: ImageFormat, side: u32) -> Vec<u8> {
    let pixels = RgbaImage::from_fn(side, side, |x, y| {
        Rgba([((x / 32) * 7) as u8, ((y / 32) * 11) as u8, 83, 255])
    });
    let mut bytes = Cursor::new(Vec::new());
    let image = DynamicImage::ImageRgba8(pixels);
    let image = if format == ImageFormat::Jpeg {
        DynamicImage::ImageRgb8(image.into_rgb8())
    } else {
        image
    };
    image.write_to(&mut bytes, format).unwrap();
    bytes.into_inner()
}

fn recoverable(image: &RenderImage) -> &RecoverableImage {
    match &image.data {
        ImageData::Recoverable(image) => image,
        ImageData::Decoded(_) => panic!("expected recoverable storage"),
    }
}

#[test]
fn static_uploads_restore_identical_pixels_and_keep_metadata_without_pixels() {
    for format in [ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::WebP] {
        let bytes = encoded_image(format, 512);
        let expected = crate::platform::decode_static_image(&bytes, format).unwrap();
        let expected = expected[0].buffer().as_raw();
        let image = RenderImage::from_static_encoded(&bytes, format).unwrap();
        let data = recoverable(&image);
        assert!(image.upload_bytes(1).unwrap().is_none());
        assert!(data.initial.lock().is_some());
        let original_ptr = data
            .initial
            .lock()
            .as_ref()
            .unwrap()
            .buffer()
            .as_raw()
            .as_ptr();
        let first = image.upload_bytes(0).unwrap().unwrap();
        assert!(matches!(first, Cow::Owned(_)));
        assert_eq!(
            first.as_ptr(),
            original_ptr,
            "first upload must move the existing allocation"
        );
        assert_eq!(first.as_ref(), expected);
        drop(first);
        assert!(data.initial.lock().is_none());
        assert!(data.borrowed.get().is_none());
        assert_eq!(image.frame_count(), 1);
        assert_eq!(image.size(0), size(512.into(), 512.into()));
        assert_eq!(image.size(1), Size::default());
        assert_eq!(image.delay(0), Delay::from_numer_denom_ms(0, 1));
        assert_eq!(image.as_bytes(1), None);
        assert!(data.borrowed.get().is_none());
        for _ in 0..3 {
            assert_eq!(image.upload_bytes(0).unwrap().unwrap().as_ref(), expected);
            assert!(data.initial.lock().is_none());
            assert!(data.borrowed.get().is_none());
        }
        assert_eq!(image.as_bytes(0).unwrap(), expected);
        let retained_ptr = image.as_bytes(0).unwrap().as_ptr();
        let upload = image.upload_bytes(0).unwrap().unwrap();
        assert!(matches!(upload, Cow::Borrowed(_)));
        assert_eq!(upload.as_ptr(), retained_ptr);
    }
}

#[test]
fn public_pixel_borrow_before_upload_keeps_its_address() {
    let image =
        RenderImage::from_static_encoded(&encoded_image(ImageFormat::Png, 256), ImageFormat::Png)
            .unwrap();
    let first = image.as_bytes(0).unwrap();
    assert!(recoverable(&image).initial.lock().is_none());
    for _ in 0..3 {
        let upload = image.upload_bytes(0).unwrap().unwrap();
        assert!(matches!(upload, Cow::Borrowed(_)));
        assert_eq!(upload.as_ptr(), first.as_ptr());
        assert_eq!(upload.as_ref(), first);
    }
}

#[test]
fn concurrent_upload_and_borrow_calls_preserve_pixels() {
    let image = Arc::new(
        RenderImage::from_static_encoded(&encoded_image(ImageFormat::WebP, 512), ImageFormat::WebP)
            .unwrap(),
    );
    let expected = image.upload_bytes(0).unwrap().unwrap().into_owned();
    std::thread::scope(|scope| {
        for index in 0..8 {
            let image = &image;
            let expected = &expected;
            scope.spawn(move || {
                for _ in 0..4 {
                    if index % 2 == 0 {
                        assert_eq!(image.as_bytes(0).unwrap(), expected);
                    } else {
                        assert_eq!(image.upload_bytes(0).unwrap().unwrap().as_ref(), expected);
                    }
                }
            });
        }
    });
}

#[test]
fn small_uncompressed_and_raw_images_keep_decoded_storage() {
    for (format, side) in [(ImageFormat::Png, 32), (ImageFormat::Bmp, 512)] {
        let image = RenderImage::from_static_encoded(&encoded_image(format, side), format).unwrap();
        assert!(matches!(image.data, ImageData::Decoded(_)));
        assert!(matches!(
            image.upload_bytes(0).unwrap().unwrap(),
            Cow::Borrowed(_)
        ));
    }
    let raw = Frame::new(RgbaImage::new(512, 512));
    let image = RenderImage::new(SmallVec::from_vec(vec![raw.clone(), raw]));
    assert!(matches!(image.data, ImageData::Decoded(_)));
    assert_eq!(image.frame_count(), 2);
    assert!(matches!(
        image.upload_bytes(1).unwrap().unwrap(),
        Cow::Borrowed(_)
    ));
}

#[test]
fn restore_errors_are_returned_to_the_atlas() {
    let mut image =
        RenderImage::from_static_encoded(&encoded_image(ImageFormat::Png, 256), ImageFormat::Png)
            .unwrap();
    drop(image.upload_bytes(0).unwrap());
    let ImageData::Recoverable(data) = &mut image.data else {
        unreachable!()
    };
    // Immutable production bytes cannot change; this injects a decoder error
    // to verify that a failed atlas restoration remains fallible.
    data.encoded = Box::new([]);
    assert!(image.upload_bytes(0).is_err());
}

#[test]
fn restored_jpeg_reapplies_exif_orientation() {
    let original = encoded_image(ImageFormat::Jpeg, 512);
    let unrotated = crate::platform::decode_static_image(&original, ImageFormat::Jpeg).unwrap();
    // EXIF little-endian TIFF, one SHORT Orientation entry, value 3 (180 degrees).
    let exif = b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x03\0\0\0\0\0\0\0";
    let mut oriented = original[..2].to_vec();
    oriented.extend_from_slice(&[0xff, 0xe1]);
    oriented.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
    oriented.extend_from_slice(exif);
    oriented.extend_from_slice(&original[2..]);
    let image = RenderImage::from_static_encoded(&oriented, ImageFormat::Jpeg).unwrap();
    let first = image.upload_bytes(0).unwrap().unwrap();
    let original_pixels = unrotated[0].buffer().as_raw();
    assert_eq!(&first[..4], &original_pixels[original_pixels.len() - 4..]);
    let restored = image.upload_bytes(0).unwrap().unwrap();
    assert_eq!(restored, first);
    assert_eq!(image.as_bytes(0).unwrap(), first.as_ref());
}
