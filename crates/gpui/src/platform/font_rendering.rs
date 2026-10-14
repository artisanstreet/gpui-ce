/// Gamma and contrast settings shared by renderers using Cosmic rasterization.
///
/// DirectWrite rendering uses the operating system's rendering parameters instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FontRenderingParameters {
    /// Polynomial coefficients used by the glyph alpha-correction shaders.
    pub gamma_ratios: [f32; 4],
    /// Contrast enhancement for grayscale glyph coverage.
    pub grayscale_enhanced_contrast: f32,
    /// Contrast enhancement for subpixel glyph coverage.
    pub subpixel_enhanced_contrast: f32,
}

impl FontRenderingParameters {
    /// Reads the existing `ZED_FONTS_*` settings, preserving WGPU defaults and bounds.
    pub fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        let mut value = |name: &str, fallback: f32| {
            lookup(name)
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(fallback)
        };
        Self {
            gamma_ratios: super::get_gamma_correction_ratios(
                value("ZED_FONTS_GAMMA", 1.8).clamp(1.0, 2.2),
            ),
            grayscale_enhanced_contrast: value("ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST", 1.0)
                .max(0.0),
            subpixel_enhanced_contrast: value("ZED_FONTS_SUBPIXEL_ENHANCED_CONTRAST", 0.5).max(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::FontRenderingParameters;

    fn settings(gamma: &str, grayscale: &str, subpixel: &str) -> FontRenderingParameters {
        FontRenderingParameters::from_lookup(|name| {
            Some(
                match name {
                    "ZED_FONTS_GAMMA" => gamma,
                    "ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST" => grayscale,
                    "ZED_FONTS_SUBPIXEL_ENHANCED_CONTRAST" => subpixel,
                    _ => unreachable!("unexpected font setting"),
                }
                .to_owned(),
            )
        })
    }

    #[test]
    fn absent_and_invalid_settings_keep_wgpu_defaults() {
        let defaults = FontRenderingParameters::from_lookup(|_| None);
        assert_eq!(defaults, settings("invalid", "invalid", "invalid"));
        assert_eq!(defaults, settings("1.8", "1.0", "0.5"));
    }

    #[test]
    fn gamma_is_bounded_and_contrast_is_nonnegative() {
        assert_eq!(settings("0.2", "-2", "-1"), settings("1.0", "0", "0"));
        assert_eq!(settings("10", "2.5", "3"), settings("2.2", "2.5", "3"));
    }

    #[test]
    fn explicit_settings_preserve_independent_contrast_values() {
        let configured = settings("2.0", "0.25", "0.75");
        assert_eq!(
            configured.gamma_ratios,
            super::super::get_gamma_correction_ratios(2.0)
        );
        assert_eq!(configured.grayscale_enhanced_contrast, 0.25);
        assert_eq!(configured.subpixel_enhanced_contrast, 0.75);
    }
}
