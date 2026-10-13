# Native path sampling references

These PNGs capture the corrected full-viewport D3D11 path renderer immediately
before bounded targets and integer texel loads were applied. They include the
shared Oklab/sRGB, dither, shadow and blend corrections. The older pre-correction
images remain historical artifacts outside the source tree and are not these
expectations.

Reference environment: vkd3d 2.1, shader models vs_4_1 and ps_4_1, Windows
10.0.26200, headless D3D11 WARP with four-sample MSAA. Reference shaders.hlsl
SHA256: `e01b28f359a4d871a307dd450ce08929e016487a5e7a422df926deaffcae4599`.
Reference test executable SHA256: `c096cad49ea8e9da624cea81a4d0b56ba188a7a013f5365a211e0a48b074e32f`.

The portable tests always require exact full-target versus compact-target pixel
equality and verify resource release/reuse plus resize after an incomplete draw.
The ignored `corrected_vkd3d_2_1_full_target_matches_reference` diagnostic also
compares against these compiler-specific reference PNGs. Run it explicitly only
with the matching compiler/adapter setup. Native Windows debug D3DCompile and
release FXC can produce different bytecode rounding, so this check is not an
ordinary portable test.

At 128 by 128 pixels, switching normalized linear sampling to an integer texel
load changes only these measured channel counts by exactly one code:

| Fixture | Blue | Green | Red | Alpha |
| --- | ---: | ---: | ---: | ---: |
| Gradient | 0 | 0 | 0 | 44 |
| Pattern | 0 | 31 | 0 | 165 |
| Overlap | 0 | 8 | 0 | 49 |

Each rounding-mask PNG identifies the exact allowed channel coordinates.
The diagnostic rejects any other changed coordinate, any change larger than one
code, or a different changed-channel count. All three 173 by 117 references are
exact. This is a measured pixel-aligned sampling effect, not a general visual
comparison tolerance. Both new full-size and bounded-target renders remain
byte-identical for all six fixtures, including gradients, patterns, fractional
clipping, overlap and antialiased edges.

The corrected reference and compact raw BGRA outputs, build logs, shader
provenance and per-fixture hashes are preserved in the Editor workspace under
`.dist/performance/native-paths/corrected-reference-*`, `corrected-compact-*`,
and `corrected-pixel-comparison.json`.
