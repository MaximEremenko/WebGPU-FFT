# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning.

## [0.1.0] - 2026-02-08

### Added
- Initial public package release of `webgpufft`.
- WebGPU FFT transforms: `c2c`, `r2c`, `c2r`.
- DCT and DST families (`dct1..dct4`, `dst1..dst4`).
- FFT convolution/correlation plan (`fftconv`) and small spatial `conv2d`.
- Large-mode routing, segmented `BufferView` support, and pipeline cache snapshot APIs.

### Notes
- `inPlace:true` is currently supported only for `c2c`.
- `r2c` is forward-only and `c2r` is inverse-only.
- `fftconv` and `conv2d` currently use `f32` compute precision.
