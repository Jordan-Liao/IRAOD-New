# Author-bound DIOR cloudy TRAIN preparation

Independent implementation of the mathematical algorithm in
[`hehaodong530/DOTA-C/clouds/Cloudy_Image_Arithmetic.m`](https://github.com/hehaodong530/DOTA-C/blob/c9ce98fad9b2fbd7218346d8b4bdba6974323ac1/clouds/Cloudy_Image_Arithmetic.m),
commit `c9ce98fad9b2fbd7218346d8b4bdba6974323ac1`. No author source is vendored.
The author describes the cloud textures as USGS Landsat8 Collection2 Tier1
true-color B4/B3/B2 imagery; retain the staged texture `NOTICE.md` and `order.json`.
The30-image archive SHA256 is
`66b15135711c62236d68ecdecbb1e45062d4b1a84363de32a9e356e80f588963`.

## Fixed algorithm and dataset binding

- Threshold0, Beta255, initial atmospheric light A=.95.
- Lexicographic TRAIN image order and lexicographic author cloud cycle
  `1,10,...,19,2,20,...,9`. No natural-number reorder.
- For each RGB channel, Gamma is the resized cloud's minimum, Lambda is
  `sum(cloud[cloud>Gamma])/sum(cloud-Gamma)`, and intensity is
  `(cloud-Gamma)*Lambda`. Synthesis is `(255-intensity)/255*land + A*intensity`.
- At the first maximum in MATLAB column-major order, reduce A to
  `(255-land_term)/intensity` only when lower. Carry A through all channels
  and subsequent images; never independently reset it on resume.
- Reproduce imresize's separable Keys cubic kernel a=-.5, pixel-center mapping,
  antialiasing for downsampling and symmetric endpoint extension. Preserve
  uint8 saturation/nearest rounding after **each** axis, not float-image
  resize followed by one conversion. Do not substitute Pillow/OpenCV resizing.
  Pixel rounding, analytic cubic stencils, and scalar author-equation/state
  regressions are tested. There is no MATLAB/Octave runtime on the compute
  host, so **bitwise comparison against that runtime is not claimed**.
- DIOR's original JPEGs decode to RGB through Pillow; results save lossless
  PNG with compression1. This dataset IO binding replaces the author's
  PNG-only directory input, not the cloud algorithm. Ignore texture alpha,
  matching MATLAB's one-output `imread`.

Only the5862 IDs from `DIOR/ImageSets/train.txt` are permitted. The tool checks
the corresponding TRAIN annotation files exist but does not parse their labels
or boxes. It reads no VAL/TEST annotations or images. Brightness and contrast
are separate already-prepared inputs; this does not alter them.

## CPU invocation

```bash
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/zechuan/miniforge3/envs/iraod/bin/python \
  -m tools.dataset.generate_dior_cloudy \
  --data-root /mnt/shared/zechuan/iraod_data/DIOR \
  --clouds-root /mnt/shared/zechuan/iraod_artifacts/third_party/dior_cloudy_clouds_v1 \
  --out-dir /mnt/shared/zechuan/iraod_artifacts/dior_oracle_train_images/JPEGImages-cloudy
```

New output roots contain `recipe.json`,5862 PNGs, and terminal `summary.json`
with per-image input/texture/shape/A-state lineage. An existing root requires
`--resume` and an identical recipe. Resume replays the state from the beginning,
requires exact pixel equality for existing images, and writes only missing
images. It refuses unknown or mismatched images rather than blindly adopting
them. Constant resized texture channels make the author's Lambda undefined
and are reported as errors, never mapped to fog or a fabricated cloud.

This prepares oracle TRAIN data only. It launches no adapter, detector,
inference, controller or GPU job, and is not an oracle result.
