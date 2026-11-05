# LODS SEM visual check, 7 more domains (alpha 0.4)

Visualization only. Target **VAL** GT boxes are read only to pick crops and draw
their outlines. Nothing here feeds a hyperparameter, checkpoint, threshold or
method choice. No detector is run and no AP is computed. The alpha (0.4) is the
value the LODS matrix already used. It is not re-chosen here (see
`experiments/comparison/lods_obb_port.md`, "Alpha decision", for how it was set).

## Setup

- Code: `tools/lods/vis_sem.py`, unchanged; its `enhance` is the
  `StyleEnhancementModule.add_style` path of `sfod/lods_sem.py`, i.e. the
  `LODSTeacher.enhance` recipe at detector scale with a random target-val style
  image. Driver: `experiments/comparison/lods_visual_supp.sh` (`--alpha 0.4 --seed 0 --n 3`).
- Run on gpu-138, 2026-11-05 (about 04:00 美东), CPU only (`CUDA_VISIBLE_DEVICES=`,
  `nice -n 19`, `ionice -c3`), no GPU claimed.
- SEMs: the seed-42 SEM of each domain (the one every LODS seed used). Before the
  run, each copy was checked against `final_inputs/lods_matrix.csv` `sem_sha256`:

| Domain | SEM sha256 |
|---|---|
| RSAR clean | `6a307204f90e043aa2b37e858eea49b6f0eee93aa64983ee8998ab71b45486ca` |
| RSAR gaussian_white_noise | `8741f0cc5beeca1cd1c749101d500e3a4914c85060a6f1de90f3cd609159ba9b` |
| RSAR point_target | `9390b578ec3e085ec047a4101e2be931d9f135d931924c398a2034a31d03e1bc` |
| DIOR clean | `1de751ad262373a68f590796e72eb3231f69cf2e8cbe3141c6488e19b8be7c75` |
| DIOR snow | `ca6100088fe1fcc2dea3ce39f5f0dbf41a11118b2c3c38a3d5f12293e7408e74` |
| DIOR gaussian_noise | `e1508c5b957aad25babf04ebb3cd2f7a696917c0effd9dcd3957da1b12887f46` |
| DIOR pixelate | `ef7f2453c9a877a0ed1dadea6f90696e503378f3a703d8c4e7aee27396c247d0` |

- Crops: `small` = val images with the most GT objects under 20×20 px (RSAR ship;
  DIOR vehicle, ship). These are also the dense clusters (up to 588 objects per
  image). `slender` = val images with the most elongated GT objects (aspect ≥ 3:
  ship, bridge, harbor). A `dense` grid (RSAR ship/car/tank; DIOR
  vehicle/ship/storagetank) was rendered too. On DIOR it picked exactly the
  `small` images, so the file is byte-identical. On RSAR it shares 2 of its 3
  images with `small`. Neither is committed; both are on the run host with the full-size
  grids and `SHA256SUMS`, in `/home/zechuan/iraod_artifacts/comparison/lods_visual/supp_20261105/`.
- Committed: 14 grids, 1000 px wide, JPEG quality 85 (2.8 MB in total; JPEG as in the
  existing `experiments/comparison/lods_visual/`), at
  `experiments/comparison/lods_visual/supp/<dataset>_<domain>_{small,slender}_alpha0.4.{jpg,json}`.
  Columns: original | SEM output | original zoom with GT outline | SEM zoom. The
  `json` files give per image the Sobel-gradient correlation inside the GT boxes
  (`g`) and the target-vs-ring contrast before and after the SEM.
- The same val images recur across the corruptions of a dataset, because the
  selection reads only the shared GT.

## What the images show

I inspected six grids by eye: RSAR clean small, RSAR gaussian_white_noise small,
RSAR point_target slender, DIOR snow small, DIOR gaussian_noise small, DIOR pixelate
slender. The other eight have only their metrics checked.

- **Fake texture, RSAR.** The SEM output overlays land and water with a
  worm/checkerboard texture. The dark river turns mid-grey or brown, so
  ship-vs-water contrast comes from blobs on a textured background, not from a
  clean background. On point_target, the SEM adds ring/halo artefacts around the
  bright point targets and replaces the speckle with a smooth brownish texture.
- **Sepia tint.** Every DIOR output, and most RSAR ones, takes a brown/sepia
  cast from the random style image. DIOR colour (water, vegetation, roofs) is
  mostly lost.
- **Lost fine structure.** In the zooms, ships keep their position and rough
  size but become smeared blobs: hull shape, superstructure and wakes are gone.
  Moored ships in dense marinas merge into continuous bands (DIOR snow and
  gaussian_noise). Pier and bridge lines survive at full scale. In DIOR pixelate,
  the SEM smooths the block edges but adds no detail.
- **Corruption partly removed, not undone.** DIOR snow streaks are fainter in the
  full view but still visible in the zoom. DIOR gaussian_noise colour noise is
  suppressed into a uniform grey-brown. RSAR gaussian_white_noise looks much like
  RSAR clean before and after the SEM.
- **Targets not erased at 0.4.** Gradient correlation inside the GT boxes is
  0.11–0.48 on DIOR and 0.29–0.78 on RSAR (committed grids). Target-vs-ring contrast mostly stays
  near its original value. It drops clearly on the densest DIOR image (10243.jpg,
  588 objects: 0.76–1.01 → 0.60–0.67) and on some RSAR slender crops (for example,
  RSAR clean 0094000.png 0.31 → 0.05).
- **Coverage gap.** On DIOR, the `small` selection is filled by marinas, so
  small *vehicles* barely appear. DIOR vehicles are not shown at a useful zoom.

This agrees with the earlier RSAR chaff and DIOR fog check in
`experiments/comparison/lods_obb_port.md`: at alpha 0.4 the SEM keeps the layout
and most objects, but adds fake texture and a sepia tint, and removes fine
structure. It is a style perturbation, not an enhancement.
