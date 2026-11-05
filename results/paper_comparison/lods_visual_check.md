# LODS SEM visual check, 7 more domains (alpha 0.4)

Visualization only. Target **VAL** GT boxes are read only to pick crops and draw
their outlines. Nothing here feeds a hyperparameter, checkpoint, threshold or
method choice. No detector is run and no AP is computed. The alpha (0.4) is the
value the LODS matrix already used; it is not re-chosen here (see
`experiments/comparison/lods_obb_port.md`, "Alpha decision", for how it was set
and why that is a disclosed gap).

## Setup

- Code: `tools/lods/vis_sem.py`. Its `enhance` is the
  `StyleEnhancementModule.add_style` path of `sfod/lods_sem.py`, i.e. the
  `LODSTeacher.enhance` recipe at detector scale with a random target-val style
  image. Driver: `experiments/comparison/lods_visual_supp.sh` (`--alpha 0.4 --seed 0 --n 3`).
  Test: `tools/cga_research/tests/test_lods_vis_select.py`.
- Crop selection (fixed 2026-11-05; the first render of this check had the
  `small` and `dense` grids pick identical DIOR images, so it was redone):
  - `small`: images with the tightest 160×160 px cluster of GT objects under
    20×20 px; the zoom centres on that cluster. RSAR: ship. DIOR: vehicle.
  - `slender`: images whose most elongated GT object has aspect ≥ 3 (ship,
    bridge, harbor); the zoom centres on that one object.
  - `dense`: images with the most GT objects inside one 160×160 px window
    (RSAR ship/car/tank; DIOR vehicle/ship/storagetank); the zoom centres there.
  - Modes of one domain never reuse an image (`--exclude`), so each grid shows
    different scenes. All 21 grids have distinct content (no duplicate md5).
  - The per-image metrics (`*.json`) are computed on the zoom's focus targets:
    Sobel-gradient correlation inside those GT boxes (`grad_corr_in_targets`) and
    target-vs-ring contrast before/after the SEM.
- Run on gpu-138, 2026-11-05, CPU only (`CUDA_VISIBLE_DEVICES=`, `nice -n 19`,
  `ionice -c3`), no GPU claimed.
- SEMs: the seed-42 SEM of each domain, the one every LODS seed used. Each copy
  was checked against `final_inputs/lods_matrix.csv` `sem_sha256`:

| Domain | SEM sha256 |
|---|---|
| RSAR clean | `6a307204f90e043aa2b37e858eea49b6f0eee93aa64983ee8998ab71b45486ca` |
| RSAR gaussian_white_noise | `8741f0cc5beeca1cd1c749101d500e3a4914c85060a6f1de90f3cd609159ba9b` |
| RSAR point_target | `9390b578ec3e085ec047a4101e2be931d9f135d931924c398a2034a31d03e1bc` |
| DIOR clean | `1de751ad262373a68f590796e72eb3231f69cf2e8cbe3141c6488e19b8be7c75` |
| DIOR snow | `ca6100088fe1fcc2dea3ce39f5f0dbf41a11118b2c3c38a3d5f12293e7408e74` |
| DIOR gaussian_noise | `e1508c5b957aad25babf04ebb3cd2f7a696917c0effd9dcd3957da1b12887f46` |
| DIOR pixelate | `ef7f2453c9a877a0ed1dadea6f90696e503378f3a703d8c4e7aee27396c247d0` |

- Committed: 21 grids (7 domains × small/slender/dense), 1000 px wide, JPEG
  quality 85, 4.2 MB, at
  `experiments/comparison/lods_visual/supp/<dataset>_<domain>_{small,slender,dense}_alpha0.4.{jpg,json}`.
  Columns: original | SEM output | original zoom with GT outline | SEM zoom.
  Full-size grids and `SHA256SUMS` stay on the run host.
- The same val images recur across the corruptions of one dataset, because the
  selection reads only the shared GT.

## What the images show

Inspected by eye: DIOR clean small/dense, DIOR snow small, RSAR clean slender,
RSAR point_target small/dense; the rest by metrics only.

- **Fake texture.** On RSAR the SEM overlays land and water with a
  worm/checkerboard texture and turns dark water mid-grey or brown; around bright
  point targets it adds ring/halo artefacts. On DIOR it adds faint embossed
  texture.
- **Sepia tint.** Every DIOR output, and most RSAR ones, takes a brown/sepia cast
  from the style image; DIOR colour is mostly lost.
- **Small targets.** DIOR small vehicles (parking lots, roads) keep their
  position as dark blobs, but the smallest cars on bright roads are reduced to
  faint dots or vanish in the zoom (DIOR snow, row 3). RSAR small ships survive
  as blobs; hull shape and wakes are gone.
- **Slender targets.** Bridges, piers and long ships keep their line at full
  scale; zoomed edges are smeared into the texture. Target contrast collapses on
  some crops (RSAR clean 0094000.png 0.31 → 0.05; DIOR snow 07007.jpg 0.95 → 0.29;
  DIOR pixelate 07007.jpg 0.82 → 0.15).
- **Dense targets.** Dense marinas and storage-tank fields keep the layout, but
  neighbouring objects merge into continuous bands. On RSAR point_target dense
  scenes the fake texture partly masks the packed point targets. The densest DIOR
  image (10243.jpg) loses about half its target contrast (0.71–0.93 → 0.30–0.45).
- **Corruption partly removed, not undone.** DIOR snow streaks are fainter but
  still visible; DIOR gaussian_noise colour noise becomes uniform grey-brown;
  RSAR gaussian_white_noise looks much like RSAR clean before and after.
- **Metrics.** Gradient correlation inside the focus targets: RSAR small
  0.18–0.49, slender 0.38–0.78, dense 0.31–0.52; DIOR small 0.09–0.51, slender
  0.10–0.50, dense 0.09–0.31. Single thin slender targets with a mask of ≤ 4 px
  give NaN and are left as NaN.

Conclusion, consistent with the earlier RSAR chaff and DIOR fog check in
`experiments/comparison/lods_obb_port.md`: at alpha 0.4 the SEM keeps the layout
and most objects, but adds fake texture and a sepia tint, removes fine structure,
merges dense neighbours and loses some of the smallest targets. It is a style
perturbation, not a structure-preserving enhancement. The requirement that small,
slender and dense targets stay intact is therefore **not fully met**; this is
reported, not rerun (no-rerun rule).
