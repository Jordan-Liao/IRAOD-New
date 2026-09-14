# RSAR Source Baseline Progress

- [x] Added explicit `--smoke-iters` support for exact bounded clean-source
  optimizer checks on a separately refused-on-reuse run directory.
- [x] Preserved the full source run's fixed `epoch_100.pth` completion policy.
- [x] Added launcher and terminal-status contract coverage.
- [x] Require an explicit `IRAOD_PYTHON` instead of a host-specific fallback.
- [x] Added an optional four-GPU distributed path for physical GPUs 4,5,6,7
  with `samples_per_gpu=1`, global batch 4, unchanged LR 0.005, and seed 42.
- [x] Record ordered physical mapping, world size, distributed argv, and
  source/global-batch/LR/epoch/seed invariant checks before ranks start.
- [x] Keep the distributed run on a separate fresh default `RUN_ROOT` and
  document explicit owner cutover without automatic process termination.
- [ ] Remote smoke and full training remain intentionally unrun.
