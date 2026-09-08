# Oracle TRAIN preparation and bounded smoke

## Real-LoRA smoke (NON_RESULT)

Run **only by the GPU owner**, from the deployed checkout, with an existing
TRAIN crop manifest produced by `tools/build_oracle_patches.py`. Set `METADATA`
to that dataset's absolute `metadata.csv` path and `OUTPUT` to a new smoke
directory. `DATASET` must be `RSAR` or `DIOR`; the adapters are dataset-specific.

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4 IRAOD_RUNTIME_READY=1 \
  /home/zechuan/miniforge3/envs/iraod/bin/python tools/train_sarclip_lora_rsar.py \
  --dataset "$DATASET" --metadata "$METADATA" \
  --sarclip-dir "$PWD" \
  --sarclip-pretrained /mnt/shared/zechuan/iraod_weights/sarclip/ViT-B-32/vit_b_32_model.safetensors \
  --output "$OUTPUT" --seed 42 --batch-size 64 --num-workers 0 \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
  --lr 0.0001 --weight-decay 0.0001 --precision fp32 --smoke-steps 2
```

The owner must bind the interpreter to its existing SARCLIP-capable environment
if it differs from the example. No GPU launch is performed by preparation.

This performs exactly two AdamW updates and 128 class-balanced replacement
draws, **not one full epoch**. It still validates the whole supplied TRAIN
manifest and builds its class weights; `--smoke-steps` bounds optimizer/data
draw work, not manifest parsing or model loading. Formal defaults and the
uncapped ten-epoch path are unchanged when the option is omitted.

Successful `smoke.json` (also reflected in `config.json`) records:

- `result_status=NON_RESULT`, zero completed/requested epochs, no selection;
- actual completed optimizer-call count;
- actual `sarclip_adapter.LoRALinear` module types, up/down tensor counts and
  factor parameter counts;
- exact pre/post equality of base parameters, no base gradients, and frozen
  `requires_grad`; the recipe's trainable `logit_scale` is explicitly separate;
- per-factor maximum absolute parameter deltas, requiring finite values and
  at least one nonzero factor delta.

No `.pth` or formal epoch log is published. Projection-only mode is rejected.
Budget mismatch, changed base, missing updates, nonfinite or all-zero factor
deltas fail without publishing successful smoke evidence. Use a fresh output
directory for another attempt; existing smoke/training artifacts are preserved.
These are engineering checks, not adapter quality or scientific result evidence.

CPU regression command (existing unittest runner):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest tools.tests.test_oracle_lora_train
```

Tests use tiny CPU models with the real LoRA injection and optimizer, not the
real SARCLIP checkpoint. GPU smoke and real-model readiness remain unverified.

## Author-bound cloudy TRAIN generator

The delegated input inventory reports 5,862 brightness TRAIN images and 5,862
contrast TRAIN images, plus author cloud textures at
`/mnt/shared/zechuan/iraod_artifacts/third_party/dior_cloudy_clouds_v1/{png,NOTICE.md}`.
Those remote assets are not mounted in this development worktree.

The separate `tools/dataset/generate_dior_cloudy.py` now implements cloudy.
`tools/dataset/DIOR_CLOUDY.md` gives the exact CPU invocation and source binding.
The staged texture provenance
names DOTA-C commit `c9ce98fad9b2fbd7218346d8b4bdba6974323ac1`,
[`clouds/Cloudy_Image_Arithmetic.m`](https://github.com/hehaodong530/DOTA-C/blob/c9ce98fad9b2fbd7218346d8b4bdba6974323ac1/clouds/Cloudy_Image_Arithmetic.m).
That MATLAB script uses antialiased bicubic `imresize`, lexicographic cyclic
texture selection and atmospheric light `A` carried across channels and images.
The independent implementation preserves these equations, uint8 axis rounding,
column-major maximum ties and stateful replay. It does not substitute Pillow
resizing, independently reset A, or map cloudy to fog. MATLAB/Octave runtime
bitwise comparison is not claimed because that runtime is unavailable.

Generation binds only5862 `DIOR/ImageSets/train.txt` IDs using the existing
`train_annotations`/`required_images` helpers. Resume requires matching lineage
and exact replayed pixels; VAL/TEST data and completed brightness/contrast
images remain untouched. Materialization is CPU preparation, not adapter
training or oracle-result completion.
