# Protocol audit snapshot

- git 0f98a48 on `/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af` (detached HEAD).
- Shared source epoch_100 SHA e489f015… ; source-val mAP 0.5613 is NOT test mAP.
- Strict overlays present for B–F. LoRA forbidden in those configs.
- Eval uses TEST GT only. Adaptation VAL images only.
- Topology: B–D 1×32, E/F 2×16 after measured OOM; not mixed-batch.
- User untracked `对比实验的plan.docx` preserved locally; not copied into this audit as a substitute for results.
