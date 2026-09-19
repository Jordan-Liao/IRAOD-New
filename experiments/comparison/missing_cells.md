# Reusable vs missing cells (snapshot 2026-09-12)

Reuse hash-matched valid EMA TEST. Do not retrain those cells.

## Reuse
- RSAR A–F EMA: 8 domains × 3 seeds complete
- DIOR A–F EMA: clean+brightness+cloudy+contrast × 3 seeds complete
- IRG/LPLD/SFUT DIOR EMA complete; RSAR EMA almost complete (NaN holes below)

## Missing (live queue)
- DRU: all domains (gate: RSAR/chaff/seed=42)
- DIOR remaining-17 official DIOR-C, seed=42, methods A–F + 6 external
- IRG/LPLD RSAR NaN: seed44 × point_target and noise_suppression (4 EMA cells; SFUT 2)
- AASFOD TEST mostly missing (RSAR local 9/24, DIOR 0/12)
- SFYOLO TEST 0; RSAR TRAIN 15/24 (TAM gaps on am_noise_horizontal/vertical/smart_suppression)

## Not in queue
Student-required TEST, LoRA oracle leftovers, B_REG / F_text_only / F_veto_only, ROI expansion, 3-seed on DIOR-17.
