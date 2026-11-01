# Faithful official SFOD ports (executability)

DRU is **in the live queue**. It is not N/A and is not method B.

| Method | Official | In-repo OBB | Executable now | Notes |
|---|---|---|---|---|
| IRG-SFDA | github.com/vibashan/irg-sfda | yes (remote ports) | reuse valid TEST; rerun 4 NaN | required |
| LPLD | ECCV 2024 | yes (remote ports) | reuse valid TEST; rerun 4 NaN | required |
| Simple-SFOD / SF-UT | SFUT-OBB | yes | independent of B | required |
| DRU | github.com/lbktrinh/DRU (ECCV 2024) | **DRUUnbiasedTeacher** | chaff/seed=42 gate then matrix | required; not UnbiasedTeacher |
| PETS | RainbowNebula/PETS (ICCV 2023) @ 1e87e45 | **PETSUnbiasedTeacher** | chaff/seed=42 gate | our reimplementation; not B; `pets_obb_port.md` |
| AASFOD | Chu et al. | yes (remote ports) | fill missing TEST | required |
| SF-YOLO | TAM + two-round detector | yes (remote ports) | fill TAM/TEST | required |

Do not substitute B numbers for DRU. Do not put official HBB numbers in the main table.
