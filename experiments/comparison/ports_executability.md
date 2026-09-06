# Faithful official SFOD ports (executability)

| Method | Official | In-repo OBB | Executable now | Reason |
|---|---|---|---|---|
| IRG-SFDA | github.com/vibashan/irg-sfda | no | no | no OBB reimplementation in IRAOD-New |
| LPLD | — | no | no | not ported |
| Simple-SFOD / SF-UT | github.com/EPFL-IMOS/simple-SFOD | ST (method B) covers equivalent Mean-Teacher ST | equivalent B executed | B already run on RSAR and DIOR |
| DRU / AASFOD / SF-YOLO | — | no | no | not ported; no approximate HBB fill |

Do not launch GPU for ports until a faithful OBB patch exists.

Updated method-specific reasons and original-plan official URLs:
`results/paper_comparison/dior_ports_executability.md`. No extra official port
is claimed by treating Simple-SFOD/SF-UT as equivalent B.
