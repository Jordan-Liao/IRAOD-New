# Faithful official SFOD ports (executability)

| Method | Official | In-repo OBB | Executable now | Reason |
|---|---|---|---|---|
| IRG-SFDA | github.com/Vibashan/irg-sfda | independent IRGOBB implementation | scientific smoke pending | Code-defined graph/KL/contrast mechanisms; common-budget differences explicit |
| LPLD | github.com/CV-Det/LPLD | independent LPLDOBB implementation | scientific smoke pending | Aligned proposal distillation, four pseudo losses and epoch-final EMA |
| SF-UT (Simple-SFOD paper) | github.com/EPFL-IMOS/simple-SFOD | distinct arm required | not satisfied by B | SF-UT retains both pseudo regression losses; actual B zeros both |
| AdaBN + Fixed SF-FixMatch | github.com/EPFL-IMOS/simple-SFOD | separate strategy, not B | scope/protocol not released | Target-only BN statistics, then fixed pseudo labels; not ordinary EMA ST |
| DRU | official evidence under review | architecture decision required | not released | Single RoI head lacks aligned multiple decoder logits |
| AASFOD | official evidence under review | source/budget decision required | not released | No source-trained dropout head; 2500-step EMA never updates within185/266 steps |
| SF-YOLO | github.com/vs-cv/sf-yolo | detector-independent core identified | not released | Required target-only TAM training and one-epoch SSM limitation; not a YOLO-anchor blocker |

Do not launch GPU for ports until a faithful OBB patch exists.

Current mechanisms, pinned evidence and implementation status:
`extensions/official_ports.json` and `extensions/protocol.md`.
The earlier seed42 port note is preserved at
`results/paper_comparison/historical/seed42/dior_ports_executability.md`.
Its B/SF-UT equivalence claim is superseded: shared self-training is not
mechanism equivalence, and no B metric is reused as an SF-UT result.
