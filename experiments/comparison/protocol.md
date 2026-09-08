# Final strict A-F protocol

RSAR: 8 domains, 8538 TEST images, 6 classes. DIOR-R: 4 domains, 11738 TEST images, 20 classes. Domains are clean plus the frozen corruption matrix in run evidence.

Shared fixed seed42 source per dataset; adaptation seeds 42/43/44. B-F adapt independently on image-only VAL for each clean/corrupted domain; no source forward, no target GT in adaptation or checkpoint selection. Global batch 32, LR .02, one epoch; B-D 1x32, E/F 2x16. Final EMA iterations: RSAR 266, DIOR 185. Final Student retained.

OBB le90 VOC AP50 (not COCO AP50:95). Class AP retains printed precision; mAP uses exact metric.mAP. mPC averages all corruptions within seed; delta_A pairs the same domains with fixed A. Across-seed mean and sample std (ddof=1); A has no SD. Historical rPC=100*mPC/fixed A clean TEST. Method-clean normalization is separate. Recovery=(method-A_corruption)/(A_clean_TEST-A_corruption), undefined for nonpositive denominator, unstable near zero and never ranked. Paired tests/CI at n=3 are low-power and conditional on the fixed source; see report.json statistical_protocol.

130 new training and 192 native-ID eval cells are complete per terminal_state.json and the complete artifact consumer. Qualitative seed42 coverage is independently accepted: 132 groups, 1267816 NPZs, 18468211 detections, 3520 visualizations, 24 embeddings. Legacy versions and frozen seed42 reports are historical, not current evidence.

Source-Free A-F only in main tables; source-available/oracle/unported baselines and extra ablations remain not run. No test-selected checkpoints or negative-result exclusions.
