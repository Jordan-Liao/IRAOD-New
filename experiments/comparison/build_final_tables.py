"""Final comparison deliverables, rebuilt from the reviewed input CSVs with an assert-based self-check.

    python3 experiments/comparison/build_final_tables.py                    # build all outputs + self-check
    python3 experiments/comparison/build_final_tables.py --import-inputs DIR
        # copy the reviewed CSVs/READMEs from DIR into final_inputs/ (original sha256 checked where known,
        # private hosts/paths redacted, sha256 of both copies recorded in final_inputs/sha256.txt), then build.
        # The DETR DIOR merge step is the same command on the directory holding detr_dior_matrix_*.

Inputs
- results/paper_comparison/raw_results.csv: the 12 original Oriented R-CNN methods (build_summary.py --scans),
  with mAP50_95 joined from existing predictions.pkl only (no pkl = not_inferred, never backfilled).
- experiments/comparison/final_inputs/: PETS and LODS (R-CNN, seeds 42/43/44), the DETR group (seed 42).

Outputs (results/paper_comparison/): main_table_rcnn_obb.tex, main_table_detr_obb.tex,
architecture_generalization.csv, dior_c_table_corrected.{md,tex}.

Conventions: AP50 and mAP50-95 are fractions in the CSVs (full precision) and percentage points with two decimals
(round half up, never truncated) in the tables. mPC = mean over the corruptions (RSAR 7; DIOR the 19 official
DIOR-C names, cloudy excluded); "all domains" adds clean (RSAR 8; DIOR clean + 19 + cloudy = 21).
Mean +- std is over seeds where a method has 3 seeds; where seeds = 1 the std is across domains.
architecture_generalization.csv status columns: ok = every needed cell present; "not_inferred (k/n missing)" =
AP50 complete but k of the n needed (seed x domain) cells have no mAP50-95 (no saved pkl), so no mean is given;
"pending (k/n missing)" = k of n AP50 cells not run yet; not_ported / user_reported as stated in the row.
"""
import argparse
import csv
import hashlib
import math
import os
import re
import statistics
import sys
from decimal import ROUND_HALF_UP, Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from build_summary import DIOR_C19, METHODS, NAMES as RCNN_NAMES, RSAR  # noqa: E402

INP = os.path.join(HERE, "final_inputs")
OUT = os.path.join(ROOT, "results", "paper_comparison")
RAW = os.path.join(OUT, "raw_results.csv")

MPC = {"RSAR": RSAR[1:], "DIOR": DIOR_C19}
ALL = {"RSAR": RSAR, "DIOR": ["clean"] + DIOR_C19 + ["cloudy"]}
RCNN_ORIG = [m for m in METHODS if m not in ("PETS", "LODS", "OURS")]  # the 12 original methods
RCNN_METHODS = RCNN_ORIG + ["PETS", "LODS"]
DETR_METHODS = ["A", "B", "E", "PETS", "SFUT", "SFYOLO", "C", "D", "F"]
NAMES = dict(RCNN_NAMES, PETS="PETS-OBB", LODS="LODS-OBB", OURS="本文方法")
TEX_NAMES = {"A": "A Source Only", "B": "B ST / Mean Teacher", "C": "C CLIP-CGA", "D": "D SARCLIP-CGA",
             "E": "E VLST", "F": "F CGA+VLST", "IRG": "IRG-OBB", "LPLD": "LPLD-OBB", "SFUT": "SF-UT-OBB",
             "DRU": "DRU-OBB", "AASFOD": "A$^2$SFOD-OBB", "SFYOLO": "SF-YOLO-OBB", "PETS": "PETS-OBB",
             "LODS": "LODS-OBB", "OURS": "本文方法"}

# Reviewed inputs (sha256 prefixes from the lead's dispatch messages).
KNOWN_SHA = {"lods_matrix.csv": "adad9f1d", "pets_matrix.csv": "1833421874", "pets_matrix_summary.csv": "9b30c2c5",
             "detr_rsar_matrix_final.csv": "c40c9a03", "detr_rsar_matrix_summary.csv": "8427dcad",
             "detr_rsar_matrix_README.md": "bd51f12c5e56", "detr_dior_matrix_final.csv": "79217836eda0",
             "detr_dior_matrix_summary.csv": "84e55ddd9811", "detr_dior_matrix_README.md": "32ee09dacd25"}
IMPORT_NAMES = list(KNOWN_SHA)
# The public repo already names these hosts and disk roots; any other host or root is redacted on import.
PUBLIC_HOSTS = {"gpu-67", "gpu-68", "gpu-138", "gpu-183"}
PUBLIC_ROOTS = ("/mnt/shared/zechuan/", "/home/zechuan/iraod_artifacts", "/home/zechuan/iraod_jobs",
                "/mnt/HDD_14TB/zechuan/", "/mnt/SSD_4TB/zechuan/", "/mnt/SSD2_8TB/zechuan/", "/mnt/SSD1_8TB/zechuan/")


def redact(text):
    text = re.sub(r"\bgpu-\d+\b", lambda m: m.group() if m.group() in PUBLIC_HOSTS else "<other-host>", text)
    return re.sub(r"(?:/home|/mnt)/[^/\s,\"']+/[^/\s,\"']+",
                  lambda m: m.group() if m.group().startswith(PUBLIC_ROOTS) or (m.group() + "/").startswith(
                      PUBLIC_ROOTS) else "<private-root>", text)

# The user's DIOR-C paper table (pasted 2026-11-01), percent, in this column order; last value = its 20-domain mAP.
USER_COLS = ["brightness", "cloudy", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
             "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
             "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"]
USER_COLS_CN = ["亮度", "cloudy", "对比度", "散焦模糊", "弹性变换", "雾", "霜", "高斯模糊", "高斯噪声", "玻璃模糊",
                "脉冲噪声", "JPEG压缩", "运动模糊", "像素化", "饱和度", "散粒噪声", "雪", "飞溅", "散斑噪声", "缩放模糊"]
USER_TABLE = {
    "A": [24.17, 29.03, 28.16] + [None] * 17 + [None],
    "IRG": [24.42, 29.33, 28.42, 9.24, 28.66, 4.20, 6.54, 9.79, 9.12, 14.29, 10.02, 29.35, 23.27, 26.73, 26.77, 8.43,
            8.48, 20.15, 10.58, 9.69, 16.87],
    "LPLD": [24.78, 29.02, 28.54, 9.54, 28.46, 4.51, 6.40, 10.02, 8.74, 15.19, 9.28, 29.16, 14.02, 26.96, 26.67, 8.01,
             8.76, 20.12, 10.30, 9.86, 16.42],
    "SFUT": [24.20, 29.39, 28.28, 9.05, 28.36, 4.06, 6.27, 9.41, 8.96, 13.96, 9.71, 29.13, 12.84, 26.54, 26.56, 8.31,
             8.32, 19.92, 10.56, 9.30, 16.16],
    "DRU": [23.94, 29.23, 27.00, 10.08, 28.11, 4.66, 5.78, 10.40, 7.37, 15.44, 8.33, 29.05, 14.49, 26.58, 25.55, 6.07,
            8.10, 19.88, 9.21, 9.77, 15.95],
    "AASFOD": [20.85, 24.54, 23.23, 9.64, 23.05, 4.68, 5.48, 10.90, 6.31, 14.51, 6.90, 23.74, 14.66, 22.58, 21.27, 5.61,
               7.12, 16.41, 7.68, 9.75, 13.95],
    "SFYOLO": [23.43, 29.64, 26.84, 9.89, 28.57, 4.48, 6.51, 10.44, 8.63, 15.59, 9.91, 29.31, 13.96, 27.04, 25.86,
               8.19, 8.47, 20.62, 10.02, 9.77, 16.36],
    "OURS": [49.70, 46.96, 49.09, 39.03, 52.74, 54.51, 36.59, 40.33, 26.86, 35.71, 28.24, 52.91, 40.03, 48.67, 53.09,
             25.72, 32.03, 41.05, 32.94, 30.51, 40.84],
}
USER_ROW_NAMES = {"A": "Direct test", "IRG": "IRG [20]", "LPLD": "LPLD [21]", "SFUT": "SF-UT [28]", "DRU": "DRU [29]",
                  "AASFOD": "A²SFOD [30]", "SFYOLO": "SF-YOLO [31]", "OURS": "本文方法"}
OURS_CLEAN = 57.71  # 本文方法's own DIOR trainval source on clean TEST, as reported by the user (no artifact found)
# Direct-test cells the lead asked to fill (percent); asserted equal to the raw seed-42 A cells.
DIRECT_TEST = {"defocus_blur": 8.63, "elastic_transform": 28.03, "fog": 4.01, "frost": 6.32, "gaussian_blur": 9.19,
               "gaussian_noise": 9.05, "glass_blur": 13.83, "impulse_noise": 9.88, "jpeg_compression": 29.06,
               "motion_blur": 12.53, "pixelate": 26.55, "saturate": 26.37, "shot_noise": 8.14, "snow": 8.27,
               "spatter": 19.95, "speckle_noise": 10.62, "zoom_blur": 9.21}


def pp(x, nd=2):
    """Fraction -> percentage points, round half up (never truncate)."""
    return str((Decimal(repr(x)) * 100).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def p1(x):
    """Plain number -> one decimal, round half up (rPC)."""
    return str(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def r2(x):
    """Percent value -> two decimals, round half up."""
    return float(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def import_inputs(src):
    os.makedirs(INP, exist_ok=True)
    rec = {}
    if os.path.exists(os.path.join(INP, "sha256.txt")):
        rec = {ln.split()[2]: ln for ln in open(os.path.join(INP, "sha256.txt")) if ln.strip()}
    for name in IMPORT_NAMES:
        p = os.path.join(src, name)
        if not os.path.exists(p):
            continue
        orig = sha(p)
        assert orig.startswith(KNOWN_SHA.get(name, "")), (name, orig)
        text = redact(open(p, encoding="utf-8").read())
        open(os.path.join(INP, name), "w", encoding="utf-8").write(text)
        rec[name] = "%s  %s  %s\n" % (orig, sha(os.path.join(INP, name)), name)
        print("imported", name, orig[:12])
    with open(os.path.join(INP, "sha256.txt"), "w") as fh:
        fh.write("".join(rec[k] for k in sorted(rec)))


def check_inputs():
    """Every committed input must match the redacted sha recorded at import (no hand edits)."""
    for ln in open(os.path.join(INP, "sha256.txt")):
        orig, red, name = ln.split()
        assert sha(os.path.join(INP, name)) == red, name
        assert orig.startswith(KNOWN_SHA.get(name, "")), name


def fnum(s):
    return float(s) if s not in ("", None, "not_inferred") else None


def load():
    """cells[(arch, dataset, method, seed)][domain] = (AP50, mAP50-95 or None)."""
    cells = {}

    def put(arch, ds, me, seed, dom, ap, mp):
        key = (arch, ds, me, int(seed))
        assert dom not in cells.setdefault(key, {}), (key, dom)
        cells[key][dom] = (float(ap), fnum(mp))

    for r in csv.DictReader(open(RAW)):
        if r["status"] == "done":
            put("R-CNN", r["dataset"], r["method"], r["seed"], r["domain"], r["mAP50"], r["mAP50_95"])
    for r in csv.DictReader(open(os.path.join(INP, "pets_matrix.csv"))):
        assert r["eval_exit"] == "0"
        put("R-CNN", r["dataset"].upper(), "PETS", r["seed"], r["domain"], r["AP50"], r["mAP50-95"])
    for r in csv.DictReader(open(os.path.join(INP, "lods_matrix.csv"))):
        if r["row_type"] == "cell":
            assert r["eval_exit"] == "0"
            put("R-CNN", r["dataset"].upper(), "LODS", r["seed"], r["domain"], r["AP50"], r["mAP50_95"])
    for ds in ("rsar", "dior"):
        p = os.path.join(INP, "detr_%s_matrix_final.csv" % ds)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if r["status"] == "ok":
                assert r["eval_exit"] == "0" and r["seed"] == "42"
                put("DETR", ds.upper(), r["method"], r["seed"], r["domain"], r["AP50"], r["mAP50_95"])
    return cells


def mean_std(values_by_seed, per_domain_42):
    """values_by_seed: {seed: metric}. 3 seeds -> sample std over seeds; 1 seed -> sample std across domains."""
    seeds = sorted(values_by_seed)
    v = [values_by_seed[s] for s in seeds]
    if len(v) >= 2:
        return statistics.mean(v), statistics.stdev(v), "seeds"
    return v[0], statistics.stdev(per_domain_42), "domains"


def aggregate(cells, arch, ds, me):
    seeds = sorted(s for (a, d, m, s) in cells if (a, d, m) == (arch, ds, me))
    if not seeds:
        return None
    row = dict(method=me, method_name=NAMES[me], architecture=arch, dataset=ds,
               seeds=",".join(map(str, seeds)), n_seeds=len(seeds))
    for k, idx in (("AP50", 0), ("mAP50_95", 1)):
        per = {s: cells[(arch, ds, me, s)] for s in seeds}
        clean = [per[s]["clean"][idx] for s in seeds if "clean" in per[s] and per[s]["clean"][idx] is not None]
        row[k + "_clean"] = statistics.mean(clean) if len(clean) == len(seeds) else ""
        for tag, doms in (("corruptions", MPC[ds]), ("all_domains", ALL[ds])):
            complete = {s: [per[s][d][idx] for d in doms] for s in seeds
                        if all(d in per[s] and per[s][d][idx] is not None for d in doms)}
            have = sum(1 for s in seeds for d in doms if d in per[s] and per[s][d][idx] is not None)
            need = len(seeds) * len(doms)
            if have == need:
                m, sd, kind = mean_std({s: statistics.mean(v) for s, v in complete.items()}, complete[seeds[0]])
                row.update({"%s_mean_%s" % (k, tag): m, "%s_std_%s" % (k, tag): sd,
                            "%s_status_%s" % (k, tag): "ok", "std_over": kind})
            else:  # AP50 gaps are runs still pending; mAP50-95 gaps behind complete AP50 are cells without a pkl
                ap_done = idx == 1 and row["AP50_status_%s" % tag] == "ok"
                row.update({"%s_mean_%s" % (k, tag): "", "%s_std_%s" % (k, tag): "",
                            "%s_status_%s" % (k, tag): "%s (%d/%d missing)"
                            % ("not_inferred" if ap_done else "pending", need - have, need)})
        row["n_corruptions"], row["n_all_domains"] = len(MPC[ds]), len(ALL[ds])
    return row


def build_rows(cells):
    rows = []
    for arch, methods in (("R-CNN", RCNN_METHODS), ("DETR", DETR_METHODS)):
        for ds in ("RSAR", "DIOR"):
            for me in methods:
                r = aggregate(cells, arch, ds, me)
                rows.append(r or dict(method=me, method_name=NAMES[me], architecture=arch, dataset=ds,
                                      seeds="", n_seeds=0, AP50_status_corruptions="pending (all missing)"
                                      if arch == "DETR" else "missing"))
            a = next(r for r in rows if (r["architecture"], r["dataset"], r["method"]) == (arch, ds, "A"))
            for r in rows:
                if (r["architecture"], r["dataset"]) != (arch, ds):
                    continue
                for tag in ("corruptions", "all_domains"):
                    x, b = r.get("AP50_mean_" + tag, ""), a.get("AP50_mean_" + tag, "")
                    r["AP50_delta_vs_A_" + tag] = x - b if x != "" and b != "" else ""
                c = a.get("AP50_clean", "")
                r["rPC"] = r["AP50_mean_corruptions"] / c * 100 if r.get("AP50_mean_corruptions", "") != "" \
                    and c != "" else ""
    # DETR: methods that were not ported (audit §10.4) get an explicit row per dataset.
    for ds in ("RSAR", "DIOR"):
        for me in [m for m in RCNN_METHODS if m not in DETR_METHODS]:
            rows.append(dict(method=me, method_name=NAMES[me], architecture="DETR", dataset=ds, seeds="", n_seeds=0,
                             AP50_status_corruptions="not_ported (detr_portability_audit.md §10.4)"))
    # 本文方法: user-reported DIOR paper row (trainval source + weight_l=1; no artifact found).
    v = dict(zip(USER_COLS, USER_TABLE["OURS"]))
    rows.append(dict(method="OURS", method_name=NAMES["OURS"], architecture="R-CNN", dataset="DIOR",
                     seeds="unknown", n_seeds="", AP50_clean=OURS_CLEAN / 100,
                     AP50_mean_corruptions=statistics.mean(v[d] for d in DIOR_C19) / 100,
                     AP50_std_corruptions=statistics.stdev(v[d] for d in DIOR_C19) / 100,
                     AP50_status_corruptions="user_reported (DIOR trainval source + weight_l=1; not strict)",
                     std_over="domains", n_corruptions=19, n_all_domains=21,
                     mAP50_95_status_corruptions="not_inferred (no artifact)"))
    return rows


FIELDS = ["architecture", "dataset", "method", "method_name", "seeds", "n_seeds", "std_over",
          "AP50_clean", "AP50_mean_corruptions", "AP50_std_corruptions", "AP50_status_corruptions",
          "AP50_mean_all_domains", "AP50_std_all_domains", "AP50_status_all_domains",
          "AP50_delta_vs_A_corruptions", "AP50_delta_vs_A_all_domains", "rPC",
          "mAP50_95_clean", "mAP50_95_mean_corruptions", "mAP50_95_std_corruptions", "mAP50_95_status_corruptions",
          "mAP50_95_mean_all_domains", "mAP50_95_std_all_domains", "mAP50_95_status_all_domains",
          "n_corruptions", "n_all_domains"]


def write_csv(rows):
    with open(os.path.join(OUT, "architecture_generalization.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})


def cell_tex(r, best, pending_word):
    """Five columns: Clean | mPC +- std | rPC | Delta ada | mPC mAP50-95."""
    if r is None:
        return ["--"] * 5
    if r.get("AP50_mean_corruptions", "") == "":
        word = pending_word if str(r.get("AP50_status_corruptions", "")).startswith("pending") else "--"
        clean = pp(r["AP50_clean"]) if r.get("AP50_clean", "") != "" else word
        return [clean, word, word, word, word]
    m = pp(r["AP50_mean_corruptions"])
    if best is not None and abs(r["AP50_mean_corruptions"] - best) < 1e-12:
        m = r"\textbf{%s}" % m
    sd = pp(r["AP50_std_corruptions"])
    mark = r"$^{\dagger}$" if r.get("std_over") == "domains" and r["method"] != "OURS" else ""
    mpc = r"%s{\tiny$\pm$%s}%s" % (m, sd, mark)
    if r["method"] == "OURS":
        return [pp(r["AP50_clean"]) + "$^{a}$", mpc + "$^{a}$", "--", "--", "n/i"]
    mp = pp(r["mAP50_95_mean_corruptions"]) if r.get("mAP50_95_mean_corruptions", "") != "" else "n/i"
    return [pp(r["AP50_clean"]), mpc, p1(r["rPC"]), "%+.2f" % float(pp(r["AP50_delta_vs_A_corruptions"])), mp]


def tex_table(rows, arch, methods, caption, label, foot, groups):
    idx = {(r["dataset"], r["method"]): r for r in rows if r["architecture"] == arch}
    best = {}
    for ds in ("RSAR", "DIOR"):
        vals = [idx[(ds, m)]["AP50_mean_corruptions"] for m in methods if m != "OURS" and (ds, m) in idx
                and idx[(ds, m)].get("AP50_mean_corruptions", "") != ""]
        best[ds] = max(vals) if vals else None
    lines = ([XELATEX] if "OURS" in methods else []) + [
             r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3.2pt}",
             r"\renewcommand{\arraystretch}{1.1}", r"\caption{%s}" % caption, r"\label{%s}" % label,
             r"\begin{tabular}{@{}l ccccc c ccccc@{}}", r"\toprule",
             r" & \multicolumn{5}{c}{\textbf{RSAR} (7 corruptions)} && \multicolumn{5}{c}{\textbf{DIOR-C} "
             r"(19 corruptions)} \\", r"\cmidrule(lr){2-6}\cmidrule(lr){8-12}",
             r"Method & Clean & mPC\,$\uparrow$ & rPC & $\Delta$ada & mPC$_{50:95}$ && Clean & mPC\,$\uparrow$ & "
             r"rPC & $\Delta$ada & mPC$_{50:95}$ \\", r"\midrule"]
    for m in methods:
        if m in groups:
            lines.append(r"\midrule")
        name = TEX_NAMES[m] + {"C": "$^{b}$", "D": "$^{b}$", "F": "$^{b}$", "SFYOLO": "$^{c}$",
                               "PETS": "$^{d}$" if arch == "R-CNN" else ""}.get(m, "")
        rs = cell_tex(idx.get(("RSAR", m)), best["RSAR"], "pending")
        rd = cell_tex(idx.get(("DIOR", m)), best["DIOR"], "pending")
        lines.append(" & ".join([name] + rs + [""] + rd) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{\textwidth}\scriptsize", foot, r"\end{minipage}", r"\end{table*}", ""]
    return "\n".join(lines)


XELATEX = "% Contains Chinese text: compile with XeLaTeX plus xeCJK (or ctex). Not compiled in this repo."
COMMON_FOOT = (r"AP50 and mAP50-95 in \%; Clean = clean TEST; mPC = mean over the corruptions (DIOR: the 19 "
               r"official DIOR-C names, cloudy excluded); rPC = mPC / A$_{\mathrm{clean}}$ $\times$ 100; "
               r"$\Delta$ada = mPC $-$ mPC of A; mPC$_{50:95}$ = mean mAP50-95 over the same corruptions, "
               r"n/i = not\_inferred (a cell has no saved predictions.pkl; never backfilled). ")


def write_tex(rows):
    topo = {}
    for r in csv.DictReader(open(os.path.join(INP, "pets_matrix.csv"))):
        if r["topology"] != "1x32":
            k = "%s s%s" % (r["dataset"].upper(), r["seed"])
            topo[k] = topo.get(k, 0) + 1
    pets_mix = "; ".join("%s: %d" % kv for kv in sorted(topo.items()))
    cga = []
    for ds in ("RSAR", "DIOR"):
        detr = {r["method"]: r for r in rows if (r["architecture"], r["dataset"]) == ("DETR", ds)}
        if detr["A"].get("AP50_mean_corruptions", "") != "":
            cga.append("%s C %s, D %s, F %s vs A %s" % ((ds,) + tuple(pp(detr[m]["AP50_mean_corruptions"])
                                                                     for m in "CDFA")))
    cga = "; ".join(cga)
    rcnn_foot = (COMMON_FOOT +
                 r"mPC is the 3-seed (42/43/44) mean $\pm$ sample std over seeds. "
                 r"$^{\dagger}$Seeds = 1 (A is the eval-only source): $\pm$ is the std across domains. "
                 r"Bold = best mPC among the strict methods. "
                 r"$^{a}$本文方法 (as reported by the user; no evaluation artifact was found) used the DIOR "
                 r"\emph{trainval} source (its own clean 57.71) plus weight\_l=1 (source labels during "
                 r"adaptation), not the strict train-only source (A clean 29.49) and weight\_l=0 of every other "
                 r"row; its $\pm$ is across domains, its rPC/$\Delta$ada are not comparable and it is not ranked. "
                 r"$^{b}$C/D/F carry the reviewed CGA flip bug (CGA crops the unflipped image while the teacher "
                 r"boxes are flipped); not rerun by the user's decision (gate $\Delta\approx$0.002). "
                 r"$^{c}$SF-YOLO has its own 2-epoch budget (all others 1 epoch). "
                 r"$^{d}$PETS cells mix DDP 2$\times$16 and 1$\times$32 runs at the same global batch 32 "
                 r"(DDP cells: " + pets_mix + r"); DDP-vs-1-GPU gate $\Delta$=0.00455 AP50 "
                 r"(RSAR chaff s42).")
    rcnn = tex_table(rows, "R-CNN", RCNN_METHODS + ["OURS"],
                     r"Strict source-free OBB comparison, Oriented R-CNN + OrthoNet-50 (shared frozen source per "
                     r"dataset, final-iteration EMA teacher, corruption TEST).", "tab:rcnn-obb-sfod", rcnn_foot,
                     groups=("IRG", "OURS"))
    detr_foot = (COMMON_FOOT +
                 r"Separate fairness group (ARS-DETR, R-50 ImageNet init, own source per dataset); never ranked "
                 r"together with the Oriented R-CNN table. Seed 42 only, so $\pm$ is the std across domains "
                 r"(seeds = 1). Bold = best mPC in this group. Mixed hardware per cell (RTX A6000 / RTX 4090D / "
                 r"A100; per-cell gpu\_type in the matrix CSVs); a cross-hardware gate (B clean, RSAR, RTX 4090D "
                 r"vs RTX A6000) differs by $\Delta$=0.0070 AP50 and is unresolved; DIOR has no cross-hardware "
                 r"replicate, so its cross-hardware agreement is unmeasured. Upstream decode limitation: "
                 r"ARS-DETR scales both x and y by the image width, which shifts y on non-square images (about "
                 r"1.4\% of RSAR images; DIOR-R is all square); kept as in the official 31.56 reproduction. "
                 r"$^{b}$CGA calibration: on DETR, C/D/F fall far below source-only (mPC: " + cga + r"). "
                 r"Whole-run train-log statistics (same cga.py, veto\_soft config and counters on both datasets): "
                 r"on RSAR, C keeps every candidate (blended 0.416--0.752), D/F keep 0.361--0.675 (blended "
                 r"0.325--0.567) at CGA label probabilities of 0.014--0.120; on DIOR, C/D/F keep $\geq$0.989, D/F "
                 r"blend 0.96--1.00 at label probabilities of 0.009--0.066. The much lower DIOR drop rate is "
                 r"observed but its cause is unverified (pred\_prob is not logged). Read as a calibration mismatch "
                 r"between the CGA "
                 r"thresholds and DETR outputs, not a confirmed bug; run as-is with no tuning, on the CGA "
                 r"flip-fixed code. "
                 r"$^{c}$SF-YOLO has its own 2-epoch budget.")
    detr = tex_table(rows, "DETR", DETR_METHODS,
                     r"Detector-architecture generalization: the same strict source-free protocol on ARS-DETR "
                     r"(DETR-based OBB), seed 42.", "tab:detr-obb-sfod", detr_foot, groups=("C",))
    open(os.path.join(OUT, "main_table_rcnn_obb.tex"), "w").write(rcnn)
    open(os.path.join(OUT, "main_table_detr_obb.tex"), "w").write(detr)
    return rcnn, detr


def corrected_user_table(cells):
    """The user's DIOR-C table recomputed from the raw cells: A = seed 42, baselines = 3-seed mean."""
    out, fixes = {}, []
    for me, vals in USER_TABLE.items():
        if me == "OURS":
            out[me] = list(vals[:-1])
            continue
        seeds = [42] if me == "A" else [42, 43, 44]
        row = [statistics.mean(cells[("R-CNN", "DIOR", me, s)][d][0] for s in seeds) * 100 for d in USER_COLS]
        out[me] = row
        for d, user, new in zip(USER_COLS, vals, row):
            if user is not None and r2(new) != user:
                fixes.append((me, d, user, r2(new), new))
    return out, fixes


def write_user_table(cells):
    out, fixes = corrected_user_table(cells)
    hdr = "| 方法 | " + " | ".join(USER_COLS_CN) + " | mAP（20 域） | mPC（19 域，无 cloudy） |"
    md = ["# DIOR-C 对比表（订正版）", "",
          "由 `experiments/comparison/build_final_tables.py` 从 `results/paper_comparison/raw_results.csv` 重算，不要手改。",
          "Direct test = A（同一 strict 源，seed 42，不适应）；IRG / LPLD / SF-UT / DRU / A²SFOD / SF-YOLO = seed "
          "42/43/44 的 3-seed 均值；本文方法整行照用户原表（找不到评测产物；DIOR trainval 源 + weight_l=1，"
          "不是 strict train-only 源）。mAP = 20 个域（含 cloudy）的算术平均，从全精度格子算，四舍五入到两位，"
          "不截断。", "",
          hdr, "|---|" + "---:|" * 22, "| clean | " + " | ".join(["-"] * 20) + " | %.2f（本文方法的 trainval 源）"
          "；strict 源 %s | |" % (OURS_CLEAN, pp(cells[("R-CNN", "DIOR", "A", 42)]["clean"][0]))]
    for me, row in out.items():
        m20, m19 = statistics.mean(row), statistics.mean(v for d, v in zip(USER_COLS, row) if d != "cloudy")
        md.append("| %s | " % USER_ROW_NAMES[me] + " | ".join("%.2f" % r2(v) for v in row)
                  + " | %.2f | %.2f |" % (r2(m20), r2(m19)))
    md += ["", "## 订正清单", "", "| 方法 | 域 | 原表 | 订正 | 全精度（%） |", "|---|---|---:|---:|---:|"]
    md += ["| %s | %s | %.2f | %.2f | %.4f |" % (USER_ROW_NAMES[me], d, u, n, full) for me, d, u, n, full in fixes]
    md += ["", "| 方法 | 原表 mAP | 订正 mAP（20 域） | 说明 |", "|---|---:|---:|---|"]
    for me, vals in USER_TABLE.items():
        if vals[-1] is None:
            md.append("| %s | - | %.2f | 原表只有 3 格；补齐 17 格（seed 42 的 A 格子） |"
                      % (USER_ROW_NAMES[me], r2(statistics.mean(out[me]))))
            continue
        new = r2(statistics.mean(out[me]))
        if new != vals[-1]:
            naive = r2(statistics.mean(vals[:-1]))
            why = "格子订正带动" if any(f[0] == me for f in fixes) else \
                "原表均值由两位小数的格子再平均（%.2f），全精度均值为 %.4f" % (naive, statistics.mean(out[me]))
            md.append("| %s | %.2f | %.2f | %s |" % (USER_ROW_NAMES[me], vals[-1], new, why))

    # bold = best, underline = second best (次优) per column over all rows, as in the user's table
    col = {d: {m: r2(out[m][i]) for m in out} for i, d in enumerate(USER_COLS)}
    col["mAP"] = {m: r2(statistics.mean(out[m])) for m in out}
    rank = {d: sorted(set(v.values()), reverse=True) for d, v in col.items()}
    # the user's rule: a Direct-test value above every adapted baseline is likely mis-measured -> flag it
    base = [m for m in out if m not in ("A", "OURS")]
    flagged = [d for d in USER_COLS if col[d]["A"] > max(col[d][m] for m in base)]
    tex = [XELATEX, r"\begin{table*}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
           r"\caption{DIOR-C (AP50, \%). Direct test = source only (seed 42); baselines = 3-seed mean; "
           r"mAP = mean over the 20 domains. 本文方法 is as reported (DIOR trainval source + weight\_l=1, "
           r"not the strict train-only source of the other rows). Bold = best, underline = second best. "
           + "".join(r"$^{\ddagger}$Direct test %s = %s is above all six adapted baselines; it equals the raw "
                     r"seed-42 cell (%s, no pkl kept) and is flagged for a re-check by the rule that a "
                     r"source-only value above every baseline is likely mis-measured. "
                     % (d.replace("_", r"\_"), "%.2f" % col[d]["A"], repr(cells[("R-CNN", "DIOR", "A", 42)][d][0]))
                     for d in flagged) + "}",
           r"\label{tab:dior-c}", r"\begin{tabular}{@{}l" + "c" * 21 + r"@{}}", r"\toprule",
           "Method & " + " & ".join(USER_COLS_CN) + r" & mAP \\", r"\midrule",
           r"clean & " + " & ".join(["--"] * 20) + r" & %.2f \\" % OURS_CLEAN]
    def mark(d, me):
        s = "%.2f" % col[d][me]
        s = r"\textbf{%s}" % s if col[d][me] == rank[d][0] else r"\underline{%s}" % s \
            if col[d][me] == rank[d][1] else s
        return s + (r"$^{\ddagger}$" if me == "A" and d in flagged else "")

    for me in out:
        tex.append(USER_ROW_NAMES[me].replace("²", "$^2$") + " & "
                   + " & ".join(mark(d, me) for d in USER_COLS + ["mAP"]) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    open(os.path.join(OUT, "dior_c_table_corrected.tex"), "w").write("\n".join(tex))
    md += [""] + ["- Direct test %s = %.2f 高于全部 6 个适应后的基线。它与 raw 里 seed 42 的格子一致（%s，没有保留 pkl）；"
                  "按用户自己的规则（source-only 某格高于所有基线，多半是测错了）标记为待复查。"
                  % (d, col[d]["A"], repr(cells[("R-CNN", "DIOR", "A", 42)][d][0])) for d in flagged]
    open(os.path.join(OUT, "dior_c_table_corrected.md"), "w").write("\n".join(md) + "\n")
    return out, fixes, flagged


def self_check(cells, rows, rcnn_tex, detr_tex, user_out, fixes):
    idx = {(r["architecture"], r["dataset"], r["method"]): r for r in rows}
    # 1. Cell counts: PETS / LODS 87 each, DETR RSAR 72 and DIOR 189.
    for me in ("PETS", "LODS"):
        n = sum(len(v) for (a, d, m, s), v in cells.items() if (a, m) == ("R-CNN", me))
        assert n == 87, (me, n)
    for ds, want in (("RSAR", 72), ("DIOR", 189)):  # 9 methods x 8 / 21 domains, seed 42
        n = sum(len(v) for (a, d, m, s), v in cells.items() if (a, d) == ("DETR", ds))
        assert n == want, (ds, n)
    for (a, d, m, s), v in cells.items():
        assert set(v) <= set(ALL[d]), (a, d, m, s)
    # 2. Recomputed means equal the producers' own summaries.
    for r in csv.DictReader(open(os.path.join(INP, "lods_matrix.csv"))):
        if r["row_type"] == "domain_3seed":
            ap = [cells[("R-CNN", r["dataset"].upper(), "LODS", s)][r["domain"]][0] for s in (42, 43, 44)]
            assert abs(statistics.mean(ap) - float(r["AP50_mean"])) < 1e-9
            assert abs(statistics.stdev(ap) - float(r["AP50_std"])) < 1e-9
        if r["row_type"] == "overall_domain_means" and r["dataset"] != "all":
            ds = r["dataset"].upper()
            assert abs(idx[("R-CNN", ds, "LODS")]["AP50_mean_all_domains"] - float(r["AP50_mean"])) < 1e-9
            assert abs(idx[("R-CNN", ds, "LODS")]["mAP50_95_mean_all_domains"] - float(r["mAP50_95_mean"])) < 1e-9
    for r in csv.DictReader(open(os.path.join(INP, "pets_matrix_summary.csv"))):
        if r["kind"] != "mean_std_3seed":
            continue
        k = 0 if r["metric"] == "AP50" else 1
        v = [cells[("R-CNN", r["dataset"].upper(), "PETS", s)][r["domain_or_ALL"]][k] for s in (42, 43, 44)]
        assert "%.4f" % statistics.mean(v) == r["mean"] and "%.4f" % statistics.stdev(v) == r["std"], r
    for ds in ("rsar", "dior"):
        p = os.path.join(INP, "detr_%s_matrix_summary.csv" % ds)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            k = 0 if r["metric"] == "AP50" else 1
            doms = [d for d in ALL[ds.upper()] if not r["kind"].startswith(("7-", "20-")) or d != "clean"]
            v = [cells[("DETR", ds.upper(), r["method"], 42)][d][k] for d in doms
                 if d in cells[("DETR", ds.upper(), r["method"], 42)]]
            assert len(v) == int(r["n"]), r
            assert "%.4f" % statistics.mean(v) == r["mean"] and "%.4f" % statistics.stdev(v) == r["std"], r
    # 3. The 12 original R-CNN methods: seed-42 mPC equals build_summary.py's method_mpc.csv.
    for r in csv.DictReader(open(os.path.join(OUT, "method_mpc.csv"))):
        c = cells[("R-CNN", r["dataset"], r["method"], 42)]
        assert abs(statistics.mean(c[d][0] for d in MPC[r["dataset"]]) - float(r["mPC_s42"])) < 1e-9, r
        if r["mPC_3seed_mean"] != "N/A":
            assert abs(idx[("R-CNN", r["dataset"], r["method"])]["AP50_mean_corruptions"]
                       - float(r["mPC_3seed_mean"])) < 1e-9, r
    # 4. Every number printed in the two main tables is re-derived from the cells by an independent path.
    for arch, tex, methods in (("R-CNN", rcnn_tex, RCNN_METHODS), ("DETR", detr_tex, DETR_METHODS)):
        body = {ln.split(" & ")[0]: ln for ln in tex.splitlines() if " & " in ln and ln.endswith(r"\\")}
        for m in methods:
            line = next(v for k, v in body.items() if k.startswith(TEX_NAMES[m]))
            cols = [c.strip() for c in line[:-2].split(" & ")]
            for ds, part in (("RSAR", cols[1:6]), ("DIOR", cols[7:12])):
                seeds = sorted(s for (a, d, mm, s) in cells if (a, d, mm) == (arch, ds, m))
                per = [statistics.mean(cells[(arch, ds, m, s)][d][0] for d in MPC[ds]) for s in seeds
                       if all(d in cells[(arch, ds, m, s)] for d in MPC[ds])]
                if len(per) != len(seeds) or not seeds:
                    assert part[1] in ("pending", "--"), (m, ds, part)
                    continue
                num = [float(x) for x in re.findall(r"-?\d+\.\d+", part[1])]
                assert num[0] == float(pp(statistics.mean(per))), (m, ds, part)
                if len(per) > 1:
                    assert num[1] == float(pp(statistics.stdev(per))), (m, ds, part)
                aclean = cells[(arch, ds, "A", 42)]["clean"][0]
                cl = statistics.mean(cells[(arch, ds, m, s)]["clean"][0] for s in seeds)
                assert float(part[0]) == float(pp(cl)), (m, ds, part)
                assert part[2] == p1(statistics.mean(per) / aclean * 100), (m, ds, part)
                amp = statistics.mean(cells[(arch, ds, "A", 42)][d][0] for d in MPC[ds])
                assert float(part[3]) == float(pp(statistics.mean(per) - amp)), (m, ds, part)
                mps = [cells[(arch, ds, m, s)][d][1] for s in seeds for d in MPC[ds]]
                if all(x is not None for x in mps):
                    assert float(part[4]) == float(pp(statistics.mean(mps))), (m, ds, part)
                else:
                    assert part[4] == "n/i", (m, ds, part)
    # 5. Re-inferred mAP50-95 pkls: every AP50 gap > 1e-4 is flagged in raw_results.csv, and none exceeds 0.0011.
    for r in csv.DictReader(open(RAW)):
        if r["mAP50_95_AP50"]:
            d = abs(float(r["mAP50_95_AP50"]) - float(r["mAP50"]))
            assert d <= 0.0011, r
            assert (d > 1e-4) == r["notes"].startswith("mAP50-95 from eval-only re-inference pkl"), r
    # 6. The user's DIOR-C table: the fixes the lead listed, and nothing else.
    assert user_out["IRG"][USER_COLS.index("motion_blur")] and r2(user_out["IRG"][12]) == 13.27
    assert r2(statistics.mean(user_out["IRG"])) == 16.37
    a = user_out["A"]
    assert r2(statistics.mean(a)) == 16.05 and r2(statistics.mean(v for d, v in zip(USER_COLS, a) if d != "cloudy")) \
        == 15.37
    for d, v in DIRECT_TEST.items():
        assert r2(a[USER_COLS.index(d)]) == v, d
    small = [(m, d) for m, d, *_ in fixes if (m, d) != ("IRG", "motion_blur")]
    assert len(small) == 6, small
    print("self-check OK: %d R-CNN + %d DETR cells; %d user-table fixes"
          % (sum(len(v) for k, v in cells.items() if k[0] == "R-CNN"),
             sum(len(v) for k, v in cells.items() if k[0] == "DETR"), len(fixes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-inputs", metavar="DIR")
    args = ap.parse_args()
    if args.import_inputs:
        import_inputs(args.import_inputs)
    check_inputs()
    cells = load()
    rows = build_rows(cells)
    write_csv(rows)
    rcnn, detr = write_tex(rows)
    user_out, fixes, flagged = write_user_table(cells)
    assert flagged == ["speckle_noise"], flagged  # the one known case; anything new needs a look
    assert cells[("R-CNN", "DIOR", "A", 42)]["speckle_noise"] == (0.106221, None)
    self_check(cells, rows, rcnn, detr, user_out, fixes)


if __name__ == "__main__":
    main()
