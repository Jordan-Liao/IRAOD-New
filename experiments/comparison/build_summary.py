"""Build the RSAR + DIOR comparison tables from per-host eval scans.

    python3 experiments/comparison/build_summary.py --scans DIR   # DIR/scan_<host>.tsv from scan_eval_cells.py
    python3 experiments/comparison/build_summary.py               # rebuild tables from raw_results.csv only

Cell = (dataset, domain, method, seed), final-EMA TEST mAP50. A cell counts only when its EMA
iteration equals the seed-42 iteration of the same (dataset, method); anything else is
`protocol_mismatch` (e.g. DIOR s43/44 trained 2 epochs) and stays out of every mean.
mAP50 == 0 is a collapsed run (`collapsed`); a non-zero copy of the same cell always wins. AASFOD counts
only runs of the standard recipe (per-seed tsd.json + work/stages.json, flagged by scan_eval_cells.py).
"""
import argparse
import collections
import csv
import glob
import math
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "paper_comparison")
CELLS = os.path.join(OUT, "raw_results.csv")

RSAR = ["clean", "chaff", "point_target", "gaussian_white_noise", "noise_suppression",
        "smart_suppression", "am_noise_horizontal", "am_noise_vertical"]
DIOR_C19 = ["brightness", "contrast", "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
            "defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur", "snow", "frost",
            "fog", "spatter", "elastic_transform", "pixelate", "jpeg_compression", "saturate"]
DOMAINS = {"RSAR": RSAR, "DIOR": ["clean"] + DIOR_C19 + ["cloudy"]}
# mPC domains: RSAR = its 7 corruptions; DIOR = the 19 official DIOR-C names (cloudy is extra).
MPC = {"RSAR": RSAR[1:], "DIOR": DIOR_C19}
METHODS = ["A", "B", "C", "D", "E", "F", "IRG", "LPLD", "SFUT", "DRU", "AASFOD", "SFYOLO"]
NAMES = {"A": "Source Only", "B": "ST / Mean Teacher", "C": "CLIP-CGA", "D": "SARCLIP-CGA",
         "E": "VLST", "F": "CGA+VLST", "IRG": "IRG-OBB", "LPLD": "LPLD-OBB", "SFUT": "SFUT-OBB",
         "DRU": "DRU-OBB", "AASFOD": "AASFOD-OBB", "SFYOLO": "SFYOLO-OBB"}
SEEDS = [42, 43, 44]
EXCLUDE = ("oracle", "b_reg", "f_deletion", "roi_", "LoRA", "B_REG", "F_text", "F_veto", "probe", "smoke",
           "qualitative", "vis_subset", "collector", "report-readiness", "tsne", "clean_val", "/val", "noop_runs")
SHARED_HOST = "gpu-67"  # holds /mnt/shared, the relay target every worker pushes to


def classify(scan_dir):
    all_domains = sorted(set(RSAR + DOMAINS["DIOR"]), key=len, reverse=True)
    rows = []
    for f in glob.glob(os.path.join(scan_dir, "scan_*.tsv")):
        host = os.path.basename(f)[5:-4]
        for line in open(f):
            path, status, m, std = (line.rstrip("\n").split("\t") + ["", "", ""])[:4]
            if not m or any(x in path for x in EXCLUDE):
                continue
            if "eval_exit=" in status and "eval_exit=0" not in status:
                continue
            segs = path.split("/")
            leaf = segs[-1]
            if "student" in leaf or "role=student" in status or leaf in ("student", "student.joint920"):
                continue
            domain = next((c for s in reversed(segs) for c in all_domains
                           if s == c or s.startswith("eval_" + c)), None)
            method = seed = None
            for s in segs:
                if s in METHODS:
                    method = s
                mm = re.match(r"^(%s)_(?:d17_)?s(4[234])$" % "|".join(METHODS), s)
                if mm:
                    method, seed = mm.group(1), int(mm.group(2))
                ms = re.match(r"^seed_?(4[234])$", s)
                if ms:
                    seed = int(ms.group(1))
            if not domain or not method:
                continue
            if domain in DOMAINS["DIOR"] and domain != "clean":
                ds = "DIOR"
            elif domain in RSAR[1:]:
                ds = "RSAR"
            else:
                low = path.lower()
                ds = "DIOR" if ("dior" in low or "d17" in low) else "RSAR" if "rsar" in low else None
            if not ds:
                continue
            it = re.search(r"iter_(\d+)_ema", status + path)
            rows.append(dict(dataset=ds, domain=domain, method=method, seed=seed or 42, host=host, std=std == "std2stage",
                             mAP50=float(m), ema_iter=it and int(it.group(1)), path=path))
    by_cell = collections.defaultdict(list)
    for r in rows:
        if not math.isnan(r["mAP50"]):
            by_cell[(r["dataset"], r["domain"], r["method"], r["seed"])].append(r)
    # Reference iteration = most common seed-42 EMA iteration per (dataset, method).
    ref = {}
    for (ds, _, me, s), v in by_cell.items():
        if s == 42:
            ref.setdefault((ds, me), collections.Counter()).update(r["ema_iter"] for r in v if r["ema_iter"])
    ref = {k: c.most_common(1)[0][0] for k, c in ref.items() if c}
    out = []
    for ds, doms in DOMAINS.items():
        for dom in doms:
            for me in METHODS:
                for s in ([42] if me == "A" else SEEDS):
                    v = by_cell.get((ds, dom, me, s))
                    row = dict(dataset=ds, domain=dom, method=me, seed=s, status="missing", mAP50="",
                               ema_iter="", ref_iter=ref.get((ds, me), ""), host="", path="")
                    if v:
                        want = ref.get((ds, me))
                        if me == "AASFOD":  # only per-seed TSD + two-stage runs are the standard recipe
                            v = [r for r in v if r["std"]] or v
                        best = sorted(v, key=lambda r: (r["mAP50"] == 0, me != "AASFOD" and r["ema_iter"] != want,
                                                        r["host"] != SHARED_HOST))[0]
                        ok = me == "A" or best["ema_iter"] in (want, None)
                        if best["mAP50"] == 0:
                            st = "collapsed"
                        elif me == "AASFOD":
                            st = "done" if best["std"] else "protocol_mismatch"
                        else:
                            st = "done" if ok else "protocol_mismatch"
                        row.update(status=st, mAP50=round(best["mAP50"], 6),
                                   ema_iter=best["ema_iter"] or "", host=best["host"], path=best["path"])
                    out.append(row)
    with open(CELLS, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


def build():
    cells = list(csv.DictReader(open(CELLS)))
    val = {(c["dataset"], c["domain"], c["method"], int(c["seed"])): float(c["mAP50"])
           for c in cells if c["status"] == "done"}

    def a(ds, dom):
        return val.get((ds, dom, "A", 42))

    # per-domain seed-42 table
    with open(os.path.join(OUT, "per_corruption_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "corruption", "method", "seed", "mAP50", "delta_ada", "status"])
        for c in cells:
            if int(c["seed"]) != 42:
                continue
            m, base = val.get((c["dataset"], c["domain"], c["method"], 42)), a(c["dataset"], c["domain"])
            w.writerow([c["dataset"], c["domain"], c["method"], 42, c["mAP50"],
                        "" if m is None or base is None else round(m - base, 6), c["status"]])

    # method summary: seed-42 mPC/rPC; 3-seed mean±std only where every mPC cell of all 3 seeds is done
    summary = []
    for ds in DOMAINS:
        a_clean = a(ds, "clean")
        for me in METHODS:
            per_seed = {}
            for s in ([42] if me == "A" else SEEDS):
                v = [val.get((ds, d, me, s)) for d in MPC[ds]]
                if all(x is not None for x in v):
                    per_seed[s] = statistics.mean(v)
            mpc = per_seed.get(42)
            a_mpc = statistics.mean(val[(ds, d, "A", 42)] for d in MPC[ds])
            three = [per_seed[s] for s in SEEDS if s in per_seed] if me != "A" else []
            summary.append(dict(
                dataset=ds, method=me, name=NAMES[me],
                clean=val.get((ds, "clean", me, 42), ""),
                mPC_s42=mpc if mpc is not None else "N/A",
                rPC_s42=mpc / a_clean * 100 if mpc is not None else "N/A",
                mean_delta_ada_s42=mpc - a_mpc if mpc is not None else "N/A",
                n_corruptions=len(MPC[ds]),
                mPC_3seed_mean=statistics.mean(three) if len(three) == 3 else "N/A",
                mPC_3seed_std=statistics.stdev(three) if len(three) == 3 else "N/A",
                seeds_complete=",".join(str(s) for s in SEEDS if s in per_seed)))
    with open(os.path.join(OUT, "method_mpc.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    def f3(x):
        return "--" if x in ("", "N/A", None) else "%.3f" % float(x)

    tex = []
    for ds in DOMAINS:
        rows = [r for r in summary if r["dataset"] == ds]
        best = max(float(r["mPC_s42"]) for r in rows if r["mPC_s42"] != "N/A")
        tex += ["%% %s seed=42 final-EMA TEST mAP50. mPC over %d corruptions; rPC = mPC / A_clean. Bold = best mPC."
                % (ds, len(MPC[ds])),
                r"\begin{tabular}{llcccc" + ("c" if ds == "RSAR" else "") + "}", r"\toprule",
                r"Method & & clean & mPC & rPC (\%) & $\Delta$ada" + (r" & mPC 3-seed" if ds == "RSAR" else "")
                + r" \\", r"\midrule"]
        for r in rows:
            mpc = f3(r["mPC_s42"])
            if r["mPC_s42"] != "N/A" and float(r["mPC_s42"]) == best:
                mpc = r"\textbf{%s}" % mpc
            cells_ = [r["method"], NAMES[r["method"]], f3(r["clean"]), mpc,
                      "--" if r["rPC_s42"] == "N/A" else "%.1f" % r["rPC_s42"],
                      "--" if r["mean_delta_ada_s42"] == "N/A" else "%+.3f" % r["mean_delta_ada_s42"]]
            if ds == "RSAR":
                cells_.append("--" if r["mPC_3seed_mean"] == "N/A"
                              else r"%.3f$\pm$%.3f" % (r["mPC_3seed_mean"], r["mPC_3seed_std"]))
            tex.append(" & ".join(cells_) + r" \\")
            if r["method"] == "F":
                tex.append(r"\midrule")
        tex += [r"\bottomrule", r"\end{tabular}", ""]
    open(os.path.join(OUT, "main_table.tex"), "w").write("\n".join(tex))

    status = collections.Counter((c["dataset"], c["status"]) for c in cells)
    md = ["# RSAR + DIOR 严格 Source-Free OBB 对比汇总（final-EMA TEST mAP50）", "",
          "由 `experiments/comparison/build_summary.py` 从 `raw_results.csv`（逐格状态）生成；不要手改。", "",
          "mPC：RSAR 取 7 种 corruption，DIOR 取 19 种官方 DIOR-C（cloudy 只列在逐格表里）；rPC = mPC / A_clean；"
          "Δada = mPC − A 的 mPC。主表只用 seed 42；3-seed 均值只有三个 seed 全部通过协议检查时才给。", ""]
    for ds in DOMAINS:
        md += ["## %s（A_clean = %.4f）" % (ds, a(ds, "clean")), "",
               "| 方法 | clean | mPC s42 | rPC | Δada | mPC 3-seed | 已完成的 seed |", "|---|---:|---:|---:|---:|---:|---|"]
        for r in (r for r in summary if r["dataset"] == ds):
            md.append("| %s %s | %s | %s | %s | %s | %s | %s |" % (
                r["method"], NAMES[r["method"]], f3(r["clean"]), f3(r["mPC_s42"]),
                "N/A" if r["rPC_s42"] == "N/A" else "%.1f%%" % r["rPC_s42"],
                "N/A" if r["mean_delta_ada_s42"] == "N/A" else "%+.4f" % r["mean_delta_ada_s42"],
                "N/A" if r["mPC_3seed_mean"] == "N/A" else "%.4f±%.4f" % (r["mPC_3seed_mean"], r["mPC_3seed_std"]),
                r["seeds_complete"] or "-"))
        md.append("")
    kinds = ["done", "protocol_mismatch", "collapsed", "missing"]
    md += ["## 格子状态", "", "| 数据集 | " + " | ".join(kinds) + " |", "|---|" + "---:|" * len(kinds)]
    for ds in DOMAINS:
        md.append("| %s | " % ds + " | ".join(str(status[(ds, k)]) for k in kinds) + " |")
    todo = [c for c in cells if c["status"] in ("collapsed", "missing")]
    md += ["", "### collapsed / missing 格子", "", "| 数据集 | 方法 | seed | 域 | 状态 |", "|---|---|---|---|---|"]
    grouped = collections.defaultdict(list)
    for c in todo:
        grouped[(c["dataset"], c["method"], c["seed"], c["status"])].append(c["domain"])
    for (ds, me, s, st), doms in sorted(grouped.items()):
        md.append("| %s | %s | %s | %s | %s |" % (ds, me, s, ", ".join(doms), st))
    md += ["", "- `protocol_mismatch`：EMA 迭代数与该方法的 seed-42 不同（DIOR-17 s43/44 跑了 2 epoch；E/F 为单卡 bs16；"
           "DIOR AASFOD s43/44 为 110 iter；RSAR DRU s43/44 为 1055 iter）。这些格子不进入任何均值。",
           "- `collapsed`：mAP50 = 0（训练发散）；同一格有非零结果时总是取非零结果。",
           "- AASFOD 只计标准做法（每个 seed 先做 TSD 切分，再做对齐 + FNS 两阶段）；其他做法的旧结果记为 protocol_mismatch。",
           "- 单 seed 主表不做显著性判断；逐格数值见 `per_corruption_summary.csv`，路径与机器见 `raw_results.csv`。", ""]
    open(os.path.join(OUT, "result_summary_cn.md"), "w").write("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans")
    args = ap.parse_args()
    if args.scans:
        classify(args.scans)
    build()
