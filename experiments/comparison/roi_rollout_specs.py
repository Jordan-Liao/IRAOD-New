"""Prepare finite approved case specs; operator owns assignment, execution and budgets."""

import argparse
import json
from pathlib import Path


EXCLUDED_PORT = {
    "DIOR/brightness/42/IRG/ema",  # Completed pre-rollout pilot.
    "RSAR/noise_suppression/44/IRG/ema",
    "RSAR/noise_suppression/44/LPLD/ema",
    "RSAR/noise_suppression/44/SFUT/ema",
    "RSAR/point_target/44/IRG/ema",
    "RSAR/point_target/44/LPLD/ema",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Existing case spec or host-profile JSON")
    parser.add_argument("--port-plan", action="append", default=[])
    parser.add_argument("--bf-input")
    parser.add_argument("--out", required=True)
    parser.add_argument("--case-root", default="/home/zechuan/iraod_artifacts/comparison/xaf_student_quant_20260908/roi_local_native_rollout_342_20260913")
    args = parser.parse_args()
    raw_profile = json.loads(Path(args.profile).read_text())
    profile = raw_profile.get("profile", raw_profile)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    prepared = []

    def emit(kind, key, fields):
        spec = {
            "scope_id": "iraod-roi-local-native-rollout-342-20260913",
            "authorized_at_utc": "2026-09-13T19:26:26.560Z", "user_quote": "我现在授权了",
            "kind": kind, "canonical_key": key,
            "case_root": str(Path(args.case_root) / str(profile["host"]) / key),
            "profile": profile, **fields,
            "budget": {"gpu_starts": 2, "allocated_gpu_wall_seconds": 1800, "automatic_retry": False},
        }
        path = output / (key.replace("/", "__") + ".json")
        if path.exists():
            if json.loads(path.read_text()) != spec:
                raise ValueError(f"Preserve existing different case spec: {path}")
        else:
            with path.open("x") as stream:
                json.dump(spec, stream, indent=2)
        prepared.append({"canonical_key": key, "kind": kind, "spec": str(path)})

    for filename in args.port_plan:
        plan = json.loads(Path(filename).read_text())
        for run in plan["runs"]:
            if run["method"] not in ("IRG", "LPLD", "SFUT") or run["role"] != "ema":
                continue
            key = "/".join(str(run[k]) for k in ("dataset", "domain", "seed", "method", "role"))
            if key in EXCLUDED_PORT:
                continue
            emit("port", key, {"source_plan": str(Path(filename).resolve()), "run_id": run["run_id"]})
    if args.bf_input:
        source = Path(args.bf_input).resolve()
        for row in json.loads(source.read_text())["records"]:
            if row["method"] not in "BCDEF" or row["seed"] not in (43, 44):
                raise ValueError("BF input is outside the approved remaining scope")
            emit("bf", row["canonical_key"], {"input_specs": str(source)})
    print(json.dumps({
        "status": "FINITE_CASE_SPECS_READY", "prepared": prepared,
        "count": len(prepared),
        "execution_policy": "pT atomically assigns pending/unclaimed keys and skips completed/failed ledger keys; this generator never launches or retries.",
    }, indent=2))


if __name__ == "__main__":
    main()
