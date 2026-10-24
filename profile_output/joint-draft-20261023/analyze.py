"""Summarize recorded joint runs; no synthetic inference measurements."""
from pathlib import Path
import json
import re
import statistics

root = Path(__file__).resolve().parent
summary = {}
for name in ["qa-prob-v2", "qa-reject", "long", "bench"]:
    p = root / name
    rows = json.loads((p / "results.json").read_text())
    status = json.loads((p / "qa-status.json").read_text())
    usage = json.loads((p / "weight-usage.json").read_text())
    assert status["functional_assertions_passed"] and status["engine_rc"] == 0
    assert usage["weights"] <= usage["limit"] == usage["total"] // 2
    assert usage["experts"] == 8300 * 2764800
    assert usage["mtp"] == 1368138752 and usage["native_draft"] == 1206131200
    counters = [dict(re.findall(r"(\w+)=([0-9.]+)", line))
                for line in (p / "stderr.log").read_text().splitlines()
                if line.startswith("strata joint draft:")]
    sums = {key: sum(int(x[key]) for x in counters) for key in
            ["windows", "mtp_offered", "mtp_accepted", "dflash_offered", "dflash_accepted", "prob_windows"]}
    summary[name] = dict(cases=len(rows), tokens=sum(len(r["generated_ids"]) for r in rows),
                         status=status, weights=usage, counters=sums)
    if name == "qa-prob-v2":
        assert len(rows) == 14 and sums["prob_windows"] > 0
        assert sums["mtp_accepted"] > 0 and sums["dflash_accepted"] > 0
    if name == "qa-reject":
        assert len(rows) == 14
        # Forced wrong proposals exercise repeated reject/catch-up, not quality loss.
        assert sums["mtp_offered"] > 0 and sums["dflash_offered"] > 0
        assert sums["mtp_accepted"] == 0 and sums["dflash_accepted"] == 0
    if name == "long":
        assert rows[0]["text"].strip() == "ORBIT" and len(rows[0]["prompt_ids"]) == 2100
        assert rows[1]["text"].strip() == "OK"
    if name == "bench":
        work = {}
        for task in ["story", "chinese", "code"]:
            values = [r["decode_tps"] for r in rows if re.fullmatch(task + r"-\d+", r["case"])]
            assert len(values) == 3
            work[task] = dict(values=values, median_tps=statistics.median(values))
        assert sums["mtp_accepted"] > 0 and sums["dflash_accepted"] > 0
        summary[name]["workloads"] = work
        timings = []
        for line in (p / "stderr.log").read_text().splitlines():
            if line.startswith("strata joint draft:"):
                timings.append(dict(re.findall(r"(\w+)=([0-9.]+)", line)))
        summary[name]["preparation_ms"] = {k: sum(float(x[k]) for x in timings)
                                             for k in ["append_ms", "mtp_ms", "dflash_ms"]}

reference = json.loads((root.parent / "q4-multitoken-20261023/summary.json").read_text())
summary["same_day_previous_build"] = {
    name: reference[name]["workloads"] for name in ["mtp-on", "native-both"]}
summary["comparison_scope"] = (
    "Same-host BS=1, three 256-token repetitions per task per engine start. "
    "Prior optimized baselines are from the same day but an earlier binary; "
    "MTP uses 8500 GPU experts, DFlash/joint 8300. All GPU weights <=50%. "
    "No statistical-significance claim.")
summary["cache"] = json.loads((root / "witness/report.json").read_text())
assert summary["cache"]["cases"] == 15 and summary["cache"]["eager_graph_equal"]
assert summary["cache"]["invalid_prefix_rejected"] == 2
summary["legacy_cache"] = json.loads((root / "legacy-cache/report.json").read_text())
assert summary["legacy_cache"]["byte_equal_previous"]
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print({k: round(v["median_tps"], 3) for k, v in summary["bench"]["workloads"].items()})
print("PASS JOINT_RESULT_AUDIT")
