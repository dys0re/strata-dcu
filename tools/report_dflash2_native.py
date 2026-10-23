"""Summarize functional DFlash2 QA first, followed by the measured speed smoke."""
import argparse
import json
from pathlib import Path


def summarize(root):
    def read(run):
        assert (root / (run + ".log")).read_text().rstrip().endswith("PASS"), run
        return json.loads((root / run / "results.json").read_text())
    reference = {r["case"]: r for r in read("qa-reference")}
    qa = {}
    for run in ["qa-full", "qa-adaptive"]:
        records = read(run)
        matches = {r["case"]: r["generated_ids"] == reference[r["case"]]["generated_ids"] for r in records}
        protocol = json.loads((root / run / "protocol.json").read_text())
        assert protocol["cancel"].split()[5] == "cancel"
        assert protocol["recovery"].split()[5] == "stop"
        qa[run] = dict(requests=len(records), output_tokens=sum(len(r["generated_ids"]) for r in records),
                       exact_target_matches=matches, protocol=protocol)
    long = read("qa-long")
    assert [r["text"].strip() for r in long] == ["ORBIT", "OK"]
    assert len(long[0]["prompt_ids"]) == 2100
    perf = read("bench-native")
    usages = {run: json.loads((root / run / "weight-usage.json").read_text())
              for run in ["qa-full", "qa-reference", "qa-adaptive", "qa-long", "bench-native"]}
    for run, u in usages.items():
        assert u["weights"] <= u["limit"] == u["total"] // 2
        assert u["weights"] == sum(u[k] for k in ["pack", "dense", "head", "experts", "mtp", "external_draft", "native_draft"])+u.get("staging",0)
    witness = json.loads((root / "witness" / "report.json").read_text())
    assert witness["eager_graph_equal"] and witness["repeat_reset_incremental_equal"]
    golden = root.parent / "dflash2" / "quality-final" / "results.json"
    comparison = {}
    if golden.exists():
        old = {r["case"]: r for r in json.loads(golden.read_text())}
        comparison = {r["case"]: r["generated_ids"] == old[r["case"]]["generated_ids"]
                      for r in perf if r["case"] in old}
    summary = dict(qa=qa, long_context=dict(prompt_tokens=2100,answers=[r["text"].strip() for r in long],
                  request_recovery=True), weights=usages,
                  speed_smoke={r["case"]:r["decode_tps"] for r in perf if r["case"]!="warmup"},
                  speed_repeats_per_workload=1, speed_output_equivalence=comparison,
                  eager_graph_equal=True, cache_reset_incremental_equal=True,
                  pytorch_hidden_relative_l2_max=max(r["relative_l2"] for r in witness["modes"]["graph"]),
                  pytorch_draft_parity_claimed=False, token_identity_informational=True)
    (root / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path)
    summarize(ap.parse_args().root)
