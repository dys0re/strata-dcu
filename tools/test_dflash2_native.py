"""Real-checkpoint numerical/cache witness for the native DFlash2 implementation."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def prepare(a):
    import torch
    from dflash2_worker import Dflash2Worker, DraftCache
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["STRATA_DFLASH_GRAPH"] = "0"
    os.environ["STRATA_DFLASH_ADAPTIVE"] = "0"
    w = Dflash2Worker(a.checkpoint, a.target)
    rng = np.random.default_rng(1729)
    features = rng.standard_normal((2100, 12800), dtype=np.float32)
    operations = []
    cases = []
    cache = DraftCache()
    pos = 0
    pending = []
    def reset():
        nonlocal cache, pos, pending
        operations.append("reset")
        cache, pos, pending = DraftCache(), 0, []
    def append(n):
        nonlocal pos
        data = features[pos:pos+n]
        path = out / f"features-{pos}-{n}.f32"
        data.tofile(path)
        operations.append(f"append {path.resolve()} {n} {pos}")
        pending.append(data)
        pos += n
    def draft(k, name):
        nonlocal pending
        anchor = 248077
        operations.append(f"draft {pos} {anchor} {k} {name}")
        new = np.concatenate(pending, axis=0) if pending else np.empty((0,12800),dtype=np.float32)
        with torch.inference_mode():
            f = torch.from_numpy(new).to("cuda", dtype=torch.bfloat16)[None]
            ids = torch.tensor([anchor]+[248077]*k)
            noise = torch.nn.functional.embedding(ids, w.embedding).to("cuda")[None]
            positions = torch.arange(pos-len(new), pos+k+1, device="cuda")[None]
            hidden = w.model(target_hidden=f, noise_embedding=noise, position_ids=positions,
                             past_key_values=cache, use_cache=True)[:, 1:]
            tokens, _, _ = w.model.propose(hidden,torch.tensor([anchor],device="cuda"),
                lambda h: torch.nn.functional.linear(h, w.head),0.)
            cache.crop(pos)
            hidden[0].float().cpu().numpy().tofile(out / (name+".reference.f32"))
            token_ids = tokens[0].tolist()
        pending = []
        cases.append(dict(name=name, pos=pos, k=k, reference_ids=token_ids))
    reset()
    append(4)
    draft(3,"short3")
    draft(3,"repeat3")
    append(2)
    draft(1,"increment1")
    append(1)
    draft(7,"full7")
    # Same final context, recomputed in one block: detects stale rejected/noise rows.
    reset()
    append(7)
    draft(7,"recompute7")
    reset()
    append(2045)
    draft(7,"window-edge")
    append(9)
    draft(7,"ring-wrap")
    append(17)
    draft(3,"after-wrap")
    reset()
    append(4)
    draft(3,"reset-short3")
    (out/"script.txt").write_text("\n".join(operations)+"\n")
    (out/"reference.json").write_text(json.dumps(dict(seed=1729,cases=cases),indent=2)+"\n")
    print("REFERENCE",len(cases),flush=True)


def check(a):
    root = Path(a.out)
    ref = json.loads((root/"reference.json").read_text())
    report = dict(seed=ref["seed"], modes={})
    for mode in ["eager","graph"]:
        out = root/mode
        out.mkdir(exist_ok=True)
        cmd = [a.binary,a.pack,a.gguf,str(root/"script.txt"),"4096",mode,str(out)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        (out/"stdout.log").write_text(result.stdout)
        (out/"stderr.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:])
        records = []
        for c in ref["cases"]:
            name = c["name"]
            expected = np.fromfile(root/(name+".reference.f32"),np.float32)
            actual = np.fromfile(out/(name+".f32"),np.float32)
            assert actual.shape==expected.shape and np.isfinite(actual).all()
            rel = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            ids = [int(v) for v in (out/(name+".ids")).read_text().split()]
            records.append(dict(**c,ids=ids,relative_l2=rel,max_abs=float(np.max(np.abs(actual-expected))),
                                tokens_equal=ids==c["reference_ids"]))
            # Draft logits/IDs need not match another backend. These distances are
            # diagnostics, not the functional acceptance gate (target QA is that gate).
        byname = {r["name"]:r for r in records}
        for x,y in [("short3","repeat3"),("short3","reset-short3"),("full7","recompute7")]:
            assert np.array_equal(np.fromfile(out/(x+".f32"),np.float32),
                                  np.fromfile(out/(y+".f32"),np.float32)), (mode,x,y)
            assert byname[x]["ids"]==byname[y]["ids"]
        report["modes"][mode] = records
        print("WITNESS",mode,"max_relative_l2",max(r["relative_l2"] for r in records),
              "matching_tokens",sum(sum(x==y for x,y in zip(r["ids"],r["reference_ids"])) for r in records),flush=True)
    for c in ref["cases"]:
        name = c["name"]
        assert (root/"eager"/(name+".ids")).read_bytes()==(root/"graph"/(name+".ids")).read_bytes()
        assert (root/"eager"/(name+".f32")).read_bytes()==(root/"graph"/(name+".f32")).read_bytes()
    report["eager_graph_equal"] = True
    report["repeat_reset_incremental_equal"] = True
    report["reference_numerics_are_diagnostic"] = True
    (root/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("PASS DFLASH2_NATIVE",flush=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=["prepare","check"])
    ap.add_argument("--checkpoint")
    ap.add_argument("--target")
    ap.add_argument("--binary")
    ap.add_argument("--pack")
    ap.add_argument("--gguf")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    (prepare if args.mode=="prepare" else check)(args)
