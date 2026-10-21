import json, os, pathlib, shutil, subprocess, sys, time
root=pathlib.Path("/root/strata-dcu")
out=root/"autoround"
logs=root/"logs"
cases=[("arithmetic","What is 2+2? Answer only the number.","4"),("chinese","中国的首都是哪座城市？只回答城市名称。","北京")]
reports=[]
ok=json.loads((logs/"autoround-ok.json").read_text())
assert ok["text"].strip()=="OK" and ok["finish_reason"]=="eos" and ok["engine_exit"]==0,ok
reports.append({"case":"ok","passed":True,**ok})
for name,prompt,expected in cases:
    cmd=[sys.executable,str(root/"Strata/tools/autoround_pack.py"),"run","--model","/root/private_data/Qwen3.8-F","--out",str(out),"--llama",str(root/"llama.cpp"),"--engine",str(root/"build-gfx936/strata"),"--max-new","16","--prompt",prompt]
    t=time.monotonic()
    with (logs/f"autoround-{name}.log").open("w") as f:
        p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        print(name,"FAILED",p.returncode,flush=True)
        print((logs/f"autoround-{name}.log").read_text()[-3000:],flush=True)
        raise SystemExit(p.returncode)
    r=json.loads((out/"generation.json").read_text())
    shutil.copy2(out/"generation.json",logs/f"autoround-{name}.json")
    shutil.copy2(out/"engine.stderr.log",logs/f"autoround-{name}.stderr.log")
    passed=r["text"].strip().rstrip("。.!")==expected and r["finish_reason"]=="eos"
    r.update(case=name,passed=passed,wall_seconds=round(time.monotonic()-t,2))
    reports.append(r)
    print(json.dumps(r,ensure_ascii=False),flush=True)
    (logs/"autoround-smoke-results.json").write_text(json.dumps(reports,ensure_ascii=False,indent=2)+"\n")
    if not passed:raise SystemExit("unexpected model answer; inspect report")
print("ALL AUTOROUND SMOKE CASES PASSED",flush=True)
