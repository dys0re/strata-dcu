"""Generate an external bench copy that remaps existing threads after model load; weights/pages stay put."""
from pathlib import Path
import hashlib,json
src=Path('/root/strata-dcu/Strata/tools/bench_dflash2_native.py');out=Path('/root/private_data/strata-dcu/cpu-slowdown-20261025/affinity_bench.py')
s=src.read_text();s=s.replace('import argparse,', 'import argparse,',1)
s='import sys\nsys.path.insert(0,"/root/strata-dcu/Strata/tools")\n'+s
needle='\nrecords=[]\n';assert s.count(needle)==1
s=s.replace(needle,'''# [PROFILE] Same engine process, same resident weights; change CPU masks only between requests.
cases=[('ok','Reply with exactly OK.',16,'OK',0),('arithmetic','What is 2+2? Answer only the number.',16,'4',0),('capital','中国的首都是哪座城市？只回答城市名称。',16,'北京',0)]
for phase in ['base0','away0','base1']:
 for repeat in range(2):cases.append((phase+'-'+str(repeat),story,64,None,0))
original_affinity={}; affinity_records=[]
records=[]
''')
needle='   prompt_ids,request=gen(prompt,n,temperature,';assert s.count(needle)==1
s=s.replace(needle,'''   # [PROFILE] Preserve the already allocated expert pages; no rebinding or migration.
   offset=16 if name.startswith('away') else 0
   for task in Path(f'/proc/{p.pid}/task').iterdir():
    try:
     tid=int(task.name)
     if tid not in original_affinity:original_affinity[tid]=set(os.sched_getaffinity(tid))
     old=original_affinity[tid]
     if len(old)==1 and min(old)<13:os.sched_setaffinity(tid,{min(old)+offset})
    except ProcessLookupError:pass
   masks={t.name:sorted(os.sched_getaffinity(int(t.name))) for t in Path(f'/proc/{p.pid}/task').iterdir()}
   affinity_records.append(dict(case=name,offset=offset,threads=masks,numa_maps=(Path(f'/proc/{p.pid}')/'numa_maps').read_text()))
   (out/'affinity-switches.json').write_text(json.dumps(affinity_records,indent=2)+'\\n')
   prompt_ids,request=gen(prompt,n,temperature,''')
out.write_text(s)
(out.parent/'affinity-source.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),probe_sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))
