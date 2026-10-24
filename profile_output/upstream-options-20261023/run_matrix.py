import os,subprocess,json,time
from pathlib import Path
out=Path('/root/private_data/strata-dcu/upstream-options-20261023');out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
status=[]
for label,adapt,pcie in [('baseline',0,0),('cache',4,0),('pcie',0,-1),('both',4,-1)]:
 cmd=['python3','-u','/root/strata-dcu/Strata/tools/bench_dflash2_native.py','--engine','/root/private_data/strata-dcu/joint-draft-20261023/strata-final','--mode','adaptive','--width','5','--repeat','3','--qa-after-bench','--timing-only','--adapt-every',str(adapt),'--pcie-frac',str(pcie),'--out',str(out/label)]
 print('START',label,flush=True)
 with (out/(label+'.log')).open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 status.append(dict(label=label,returncode=r.returncode));(out/'status.json').write_text(json.dumps(status,indent=2))
 print('FINISHED',label,r.returncode,flush=True)
 if r.returncode:break
