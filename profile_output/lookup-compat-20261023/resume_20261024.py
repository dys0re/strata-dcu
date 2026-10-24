import json,os,signal,subprocess,time
from pathlib import Path
R=Path('/root/private_data/strata-dcu'); O=R/'lookup-compat-20261023'; M=R/'upstream-options-20261023'
env=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
status=[]
def run(name,mode,extra,*,dma=False,deadline=900):
 out=(M if dma else O)/name
 assert not out.exists(),out
 cmd=['python3','-u','/root/strata-dcu/Strata/tools/bench_dflash2_native.py','--engine',str(R/'joint-draft-20261023/strata-final' if dma else O/'strata'),'--mode',mode,'--width','5' if mode=='adaptive' else '4' if mode=='ipc' else '8','--graph','--draft-adaptive','--timing-only','--adapt-every','0','--pcie-frac','-1' if dma else '0','--pcie-mode','dma' if dma else 'auto','--resident-pin' if dma else '--no-resident-pin','--suffix-draft','0' if dma else '3','--out',str(out)]
 if not dma:cmd+=['--lookup-checks']
 cmd+=extra
 print('START',name,flush=True)
 with out.with_suffix('.log').open('w') as f:
  p=subprocess.Popen(cmd,env=dict(env,STRATA_HC_SPLIT='0' if dma else '2'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=deadline)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   rc=124
 status.append(dict(name=name,returncode=rc));(O/'resume-status.json').write_text(json.dumps(status,indent=2))
 print('FINISHED',name,rc,flush=True)
 return rc
jobs=[('native-lookup-long-resume','native',['--repeat','0','--context','4096','--long-checks']),('joint-lookup-long','joint',['--repeat','0','--context','4096','--long-checks']),('joint-lookup-reject','joint',['--qa-only','--corrupt','1']),('ipc-lookup','ipc',['--qa-only'])]
for name,mode,args in jobs:
 if run(name,mode,args):raise SystemExit(1)
if not run('dma-smoke','adaptive',['--smoke'],dma=True,deadline=300):
 if run('dma','adaptive',['--repeat','3','--qa-after-bench'],dma=True,deadline=1200):raise SystemExit(1)
print('REMAINING CHECKS FINISHED',flush=True)
