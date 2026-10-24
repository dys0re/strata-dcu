"""Validate the existing DMA alternative after the one-GPU compatibility suite."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/root/private_data/strata-dcu');OUT=ROOT/'upstream-options-20261023'
while True:
 try:
  state=Path('/proc/50745/stat').read_text().split(') ')[1].split()[0]
  if state=='Z':break
 except FileNotFoundError:break
 time.sleep(5)
assert 'SUITE PASS' in (ROOT/'lookup-compat-20261023/suite.log').read_text(), 'compatibility suite did not pass; leave GPU for diagnosis'
env=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
status=[]
for label,extra,deadline in [('dma-smoke',['--smoke'],300),('dma',['--repeat','3','--qa-after-bench'],1200)]:
 cmd=['python3','-u','/root/strata-dcu/Strata/tools/bench_dflash2_native.py',
      '--engine',str(ROOT/'joint-draft-20261023/strata-final'),'--mode','adaptive','--width','5',
      '--adapt-every','0','--pcie-frac','-1','--pcie-mode','dma','--resident-pin','--timing-only','--out',str(OUT/label)]+extra
 print('START',label,flush=True)
 with (OUT/(label+'.log')).open('w') as f:
  p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=p.wait(timeout=deadline)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   rc=124
 status.append(dict(label=label,returncode=rc));(OUT/'dma-status.json').write_text(json.dumps(status,indent=2))
 print('FINISHED',label,rc,flush=True)
 if rc:raise SystemExit(rc)
print('DMA PASS',flush=True)
