import subprocess,shlex,json,os,hashlib
from pathlib import Path
b=Path('/root/private_data/strata-dcu/p1/build');o=Path('/root/private_data/strata-dcu/hc-diagnosis-20261023')
lines=subprocess.check_output(['ninja','-C',str(b),'-t','commands','fused_gr_bench'],text=True).splitlines()
base=shlex.split(next(x for x in lines if ' -c ' in x and x.endswith('/src/kernels/cuda/fused_gr.cu')))
link=shlex.split(lines[-1]);link=link[2:-2] if link[:2]==[':', '&&'] else link
commands={}
for tag,extra in [('default',[]),('no-contract',['-ffp-contract=off']),('norm-no-contract',[])]:
 cmd=[];i=0
 while i<len(base):
  v=base[i]
  if v in ['-MT','-MF']:i+=2;continue
  if v=='-MD':i+=1;continue
  if v=='-o':cmd+=['-o',str(o/(tag+'.o'))];i+=2;continue
  if v=='-c':cmd+=['-c',str(o/('hc_norm_probe.cu' if tag=='norm-no-contract' else 'hc_probe.cu'))];i+=2;continue
  cmd.append(v);i+=1
 cmd+=['-I/root/strata-dcu/Strata/src/kernels/cuda']+extra
 l=[str(o/(tag+'.o')) if x.endswith('fused_gr_bench.cpp.o') else x for x in link]
 l[l.index('-o')+1]=str(o/('hc-'+tag))
 commands[tag]={'compile':cmd,'link':l}
 with (o/(tag+'-build.log')).open('w') as log:
  subprocess.run(cmd,cwd=b,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(l,cwd=b,stdout=log,stderr=subprocess.STDOUT,check=True)
 for fast in ['0','1']:
  name=tag+'-fast'+fast
  with (o/(name+'.csv')).open('w') as stdout,(o/(name+'.log')).open('w') as stderr:
   subprocess.run([str(o/('hc-'+tag))],env=dict(os.environ,STRATA_GR_FAST=fast),stdout=stdout,stderr=stderr,check=True)
  print('DONE',name,flush=True)
(o/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
print('PASS HC_PROBE_BUILD_RUN',flush=True)
