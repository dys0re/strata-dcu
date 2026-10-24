import os,json,time,subprocess,hashlib
from pathlib import Path
root=Path('/root/private_data/strata-dcu/lookup-compat-20261023')
out=Path('/root/private_data/strata-dcu/thread-util-20261024');out.mkdir(exist_ok=False)
base=json.loads((root/'hc-staged/command.json').read_text())
cmd=base['command'];cmd[0]=str(root/'strata-final')
env=os.environ.copy();env.update(base['overrides']);env.update(STRATA_HC_SPLIT='2',STRATA_DECODE_TIMING='1')
(out/'command.json').write_text(json.dumps({'command':cmd,'overrides':base['overrides'],'extra':{'STRATA_HC_SPLIT':'2','STRATA_DECODE_TIMING':'1'},'sha256':hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest()},indent=2))
err=(out/'stderr.log').open('w');log=(out/'stdout.log').open('w')
p=subprocess.Popen(cmd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
records=[]
def snap():
 t=time.monotonic_ns();threads={}
 for d in Path(f'/proc/{p.pid}/task').iterdir():
  try:
   st=d.joinpath('stat').read_text().rsplit(')',1)[1].split();ss=list(map(int,d.joinpath('schedstat').read_text().split()))
   status=d.joinpath('status').read_text().splitlines()
   threads[d.name]={'runtime_ns':ss[0],'runqueue_ns':ss[1],'slices':ss[2],'utime':int(st[11]),'stime':int(st[12]),'cpu':int(st[36]),'affinity':next(x.split(':',1)[1].strip() for x in status if x.startswith('Cpus_allowed_list:'))}
  except (FileNotFoundError,ProcessLookupError):pass
 return {'time_ns':t,'threads':threads,'cgroup_cpu':Path('/sys/fs/cgroup/cpu.stat').read_text()}
def line():
 s=p.stdout.readline()
 if not s:raise RuntimeError(f'EOF {p.poll()}')
 log.write(s);log.flush()
 if s.startswith('ERR '):raise RuntimeError(s)
 return s.strip()
try:
 while not line().startswith('READY '):pass
 print('READY',p.pid,flush=True)
 for name in ['warmup']+[f'{case}-{i}' for i in range(3) for case in ['story','chinese','code']]+['ok']:
  request=(root/'hc-staged'/f'{name}.request.txt').read_text();r={'case':name,'pid':p.pid,'before':snap(),'request':request};ids=[]
  p.stdin.write(request);p.stdin.flush()
  while True:
   s=line()
   if s.startswith('REUSED '):r['decode_start']=snap()
   if s.startswith('T '):
    ids.append(int(s.split()[1]))
    if 'first_token' not in r:r['first_token']=snap()
   if s.startswith('DONE '):r['after']=snap();r['done']=s;break
  r['generated_ids']=ids;records.append(r)
  (out/'results.json').write_text(json.dumps(records,indent=2))
  a=r['decode_start'];b=r['after'];dt=(b['time_ns']-a['time_ns'])/1e9
  tid=str(p.pid);v=(b['threads'][tid]['runtime_ns']-a['threads'][tid]['runtime_ns'])/1e9
  print(name,'host_cpu_pct',round(v/dt*100,2),'decode_ms',r['done'].split()[4],flush=True)
 p.stdin.write('QUIT\n');p.stdin.flush();p.wait(timeout=30)
finally:
 if p.poll() is None:p.terminate();p.wait(timeout=30)
 err.close();log.close()
print('EXIT',p.returncode,flush=True)
