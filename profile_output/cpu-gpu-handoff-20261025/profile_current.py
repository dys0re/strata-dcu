"""External current-engine lifecycle probe; no production changes or TPS claim."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

ROOT=Path('/root/private_data/strata-dcu/cpu-gpu-handoff-20261025')
PREV=Path('/root/private_data/strata-dcu/idle-core-retest-20261025/v2/idle-selected')
sys.path.insert(0,str(PREV.parent))
from idle_cores import choose_and_pin

out=ROOT/'trace-current'
out.mkdir(exist_ok=False)
assert 'Used Memory (MiB): 2' in subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
cfg=json.loads((PREV/'command.json').read_text())
cmd=cfg['command']
cmd[0]='/root/strata-dcu/build-gfx936/strata'
sha=hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest()
assert sha=='4fc8304ec17c7e6b0df4a843f7f60ebe21934e6a7e88004a358e6b2356edb8cd'
env=dict(os.environ,**cfg['overrides'])
for k,v in cfg['hc_environment'].items():
    if v is None: env.pop(k,None)
    else: env[k]=v
for k in ['STRATA_VERIFY_PROFILE','STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_SPIN_US']:
    env.pop(k,None)
flags=dict(STRATA_CPU_Q4_INPUT_BIAS='1',STRATA_POOL_TRACE='1',STRATA_VERIFY_TRACE='1',STRATA_VERIFY_TRACE_FULL='1',STRATA_DECODE_TIMING='1')
env.update(flags)
(out/'command.json').write_text(json.dumps(dict(command=cmd,flags=cfg['overrides'],trace_flags=flags,engine_sha256=sha,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
records=[]
with (out/'stderr.log').open('w') as err,(out/'stdout.log').open('w') as log:
    p=subprocess.Popen(cmd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
    (ROOT/'active.json').write_text(json.dumps(dict(pid=p.pid,driver_pid=os.getpid(),out=str(out))))
    def line():
        s=p.stdout.readline()
        if not s: raise RuntimeError(f'EOF rc={p.poll()}')
        log.write(s);log.flush()
        if s.startswith('ERR '): raise RuntimeError(s)
        return s.strip()
    try:
        while not line().startswith('READY '): pass
        choose_and_pin(p.pid,os.sched_getaffinity(0),13,out/'idle-core-selection.json')
        for case,file,n in [('warmup','warmup',128),('story','story-0',64),('chinese','chinese-0',64),('code','code-0',64),('recovery','ok',16)]:
            req=(PREV/(file+'.request.txt')).read_text()
            req=req.replace(req.split(' ',2)[0]+' '+req.split(' ',2)[1]+' ',f'GEN {n} ',1)
            proc=Path(f'/proc/{p.pid}')
            before=(proc/'numa_maps').read_text()
            (out/(case+'.numa-before.txt')).write_text(before)
            def threads():
                data={}
                for task in (proc/'task').iterdir():
                    try:data[task.name]=dict(schedstat=list(map(int,(task/'schedstat').read_text().split())),affinity=sorted(os.sched_getaffinity(int(task.name))))
                    except (ProcessLookupError,FileNotFoundError):pass
                return data
            t0=threads();start=time.monotonic_ns()
            p.stdin.write(req);p.stdin.flush();ids=[]
            while True:
                s=line()
                if s.startswith('T '): ids.append(int(s.split()[1]))
                if s.startswith('DONE '):done=s;break
            wall=time.monotonic_ns()-start;t1=threads()
            fields=done.split()
            assert int(fields[1])==len(ids) and int(fields[8])==0 and fields[5] in ['length','stop'],done
            records.append(dict(case=case,request=req,done=done,generated_ids=ids,wall_ns=wall,threads_before=t0,threads_after=t1))
            (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
            (out/(case+'.numa-after.txt')).write_text((proc/'numa_maps').read_text())
            print('REQUEST_COMPLETE',case,done,flush=True)
        p.stdin.write('QUIT\n');p.stdin.flush();p.wait(timeout=30)
        assert p.returncode==0
    finally:
        if p.poll() is None:p.terminate();p.wait(timeout=30)
(out/'complete.json').write_text(json.dumps(dict(returncode=p.returncode,qa='finite generation counters only; not full functional QA or throughput evidence'))+'\n')
print('PROFILE_COMPLETE',flush=True)
