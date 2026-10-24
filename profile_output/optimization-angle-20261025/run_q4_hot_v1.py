"""Build isolated CPU probes only after the lifecycle engine has actually exited."""
from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parent
active=json.loads((R/'active.json').read_text())
assert not Path(f"/proc/{active['pid']}").exists(),'engine still alive'
assert json.loads((R/'trace-current/complete.json').read_text())['returncode']==0
S=Path('/root/strata-dcu/Strata');B=Path('/root/private_data/strata-dcu/p1/build')
subprocess.run(['python3',str(R/'prepare_q4_probe.py')],check=True)
cmd=['g++','-O3','-DNDEBUG','-no-pie','-std=c++20','-mavx512f','-mavx512bw','-mavx512vl','-mavx512dq','-mavx512vnni','-mavx2','-mf16c','-mfma',
     '-I'+str(S/'include'),'-I/root/strata-dcu/llama.cpp/ggml/include','-I/root/strata-dcu/llama.cpp/ggml/src',
     str(R/'q4_probe.cpp'),str(R/'q4_unsigned_probe.cpp'),str(B/'libstrata_kernels_cpu.a'),
     str(B/'ggml/src/libggml-cpu.a'),str(B/'ggml/src/libggml-base.a'),'-lpthread','-lm','-o',str(R/'q4-probe')]
with (R/'q4-build.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
def snapshot():
    return {int(s.split()[0][3:]):list(map(int,s.split()[1:])) for s in Path('/proc/stat').read_text().splitlines() if s.startswith('cpu') and s.split()[0]!='cpu'}
a=snapshot();time.sleep(3);b=snapshot();busy={}
for c in os.sched_getaffinity(0):
    d=[v-u for u,v in zip(a[c],b[c])];busy[c]=(sum(d)-d[3]-d[4])/max(1,sum(d))
core=min(busy,key=lambda c:(busy[c],c))
info=dict(build_command=cmd,compiler=subprocess.check_output(['g++','--version'],text=True),cpu=core,busy=busy,initial_cpu=a,final_cpu=b,
          scope='Single core, synthetic normally quantized Q4/Q8, hot weights; primitive timings, not engine TPS.')
(R/'q4-probe-command.json').write_text(json.dumps(info,indent=2)+'\n')
with (R/'q4-results.jsonl').open('w') as log:
    rc=subprocess.run(['taskset','-c',str(core),str(R/'q4-probe')],stdout=log,stderr=subprocess.STDOUT).returncode
(R/'q4-status.json').write_text(json.dumps(dict(returncode=rc))+'\n')
assert rc==0,rc
print('Q4_PROBE_COMPLETE',core,flush=True)
