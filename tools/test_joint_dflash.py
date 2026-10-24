"""Real-weight seeded-prefix/cache witness. Reuses existing seeded feature fixtures."""
import argparse,json,subprocess,resource
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
ap=argparse.ArgumentParser(__doc__)
for key in ['binary','pack','gguf','reference','out']:ap.add_argument('--'+key,required=True)
a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
ref=Path(a.reference);sources={}
for line in (ref/'script.txt').read_text().splitlines():
 p=line.split()
 if p and p[0]=='append':sources[(int(p[3]),int(p[2]))]=p[1]
ops=[];cases=[]
def reset():ops.append('reset')
def append(pos,n):
 path=Path(sources[(pos,n)]);assert path.stat().st_size==n*12800*4
 ops.append(f'append {path} {n} {pos}')
def draft(pos,k,name,prefix=None):
 if prefix is None:ops.append(f'draft {pos} 553 {k} {name}')
 else:ops.append(f'seed {pos} 553 {k} '+(','.join(map(str,prefix)) or '-')+f' {name}')
 cases.append(dict(name=name,k=k,prefix=prefix or []))
reset();append(0,4)
draft(4,7,'plain');draft(4,7,'zero',[])
draft(4,7,'prefix2',[393,85]);ops.append('discard');draft(4,7,'repeat2',[393,85]);draft(4,7,'prefix3',[17,42,99])
append(4,2);draft(6,7,'increment',[393,85]);append(6,1);draft(7,7,'full',[393,85])
reset();append(0,7);draft(7,7,'recompute',[393,85])
reset();append(0,4);draft(4,2,'short',[393]);draft(4,7,'all',[393,85,17,42,99,100,101])
reset();append(0,2045);draft(2045,7,'edge',[393,85]);append(2045,9)
draft(2054,7,'wrap',[393,85]);draft(2054,7,'wrap-repeat',[393,85]);append(2054,17)
draft(2071,3,'after-wrap',[393,85]);reset();append(0,4);draft(4,7,'reset2',[393,85])
(out/'script.txt').write_text('\n'.join(ops)+'\n');(out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
report={'cases':len(cases),'modes':{}}
for mode in ['eager','graph']:
 d=out/mode;d.mkdir()
 cmd=[a.binary,a.pack,a.gguf,str(out/'script.txt'),'4096',mode,str(d)]
 r=subprocess.run(cmd,capture_output=True,text=True)
 (d/'stdout.log').write_text(r.stdout);(d/'stderr.log').write_text(r.stderr)
 assert r.returncode==0,r.stderr[-3000:]
 h={};ids={}
 for c in cases:
  n=c['name'];h[n]=np.fromfile(d/(n+'.f32'),np.float32);ids[n]=[int(x) for x in (d/(n+'.ids')).read_text().split()]
  assert len(ids[n])==c['k'] and h[n].size==c['k']*2560 and np.isfinite(h[n]).all()
  assert ids[n][:len(c['prefix'])]==c['prefix'],(mode,n,ids[n],c['prefix'])
 for x,y in [('plain','zero'),('prefix2','repeat2'),('prefix2','reset2'),('full','recompute'),('wrap','wrap-repeat')]:
  assert np.array_equal(h[x],h[y]) and ids[x]==ids[y],(mode,x,y)
 assert not np.array_equal(h['prefix2'],h['plain']), 'prefix was not supplied to the network'
 assert not np.array_equal(h['prefix2'],h['prefix3']), 'changing prefix did not change network state'
 report['modes'][mode]={'prefix_locked':True,'prefix_conditions_network':True,'repeat_reset_recompute':True,'ids':ids}
 print('WITNESS JOINT',mode,len(cases),flush=True)
for c in cases:
 for suffix in ['.f32','.ids']:assert (out/'eager'/(c['name']+suffix)).read_bytes()==(out/'graph'/(c['name']+suffix)).read_bytes()
report['eager_graph_equal']=True
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
# Bad prefixes must be rejected before any invalid embedding access.
for name,k,prefix in [('bad-id',3,'-1'),('too-long',1,'393,85')]:
 script=out/(name+'.txt');script.write_text('reset\n'+ops[1]+f'\nseed 4 553 {k} {prefix} rejected\n')
 r=subprocess.run([a.binary,a.pack,a.gguf,str(script),'4096','eager',str(out)],capture_output=True,text=True)
 assert r.returncode!=0 and 'invalid' in r.stderr,(name,r.returncode,r.stderr)
 (out/(name+'.log')).write_text(r.stderr)
report['invalid_prefix_rejected']=2
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS JOINT_SEEDED_CACHE',flush=True)
