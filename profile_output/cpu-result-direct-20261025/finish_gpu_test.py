"""Finish expanded position coverage after normal benchmarks, without changing the tested engine."""
from pathlib import Path
import hashlib,json,os,subprocess
R=Path('/root/private_data/strata-dcu/cpu-result-direct-20261025')
S=Path('/root/strata-dcu/Strata');B=Path('/root/private_data/strata-dcu/p1/build')
assert len(json.loads((R/'bench-status.json').read_text()))==4
assert all(x['returncode']==0 for x in json.loads((R/'bench-status.json').read_text()))
assert 'Used Memory (MiB): 2' in subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
binary=hashlib.sha256((R/'strata-candidate').read_bytes()).hexdigest()
status=[]
for label,cmd in [('expanded-test-build',['cmake','--build',str(B),'--target','mapped_moe_parity','-j','2']),('expanded-test',['ctest','--test-dir',str(B),'-R','^mapped_moe_parity$','--output-on-failure'])]:
 with (R/(label+'.log')).open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
 status.append(dict(label=label,command=cmd,returncode=rc));(R/'extra-gates.json').write_text(json.dumps(status,indent=2)+'\n')
 assert rc==0
(R/'expanded-test-full.log').write_bytes((B/'Testing/Temporary/LastTest.log').read_bytes())
assert 'MAPPED_MOE_PARITY cases=960 failures=0' in (R/'expanded-test-full.log').read_text()
assert hashlib.sha256((R/'strata-candidate').read_bytes()).hexdigest()==binary
before=json.loads((R/'source-manifest.json').read_text())
(R/'initial-source-manifest.json').write_text(json.dumps(before,indent=2)+'\n')
after={f:hashlib.sha256((S/f).read_bytes()).hexdigest() for f in before}
changed=[f for f in before if before[f]!=after[f]]
assert changed==['src/kernels/mapped_moe_parity.cpp'],changed
(R/'source-manifest.json').write_text(json.dumps(after,indent=2)+'\n')
(R/'test-expansion.json').write_text(json.dumps(dict(changed=changed,engine_unchanged=binary,coverage='all 1..8 positions; previously1,2,3,5,8'),indent=2)+'\n')
print('EXPANDED GPU TEST PASS; ENGINE UNCHANGED',flush=True)
