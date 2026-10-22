"""Reject the historical 19,000-slot configuration under the new weight budget."""
import json,os,subprocess
from pathlib import Path
out=Path('/root/private_data/strata-dcu/weight50')
meta=json.loads((out/'run1/command.json').read_text())
cmd=meta['command'];cmd[cmd.index('--expert-cache')+1]='19000'
env=dict(os.environ,**meta['overrides'])
(out/'over-budget-command.json').write_text(json.dumps(dict(command=cmd,overrides=meta['overrides']),indent=2)+'\n')
with (out/'over-budget.stderr.log').open('w') as err,(out/'over-budget.stdout.log').open('w') as stdout:
 p=subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=stdout,stderr=err,env=env,timeout=180)
errors=(out/'over-budget.stderr.log').read_text()
assert p.returncode==2,p.returncode
assert 'requested weights exceed the weight VRAM budget' in errors
assert 'resident cache complement ready' not in errors
assert 'READY ' not in (out/'over-budget.stdout.log').read_text()
(out/'over-budget-result.json').write_text(json.dumps(dict(exit_code=p.returncode,rejected_before_generation=True,rejected_before_resident_ram_allocation=True,diagnostic=[s for s in errors.splitlines() if 'weight VRAM budget' in s]),indent=2)+'\n')
print('PASS: 19000 slots rejected before generation and host expert allocation')
