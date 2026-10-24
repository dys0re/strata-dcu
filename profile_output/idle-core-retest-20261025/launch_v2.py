"""Wait for the already-live predecessor, then run the corrected selector matrix serially."""
from pathlib import Path
import time,subprocess,json,os
R=Path('/root/private_data/strata-dcu/idle-core-retest-20261025')
pid=99180
while Path(f'/proc/{pid}/stat').exists():
 try:
  if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(1)
qa=json.loads((R/'idle-bf16-proxy/qa-status.json').read_text());assert qa['functional_assertions_passed'] and qa['engine_rc']==0
assert (R/'idle-bf16-proxy.log').read_text().rstrip().endswith('PASS')
print('PREDECESSOR_DONE',qa,flush=True)
rc=subprocess.run(['python3','-u',str(R/'v2/run.py')]).returncode
(R/'v2/job-status.json').write_text(json.dumps(dict(returncode=rc))+'\n')
raise SystemExit(rc)
