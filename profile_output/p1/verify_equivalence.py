"""Strict model-output and first-token-logit checks, separate from performance."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
base=json.loads((root/'baseline-checks/results.json').read_text())
summary={}
for name in ['final-checks','fixed8-checks']:
 rows=json.loads((root/name/'results.json').read_text())
 assert len(rows)==len(base)==11
 for a,b in zip(base,rows):
  assert a['case']==b['case'] and a['prompt_ids']==b['prompt_ids']
  assert a['generated_ids']==b['generated_ids'],(name,a['case'])
 by_case={r['case']:r for r in rows}
 assert by_case['one_token']['done'].split()[5]=='length' and len(by_case['one_token']['generated_ids'])==1
 assert len(by_case['context_tail']['prompt_ids'])==501 and len(by_case['context_tail']['generated_ids'])==3
 assert by_case['context_tail']['done'].split()[5]=='length'
 assert by_case['sample-0']['generated_ids']==by_case['sample-1']['generated_ids']
 assert {k:by_case[k]['text'].strip() for k in ['ok','arithmetic','capital']}=={'ok':'OK','arithmetic':'4','capital':'北京'}
 cfg=json.loads((root/name/'command.json').read_text())
 assert cfg['overrides']['STRATA_BF16_Q8_COLS']=='4' and cfg['checks']
 if name=='final-checks':assert '--mtp-adaptive' in cfg['command']
 else:
  cmd=cfg['command'];assert cmd[cmd.index('--spec')+1]=='8' and cmd[cmd.index('--mtp-max-t')+1]=='8'
 baseline_logits=sorted((root/'baseline-checks').glob('logits.f32.*'))
 logits=sorted((root/name).glob('logits.f32.*'))
 assert len(logits)==len(baseline_logits)==13
 for p in baseline_logits:
  a=np.fromfile(p,np.float32);b=np.fromfile(root/name/p.name,np.float32)
  assert a.size==b.size==248320 and np.isfinite(a).all() and np.isfinite(b).all()
  assert np.array_equal(a.view('u4'),b.view('u4')),(name,p.name)
 protocol=json.loads((root/name/'protocol.json').read_text())
 assert protocol['cancel'].split()[5]=='cancel' and protocol['recovery'].split()[5]=='stop'
 assert protocol['seed_repeat'] and protocol['one_token_limit'] and protocol['context_tail']
 summary[name]={'matched_requests':len(rows),'first_token_logit_arrays_bit_identical':len(logits),'stop_recovery_seed_limits':True}
(root/'equivalence-results.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS: final adaptive and fixed T=8 each match 11 P0 requests and 13 first-token-logit arrays; STOP, recovery, seed and limits pass')
