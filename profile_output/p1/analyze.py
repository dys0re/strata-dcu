"""Verify primary adaptive-MTP comparisons and summarize their measured samples."""
import csv,json,math,statistics
from pathlib import Path

root=Path(__file__).resolve().parent
runs={name:json.loads((root/name/'results.json').read_text()) for name in ['a0','b0','b1','a1']}
commands={name:json.loads((root/name/'command.json').read_text()) for name in runs}
reference=runs['a0'];ref={r['case']:r for r in reference}
matched=0
for name,rows in runs.items():
 cfg=commands[name];cmd=cfg['command']
 assert '--mtp-adaptive' in cmd and '--mtp' in cmd
 assert cmd[1:]==commands['a0']['command'][1:]
 assert cmd[cmd.index('--expert-cache')+1]=='19000'
 assert not cfg['profile'] and not cfg['dump_logits']
 assert cfg['overrides']['STRATA_BF16_Q8_COLS']==('4' if name.startswith('b') else '0')
 assert cfg['overrides']['STRATA_QFUSE']=='0' and cfg['overrides']['STRATA_VERIFY_QDEDUP']=='0'
 for r in rows:
  expected=ref[r['case']]
  assert r['prompt_ids']==expected['prompt_ids'] and r['generated_ids']==expected['generated_ids'],(name,r['case'])
  assert math.isfinite(r['decode_tps']) and r['decode_tps']>0
  matched+=1
summary={'baseline':'089c5b4','mtp':'adaptive in both arms','cache_slots':19000,
 'bs':1,'max_context':512,'performance_instrumentation':False,'matched_requests':matched,'performance':{}}
for case in ['story','chinese','code']:
 groups={arm:[r['decode_tps'] for name,rows in runs.items() if name.startswith(arm) for r in rows if r['case'].startswith(case+'-')] for arm in ['a','b']}
 for samples in groups.values():assert len(samples)==6 and all(math.isfinite(s) for s in samples)
 stats={arm:{'samples':s,'median':statistics.median(s),'min':min(s),'max':max(s)} for arm,s in groups.items()}
 stats['gain_percent']=(stats['b']['median']/stats['a']['median']-1)*100
 summary['performance'][case]=stats
for name in ['a0','b0']:
 p=json.loads((root/name/'protocol.json').read_text())
 assert p['cancel'].split()[5]=='cancel' and p['recovery'].split()[5]=='stop'
 assert p['seed_repeat'] and p['one_token_limit'] and p['context_tail']
summary['stop_recovery_seed_limits_passed']=True
micro={str(c):list(csv.DictReader((root/f'micro-tuned-cols{c}.csv').open())) for c in [0,4]}
for rows in micro.values():
 assert len(rows)==96
 assert all(r['bitwise']=='1' and r['guard']=='1' for r in rows)
 assert all(0<float(r['min_us'])<=float(r['median_us'])<=float(r['max_us']) for r in rows)
summary['micro_shapes_per_config']=96
(root/'primary-results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print('PASS:',matched,'requests match P0 adaptive output; all primary runs use adaptive MTP, matched cache and no profiling/logit dumps')
for case,s in summary['performance'].items():print(case,round(s['a']['median'],2),'->',round(s['b']['median'],2),f"{s['gain_percent']:+.2f}%")
