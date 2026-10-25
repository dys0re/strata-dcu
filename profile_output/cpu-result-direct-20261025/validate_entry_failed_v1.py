"""One-shot public-entry validation for a fresh agent, after deployment and GPU idle."""
from pathlib import Path
import hashlib,json,os,re,resource,subprocess,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
R=Path('/root/private_data/strata-dcu/cpu-result-direct-20261025')
meta=json.loads((R/'deployment.json').read_text());out=R/'entry';out.mkdir(exist_ok=False)
engine=Path('/root/strata-dcu/build-gfx936/strata');cache=Path('/root/private_data/strata-dcu/autoround')
assert str(engine.resolve())==meta['engine']
assert hashlib.sha256(engine.read_bytes()).hexdigest()==meta['sha256']
def gpu():return subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
before=gpu();assert 'Used Memory (MiB): 2' in before;(out/'gpu-before.txt').write_text(before)
env=os.environ.copy()
for k in ['STRATA_DCU_MODEL_DEFAULTS','STRATA_DCU_MODEL_CONFIG','STRATA_DENSE_Q8_VARIANT','STRATA_HC_Q8','STRATA_DRAFT_HEAD_PROXY','STRATA_DCU_SCHEDULING_DEFAULTS','STRATA_VERIFY_HIT_PLAN','STRATA_CPU_EXPERT_PIPELINE','STRATA_CPU_EARLY_RESULT','STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_SPIN_US','STRATA_POOL_TEST_DELAY_US','STRATA_POOL_TRACE','STRATA_VERIFY_TRACE','STRATA_VERIFY_TRACE_FULL','STRATA_VERIFY_PROFILE','STRATA_PCIE_BALANCE','STRATA_SH_STREAM','STRATA_HC_SPLIT','STRATA_CPU_Q4_MT','STRATA_CPU_Q4_INPUT_BIAS','STRATA_CPU_RESULT_DIRECT','STRATA_DFLASH_SMALL_GEMV']:
    env.pop(k,None)
phrase='The copper owl watches the quiet river.';answer=' '.join([phrase]*8)
prompt='Copy the text between <text> tags exactly, without tags or commentary.\n<text>'+answer+'</text>'
native=['--dflash2','/root/private_data/strata-dcu/dflash2/checkpoint']
default=bool(meta['mtp_direct_result_default'])
cases=[('mtp-default',[],None,default,29373360352,8500),('mtp-override',[],not default,not default,29373360352,8500),
       ('native-opt-in',native,True,True,32827238400,8300),('joint-opt-in',native+['--dflash-mtp-prefix','2'],True,True,34195377152,8300)]
checks=[]
for name,args,override,direct,weights,slots in cases:
    d=out/name;d.mkdir();runenv=dict(env)
    if override is not None:runenv['STRATA_CPU_RESULT_DIRECT']=str(int(override))
    cmd=['bash','/root/strata-dcu/run-autoround.sh',*args,'--context','512','--suffix-draft','3','--prompt',prompt,'--max-new','128']
    (d/'command.json').write_text(json.dumps(dict(command=cmd,direct_result_override=override),indent=2)+'\n')
    with (d/'invocation.log').open('w') as log:rc=subprocess.run(cmd,env=runenv,stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0,(name,rc)
    for f in ['generation.json','run-command.json','engine.stdout.log','engine.stderr.log']:(d/f).write_bytes((cache/f).read_bytes())
    mtp=name.startswith('mtp')
    report=json.loads((d/'generation.json').read_text());err=(d/'engine.stderr.log').read_text()
    assert report['text'].strip()==answer,(name,report['text'])
    assert report['kernel_options']==dict(hc_split='2',cpu_q4_mt=True,cpu_q4_input_bias=mtp,dflash_small_gemv=True,cpu_result_direct=direct)
    assert ('strata verify: direct CPU result combine selected' in err)==direct
    assert ('strata cpu q4: shared input direct and single-position rows active' in err)==mtp
    assert f'expert_slots={slots}' in err and 'participating threads: 13' in err
    w=report['weight_vram'];assert w['weights']==weights and w['weights']<=w['limit']==34351349760
    assert w['experts']==(23500800000 if slots==8500 else 22947840000)
    assert w['total']==68702699520 and w.get('mtp_proxy',0)==0
    assert sum(w.get(k,0) for k in ['pack','dense','head','experts','mtp','external_draft','native_draft','staging'])==w['weights']
    v=report['model_variants'];mtp=name.startswith('mtp')
    assert v['hc_q8']==mtp and not v['draft_head_proxy']
    if mtp:
        assert v['dense_gguf']=='/root/private_data/strata-dcu/model-specific-20261024/dense-q8.gguf'
        assert v['mtp_runtime']=='/root/private_data/strata-dcu/model-specific-20261024/mtp-tuned'
        assert report['mtp_mode']=='adaptive' and report['mtp_max_t']==5
    else:
        assert v['dense_gguf']==str(cache/'dense.gguf') and v['mtp_runtime'] is None
        assert report['draft_backend']=='native' and report['draft_method']==('mtp+dflash2' if name.startswith('joint') else 'dflash2')
    assert all(report['scheduling_options'][k]==mtp for k in ['gpu_hit_plan','cpu_expert_pipeline','early_result'])
    assert report['expert_execution']==dict(resident_pin=False,adapt_every=0,adapt_swaps=96,pcie_mode='auto',pcie_frac=0)
    assert sum(map(int,re.findall(r'suffix drafts\s*:?\s+(\d+) windows',err+(d/'engine.stdout.log').read_text())))>0
    checks.append(dict(name=name,answer_correct=True,lookup_exercised=True,direct_result=direct,weights=weights,expert_slots=slots))
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('PASS ENTRY',name,flush=True)
after=gpu();assert 'Used Memory (MiB): 2' in after;(out/'gpu-after.txt').write_text(after)
ps=subprocess.check_output(['ps','-eo','pid,comm'],text=True)
assert not any(re.search(r'\bstrata(?:-|$)',s) for s in ps.splitlines()),ps
assert str(engine.resolve())==meta['engine'] and hashlib.sha256(engine.read_bytes()).hexdigest()==meta['sha256']
print('PASS CPU RESULT DIRECT PUBLIC ENTRIES; GPU idle',flush=True)
