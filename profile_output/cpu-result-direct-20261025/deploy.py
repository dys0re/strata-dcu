"""Deploy only the gated candidate; preserve the previous engine and model config."""
from pathlib import Path
import ast,hashlib,json,os,shutil,subprocess
R=Path('/root/private_data/strata-dcu/cpu-result-direct-20261025');S=Path('/root/strata-dcu/Strata')
assert [s['label'] for s in json.loads((R/'bench-status.json').read_text())]==['baseline-a','candidate-b','candidate-b2','baseline-a2']
assert all(s['returncode']==0 for s in json.loads((R/'bench-status.json').read_text()))
assert len(json.loads((R/'modes-status.json').read_text()))==4
assert all(s['returncode']==0 for s in json.loads((R/'modes-status.json').read_text()))
assert len(json.loads((R/'after-bench-status.json').read_text()))==4
assert all(x['returncode']==0 for x in json.loads((R/'after-bench-status.json').read_text()))
assert 'Used Memory (MiB): 2' in subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
analysis=json.loads((R/'analysis.json').read_text());enable=analysis['enable_machine_default']
manifest=json.loads((R/'source-manifest.json').read_text())
for name,digest in manifest.items():
    if name.endswith(('.cpp','.hpp')) or name=='CMakeLists.txt':assert hashlib.sha256((S/name).read_bytes()).hexdigest()==digest,name
ast.parse((S/'tools/autoround_pack.py').read_text())
candidate=R/'strata-candidate';sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
assert sha==json.loads((R/'candidate.json').read_text())['sha256']
link=Path('/root/strata-dcu/build-gfx936/strata');previous=str(link.resolve())
assert hashlib.sha256(link.read_bytes()).hexdigest()=='4fc8304ec17c7e6b0df4a843f7f60ebe21934e6a7e88004a358e6b2356edb8cd'
config=Path('/root/private_data/strata-dcu/model-specific-20261024/selected-model.json')
before=config.read_bytes();(R/'selected-model-before.json').write_bytes(before)
model=json.loads(before);assert model['cpu_q4_input_bias'] is True
model['cpu_result_direct']=enable
model['engine_sha256']=sha
temp=config.with_name('selected-model.q4-input-bias.tmp');assert not temp.exists()
temp.write_text(json.dumps(model,indent=2)+'\n');temp.replace(config)
shutil.copy2(candidate,R/'strata-final')
pending=link.with_name('strata.q4-input-bias-new');assert not pending.exists() and not pending.is_symlink()
pending.symlink_to(R/'strata-final');pending.replace(link)
meta=dict(engine=str(R/'strata-final'),sha256=sha,previous_engine=previous,mtp_direct_result_default=enable,default_rule=analysis['default_rule'],geometric_mean_ratio=analysis['geometric_mean_ratio'])
(R/'deployment.json').write_text(json.dumps(meta,indent=2)+'\n')
(R/'selected-model.json').write_bytes(config.read_bytes())
(R/'deployed-source-manifest.json').write_text(json.dumps({name:hashlib.sha256((S/name).read_bytes()).hexdigest() for name in manifest},indent=2)+'\n')
print(json.dumps(meta,indent=2),flush=True)
