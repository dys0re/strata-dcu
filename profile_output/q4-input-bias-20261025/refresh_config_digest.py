"""After entry validation, refresh descriptive engine metadata without changing options."""
from pathlib import Path
import hashlib,json
R=Path('/root/private_data/strata-dcu/q4-input-bias-20261025')
assert len(json.loads((R/'entry/checks.json').read_text()))==4
meta=json.loads((R/'deployment.json').read_text())
p=Path('/root/private_data/strata-dcu/model-specific-20261024/selected-model.json')
before=json.loads(p.read_text());after=dict(before)
assert before['cpu_q4_input_bias'] is True
assert before['engine_sha256']=='3ae138ead7fdbab0324bb17b423022d32c497eb18e3fb5aa82901011b98c9450'
assert hashlib.sha256(Path(meta['engine']).read_bytes()).hexdigest()==meta['sha256']
after['weight_selection_engine_sha256']=before['engine_sha256']
after['engine_sha256']=meta['sha256']
assert all(after[k]==v for k,v in before.items() if k!='engine_sha256')
tmp=p.with_suffix('.digest.tmp');assert not tmp.exists()
tmp.write_text(json.dumps(after,indent=2)+'\n');tmp.replace(p)
(R/'selected-model.json').write_bytes(p.read_bytes())
(R/'config-metadata-refresh.json').write_text(json.dumps(dict(before=before,after=after,changed_runtime_options=False),indent=2)+'\n')
print('CONFIG DIGEST REFRESHED; runtime options unchanged')
