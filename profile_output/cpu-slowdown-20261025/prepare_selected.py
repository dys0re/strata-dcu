from pathlib import Path
import ast,json,hashlib
r=Path('/root/private_data/strata-dcu/cpu-slowdown-20261025')
s=(r/'affinity_bench.py').read_text().replace("for repeat in range(2):cases.append((phase+'-'+str(repeat),story,64,None,0))","for repeat in range(1):cases.append((phase+'-'+str(repeat),story,256,None,0))")
assert 'story,256,None,0' in s
ast.parse(s);(r/'selected_affinity_bench.py').write_text(s)
s=(r/'run_affinity.py').read_text().replace("STRATA_HC_Q8='0'","STRATA_HC_Q8='1'").replace('affinity_bench.py','selected_affinity_bench.py')
s=s.replace("if os.environ.get('AFFINITY'):","cmd += ['--dense-gguf',str(P/'dense-q8.gguf'),'--mtp-runtime',str(P/'mtp-tuned')]\nif os.environ.get('AFFINITY'):")
ast.parse(s);(r/'run_selected.py').write_text(s)
(r/'selected-probe-source.json').write_text(json.dumps({f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['selected_affinity_bench.py','run_selected.py']},indent=2))
