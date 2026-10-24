"""Run after the configuration matrix exits. One engine at a time; persistent output."""
import json,math,os,statistics,subprocess
from pathlib import Path
ROOT=Path('/root/private_data/strata-dcu')
OUT=ROOT/'lookup-compat-20261023'
ENGINE=OUT/'strata'
DRIVER='/root/strata-dcu/Strata/tools/bench_dflash2_native.py'
WORKLOADS=['story','chinese','code']
ENV=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
status=[]
def med(path):
 records=json.loads((path/'results.json').read_text())
 return {w:statistics.median(r['decode_tps'] for r in records if r['case'].startswith(w+'-') and r['case'][len(w)+1:].isdigit()) for w in WORKLOADS}
def select(rows,baseline):
 b=rows[baseline]
 eligible=[k for k,r in rows.items() if all(r[w]>=b[w]*.97 for w in WORKLOADS)]
 winner=max(eligible,key=lambda k:math.prod(rows[k][w]/b[w] for w in WORKLOADS))
 return winner if math.prod(rows[winner][w]/b[w] for w in WORKLOADS)**(1/3)>1.03 else baseline
matrix=ROOT/'upstream-options-20261023'
assert json.loads((matrix/'status.json').read_text())==[dict(label=k,returncode=0) for k in ['baseline','cache','pcie','both']]
print('START seeded-cache-witness',flush=True)
with (OUT/'witness.log').open('w') as f:
 r=subprocess.run(['python3','-u','/root/strata-dcu/Strata/tools/test_joint_dflash.py',
    '--binary',str(ROOT/'p1/build/dflash_native_witness'),
    '--pack',str(ROOT/'dflash-native/pack'),'--gguf',str(ROOT/'autoround/dense.gguf'),
    '--reference',str(ROOT/'dflash-native/witness'),'--out',str(OUT/'witness')],
    env=ENV,stdout=f,stderr=subprocess.STDOUT)
if r.returncode:raise SystemExit(r.returncode)
print('FINISHED seeded-cache-witness',flush=True)
settings={'baseline':(0,0,False),'cache':(4,0,False),'cache-pinned':(4,0,True),'pcie-pinned':(0,-1,True),'both':(4,-1,True)}
# Repeat the old-binary baseline after all compiling has stopped. Keep the first
# run as evidence rather than depending on the overlap's exact boundary.
clean=matrix/'baseline-clean'
cmd=['python3','-u',DRIVER,'--engine',str(ROOT/'joint-draft-20261023/strata-final'),
     '--mode','adaptive','--width','5','--repeat','3','--qa-after-bench','--timing-only',
     '--adapt-every','0','--pcie-frac','0','--out',str(clean)]
print('START baseline-clean',flush=True)
with (matrix/'baseline-clean.log').open('w') as f:
 r=subprocess.run(cmd,env=ENV,stdout=f,stderr=subprocess.STDOUT)
if r.returncode:raise SystemExit(r.returncode)
# The initial pcie arm used pageable RAM and logged zero PCIe experts. It is a
# diagnostic negative control, not evidence about working PCIe offload.
for label in ['cache-pinned','pcie-pinned']:
 ae,pf,pin=settings[label]
 cmd=['python3','-u',DRIVER,'--engine',str(ROOT/'joint-draft-20261023/strata-final'),
      '--mode','adaptive','--width','5','--repeat','3','--qa-after-bench','--timing-only',
      '--adapt-every',str(ae),'--pcie-frac',str(pf),'--resident-pin','--out',str(matrix/label)]
 print('START',label,flush=True)
 with (matrix/(label+'.log')).open('w') as f:
  r=subprocess.run(cmd,env=ENV,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise SystemExit(r.returncode)
 print('FINISHED',label,r.returncode,flush=True)
rows={k:med(clean if k=='baseline' else matrix/k) for k in settings}
chosen=select(rows,'baseline');adapt,pcie,pinned=settings[chosen]
selection=dict(config=chosen,config_medians=rows,adapt_every=adapt,pcie_frac=pcie,resident_pin=pinned)
(OUT/'selection.json').write_text(json.dumps(selection,indent=2))
print('CONFIG',chosen,rows,flush=True)
def run(label,mode,*,hc=2,suffix=0,repeat=3,qa=True,extra=()):
 cmd=['python3','-u',DRIVER,'--engine',str(ENGINE),'--mode',mode,'--width','5' if mode=='adaptive' else '4' if mode=='ipc' else '8',
      '--graph','--draft-adaptive','--repeat',str(repeat),'--adapt-every',str(adapt),'--pcie-frac',str(pcie),'--suffix-draft',str(suffix),
      '--timing-only','--resident-pin' if pinned else '--no-resident-pin','--out',str(OUT/label)]
 if qa:cmd+=['--qa-after-bench'] if repeat else ['--qa-only']
 if suffix:cmd+=['--lookup-checks']
 cmd+=list(extra)
 print('START',label,flush=True)
 with (OUT/(label+'.log')).open('w') as f:
  r=subprocess.run(cmd,env=dict(ENV,STRATA_HC_SPLIT=str(hc)),stdout=f,stderr=subprocess.STDOUT)
 status.append(dict(label=label,returncode=r.returncode));(OUT/'status.json').write_text(json.dumps(status,indent=2))
 print('FINISHED',label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
for hc,name in [(0,'plain'),(1,'split'),(2,'staged')]:
 run('hc-'+name,'adaptive',hc=hc,qa=(hc==2))
hc_rows={k:med(OUT/('hc-'+k)) for k in ['plain','split','staged']}
hc_choice=select(hc_rows,'plain');hc={'plain':0,'split':1,'staged':2}[hc_choice]
selection.update(hc=hc,hc_name=hc_choice,hc_medians=hc_rows)
(OUT/'selection.json').write_text(json.dumps(selection,indent=2))
print('HC',hc_choice,hc_rows,flush=True)
if hc!=2:run('hc-selected-qa','adaptive',hc=hc,repeat=0)
run('mtp-lookup','adaptive',hc=hc,suffix=3)
for mode in ['native','joint']:
 run(mode+'-plain','%s'%mode,hc=0)
 if hc:run(mode+'-hc',mode,hc=hc)
 run(mode+'-lookup',mode,hc=hc,suffix=3)
run('joint-lookup-prob','joint',hc=hc,suffix=3,repeat=0,extra=['--spec-prob'])
run('joint-lookup-reject','joint',hc=hc,suffix=3,repeat=0,extra=['--corrupt','1'])
run('native-lookup-long','native',hc=hc,suffix=3,repeat=0,qa=False,extra=['--context','4096','--long-checks'])
run('joint-lookup-long','joint',hc=hc,suffix=3,repeat=0,qa=False,extra=['--context','4096','--long-checks'])
# The IPC fallback has a separate feedback-reset command and must exercise it too.
run('ipc-lookup','ipc',hc=hc,suffix=3,repeat=0)
print('SUITE PASS',flush=True)
