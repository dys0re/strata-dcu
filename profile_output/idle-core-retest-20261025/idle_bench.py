import sys
sys.path.insert(0,"/root/strata-dcu/Strata/tools")
"""Qwen3.8-F BS=1 speed test; all GPU weight allocations must fit 50% of physical VRAM."""
import argparse,ast,json,os,re,subprocess,time,hashlib,resource
from pathlib import Path

resource.setrlimit(resource.RLIMIT_CORE,(0,0))

ap=argparse.ArgumentParser(__doc__)
ap.add_argument('--engine',required=True)
ap.add_argument('--dense-gguf',help='optional non-expert weight variant; expert pack stays fixed')
ap.add_argument('--mtp-runtime',help='optional compatible MTP runtime variant')
ap.add_argument('--smoke',action='store_true')
ap.add_argument('--qa-only',action='store_true',help='functional QA first; no throughput workload warmups')
ap.add_argument('--qa-after-bench',action='store_true',help='run semantic/code/protocol QA after throughput workloads')
ap.add_argument('--extended-qa',action='store_true',help='also execute generated search/interval/encoding functions and check a data transform')
ap.add_argument('--graph',action='store_true')
ap.add_argument('--draft-adaptive',action='store_true')
ap.add_argument('--expert-cache',type=int)
ap.add_argument('--pcie-mode',choices=['auto','kernel','dma','direct'],default='auto')
ap.add_argument('--resident-pin',action=argparse.BooleanOptionalAction,default=None)
ap.add_argument('--adapt-every',type=int,default=0)
ap.add_argument('--adapt-swaps',type=int,default=96)
ap.add_argument('--pcie-frac',type=float,default=0,help='-1 uses engine auto policy; 0 CPU-only misses')
ap.add_argument('--suffix-draft',type=int,default=0)
ap.add_argument('--lookup-checks',action='store_true',help='exercise repeated-text lookup and recovery')
ap.add_argument('--out',required=True)
ap.add_argument('--cols',type=int,choices=[0,2,4],default=4)
ap.add_argument('--mode',choices=['native','ipc','off','adaptive','joint'],default='native')
ap.add_argument('--joint-prefix',type=int,choices=[1,2,3],default=2)
ap.add_argument('--joint-fixed-prefix',action='store_true')
ap.add_argument('--spec-prob',action='store_true')
ap.add_argument('--corrupt',type=int,default=0)
ap.add_argument('--timing-only',action='store_true')
ap.add_argument('--thread-usage',action='store_true',help='sample per-thread scheduled CPU time at decode protocol boundaries')
ap.add_argument('--process-placement',action='store_true',help='record anonymous NUMA page placement and CPU affinity after load')
ap.add_argument('--shared-stream',choices=['auto','on','off'],default='auto',help='explicit shared-expert GPU stream placement; auto preserves backend defaults')
ap.add_argument('--request-pcie-frac',type=float,help='explicit per-request fraction; exercises the fixed bounded-offload path')
ap.add_argument('--width',type=int,choices=range(2,9),default=8)
ap.add_argument('--context',type=int,default=512)
ap.add_argument('--long-checks',action='store_true')
ap.add_argument('--repeat',type=int,default=3)
ap.add_argument('--checks',action='store_true')
ap.add_argument('--profile',action='store_true')
ap.add_argument('--dump-logits',action='store_true')
ap.add_argument('--fuse',choices=['off','qfuse','qdedup','both','hc'],default='off')
a=ap.parse_args()
if a.resident_pin is None:a.resident_pin=a.pcie_frac!=0
assert a.adapt_every >= 0 and a.adapt_swaps >= 0 and a.suffix_draft >= 0
assert a.pcie_frac == -1 or 0 <= a.pcie_frac <= 1
assert a.request_pcie_frac is None or 0 <= a.request_pcie_frac <= 1
assert not a.long_checks or a.context>=2300
out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
root=Path('/root/strata-dcu'); cache=root/'autoround'
source_files=['CMakeLists.txt','src/program/generate.cpp','src/core/verify.cpp','src/prefill/prefill.cpp','include/strata/core/dflash_bridge.hpp','include/strata/core/verify.hpp','include/strata/prefill/prefill.hpp','src/kernels/cuda/native_mmvq.cu','include/strata/kernels/native_mmvq.hpp','src/core/mtp.cpp','include/strata/core/mtp.hpp','tools/dflash2_worker.py','tools/strata_dflash2.py','third_party/dflash/model.py','third_party/dflash/LICENSE','third_party/dflash/VERSION','include/strata/core/dflash_native.hpp','src/prefill/dflash.cu','src/prefill/dflash_witness.cpp','tools/dflash2_pack.py','tools/test_dflash2_native.py']
source_files+=['src/kernels/cpu/native_expert.cpp','src/kernels/cpu/q4_avx512.cpp',
 'include/strata/kernels/cpu/q4_avx512.hpp','src/kernels/cuda/bf16_small.cu',
 'include/strata/kernels/bf16_small.hpp']
source_files+=['include/strata/spec/joint_budget.hpp']
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
(out/'provenance.json').write_text(json.dumps({'engine_sha256':sha(a.engine),'source_sha256':{f:sha(root/'Strata'/f) for f in source_files},'driver_sha256':sha(__file__)},indent=2)+'\n')

cmd=json.loads((cache/'run-command.json').read_text());cmd[0]=a.engine
for flag,n in [('--mtp',2),('--mtp-adaptive',1),('--dflash2',2),('--dflash-adaptive',1),('--dflash-eager',1),('--dflash-mtp-prefix',2)]:
 if flag in cmd:
  i=cmd.index(flag);del cmd[i:i+n]
for flag,value in [('--spec',str(min(8,a.width+(2 if a.suffix_draft and a.mode=='adaptive' else 0)))),('--mtp-max-t','1' if a.mode=='off' else str(a.width)),('--max-context',str(a.context)),('--expert-cache',str(a.expert_cache or (8500 if a.mode=='adaptive' else 8300)))]:
 cmd[cmd.index(flag)+1]=value
if a.mode in ['adaptive','joint']:
 cmd+=['--mtp',str(Path(a.mtp_runtime).resolve()) if a.mtp_runtime else str(cache/'mtp')]
 if a.mode!='joint' or not a.joint_fixed_prefix:cmd+=['--mtp-adaptive']
# Normalize inherited flags: run-command.json may come from any public entry mode.
for flag,n in [('--serve',1),('--prompt-cache',2),('--conversation-cache-mib',2),
               ('--pcie-frac',2),('--pcie-mode',2),('--adapt-every',2),('--adapt-swaps',2),('--suffix-draft',2),
               ('--resident-cpu-experts',1),('--resident-experts',1)]:
 while flag in cmd:
  i=cmd.index(flag);del cmd[i:i+n]
cmd+=['--resident-experts' if a.resident_pin else '--resident-cpu-experts']
cmd[cmd.index('--native')+1]=str(Path(a.dense_gguf).resolve()) if a.dense_gguf else str(cache/'dense.gguf')
cmd+=['--serve','--prompt-cache','0','--conversation-cache-mib','0',
      '--pcie-frac',str(a.pcie_frac),'--pcie-mode',a.pcie_mode,'--adapt-every',str(a.adapt_every),
      '--adapt-swaps',str(a.adapt_swaps),'--suffix-draft',str(a.suffix_draft)]
if a.mode in ['native','joint']:
 cmd+=['--dflash2','/root/private_data/strata-dcu/dflash-native/pack']
 if a.draft_adaptive:cmd+=['--dflash-adaptive']
 if not a.graph:cmd+=['--dflash-eager']
 if a.mode=='joint':cmd+=['--dflash-mtp-prefix',str(a.joint_prefix)]
if a.corrupt:cmd+=['--spec-corrupt',str(a.corrupt)]
if a.mode=='ipc':
 cmd=['python3',str(root/'Strata/tools/strata_dflash2.py'),'--checkpoint','/root/private_data/strata-dcu/dflash2/checkpoint','--model','/root/private_data/Qwen3.8-F','--stats-file',str(out/'worker.json')]+(['--graph'] if a.graph else [])+(['--adaptive'] if a.draft_adaptive else [])+['--']+cmd
env=os.environ.copy()
for key in ['STRATA_VERIFY_PROFILE','STRATA_DECODE_TIMING','STRATA_DUMP_FIRST_LOGITS','STRATA_SPEC_PROB','STRATA_MTP_FRONT_CHECK','STRATA_SH_STREAM','STRATA_DFLASH_FD','STRATA_DFLASH_NATIVE']:
 env.pop(key,None)
settings=dict(STRATA_MAX_WEIGHT_VRAM_FRACTION='0.5',STRATA_TRACE='1',STRATA_BF16_Q8_VEC4='1',STRATA_BF16_Q8_COLS=str(a.cols),STRATA_ROUTE_TAIL_SKIP='0',STRATA_SPEC_COUPLED='1',
 STRATA_QFUSE=str(int(a.fuse in ['qfuse','both'])),STRATA_VERIFY_QDEDUP=str(int(a.fuse in ['qdedup','both'])),STRATA_QFUSE_HC=str(int(a.fuse=='hc')))
env.update(settings)
if a.dense_gguf:
 settings['STRATA_DENSE_Q8_VARIANT']='1'
 env['STRATA_DENSE_Q8_VARIANT']='1'
if a.shared_stream != 'auto':
 settings['STRATA_SH_STREAM']='1' if a.shared_stream=='on' else '0'
 env['STRATA_SH_STREAM']=settings['STRATA_SH_STREAM']
for feature in ['STRATA_CPU_Q4_MT','STRATA_DFLASH_SMALL_GEMV']:
 settings[feature]=env.get(feature,'0')
 env[feature]=settings[feature]
for feature in ['STRATA_VERIFY_HIT_PLAN','STRATA_CPU_EXPERT_PIPELINE','STRATA_CPU_EARLY_RESULT',
                'STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_SPIN_US',
                'STRATA_HC_Q8','STRATA_DRAFT_HEAD_PROXY']:
 if feature in env:settings[feature]=env[feature]
if a.profile:env.update(STRATA_VERIFY_PROFILE='1',STRATA_DECODE_TIMING='1',STRATA_SPEC_DEPTH='1')
if a.timing_only:env['STRATA_DECODE_TIMING']='1'
if a.spec_prob:env['STRATA_SPEC_PROB']='1'
if a.checks and (a.profile or a.dump_logits):env['STRATA_DUMP_FIRST_LOGITS']=str(out/'logits.f32')
(out/'command.json').write_text(json.dumps(dict(command=cmd,overrides=settings,hc_environment={k:env.get(k) for k in ['STRATA_HC_SPLIT','STRATA_GR_FAST']},profile=a.profile,timing_only=a.timing_only,spec_prob=a.spec_prob,checks=a.checks,dump_logits='STRATA_DUMP_FIRST_LOGITS' in env),indent=2)+'\n')
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/root/private_data/Qwen3.8-F',local_files_only=True)
tok.chat_template=Path('/root/private_data/Qwen3.8-F/chat_template.jinja').read_text()
story='Write a detailed 800-word story about a robot exploring an abandoned observatory. Begin the story immediately and continue with rich descriptions and dialogue.'
workloads=[('story',story),('chinese','请详细解释大语言模型的 prefill 和 decode 阶段有什么区别，以及它们各自的性能瓶颈。请分段说明并举例。'),('code','Write a Python LRU cache class using OrderedDict. Include get and put methods, explain their complexity, and give a usage example.')]
cases=[('warmup',story,128,None,0)]
for rep in range(a.repeat):cases += [(name+f'-{rep}',prompt,256,None,0) for name,prompt in workloads]
if a.checks:
 cases += [('ok','Reply with exactly OK.',16,'OK',0),('arithmetic','What is 2+2? Answer only the number.',16,'4',0),('capital','中国的首都是哪座城市？只回答城市名称。',16,'北京',0)]
 cases += [(f'sample-{i}','Write a short story about a lost compass and a mountain village.',64,None,.7) for i in range(2)]
 cases += [('one_token','Reply with exactly OK.',1,None,0),('context_tail','Continue a long story for several paragraphs.',3,None,0)]
if a.long_checks:
 cases=[('window-crossing','Reply with exactly ORBIT.',16,'ORBIT',0),('long-recovery','Reply with exactly OK.',16,'OK',0)]
if a.smoke:
 cases=[('ok','Reply with exactly OK.',16,'OK',0),('arithmetic','What is 2+2? Answer only the number.',16,'4',0),('capital','中国的首都是哪座城市？只回答城市名称。',16,'北京',0),('story-short',story,64,None,0)]
if a.qa_only or a.qa_after_bench:
 qa_cases=[('ok','Reply with exactly OK.',16,'OK',0),
  ('arithmetic','What is 2+2? Answer only the number.',16,'4',0),
  ('multiply','What is 17 times 23? Answer only the integer.',32,'391',0),
  ('sum','What is the sum of the integers from 1 to 100 inclusive? Answer only the integer.',32,'5050',0),
  ('capital','中国的首都是哪座城市？只回答城市名称。',16,'北京',0),
  ('uppercase','Convert abc to uppercase. Output only the result.',16,'ABC',0),
  ('sort-json','Sort [9,2,2,-1] ascending. Output only a JSON array.',64,None,0),
  ('json','Output exactly one JSON object with boolean key ok set to true and integer key count set to 3. No Markdown.',64,None,0),
  ('code-function','Return only Python code defining is_even(n), which is True exactly for even integers. No Markdown or explanation.',96,None,0),
  ('multi-turn',[{'role':'user','content':'Remember the code word ORBIT. Reply OK.'},{'role':'assistant','content':'OK.'},{'role':'user','content':'What code word did I ask you to remember? Reply with only that word.'}],16,'ORBIT',0),
  ('sample-0','Write a short story about a lost compass and a mountain village.',64,None,.7),
  ('sample-1','Write a short story about a lost compass and a mountain village.',64,None,.7),
  ('one_token','Reply with exactly OK.',1,None,0),
  ('context_tail','Continue a long story for several paragraphs.',3,None,0)]
 if a.qa_only:cases=qa_cases
 else:cases+=qa_cases
if a.lookup_checks:
 phrase='The copper owl watches the quiet river.'
 repeated=' '.join([phrase]*8)
 cases += [('lookup-copy','Copy the text between <text> tags exactly, without tags or commentary.\n<text>'+repeated+'</text>',128,repeated,0),
           ('lookup-recovery','What is 17 times 23? Answer only the integer.',32,'391',0)]
if a.extended_qa:
 from dcu_functional_qa import CASES as extra_cases
 cases += extra_cases
records=[]
mem=Path('/sys/fs/cgroup')
def memory_snapshot():
 return {n:(mem/n).read_text() for n in ['memory.current','memory.peak','memory.events','memory.max'] if (mem/n).exists()}
(out/'memory-before.json').write_text(json.dumps(memory_snapshot(),indent=2)+'\n')
with (out/'stderr.log').open('w') as err,(out/'stdout.log').open('w') as stdout:
 p=subprocess.Popen(cmd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
 thread_records=[]
 def thread_snapshot():
  stamp=time.monotonic_ns();threads={}
  for task in Path(f'/proc/{p.pid}/task').iterdir():
   try:
    sched=list(map(int,(task/'schedstat').read_text().split()))
    status=(task/'status').read_text().splitlines()
    threads[task.name]=dict(runtime_ns=sched[0],runqueue_ns=sched[1],affinity=next(s.split(':',1)[1].strip() for s in status if s.startswith('Cpus_allowed_list:')))
   except FileNotFoundError:pass
  return dict(time_ns=stamp,threads=threads)
 thread_start=None
 def line():
  global thread_start
  s=p.stdout.readline()
  if not s:raise RuntimeError(f'engine EOF: {p.poll()}')
  stdout.write(s);stdout.flush()
  if a.thread_usage:
   if s.startswith('REUSED '):thread_start=thread_snapshot()
   if s.startswith('DONE ') and thread_start is not None:
    thread_records.append(dict(pid=p.pid,done=s.strip(),start=thread_start,end=thread_snapshot()))
    (out/'thread-usage.json').write_text(json.dumps(thread_records,indent=2)+'\n')
    thread_start=None
  if s.startswith('ERR '):raise RuntimeError(s)
  return s.strip()
 def send(s):p.stdin.write(s);p.stdin.flush()
 def finish(stop=False):
  ids=[];sent=False
  while True:
   s=line()
   if s.startswith('T '):
    ids.append(int(s.split()[1]))
    if stop and len(ids)>=8 and not sent:send('STOP\n');sent=True
   if s.startswith('DONE '):return ids,s
 def gen(prompt,n,temperature=0,tail=False):
  messages=prompt if isinstance(prompt,list) else [{'role':'user','content':prompt}]
  ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
  if tail:ids=ids[:3]+[tok.encode(' word',add_special_tokens=False)[0]]*((2100 if a.long_checks else a.context-11)-len(ids))+ids[3:]
  assert len(ids)+n<=a.context
  sampling=f'temperature={temperature} top_p=0.9 top_k=20 seed=1729' if temperature else 'temperature=0'
  if a.request_pcie_frac is not None:sampling+=f' pcie_frac={a.request_pcie_frac}'
  return ids,f'GEN {n} ckpt=0 {sampling} '+','.join(map(str,ids))+'\n'
 try:
  while not line().startswith('READY '):pass
  usage_lines=[s for s in (out/'stderr.log').read_text().splitlines() if s.startswith('strata weight usage:')]
  assert usage_lines, 'engine did not enforce/report the weight budget'
  usage={k:float(v) if '.' in v else int(v) for k,v in re.findall(r'(\w+)=([0-9.]+)',usage_lines[-1])}
  assert usage['limit']==usage['total']//2 and usage['weights']<=usage['limit']
  assert usage['weights']==sum(usage[k] for k in ['pack','dense','head','experts','mtp'])+usage.get('external_draft',0)+usage.get('native_draft',0)+usage.get('staging',0)
  if a.mode=='ipc':assert usage['external_draft']>0
  if a.mode in ['native','joint']:assert usage['native_draft']==1206131200 and usage['external_draft']==0
  if a.mode=='joint':assert usage['mtp']>0
  (out/'weight-usage.json').write_text(json.dumps(usage,indent=2)+'\n')
  print('WEIGHTS',usage,flush=True)
  print('READY',a.mode,a.cols,a.fuse,flush=True)
  if a.process_placement:
   proc=Path(f'/proc/{p.pid}');nodes={};large=[]
   for row in (proc/'numa_maps').read_text().splitlines():
    fields=dict(s.split('=',1) for s in row.split() if '=' in s)
    if 'anon' not in fields:continue
    places={k:int(v) for k,v in fields.items() if re.fullmatch(r'N\d+',k)}
    for key,value in places.items():nodes[key]=nodes.get(key,0)+value
    if int(fields['anon'])>100000:large.append(dict(anon_pages=int(fields['anon']),nodes=places,policy=row.split()[1]))
   status=[s for s in (proc/'status').read_text().splitlines() if s.startswith(('VmRSS:','VmSwap:','Cpus_allowed_list:','Mems_allowed_list:'))]
   (out/'process-placement.json').write_text(json.dumps(dict(pid=p.pid,status=status,anonymous_nodes=nodes,large_regions=large,cpu_stat=(mem/'cpu.stat').read_text()),indent=2)+'\n')
  # [PROFILE] Select once after load, before warmup. No active selection during decode.
  from idle_cores import choose_and_pin
  selected=choose_and_pin(p.pid,os.sched_getaffinity(0),13,out/'idle-core-selection.json')
  for name,prompt,n,expected,temperature in cases:
   prompt_ids,request=gen(prompt,n,temperature,name in ['context_tail','window-crossing'] or (a.long_checks and name=='lookup-copy'))
   (out/(name+'.request.txt')).write_text(request)
   started=time.monotonic();send(request);ids,done=finish();fields=done.split()
   text=tok.decode(ids,skip_special_tokens=True)
   assert int(fields[1])==len(ids) and int(fields[2])==len(prompt_ids) and len(ids)<=n and int(fields[8])==0
   assert fields[5] in ['length','stop']
   if expected is not None:assert text.strip()==expected,(name,text)
   if name=='sort-json':assert json.loads(text)==[-1,2,2,9],text
   if name=='json':assert json.loads(text)==dict(ok=True,count=3),text
   if a.extended_qa:
    from dcu_functional_qa import check as check_answer
    check_answer(name,text)
   if name=='code-function':
    code=text.strip()
    if code.startswith('```'):code='\n'.join(code.splitlines()[1:-1])
    tree=ast.parse(code)
    allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.Compare,ast.BinOp,ast.Mod,ast.Constant,ast.Name,ast.Load,ast.Eq,ast.Expr)
    assert all(isinstance(n,allowed) for n in ast.walk(tree)),code
    ns={'__builtins__':{}};exec(compile(tree,'qa-generated-code','exec'),ns)
    assert all(ns['is_even'](n)==(n%2==0) for n in range(-20,21)),code
   records.append(dict(case=name,prompt_ids=prompt_ids,generated_ids=ids,text=text,done=done,
    decode_ms=float(fields[4]),decode_tps=1000*len(ids)/float(fields[4]),wall_s=time.monotonic()-started))
   (out/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
   print('DONE',name,len(ids),round(records[-1]['decode_tps'],2),flush=True)
  if (a.checks or a.qa_only or a.qa_after_bench) and not a.smoke and not a.long_checks:
   by_case={r['case']:r for r in records}
   seed_repeat=by_case['sample-0']['generated_ids']==by_case['sample-1']['generated_ids']
   print('SEED_REPEAT',seed_repeat,flush=True)
   assert len(by_case['one_token']['generated_ids'])==1 and len(by_case['context_tail']['generated_ids'])<=3
   _,request=gen(story,256);send(request);ids,done=finish(stop=True);assert done.split()[5]=='cancel'
   _,request=gen('Reply with exactly OK.',16);send(request);ids,recovery=finish();assert tok.decode(ids,skip_special_tokens=True).strip()=='OK'
   (out/'protocol.json').write_text(json.dumps(dict(cancel=done,recovery=recovery,seed_repeat=seed_repeat,one_token_limit=True,context_tail=True),indent=2)+'\n')
  send('QUIT\n');p.wait(timeout=30);assert p.returncode==0
 finally:
  if p.poll() is None:p.kill();p.wait()
errors=(out/'stderr.log').read_text()
if a.spec_prob and a.mode=='joint':
 prob=[int(x) for x in re.findall(r'prob_windows=(\d+)',errors)]
 assert prob and max(prob)>0,'probabilistic verification was not exercised'
if a.resident_pin and a.pcie_frac!=0 and a.timing_only:
 moved=[float(x) for x in re.findall(r'VRAM hits [0-9.]+, PCIe ([0-9.]+);',errors)]
 assert moved and max(moved)>0,'PCIe was requested but no experts used PCIe offload; inspect resident pinning'
if a.lookup_checks and a.suffix_draft:
 counts=re.findall(r'suffix drafts: (\d+) windows, (\d+) of (\d+) drafts accepted',errors)
 assert counts and sum(int(c[0]) for c in counts)>0,'lookup path was not exercised'
 (out/'lookup-counters.json').write_text(json.dumps(counts)+'\n')
 if a.mode in ['native','joint']:
  discarded=re.findall(r'discarded_feedback=(\d+)',errors)
  assert discarded and int(discarded[-1])==sum(int(c[0]) for c in counts), 'lookup feedback was not discarded exactly once per replaced draft'

assert not any(s in errors for s in ['Launch params','larger than launch bounds','invalid device function','out of memory'])
(out/'memory-after.json').write_text(json.dumps(memory_snapshot(),indent=2)+'\n')
(out/'qa-status.json').write_text(json.dumps(dict(functional_assertions_passed=True,qa_scope='full' if a.qa_only or a.qa_after_bench else 'smoke' if a.smoke else 'long' if a.long_checks else 'checks' if a.checks else 'throughput',extended_qa=a.extended_qa,engine_rc=p.returncode,token_identity_informational=True))+'\n')
print('PASS',flush=True)
