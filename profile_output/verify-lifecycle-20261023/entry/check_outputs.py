import pathlib,json,csv,collections,hashlib,subprocess
p=pathlib.Path('/root/private_data/strata-dcu/verify-lifecycle-20261023/mtp-profile')
expected_diag='7060a916d2954b6904e41faf9617a465d549f70244168fe0e78ea2749394e28e'
expected_prod='1551c2b1cc1624a29ab9b17f2276599c2a35648c58babc417e2c6ad98947b1b1'
checks={};details={}
def check(k,v): checks[k]=bool(v)
def read(n):return json.loads((p/n).read_text())
def sha(f):
 with open(f,'rb') as x:return hashlib.file_digest(x,'sha256').hexdigest()
files=['command.json','provenance.json','memory-before.json','memory-after.json','weight-usage.json','results.json','protocol.json','qa-status.json','stderr.log','stdout.log','lifecycle.csv','cpu.csv']
check('expected_files',all((p/f).is_file() and (p/f).stat().st_size>0 for f in files))
details['files']={f.name:f.stat().st_size for f in p.iterdir() if f.is_file()}
r=read('results.json');by={x['case']:x for x in r}
check('eleven_request_records',len(r)==11)
check('expected_cases',list(by)==['warmup','story-0','chinese-0','code-0','ok','arithmetic','capital','sample-0','sample-1','one_token','context_tail'])
check('three_256_token_workloads',all(len(by[n]['generated_ids'])==256 for n in ['story-0','chinese-0','code-0']))
check('semantic_qa',all(by[n]['text'].strip()==t for n,t in [('ok','OK'),('arithmetic','4'),('capital','北京')]))
check('samples',all(0<len(by[n]['generated_ids'])<=64 for n in ['sample-0','sample-1']))
check('output_context_limits',len(by['one_token']['generated_ids'])==1 and len(by['context_tail']['generated_ids'])<=3 and all(len(x['prompt_ids'])+len(x['generated_ids'])<=512 for x in r))
q=read('qa-status.json');check('qa_status',q['functional_assertions_passed'] and q['engine_rc']==0)
protocol=read('protocol.json');check('cancellation',protocol['cancel'].split()[5]=='cancel');check('recovery_done',protocol['recovery'].startswith('DONE '));check('protocol_limits',protocol['one_token_limit'] and protocol['context_tail'])
u=read('weight-usage.json');check('weights_within_50_percent',u['weights']<=u['limit']==u['total']//2)
check('weight_sum',u['weights']==sum(u.get(k,0) for k in ['pack','dense','head','experts','mtp','external_draft','native_draft','staging']))
c=read('command.json');args=c['command'];o=c['overrides']
check('8500_gpu_experts',args[args.index('--expert-cache')+1]=='8500')
check('kernel_flags_1',all(o[n]=='1' for n in ['STRATA_CPU_Q4_MT','STRATA_DFLASH_SMALL_GEMV']))
check('lifecycle_without_verify_profile',c['lifecycle'] and c['timing_only'] and not c['profile'] and 'STRATA_VERIFY_PROFILE' not in o)
check('fusion_defaults',all(o[n]=='0' for n in ['STRATA_QFUSE','STRATA_VERIFY_QDEDUP','STRATA_QFUSE_HC']) and 'STRATA_SH_STREAM' not in o)
diag=sha('/root/private_data/strata-dcu/verify-lifecycle-20261023/strata-profile');prod=pathlib.Path('/root/strata-dcu/build-gfx936/strata');prodhash=sha(prod)
check('diagnostic_sha_matches_build_provenance',diag==expected_diag)
check('driver_recorded_actual_diagnostic_sha',read('provenance.json')['engine_sha256']==diag)
check('production_sha_unchanged',prodhash==expected_prod)
check('production_symlink_unchanged',str(prod.resolve())=='/root/private_data/strata-dcu/joint-draft-20261023/strata-final')
groups=collections.defaultdict(list);widths=collections.Counter();all_int=True;rowcount=0;host_order=True
with (p/'lifecycle.csv').open() as f:
 for row in csv.reader(f):
  widths[len(row)]+=1;rowcount+=1
  try:v=list(map(int,row))
  except ValueError:all_int=False;continue
  if len(v)!=43:continue
  groups[tuple(v[:4])].append(v[4]);host=v[40:43]
  if v[4]<48 and any(host):host_order &= host[0]<=host[1]<=host[2]<=v[0]
check('lifecycle_43_integer_columns',all_int and set(widths)=={43})
check('lifecycle_49_rows_each_window',bool(groups) and all(sorted(ls)==list(range(49)) for ls in groups.values()))
check('lifecycle_host_timestamps_ordered',host_order)
with (p/'cpu.csv').open() as f:
 reader=csv.DictReader(f);header=reader.fieldnames;cpucount=0;valid=True
 for row in reader:
  cpucount+=1
  valid &= int(row['end_ns'])>=int(row['begin_ns']) and all(float(row[n])>=0 for n in ['plan_ms','actq_ms','jobs_ms','compute_ms'])
check('cpu_named_columns',header==['begin_ns','end_ns','layer','T','plan_ms','actq_ms','jobs_ms','compute_ms','cpu_experts','cpu_entries','gpu_hits','pcie_experts'])
check('cpu_nonempty_valid_timings',cpucount>0 and valid)
ps=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True)
processes=[line for line in ps.splitlines() if ('strata-profile' in line or 'dflash2_worker.py' in line or ('verify-lifecycle-20261023/run.py' in line))]
check('inference_processes_exited',not processes)
details.update(requests=[{'case':x['case'],'tokens':len(x['generated_ids']),'decode_tps':x['decode_tps']} for x in r],weight_usage=u,protocol=protocol,qa_status=q,lifecycle_rows=rowcount,lifecycle_windows=len(groups),lifecycle_widths=dict(widths),cpu_rows=cpucount,cpu_header=header,diagnostic_sha256=diag,production_sha256=prodhash,production_target=str(prod.resolve()),remaining_processes=processes)
print(json.dumps({'pass':all(checks.values()),'checks':checks,'details':details},ensure_ascii=False,indent=2))
