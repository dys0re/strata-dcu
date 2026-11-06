"""Recover text expert GGUFs from surviving AutoRound shards; never deletes sources.
Each layer is independently atomic and verified before publication. PLE is external.
"""
import argparse, concurrent.futures, hashlib, json, os, struct, sys
from pathlib import Path
import numpy as np
from nvfp4_local_pack import Checkpoint
from autoround_pack import AutoRoundCheckpoint, q4_0_reblock, decode_gptq_reference, decode_q4_0_reference
import gguf_reader as G

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--out',required=True);a.add_argument('--spill',required=True);a.add_argument('--native',required=True);a.add_argument('--dense',required=True);a.add_argument('--pack-dense',required=True);args=a.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True);spill=Path(args.spill);spill.mkdir(parents=True,exist_ok=True)
 ck=AutoRoundCheckpoint.__new__(AutoRoundCheckpoint);Checkpoint.__init__(ck,args.model,partial_ok=True)
 c=json.loads((ck.dir/'config.json').read_text());t=c['text_config'];q=c['quantization_config']
 assert q['bits']==4 and q['sym'] and q['packing_format']=='auto_round:auto_gptq'
 ck.group_size=q['group_size'];ck.n_embd=t['hidden_size'];ck.n_ff=t['moe_intermediate_size'];ck.experts=t['num_experts'];ck.layers=t['num_hidden_layers']
 w=json.loads((ck.dir/'model.safetensors.index.json').read_text())['weight_map']
 missing=[n for n in w if n not in ck.meta]
 assert missing and all('.ple.ple_embedding.ngram_embedding.' in n for n in missing),missing[:3]
 native=Path(args.native);records=[];lines=[];offset=0
 for layer in range(ck.layers):
  name=f'experts-layer-{layer:02d}.gguf';p=(out if layer<24 else spill)/name;receipt=p.with_suffix('.verified.json')
  if receipt.exists() and p.exists() and digest(p)==json.loads(receipt.read_text())['sha256']:
   print('reuse',layer,flush=True)
  else:
   shapes=[(ck.n_embd,ck.n_ff,ck.experts),(ck.n_embd,ck.n_ff,ck.experts),(ck.n_ff,ck.n_embd,ck.experts)]
   def string(s):
    b=s.encode();return struct.pack('<Q',len(b))+b
   header=struct.pack('<IIQQ',0x46554747,3,3,0);at=0
   for role,shape in zip(['gate','up','down'],shapes):
    header+=string(f'blk.{layer}.ffn_{role}_exps.weight')+struct.pack('<I3QIQ',3,*shape,2,at);at+=int(np.prod(shape))*18//32
   header+=b'\0'*(-len(header)%32)
   tmp=p.with_suffix('.tmp');hashes=[]
   with open(tmp,'wb') as f:
    f.write(header)
    for role in ['gate','up','down']:
     def one(e):return q4_0_reblock(*ck.projection(layer,e,role),ck.group_size)
     h=hashlib.sha256()
     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
      for e,arr in enumerate(pool.map(one,range(ck.experts))):
       if e in (0,ck.experts//2,ck.experts-1):
        np.testing.assert_array_equal(decode_q4_0_reference(arr),decode_gptq_reference(*ck.projection(layer,e,role),ck.group_size))
       b=arr.tobytes();f.write(b);h.update(b)
     hashes.append(h.hexdigest())
    f.flush();os.fsync(f.fileno())
   g=G.GGUFFile(tmp)
   with open(tmp,'rb') as f:
    for tensor,h in zip(g.tensors,hashes):
     f.seek(g.data_start+tensor.offset);hh=hashlib.sha256();remaining=tensor.expected_bytes()
     while remaining:
      b=f.read(min(8<<20,remaining));assert b;hh.update(b);remaining-=len(b)
     assert hh.hexdigest()==h
   tmp.replace(p);receipt.write_text(json.dumps({'sha256':digest(p),'tensor_sha256':hashes,'layer':layer,'reference_checks':9},indent=2))
   print('verified layer',layer,str(p),flush=True)
  link=native.parent/name
  if not link.exists():link.symlink_to(p)
  g=G.GGUFFile(p);per=[x.expected_bytes()//ck.experts for x in g.tensors];blob=sum(per)
  lines.append(f'{layer} 2 2 {offset} {blob} '+ ' '.join(str(g.data_start+x.offset) for x in g.tensors)+' '+name+'\n');offset+=blob*ck.experts
  records.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
 # Build the same dense pack ABI as the original importer. Hardlink its surviving data after checking it.
 import iq_pack
 if not (out/'index.txt').exists():
  stage=spill/'dense-stage';stage.mkdir(exist_ok=True)
  assert iq_pack.index_standalone(Path(args.dense),stage,iq_pack.Model(Path(args.dense)))==0
  assert digest(stage/'dense.bin')==digest(args.pack_dense)
  (stage/'dense.bin').unlink()
  import shutil
  for item in stage.iterdir(): shutil.move(str(item),str(out/item.name))
  os.link(args.pack_dense,out/'dense.bin')
 head=f'# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert {ck.experts}, total {offset}; absolute offsets in {native.name}, or in the named shard beside it)\n'
 (out/'native_experts.txt').write_text(head+''.join(lines))
 (out/'recovery.json').write_text(json.dumps({'layers':ck.layers,'experts':ck.experts,'expert_bytes':offset,'shards':records,'missing_source_is_ple_only':True},indent=2))
 print('RECOVERY_CONVERSION_PASS',flush=True)
if __name__=='__main__':main()
