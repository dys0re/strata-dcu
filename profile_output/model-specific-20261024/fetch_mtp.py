"""Fetch only Basalt's published MTP dense section through HF Mirror; never whole model packs."""
import hashlib,json,time,urllib.request,sys
from pathlib import Path
root=Path(sys.argv[1] if len(sys.argv)>1 else '/root/private_data/strata-dcu/model-specific-20261024/basalt-mtp-source')
root.mkdir(parents=True,exist_ok=True)
base='https://hf-mirror.com';repo='jesdga/Qwen3.8-Flash-Next-Basalt'
meta=json.load(urllib.request.urlopen(f'{base}/api/models/{repo}',timeout=30));rev=meta['sha']
entries=json.load(urllib.request.urlopen(f'{base}/api/models/{repo}/tree/{rev}/convert/mtp/rt',timeout=30))
wanted=['dense.txt','dense-tuned.json','dense.bin']
receipt={'repo':repo,'revision':rev,'endpoint':base,'files':{}}
for name in wanted:
 info=next(x for x in entries if x['path']==f'convert/mtp/rt/{name}')
 url=f'{base}/{repo}/resolve/{rev}/convert/mtp/rt/{name}'
 path=root/name;tmp=root/(name+'.part');t=time.monotonic();h=hashlib.sha256();n=0
 with urllib.request.urlopen(url,timeout=45) as r,tmp.open('wb') as f:
  while block:=r.read(1024*1024):
   f.write(block);h.update(block);n+=len(block)
   if n%(16*1024*1024)==0:print(name,n,round(n/(time.monotonic()-t)/1048576,2),'MiB/s',flush=True)
 assert n==info['size'],(name,n,info['size'])
 if 'lfs' in info:assert h.hexdigest()==info['lfs']['oid'],name
 tmp.replace(path);receipt['files'][name]={'bytes':n,'sha256':h.hexdigest(),'source':info}
 print('VERIFIED',name,n,h.hexdigest(),flush=True)
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('FETCH_COMPLETE',rev,flush=True)
