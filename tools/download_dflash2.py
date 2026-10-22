"""Download the pinned DFlash2 checkpoint through HF Mirror, then ModelScope; verify SHA-256."""
import argparse,hashlib,json,time
from pathlib import Path
import requests
REPO='incoai/Qwen3.8-Flash-Next-DFlash2'
REV='a22a1744e85c17364dcc54a6a420af7135d4c7f6'
FILES={'config.json':('2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9',1236),'model.safetensors':('d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f',1206140008)}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser(__doc__);ap.add_argument('--out',required=True);ap.add_argument('--source',choices=['auto','hf-mirror','modelscope'],default='auto');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 records={}
 for name,(expected,size) in FILES.items():
  dst=out/name
  if dst.exists() and dst.stat().st_size==size and sha(dst)==expected:
   records[name]={'bytes':size,'sha256':expected,'source':'verified existing'};continue
  tmp=dst.with_suffix(dst.suffix+'.partial')
  origins=[('hf-mirror',f'https://hf-mirror.com/{REPO}/resolve/{REV}/{name}'),('modelscope',f'https://modelscope.cn/models/{REPO}/resolve/master/{name}')]
  if a.source!='auto':origins=[x for x in origins if x[0]==a.source]
  for origin,url in origins:
   try:
    offset=tmp.stat().st_size if tmp.exists() else 0
    print('DOWNLOAD',origin,name,'resume_bytes',offset,flush=True)
    with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(15,45)) as r:
     r.raise_for_status()
     if offset and r.status_code!=206:offset=0
     if offset and not r.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('invalid range')
     done=offset;next_log=done+(32<<20);start=time.monotonic()
     with tmp.open('ab' if offset else 'wb') as f:
      for chunk in r.iter_content(1<<20):
       f.write(chunk);done+=len(chunk)
       if time.monotonic()-start>30 and (done-offset)/(time.monotonic()-start)<128*1024:raise TimeoutError('source too slow')
       if done>size:raise ValueError('oversized download')
       if done>=next_log:print('PROGRESS',origin,name,done,'/',size,'MiB/s',round((done-offset)/(time.monotonic()-start)/2**20,2),flush=True);next_log=done+(32<<20)
    if tmp.stat().st_size!=size or sha(tmp)!=expected:raise ValueError('size/SHA-256 mismatch')
    tmp.replace(dst);records[name]={'bytes':size,'sha256':expected,'source':origin};print('VERIFIED',name,size,flush=True);break
   except Exception as e:
    # Network exceptions can contain authenticated proxy URLs. Log their class only.
    print('FAILED',origin,name,type(e).__name__,flush=True)
  else:raise RuntimeError('both remote sources failed; use local transit')
 (out/'download-manifest.json').write_text(json.dumps({'repo':REPO,'revision':REV,'files':records},indent=2)+'\n')
 print('DOWNLOAD COMPLETE',flush=True)
if __name__=='__main__':main()
