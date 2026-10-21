#!/usr/bin/env python3
import concurrent.futures, hashlib, json, os, time, urllib.parse
import requests
CAP=8*1024*1024
GH="https://github.com/BurntSushi/ripgrep/releases/download/14.1.0/ripgrep-14.1.0-x86_64-unknown-linux-musl.tar.gz"
HF="/google-bert/bert-base-uncased/resolve/main/model.safetensors"
TARGETS=[("GitHub",h,h+"/"+GH,"gzip") for h in ["https://ghfast.top","https://ghproxy.net","https://gh-proxy.com","https://gh-proxy.org"]]
TARGETS += [("GitHub","github.com",GH,"gzip")]
TARGETS += [("HF",h,h+HF,"safetensors") for h in ["https://hf-mirror.com","https://hf-mirror.net","https://alpha.hf-mirror.com","https://huggingface.co"]]
def bench(target,round_no):
 group,name,url,fmt=target
 result=dict(group=group,site=name,round=round_no,url=url)
 session=requests.Session(); session.trust_env=False
 session.proxies=requests.utils.get_environ_proxies(url)
 start=time.monotonic(); received=0; data=bytearray(); response=None; error=None; first=None
 try:
  response=session.get(url,headers={"Range":f"bytes=0-{CAP-1}","Accept-Encoding":"identity","User-Agent":"network-benchmark/1.0"},stream=True,timeout=(5,8))
  result.update(status=response.status_code,content_type=response.headers.get("Content-Type"),content_range=response.headers.get("Content-Range"),final_host=urllib.parse.urlsplit(response.url).hostname)
  if response.status_code not in (200,206):
   result["error"]="HTTP "+str(response.status_code)
  else:
   for chunk in response.iter_content(65536):
    if not chunk: continue
    now=time.monotonic()
    if first is None:first=now
    chunk=chunk[:CAP-received];received+=len(chunk);data.extend(chunk)
    if received>=CAP or now-start>30:break
 except Exception as e:error=type(e).__name__;result["error"]=error
 finally:
  if response is not None:response.close()
  session.close()
 elapsed=time.monotonic()-start
 valid=False
 if fmt=="gzip":valid=data[:2]==b"\x1f\x8b"
 elif len(data)>12:
  headerlen=int.from_bytes(data[:8],"little");valid=0<headerlen<10*1024*1024 and data[8:9]==b"{"
 result.update(bytes=received,seconds=round(elapsed,3),ttfb=round(first-start,3) if first else None,mib_per_s=round(received/(1024**2)/elapsed,3),valid_payload=bool(valid),sha256=hashlib.sha256(data).hexdigest() if received else None,prefix128k_sha256=hashlib.sha256(data[:131072]).hexdigest() if received>=131072 else None,success=bool(valid and received>=1024**2 and error is None))
 return result
def main():
 results=[]
 print(json.dumps({"event":"start","utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"cap_mib":CAP//1024**2,"proxy_hosts":sorted({urllib.parse.urlsplit(v).hostname for k,v in requests.utils.get_environ_proxies("https://example.com").items() if k!="no"})}),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  futures=[ex.submit(bench,t,1) for t in TARGETS]
  for f in concurrent.futures.as_completed(futures):
   r=f.result();results.append(r);print(json.dumps(r),flush=True)
 # Repeat usable endpoints one at a time to avoid competition between mirrors.
 usable=[t for t in TARGETS if any(r["site"]==t[1] and r["success"] for r in results)]
 for round_no in (2,3):
  for target in usable:
   r=bench(target,round_no);results.append(r);print(json.dumps(r),flush=True)
 path="/tmp/scnet-mirror-benchmark-results.json"
 with open(path,"w") as f:json.dump(results,f,indent=2)
 print(json.dumps({"event":"complete","results":path,"tests":len(results)}),flush=True)
if __name__=="__main__":main()
