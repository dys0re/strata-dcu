"""Experimental PyTorch DFlash2 adapter for Strata's verified single-request decode loop.

Uses the pinned upstream model implementation, a bounded context cache, and the
main checkpoint's raw BF16 embedding/head. All persistent GPU weight copies are
reported to the C++ engine before it sizes/checks its expert cache.
"""
import argparse,gc,hashlib,importlib.util,json,os,socket,struct,time,traceback
from pathlib import Path
import numpy as np
MAGIC=0x44464c32
class DraftCache:
    """Full context cache for the prototype's <=512-token witness/benchmark."""
    def __init__(self):self.layers={}
    def update(self,k,v,layer_idx,cache_kwargs=None):
        import torch
        if layer_idx in self.layers:
            pk,pv=self.layers[layer_idx];k=torch.cat([pk,k],dim=-2);v=torch.cat([pv,v],dim=-2)
        self.layers[layer_idx]=(k,v);return k,v
    def crop(self,length):
        self.layers={i:(k[...,:length,:].contiguous(),v[...,:length,:].contiguous()) for i,(k,v) in self.layers.items()}
    def length(self):return next(iter(self.layers.values()))[0].shape[-2] if self.layers else 0
class StaticDraftCache:
    """Address-stable K/V for graph replay; noise rows are overwritten and masked."""
    def __init__(self,torch):
        self.layers={i:(torch.zeros((1,8,520,128),device='cuda',dtype=torch.bfloat16),torch.zeros((1,8,520,128),device='cuda',dtype=torch.bfloat16)) for i in range(5)}
        self.positions=None;self.n=0
    def update(self,k,v,layer_idx,cache_kwargs=None):
        pk,pv=self.layers[layer_idx];pk.index_copy_(2,self.positions[0],k);pv.index_copy_(2,self.positions[0],v);return pk,pv
    def crop(self,length):self.n=length
    def length(self):return self.n
    def clear(self):
        for k,v in self.layers.values():k.zero_();v.zero_()
        self.n=0
class DraftBudget:
    """Choose 1..3 proposals using only completed draft+verification rounds."""
    def __init__(self):self.reset()
    def reset(self):self.samples={k:[] for k in range(1,4)};self.rounds=0;self.last=None
    def observe(self,rows,now):
        if self.last is None:return
        k,start=self.last;self.last=None
        if not 1<=rows<=k+1:raise ValueError('invalid accepted draft prefix')
        ms=1000*(now-start)
        self.samples[k].append((rows,ms));self.samples[k]=self.samples[k][-12:];self.rounds+=1
    def choose(self,cap):
        sizes=list(range(1,min(3,cap)+1))
        for k in sizes:
            if len(self.samples[k])<3:return k
        if self.rounds%32==0:return sizes[(self.rounds//32)%len(sizes)]
        return max(sizes,key=lambda k:sum(n for n,ms in self.samples[k])/sum(ms for n,ms in self.samples[k]))
    def report(self):return {'rounds':self.rounds,'samples':self.samples}
class Dflash2Worker:
    def __init__(self,checkpoint,target,reference=None):
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers import Qwen3Config
        self.torch=torch;torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        reference=Path(reference or Path(__file__).resolve().parents[1]/'third_party/dflash/model.py')
        spec=importlib.util.spec_from_file_location('strata_dflash_reference',reference);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        checkpoint=Path(checkpoint);target=Path(target)
        expected='d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f'
        with (checkpoint/'model.safetensors').open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:raise ValueError('DFlash2 checkpoint SHA-256 mismatch')
        cfg=Qwen3Config(**json.loads((checkpoint/'config.json').read_text()));cfg._attn_implementation='sdpa'
        if cfg.dflash_config['target_layer_ids']!=[5,15,25,35,45] or cfg.hidden_size!=2560 or cfg.vocab_size!=248320:raise ValueError('unsupported draft configuration')
        self.model=mod.DFlash2DraftModel(cfg).eval()
        state=load_file(str(checkpoint/'model.safetensors'))
        for k in ['predecessor_codebook','successor_codebook']:
            state['candidate_selector.'+k+'.weight']=state.pop('candidate_selector.'+k)
        self.model.load_state_dict(state,strict=True,assign=True)
        self.model=self.model.to(device='cuda',dtype=torch.bfloat16)
        self.model.rotary_emb=mod.Qwen3RotaryEmbedding(cfg,device='cuda')
        del state;gc.collect()
        idx=json.loads((target/'model.safetensors.index.json').read_text())['weight_map']
        with safe_open(str(target/idx['lm_head.weight']),framework='pt') as f:self.head=f.get_tensor('lm_head.weight').to('cuda')
        key='model.language_model.embed_tokens.weight'
        with safe_open(str(target/idx[key]),framework='pt') as f:self.embedding=f.get_tensor(key)
        assert self.embedding.device.type=='cpu' and self.embedding.dtype==torch.bfloat16
        assert self.head.dtype==torch.bfloat16 and self.head.shape==(248320,2560)
        self.weight_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters())+self.head.numel()*self.head.element_size()
        self.use_graph=os.environ.get('STRATA_DFLASH_GRAPH','0')=='1'
        self.graphs={}
        self.adaptive=os.environ.get('STRATA_DFLASH_ADAPTIVE','0')=='1'
        self.budget=DraftBudget();self.budget_history=[]
        if self.use_graph:
            self.static_cache=StaticDraftCache(torch)
            self._capture_graphs()
        self.timings=[]
        self.reset();self.totals={'calls':0,'draft_ms':0.0,'feature_rows':0,'resets':0}
        torch.cuda.synchronize()
    def reset(self):
        if getattr(self,'use_graph',False):
            self.cache=self.static_cache;self.cache.clear()
        else:self.cache=DraftCache()
        self.pending=[];self.received=0
        if hasattr(self,'budget'):
            if self.budget.rounds:self.budget_history.append(self.budget.report())
            self.budget.reset()
    def _graph_forward(self,features,noise,positions,anchor):
        torch=self.torch
        self.static_cache.positions=positions
        limit=positions[0,-1]+1
        mask=(torch.arange(520,device='cuda')<limit)[None,None,None,:]
        hidden=self.model(target_hidden=features,noise_embedding=noise,position_ids=positions,attention_mask=mask,past_key_values=self.static_cache,use_cache=True)[:,1:,:]
        tokens,_,_=self.model.propose(hidden,anchor,lambda x:torch.nn.functional.linear(x,self.head),0.0)
        return hidden,tokens
    def _capture_graphs(self):
        torch=self.torch
        # Warm first, outside capture: lazy HIP/BLAS kernels cannot initialize in a graph.
        with torch.inference_mode():
            for rows in range(1,5):
                for k in range(1,4):
                    features=torch.zeros((1,rows,12800),device='cuda',dtype=torch.bfloat16)
                    noise=torch.nn.functional.embedding(torch.tensor([248077]*(k+1)),self.embedding).to('cuda')[None]
                    positions=torch.arange(rows+k+1,device='cuda')[None]
                    anchor=torch.tensor([248077],device='cuda')
                    self._graph_forward(features,noise,positions,anchor);torch.cuda.synchronize()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):hidden,tokens=self._graph_forward(features,noise,positions,anchor)
                    self.graphs[(rows,k)]=(graph,features,noise,positions,anchor,hidden,tokens)
            torch.cuda.synchronize()

    def append(self,features,pos):
        if pos!=self.received:raise ValueError(f'noncontiguous target features: {pos} vs {self.received}')
        if pos+features.shape[0]>512:raise ValueError('prototype supports context <=512 only')
        self.pending.append(features.copy());self.received+=features.shape[0]
    @staticmethod
    def _finite(x):return bool(x.isfinite().all())
    def propose(self,pos,anchor,k):
        torch=self.torch
        if pos!=self.received or self.cache.length()+sum(x.shape[0] for x in self.pending)!=pos:raise ValueError('draft/target context mismatch')
        if not 1<=k<=7 or not 0<=anchor<248320 or not self.pending:raise ValueError('invalid draft request')
        if self.adaptive:k=self.budget.choose(k)
        started=time.monotonic()
        with torch.inference_mode():
            features=torch.from_numpy(np.concatenate(self.pending,axis=0)).to(device='cuda',dtype=torch.bfloat16)[None]
            ids=torch.tensor([anchor]+[248077]*k,dtype=torch.long)
            noise=torch.nn.functional.embedding(ids,self.embedding).to('cuda')[None]
            positions=torch.arange(pos-features.shape[1],pos+k+1,device='cuda')[None]
            prepared=time.monotonic()
            entry=self.graphs.get((features.shape[1],k)) if self.use_graph else None
            if entry is not None:
                graph,fi,ni,pi,ai,hidden,tokens=entry
                fi.copy_(features);ni.copy_(noise);pi.copy_(positions);ai.fill_(anchor)
                graph.replay()
                modeled=time.monotonic()
                if not self._finite(hidden):raise ValueError('nonfinite graphed drafter activations')
                modeled=time.monotonic()
            else:
                if self.use_graph:
                    self.cache.positions=positions
                    mask=(torch.arange(520,device='cuda')<pos+k+1)[None,None,None,:]
                else:mask=None
                hidden=self.model(target_hidden=features,noise_embedding=noise,position_ids=positions,attention_mask=mask,past_key_values=self.cache,use_cache=True)[:,1:,:]
                if not self._finite(hidden):raise ValueError('nonfinite drafter activations')
                modeled=time.monotonic()
                tokens,_,_=self.model.propose(hidden,torch.tensor([anchor],device='cuda'),lambda x:torch.nn.functional.linear(x,self.head),0.0)
            self.cache.crop(pos);self.pending.clear()
            result=tokens[0].to(device='cpu',dtype=torch.int32).numpy().copy()
        elapsed=1000*(time.monotonic()-started)
        self.timings.append({'pos':pos,'k':k,'context_rows':features.shape[1],'prep_ms':1000*(prepared-started),'model_ms':1000*(modeled-prepared),'head_selector_ms':elapsed-1000*(modeled-started),'draft_ms':elapsed,'graph':entry is not None})
        self.totals['calls']+=1;self.totals['draft_ms']+=elapsed
        if self.adaptive:self.budget.last=(k,started)
        return result
    def serve(self,channel,error_path):
        def read(n):
            buf=bytearray()
            while len(buf)<n:
                s=channel.recv(n-len(buf))
                if not s:raise EOFError
                buf.extend(s)
            return buf
        try:
            channel.sendall(struct.pack('<QQ',MAGIC,self.weight_bytes))
            while True:
                magic,kind,rows,pos,anchor,k=struct.unpack('<6q',read(48))
                if magic!=MAGIC or not 0<=rows<=512:raise ValueError('invalid IPC header')
                if kind==0:
                    self.reset();self.totals['resets']+=1;result=np.empty(0,dtype=np.int32)
                elif kind==1:
                    features=np.frombuffer(read(rows*12800*4),dtype='<f4').reshape(rows,12800)
                    if self.adaptive:self.budget.observe(rows,time.monotonic())
                    self.append(features,pos);self.totals['feature_rows']+=rows;result=np.empty(0,dtype=np.int32)
                elif kind==2:result=self.propose(pos,anchor,k)
                elif kind==3:
                    self.budget.last=None;result=np.empty(0,dtype=np.int32)
                else:raise ValueError('invalid IPC command')
                channel.sendall(struct.pack('<2q',MAGIC,len(result))+result.tobytes())
        except (EOFError,ConnectionResetError,BrokenPipeError):pass
        except Exception:
            Path(error_path).write_text(traceback.format_exc())
            try:channel.sendall(struct.pack('<2q',MAGIC,-1))
            except OSError:pass
        finally:channel.close()
def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--checkpoint',required=True);ap.add_argument('--target',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    worker=Dflash2Worker(a.checkpoint,a.target)
    rng=np.random.default_rng(1729);features=rng.standard_normal((4,12800),dtype=np.float32)
    results=[]
    for _ in range(2):
        worker.reset();worker.append(features,0);results.append(worker.propose(4,248044,3).tolist())
    assert results[0]==results[1]
    extra=rng.standard_normal((1,12800),dtype=np.float32)
    worker.append(extra,4);incremental=worker.propose(5,553,3).tolist()
    worker.reset();worker.append(np.concatenate([features,extra],axis=0),0)
    full=worker.propose(5,553,3).tolist()
    assert incremental==full,('incremental draft cache mismatch',incremental,full)
    report={'seed':1729,'draft_ids':results,'finite_forward':True,'repeat_equal':True,'incremental_cache_equal':True,'incremental_ids':incremental,'weight_bytes':worker.weight_bytes,'torch':worker.torch.__version__,'hip':worker.torch.version.hip,'device':worker.torch.cuda.get_device_name(0),'graphs':len(worker.graphs),'timings':worker.timings,'totals':worker.totals}
    (out/'witness.json').write_text(json.dumps(report,indent=2)+'\n');print('WITNESS DFLASH2',json.dumps(report),flush=True)
if __name__=='__main__':main()
