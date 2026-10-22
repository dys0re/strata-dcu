"""Live STOP/recovery, seeded repeat and invalid-option checks for adaptive MTP."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/root/strata-dcu')
out=root/'profiles/p0-protocol'
out.mkdir(exist_ok=False)
cmd=json.loads((root/'profiles/p0-final-matrix/adaptive.command.json').read_text())['command']
env=os.environ.copy()
for key in ['STRATA_MTP_FRONT_CHECK','STRATA_SPEC_PROB','STRATA_VERIFY_PROFILE','STRATA_DUMP_FIRST_LOGITS']:
    env.pop(key,None)
env.update(STRATA_BF16_Q8_VEC4='1',STRATA_ROUTE_TAIL_SKIP='0',STRATA_SPEC_COUPLED='1')
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('/root/private_data/Qwen3.8-F',local_files_only=True)
t.chat_template=Path('/root/private_data/Qwen3.8-F/chat_template.jinja').read_text()
def request(prompt,n=64,sampling='temperature=0'):
    ids=t.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    return f'GEN {n} ckpt=0 {sampling} '+','.join(map(str,ids))+'\n'
records=[]
with (out/'stderr.log').open('w') as stderr,(out/'stdout.log').open('w') as stdout:
    p=subprocess.Popen(cmd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)
    def read():
        s=p.stdout.readline()
        if not s:raise RuntimeError(f'EOF, exit {p.poll()}')
        stdout.write(s);stdout.flush();return s.strip()
    def send(s):p.stdin.write(s);p.stdin.flush()
    def finish(stop_at=0):
        ids=[];sent=False
        while True:
            s=read()
            if s.startswith('ERR '):raise RuntimeError(s)
            if s.startswith('T '):
                ids.append(int(s.split()[1]))
                if stop_at and len(ids)>=stop_at and not sent:send('STOP\n');sent=True
            if s.startswith('DONE '):return {'ids':ids,'done':s,'text':t.decode(ids,skip_special_tokens=True)}
    try:
        while not read().startswith('READY '):pass
        send(request('Write a detailed 800-word story about a robot exploring an abandoned observatory.',256))
        cancelled=finish(stop_at=8)
        assert cancelled['done'].split()[5]=='cancel',cancelled
        records.append({'case':'cancel','result':cancelled})
        send(request('Reply with exactly OK.',16));recovered=finish()
        assert recovered['text'].strip()=='OK',recovered
        records.append({'case':'recovery','result':recovered})
        send(request('Reply with exactly OK.',4,'spec_min_p=0.5'))
        invalid=read();assert invalid.startswith('ERR --mtp-adaptive'),invalid
        records.append({'case':'invalid_request','result':invalid})
        samples=[]
        for _ in range(2):
            send(request('Write a short story about a lost compass and a mountain village.',64,
                         'temperature=0.7 top_p=0.9 top_k=20 seed=1729'))
            samples.append(finish())
        assert samples[0]['ids']==samples[1]['ids'],'seeded repeats changed with policy history'
        records.append({'case':'seeded_repeat','results':samples})
        send(request('Reply with exactly OK.',1));limited=finish()
        assert len(limited['ids'])==1 and limited['done'].split()[5]=='length',limited
        records.append({'case':'one_token_limit','result':limited})
        send('QUIT\n');p.wait(timeout=30);assert p.returncode==0
    finally:
        if p.poll() is None:p.kill();p.wait()
negatives=[]
for name in ['probabilistic','lookup','missing_mtp']:
    args=cmd.copy();negative_env=env.copy()
    if name=='probabilistic':negative_env['STRATA_SPEC_PROB']='1'
    elif name=='lookup':args[args.index('--suffix-draft')+1]='3'
    else:
        i=args.index('--mtp');del args[i:i+2]
    run=subprocess.run(args,env=negative_env,input='',text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    assert run.returncode==2,(name,run.returncode,run.stderr[-1000:])
    negatives.append({'case':name,'exit':run.returncode,'stderr':run.stderr})
(out/'results.json').write_text(json.dumps({'protocol':records,'invalid_options':negatives,'engine_exit':0},ensure_ascii=False,indent=2)+'\n')
print('PASS: STOP/recovery, invalid request, seeded repeat, one-token limit, three invalid configurations',flush=True)
