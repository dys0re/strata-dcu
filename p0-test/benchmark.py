"""Single-request MTP comparison, fixed target cache and exact routing settings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ap = argparse.ArgumentParser(__doc__)
ap.add_argument('--root', default='/root/strata-dcu')
ap.add_argument('--out', required=True)
ap.add_argument('--modes', default='off,fixed2,fixed3,fixed4,fixed5,adaptive')
a = ap.parse_args()
root, out = Path(a.root), Path(a.out)
out.mkdir(parents=True, exist_ok=False)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/root/private_data/Qwen3.8-F', local_files_only=True)
tok.chat_template = Path('/root/private_data/Qwen3.8-F/chat_template.jinja').read_text()
story = 'Write a detailed 800-word story about a robot exploring an abandoned observatory. Begin the story immediately and continue with rich descriptions and dialogue.'
cases = [('warmup', story, 128, None, 0),
         ('ok', 'Reply with exactly OK.', 16, 'OK', 0),
         ('arithmetic', 'What is 2+2? Answer only the number.', 16, '4', 0),
         ('capital', '中国的首都是哪座城市？只回答城市名称。', 16, '北京', 0),
         ('story', story, 256, None, 0),
         ('chinese', '请详细解释大语言模型的 prefill 和 decode 阶段有什么区别，以及它们各自的性能瓶颈。请分段说明并举例。', 256, None, 0),
         ('code', 'Write a Python LRU cache class using OrderedDict. Include get and put methods, explain their complexity, and give a usage example.', 256, None, 0),
         ('sample', 'Write a short story about a lost compass and a mountain village.', 64, None, .7)]
base = json.loads((root/'autoround/run-command.json').read_text())
for flag, takes_value in [('--mtp', True), ('--mtp-adaptive', False)]:
    if flag in base:
        i = base.index(flag)
        del base[i:i + (2 if takes_value else 1)]
for flag, value in [('--spec', '5'), ('--mtp-max-t', '1'), ('--expert-cache', '19000')]:
    base[base.index(flag)+1] = value
base += ['--serve', '--prompt-cache', '0', '--conversation-cache-mib', '0', '--pcie-frac', '0', '--adapt-swaps', '0']
results = []
for mode in a.modes.split(','):
    assert mode in ['off', 'fixed2', 'fixed3', 'fixed4', 'fixed5', 'adaptive']
    cmd = base.copy()
    if mode != 'off':
        cmd += ['--mtp', str(root/'autoround/mtp')]
        cmd[cmd.index('--mtp-max-t')+1] = '5' if mode == 'adaptive' else mode[-1]
        if mode == 'adaptive': cmd += ['--mtp-adaptive']
    env = os.environ.copy()
    for key in ['STRATA_VERIFY_PROFILE', 'STRATA_DUMP_FIRST_LOGITS', 'STRATA_SPEC_PROB', 'STRATA_SH_STREAM']:
        env.pop(key, None)
    env.update(STRATA_BF16_Q8_VEC4='1', STRATA_ROUTE_TAIL_SKIP='0', STRATA_SPEC_COUPLED='1',
               STRATA_DECODE_TIMING='1', STRATA_SPEC_DEPTH='1')
    env['STRATA_DUMP_FIRST_LOGITS'] = str(out/(mode+'.logits.f32'))
    (out/(mode+'.command.json')).write_text(json.dumps({'command': cmd,
        'overrides': {k:env[k] for k in ['STRATA_BF16_Q8_VEC4', 'STRATA_ROUTE_TAIL_SKIP', 'STRATA_SPEC_COUPLED']}},indent=2)+'\n')
    print('START', mode, flush=True)
    with (out/(mode+'.stderr.log')).open('w') as stderr, (out/(mode+'.stdout.log')).open('w') as stdout:
        p = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
        def line():
            s = p.stdout.readline()
            if not s: raise RuntimeError(f'{mode} engine EOF, exit {p.poll()}')
            stdout.write(s); stdout.flush()
            if s.startswith('ERR '): raise RuntimeError(s)
            return s.strip()
        try:
            while not line().startswith('READY '): pass
            selected = list(cases)
            if mode == 'adaptive':
                selected += [('context_tail', 'Continue a long story for several paragraphs.', 3, None, 0)]
            for case, prompt, length, expected, temperature in selected:
                ids = tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False)
                if case == 'context_tail':
                    filler = tok.encode(' word', add_special_tokens=False)[0]
                    ids = ids[:3] + [filler] * (501-len(ids)) + ids[3:]
                    assert len(ids)==501  # Existing service admission reserves eight tokens.
                assert len(ids)+length<=512
                sampling = f'temperature={temperature} top_p=0.9 top_k=20 seed=1729' if temperature else 'temperature=0'
                request = f'GEN {length} ckpt=0 {sampling} '+','.join(map(str,ids))+'\n'
                (out/(mode+'-'+case+'.request.txt')).write_text(request)
                t0 = time.monotonic(); p.stdin.write(request); p.stdin.flush()
                generated=[]
                while True:
                    s=line()
                    if s.startswith('T '): generated.append(int(s.split()[1]))
                    if s.startswith('DONE '): done=s.split(); break
                assert int(done[1])==len(generated) and int(done[2])==len(ids) and len(generated)<=length
                assert int(done[8])==0 and done[5] in ['length','stop'],done
                text=tok.decode(generated,skip_special_tokens=True)
                if expected is not None: assert text.strip()==expected,(mode,case,text)
                result={'mode':mode,'case':case,'prompt_ids':ids,'generated_ids':generated,'text':text,'done':s,
                    'decode_ms':float(done[4]),'decode_tps':1000*len(generated)/float(done[4]),
                    'wall_s':time.monotonic()-t0,'expected':expected}
                results.append(result)
                (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
                print('DONE',mode,case,len(generated),'tokens',round(result['decode_tps'],2),'TPS',flush=True)
            p.stdin.write('QUIT\n');p.stdin.flush();p.wait(timeout=30)
            assert p.returncode==0
        finally:
            if p.poll() is None:p.kill();p.wait()
    err=(out/(mode+'.stderr.log')).read_text()
    assert 'Launch params' not in err and 'larger than launch bounds' not in err
print('All modes completed, short answers and request bounds passed',flush=True)
