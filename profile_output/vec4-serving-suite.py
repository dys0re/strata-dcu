"""Compare isolated stdin engine sessions: no HTTP listener or prefix reuse."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ap = argparse.ArgumentParser(__doc__)
ap.add_argument('--root', default='/root/strata-dcu')
ap.add_argument('--out', required=True)
a = ap.parse_args()
root, out = Path(a.root), Path(a.out)
out.mkdir(parents=True, exist_ok=False)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/root/private_data/Qwen3.8-F', local_files_only=True)
tok.chat_template = Path('/root/private_data/Qwen3.8-F/chat_template.jinja').read_text()
cases = [('ok', 'Reply with exactly OK.', 16, 'OK'),
         ('arithmetic', 'What is 2+2? Answer only the number.', 16, '4'),
         ('capital', '中国的首都是哪座城市？只回答城市名称。', 16, '北京'),
         ('code', 'Write a Python LRU cache class using OrderedDict. Include get and put methods, explain their complexity, and give a usage example.', 256, None),
         ('chinese', '请详细解释大语言模型的 prefill 和 decode 阶段有什么区别，以及它们各自的性能瓶颈。请分段说明并举例。', 256, None)]
cmd = json.loads((root/'autoround/run-command.json').read_text())
cmd += ['--serve', '--prompt-cache', '0', '--conversation-cache-mib', '0']
report = []
for arm in ['baseline', 'vec4']:
    env = os.environ.copy()
    for key in ['STRATA_VERIFY_PROFILE', 'STRATA_LOOKAHEAD', 'STRATA_ONE_TOKEN_COMMIT', 'STRATA_DOORBELL_STORE']:
        env.pop(key, None)
    env['STRATA_BF16_Q8_VEC4'] = '1' if arm == 'vec4' else '0'
    env['STRATA_DUMP_FIRST_LOGITS'] = str(out/(arm+'.logits.f32'))
    (out/(arm+'.command.json')).write_text(json.dumps({'command': cmd,
        'overrides': {k:env[k] for k in ['STRATA_BF16_Q8_VEC4', 'STRATA_DUMP_FIRST_LOGITS']}}, indent=2)+'\n')
    print('START', arm, flush=True)
    with (out/(arm+'.stderr.log')).open('w') as stderr, (out/(arm+'.stdout.log')).open('w') as stdout:
        p = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
        def line():
            s = p.stdout.readline()
            if not s:
                raise RuntimeError(f'Engine EOF, {arm}, exit {p.poll()}')
            stdout.write(s)
            stdout.flush()
            if s.startswith('ERR '):
                raise RuntimeError(s)
            return s.strip()
        try:
            while not line().startswith('READY '):
                pass
            for case, prompt, length, expected in cases:
                ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
                assert len(ids)+length <= 512
                request = f'GEN {length} ckpt=0 '+','.join(map(str, ids))+'\n'
                (out/(arm+'-'+case+'.request.txt')).write_text(request)
                p.stdin.write(request)
                p.stdin.flush()
                generated = []
                while True:
                    s = line()
                    if s.startswith('T '):
                        generated.append(int(s.split()[1]))
                    if s.startswith('DONE '):
                        done = s.split()
                        break
                assert int(done[1]) == len(generated) and int(done[2]) == len(ids)
                assert done[5] in ['length', 'stop'] and int(done[8]) == 0, done
                text = tok.decode(generated, skip_special_tokens=True)
                if expected is not None:
                    assert text.strip() == expected, (arm, case, text)
                result = {'arm': arm, 'case': case, 'prompt_ids': ids, 'generated_ids': generated,
                          'text': text, 'expected': expected, 'done': s,
                          'decode_ms': float(done[4]), 'decode_tps': 1000*len(generated)/float(done[4])}
                report.append(result)
                (out/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                print('DONE', arm, case, len(generated), 'tokens',round(result['decode_tps'],2),'TPS',flush=True)
            p.stdin.write('QUIT\n')
            p.stdin.flush()
            p.wait(timeout=30)
            assert p.returncode == 0
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
    errors = (out/(arm+'.stderr.log')).read_text()
    assert 'Launch params' not in errors and 'larger than launch bounds' not in errors
print('Both engine sessions exited 0; all short-answer assertions passed',flush=True)
