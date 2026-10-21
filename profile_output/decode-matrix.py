"""Serial full-engine A/B runs; stores commands, raw logs, IDs and optional logits."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

ap = argparse.ArgumentParser(__doc__)
ap.add_argument('--root', default='/root/strata-dcu')
ap.add_argument('--out', required=True)
ap.add_argument('--engine')
ap.add_argument('--arms', default='baseline,vec4')
ap.add_argument('--cases', default='story')
ap.add_argument('--runs', type=int, default=1)
ap.add_argument('--logits', action='store_true')
a = ap.parse_args()
root, out = Path(a.root), Path(a.out)
out.mkdir(parents=True, exist_ok=False)
arms = {'baseline': {}, 'no_lookahead': {'STRATA_LOOKAHEAD': '0'},
        'vec4': {'STRATA_BF16_Q8_VEC4': '1'},
        'vec4_serial': {'STRATA_BF16_Q8_VEC4': '1', 'STRATA_SH_STREAM': '0'},
        'combined': {'STRATA_BF16_Q8_VEC4': '1', 'STRATA_LOOKAHEAD': '0'},
        'commit': {'STRATA_BF16_Q8_VEC4': '1', 'STRATA_LOOKAHEAD': '0', 'STRATA_ONE_TOKEN_COMMIT': '1'},
        'doorbell': {'STRATA_BF16_Q8_VEC4': '1', 'STRATA_LOOKAHEAD': '0', 'STRATA_DOORBELL_STORE': '1'}}
cases = {'story': ('Write a detailed 800-word story about a robot exploring an abandoned observatory. Begin the story immediately and continue with rich descriptions and dialogue.', 256),
         'code': ('Write a Python LRU cache class using OrderedDict. Include get and put methods, explain their complexity, and give a usage example.', 256),
         'chinese': ('请详细解释大语言模型的 prefill 和 decode 阶段有什么区别，以及它们各自的性能瓶颈。请分段说明并举例。', 256),
         'ok': ('Reply with exactly OK.', 16),
         'arithmetic': ('What is 2+2? Answer only the number.', 16),
         'capital': ('中国的首都是哪座城市？只回答城市名称。', 16)}
selected_arms, selected_cases = a.arms.split(','), a.cases.split(',')
assert all(x in arms for x in selected_arms) and all(x in cases for x in selected_cases)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/root/private_data/Qwen3.8-F', local_files_only=True)
tokenizer.chat_template = Path('/root/private_data/Qwen3.8-F/chat_template.jinja').read_text()
base = json.loads((root/'autoround/run-command.json').read_text())
if a.engine:
    base[0] = a.engine
report = []
for repeat in range(a.runs):
    # Alternate the order for repeated pairs.
    for arm in selected_arms if repeat % 2 == 0 else selected_arms[::-1]:
        for case in selected_cases:
            prompt, length = cases[case]
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False)
            assert len(ids) + length <= 512
            tag = f'{repeat}-{arm}-{case}'
            token_file = out/(tag+'.ids')
            token_file.write_text(','.join(map(str, ids))+'\n')
            cmd = base.copy()
            cmd[cmd.index('--tokens-file')+1] = str(token_file)
            cmd[cmd.index('--max-new')+1] = str(length)
            env = os.environ.copy()
            for key in ['STRATA_VERIFY_PROFILE', 'STRATA_BF16_Q8_VEC4', 'STRATA_LOOKAHEAD',
                        'STRATA_ONE_TOKEN_COMMIT', 'STRATA_DOORBELL_STORE', 'STRATA_SH_STREAM', 'STRATA_DUMP_FIRST_LOGITS']:
                env.pop(key, None)
            env.update(arms[arm])
            if a.logits:
                env['STRATA_DUMP_FIRST_LOGITS'] = str(out/(tag+'.logits.f32'))
            print('START', tag, flush=True)
            (out/(tag+'.command.json')).write_text(json.dumps({'command': cmd, 'overrides': arms[arm]}, indent=2)+'\n')
            started = time.monotonic()
            with (out/(tag+'.log')).open('w') as stdout, (out/(tag+'.stderr.log')).open('w') as stderr:
                run = subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr)
            raw = (out/(tag+'.log')).read_text()
            errors = (out/(tag+'.stderr.log')).read_text()
            assert run.returncode == 0, (tag, run.returncode, errors[-2000:])
            assert 'Launch params' not in errors and 'larger than launch bounds' not in errors
            generated = list(map(int, re.search(r'^output\s*:(.*)$', raw, re.M).group(1).split()))
            m = re.search(r'^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s', raw, re.M)
            assert m and int(m[1]) == len(generated)
            result = {'repeat': repeat, 'arm': arm, 'case': case, 'engine_exit': run.returncode,
                      'prompt_ids': ids, 'generated_ids': generated, 'text': tokenizer.decode(generated, skip_special_tokens=True),
                      'decode_ms': float(m[2]), 'decode_tps': float(m[3]), 'wall_s': time.monotonic()-started}
            (out/(tag+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            report.append(result)
            (out/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print('DONE',tag,'tokens',len(generated),'TPS',result['decode_tps'],'wall_s',round(result['wall_s'],1),flush=True)
