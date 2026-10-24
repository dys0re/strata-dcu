"""Additional executable answer checks for non-expert weight changes."""
import json
import subprocess
import sys

CASES = [
    ('code-search', 'Return only Python code defining lower_bound(a, x): a is a sorted list of integers. Return the first index whose value is >= x, or len(a) if none. Use binary search. No imports, Markdown or explanation.', 256, None, 0),
    ('code-intervals', 'Return only Python code defining merge_intervals(intervals), accepting integer [start,end] pairs in any order. Return sorted non-overlapping pairs, merging overlaps and touching endpoints. Empty input returns []. No imports, Markdown or explanation.', 256, None, 0),
    ('code-encoding', 'Return only Python code defining rle(s), returning a list of [character,count] pairs for consecutive runs. Empty string returns []. No imports, Markdown or explanation.', 256, None, 0),
    ('data-transform', 'For records [{"name":"c","score":2},{"name":"a","score":5},{"name":"b","score":5}], sort by score descending, then name ascending. Return only a JSON array of names.', 64, None, 0),
]

HARNESS = '''import ast,bisect,json,sys
x=json.loads(sys.argv[1]);tree=ast.parse(x['code'])
assert not any(isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.Attribute) and n.attr.startswith('_') for n in ast.walk(tree))
ns={'__builtins__':{'len':len,'range':range,'min':min,'max':max,'sorted':sorted,'enumerate':enumerate,'list':list,'tuple':tuple,'zip':zip}}
exec(compile(tree,'generated-functional-qa','exec'),ns)
if x['case']=='code-search':
 for a in [[],[1],[1,1,3,5],[-5,-2,0,0,9],list(range(100))]:
  for q in range(-7,103):assert ns['lower_bound'](a,q)==bisect.bisect_left(a,q)
elif x['case']=='code-intervals':
 for a,b in [([],[]),([[1,2]],[[1,2]]),([[5,7],[1,3],[3,5]],[[1,7]]),([[8,9],[-3,-1],[1,2]],[[-3,-1],[1,2],[8,9]]),([[1,10],[2,3],[10,11]],[[1,11]])]:assert ns['merge_intervals'](a)==b
else:
 for a,b in [('',[]),('a',[['a',1]]),('aaabbcca',[['a',3],['b',2],['c',2],['a',1]]),('你好好你',[['你',1],['好',2],['你',1]])]:assert ns['rle'](a)==b
'''


def check(name, text):
    if name == 'data-transform':
        value = text.strip()
        if value.startswith('```'):
            value = '\n'.join(value.splitlines()[1:-1])
        assert json.loads(value) == ['a', 'b', 'c'], text
    elif name in ('code-search', 'code-intervals', 'code-encoding'):
        code = text.strip()
        if code.startswith('```'):
            code = '\n'.join(code.splitlines()[1:-1])
        result = subprocess.run([sys.executable, '-c', HARNESS, json.dumps(dict(case=name, code=code))],
                                capture_output=True, text=True, timeout=5)
        assert result.returncode == 0, (name, code, result.stderr)
