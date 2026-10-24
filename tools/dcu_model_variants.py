"""Prepare optional non-expert Q8 weights and a compatible, published tuned MTP runtime.

Expert weights and the source checkpoint are never rewritten. Outputs belong on persistent storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dense_layout(path, size):
    result = {}
    spans = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, kind, rows, cols, off, count = line.split()
        rows, cols, off, count = map(int, (rows, cols, off, count))
        if kind not in ('q8_0', 'bf16', 'f32') or rows <= 0 or cols <= 0 or off < 0 or off % 2:
            raise ValueError(f'invalid MTP tensor: {name}')
        expected = rows * cols * {'bf16': 2, 'f32': 4}.get(kind, 1)
        if kind == 'q8_0':
            if cols % 32:
                raise ValueError(f'partial Q8 block: {name}')
            expected = rows * (cols // 32) * 34
        if count != expected or off + count > size or name in result:
            raise ValueError(f'invalid MTP extent: {name}')
        result[name] = (kind, rows, cols, off, count)
        spans.append((off, off + count))
    spans.sort()
    if not spans or any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError('empty or overlapping MTP dense layout')
    return result


def prepare_mtp(base, source, out):
    base, source, out = map(Path, (base, source, out))
    receipt = json.loads((source / 'receipt.json').read_text())
    for name in ('dense.bin', 'dense.txt'):
        entry = receipt['files'][name]
        if (source / name).stat().st_size != entry['bytes'] or sha(source / name) != entry['sha256']:
            raise ValueError('published source digest mismatch: ' + name)
    src = dense_layout(source / 'dense.txt', (source / 'dense.bin').stat().st_size)
    dst = dense_layout(base / 'dense.txt', (base / 'dense.bin').stat().st_size)
    if src.keys() != dst.keys() or any(src[n][:3] != dst[n][:3] for n in dst):
        raise ValueError('tuned MTP names, kinds or shapes differ from the local runtime')
    ready = json.loads((base / 'ready.json').read_text())
    for name in ('experts.bin', 'dense.bin', 'dense.txt', 'draft_vocab.bin'):
        expected = ready['files'][name]
        if (base / name).stat().st_size != expected['bytes'] or sha(base / name) != expected['sha256']:
            raise ValueError('local MTP source digest mismatch: ' + name)
    out.mkdir(parents=True, exist_ok=False)
    with (source / 'dense.bin').open('rb') as f, (out / 'dense.bin').open('wb') as g:
        g.truncate((base / 'dense.bin').stat().st_size)
        for name, (_, _, _, off, count) in dst.items():
            f.seek(src[name][3])
            raw = f.read(count)
            if len(raw) != count:
                raise ValueError('truncated MTP tensor: ' + name)
            g.seek(off)
            g.write(raw)
    shutil.copyfile(base / 'dense.txt', out / 'dense.txt')
    for name in ('experts.bin', 'draft_vocab.bin'):
        (out / name).symlink_to((base / name).resolve())
    files = {}
    for name in ('experts.bin', 'dense.bin', 'dense.txt', 'draft_vocab.bin'):
        p = out / name
        st = p.stat()
        files[name] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns, sha256=sha(p))
    ready['files'] = files
    ready['variant'] = dict(kind='self-distilled-mtp-dense', source=receipt,
                            base_dense_sha256=sha(base / 'dense.bin'), tensor_count=len(dst))
    (out / 'ready.json').write_text(json.dumps(ready, indent=2) + '\n')
    print('MTP_VARIANT_READY', out, files['dense.bin']['sha256'], flush=True)


_PROJECTION = re.compile(r'blk\.\d+\.(?:attn_(?:q|k|v|qkv|gate|output)|ssm_out|ffn_(?:gate|up|down)_shexp)\.weight$')


def q8_blocks(bf16):
    """ggml Q8_0 block layout, quantized directly from original BF16, bounded temporary memory."""
    from mtp_rt import q8_0
    flat = np.asarray(bf16).view(np.uint16).reshape(-1)
    if flat.size % 32:
        raise ValueError('Q8 weights require whole blocks')
    out = np.empty(flat.size // 32 * 34, np.uint8)
    chunk = 32 * 65536
    for at in range(0, flat.size, chunk):
        x = (flat[at:at + chunk].astype(np.uint32) << 16).view(np.float32)
        if not np.isfinite(x).all():
            raise ValueError('nonfinite source weight')
        raw = np.frombuffer(q8_0(x), dtype=np.uint8)
        start = at // 32 * 34
        out[start:start + raw.size] = raw
    return out


def prepare_dense(source, out, scope):
    from _paths import add_gguf_py
    add_gguf_py()
    import gguf
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(out)
    rd = gguf.GGUFReader(source)
    wr = gguf.GGUFWriter(out, rd.fields['general.architecture'].contents())
    for name, field in rd.fields.items():
        if name.startswith('GGUF.') or name == 'general.architecture':
            continue
        wr.add_key_value(name, field.contents(), field.types[0],
                         sub_type=field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
    changed = []
    for tensor in rd.tensors:
        chosen = ((scope in ('all', 'projections') and _PROJECTION.fullmatch(tensor.name)) or
                  (scope in ('all', 'head') and tensor.name == 'output.weight'))
        data = np.asarray(tensor.data)
        if chosen:
            if tensor.tensor_type != gguf.GGMLQuantizationType.BF16 or len(tensor.shape) != 2 or tensor.shape[0] % 32:
                raise ValueError('expected whole-row BF16 projection: ' + tensor.name)
            quant = q8_blocks(data).reshape(int(tensor.shape[1]), int(tensor.shape[0]) // 32 * 34)
            wr.add_tensor(tensor.name, quant, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            changed.append(dict(name=tensor.name, shape=list(map(int, tensor.shape)),
                                old_bytes=int(data.nbytes), new_bytes=int(quant.nbytes)))
        elif tensor.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            wr.add_tensor(tensor.name, data)
        else:
            raw = np.ascontiguousarray(data).view(np.uint8)
            wr.add_tensor(tensor.name, raw, raw_dtype=tensor.tensor_type, raw_shape=list(raw.shape))
    if not changed:
        raise ValueError('no eligible matrices found')
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()
    report = dict(source=str(source.resolve()), source_sha256=sha(source), output_sha256=sha(out),
                  scope=scope, format='q8_0', changed=changed,
                  saved_weight_bytes=sum(x['old_bytes'] - x['new_bytes'] for x in changed))
    Path(str(out) + '.json').write_text(json.dumps(report, indent=2) + '\n')
    print('DENSE_VARIANT_READY', len(changed), report['saved_weight_bytes'], out, flush=True)


def main():
    ap = argparse.ArgumentParser(__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    dense = sub.add_parser('dense')
    dense.add_argument('--source', required=True)
    dense.add_argument('--out', required=True)
    dense.add_argument('--scope', choices=['all', 'projections', 'head'], default='all')
    mtp = sub.add_parser('mtp')
    mtp.add_argument('--base-runtime', required=True)
    mtp.add_argument('--source', required=True)
    mtp.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.command == 'dense':
        prepare_dense(a.source, a.out, a.scope)
    else:
        prepare_mtp(a.base_runtime, a.source, a.out)


if __name__ == '__main__':
    main()
