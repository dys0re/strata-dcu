"""Checks layout remapping, input integrity, Q8 error and preservation of unmodified GGUF data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from _paths import add_gguf_py
add_gguf_py()
import gguf
from dcu_model_variants import dense_layout, prepare_mtp, prepare_dense, q8_blocks


class Variants(unittest.TestCase):
    def test_q8_roundtrip(self):
        source = np.array([0., 1., -1., .125, -2., .01, .002, .05] * 16, np.float32)
        bits = (source.view(np.uint32) >> 16).astype(np.uint16)
        raw = q8_blocks(bits).reshape(-1, 34)
        scale = raw[:, :2].copy().view('<f2').astype(np.float32)
        decoded = (scale * raw[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)
        original = (bits.astype(np.uint32) << 16).view(np.float32)
        self.assertLess(np.max(np.abs(decoded - original)), .009)
        with self.assertRaises(ValueError): q8_blocks(bits[:33])

    def test_layout_and_tuned_reorder(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); base = root / 'base'; src = root / 'published'; base.mkdir(); src.mkdir()
            (base / 'dense.txt').write_text('a bf16 1 32 0 64\nb f32 1 8 64 32\n')
            (base / 'dense.bin').write_bytes(bytes(96))
            (base / 'experts.bin').write_bytes(b'local experts')
            (base / 'draft_vocab.bin').write_bytes(bytes(4))
            original = {n:dict(bytes=(base/n).stat().st_size, sha256=hashlib.sha256((base/n).read_bytes()).hexdigest()) for n in ('experts.bin','dense.bin','dense.txt','draft_vocab.bin')}
            (base / 'ready.json').write_text(json.dumps(dict(recipe=1, source={'checkpoint': 'local'}, files=original)))
            (src / 'dense.txt').write_text('b f32 1 8 0 32\na bf16 1 32 32 64\n')
            (src / 'dense.bin').write_bytes(b'b' * 32 + b'a' * 64)
            files = {n: dict(bytes=(src/n).stat().st_size, sha256=hashlib.sha256((src/n).read_bytes()).hexdigest()) for n in ('dense.bin', 'dense.txt')}
            (src / 'receipt.json').write_text(json.dumps(dict(repo='fixture', revision='fixed', files=files)))
            out = root / 'variant'; prepare_mtp(base, src, out)
            self.assertEqual((out / 'dense.bin').read_bytes(), b'a' * 64 + b'b' * 32)
            self.assertEqual((out / 'experts.bin').resolve(), (base / 'experts.bin').resolve())
            self.assertEqual(json.loads((out/'ready.json').read_text())['source'], {'checkpoint':'local'})
            (src / 'dense.bin').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'): prepare_mtp(base, src, root/'bad')
            (src / 'bad.txt').write_text('a bf16 1 32 0 64\nb f32 1 8 32 32\n')
            with self.assertRaisesRegex(ValueError, 'overlapping'): dense_layout(src/'bad.txt', 96)

    def test_dense_preserves_other_tensors(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'source.gguf'; out=root/'q8.gguf'
            writer=gguf.GGUFWriter(source,'qwen4exp')
            writer.add_uint32('qwen4exp.embedding_length',32)
            a=np.arange(64,dtype=np.float32).reshape(2,32)/128
            bits=(a.view(np.uint32)>>16).astype(np.uint16)
            writer.add_tensor('blk.0.attn_qkv.weight',bits,raw_dtype=gguf.GGMLQuantizationType.BF16)
            writer.add_tensor('blk.0.ffn_gate_inp.weight',bits,raw_dtype=gguf.GGMLQuantizationType.BF16)
            writer.add_tensor('output.weight',bits,raw_dtype=gguf.GGMLQuantizationType.BF16)
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            prepare_dense(source,out,'projections')
            rd=gguf.GGUFReader(out);t={x.name:x for x in rd.tensors}
            self.assertEqual(rd.fields['qwen4exp.embedding_length'].contents(),32)
            self.assertEqual(t['blk.0.attn_qkv.weight'].tensor_type,gguf.GGMLQuantizationType.Q8_0)
            for name in ['blk.0.ffn_gate_inp.weight','output.weight']:
                self.assertEqual(t[name].tensor_type,gguf.GGMLQuantizationType.BF16)
                self.assertEqual(np.asarray(t[name].data).tobytes(),bits.tobytes())
            self.assertEqual(list(t['blk.0.attn_qkv.weight'].shape),[32,2])


if __name__=='__main__':unittest.main()
