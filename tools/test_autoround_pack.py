"""Lossless layout conversion against independent GPTQ and gguf decoders."""
import sys
import unittest
from pathlib import Path
import numpy as np
import hashlib
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoround_pack import q4_0_reblock, decode_gptq_reference, decode_q4_0_reference, extract_bf16_mtp


class AutoRoundPackingTests(unittest.TestCase):
    def fixture(self, k=256, n=16):
        rng = np.random.default_rng(17)
        words = rng.integers(0, 2**32, size=(k // 8, n), dtype=np.uint32).view(np.int32)
        zeros = np.full((k // 128, n // 8), 0x77777777, dtype=np.uint32).view(np.int32)
        scales = rng.uniform(-0.1, 0.1, (k // 128, n)).astype('<f2')
        scales.flat[:4] = [0, -0.0, np.finfo(np.float16).tiny, -np.finfo(np.float16).tiny]
        return words, zeros, scales

    def test_signed_scales_and_every_nibble_position(self):
        for k in (128, 256, 640, 2560):
            args = self.fixture(k)
            blocks = q4_0_reblock(*args)
            actual = decode_q4_0_reference(blocks)
            expected = decode_gptq_reference(*args)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))

    def test_gguf_reference_decoder(self):
        from _paths import add_gguf_py
        add_gguf_py()
        import gguf
        args = self.fixture()
        packed = q4_0_reblock(*args)
        actual = gguf.quants.dequantize(packed.reshape(16, -1), gguf.GGMLQuantizationType.Q4_0)
        np.testing.assert_array_equal(actual.view(np.uint32), decode_gptq_reference(*args).view(np.uint32))

    def test_refuse_unsupported_inputs(self):
        w, z, s = self.fixture()
        z[0, 0] = 0x66666666
        with self.assertRaisesRegex(ValueError, 'zero point'):
            q4_0_reblock(w, z, s)
        w, z, s = self.fixture()
        s[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite FP16'):
            q4_0_reblock(w, z, s)
        with self.assertRaisesRegex(ValueError, 'int32'):
            q4_0_reblock(w.astype(np.int16), z, s)


class LocalMtpTests(unittest.TestCase):
    def fixture(self):
        class Checkpoint:
            experts, n_embd, n_ff = 3, 64, 32
            meta = {}
            def dtype(self, name): return 'BF16'
            def shape(self, name): return list(self.meta[name].shape)
            def raw(self, name): return self.meta[name].view(np.uint8).reshape(-1)
        ck = Checkpoint()
        ck.meta = {'mtp.pre_fc_norm_embedding.weight': np.arange(64, dtype=np.uint16)}
        for e in range(3):
            for i, role in enumerate(('gate', 'up', 'down')):
                shape = (64, 32) if role == 'down' else (32, 64)
                ck.meta[f'mtp.layers.0.mlp.experts.{e}.{role}_proj.weight'] = np.full(shape, e * 10 + i, np.uint16)
        return ck

    def test_exact_expert_fusion(self):
        ck = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            rows = extract_bf16_mtp(ck, Path(directory))
            self.assertEqual(len(rows), 3)
            for record in rows:
                data = (Path(directory) / record['file']).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])
                if record['name'].endswith('gate_up_proj'):
                    actual = np.frombuffer(data, '<u2').reshape(3, 64, 64)
                    for e in range(3):
                        np.testing.assert_array_equal(actual[e, :32], ck.meta[f'mtp.layers.0.mlp.experts.{e}.gate_proj.weight'])
                        np.testing.assert_array_equal(actual[e, 32:], ck.meta[f'mtp.layers.0.mlp.experts.{e}.up_proj.weight'])
                elif record['name'].endswith('down_proj'):
                    actual = np.frombuffer(data, '<u2').reshape(3, 64, 32)
                    for e in range(3):
                        np.testing.assert_array_equal(actual[e], ck.meta[f'mtp.layers.0.mlp.experts.{e}.down_proj.weight'])

    def test_missing_or_wrong_shape_rejected(self):
        ck = self.fixture()
        del ck.meta['mtp.layers.0.mlp.experts.2.down_proj.weight']
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'complete layer'):
                extract_bf16_mtp(ck, Path(directory))
        ck = self.fixture()
        ck.meta['mtp.layers.0.mlp.experts.1.up_proj.weight'] = np.zeros((64, 32), np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'unexpected MTP shape'):
                extract_bf16_mtp(ck, Path(directory))


if __name__ == '__main__':
    unittest.main()
