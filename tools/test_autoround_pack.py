"""Lossless layout conversion against independent GPTQ and gguf decoders."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoround_pack import q4_0_reblock, decode_gptq_reference, decode_q4_0_reference


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


if __name__ == '__main__':
    unittest.main()
