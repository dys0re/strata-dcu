"""Stage the pinned Flash-Next DFlash2 tensors for native CUDA/HIP execution.

No quantization or head/embedding copies. Keep --out on persistent storage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

WEIGHT_SHA = "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"


def prepare(checkpoint, out):
    checkpoint, out = Path(checkpoint), Path(out)
    cfg = json.loads((checkpoint / "config.json").read_text())
    expected = dict(hidden_size=2560, intermediate_size=7680, num_hidden_layers=5,
                    num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                    vocab_size=248320, is_causal=False, sliding_window=2048)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported DFlash2 target geometry")
    dc = cfg["dflash_config"]
    if dc != dict(block_size=8, conv_group_size=16, conv_kernel_size=2,
                  mask_token_id=248077, selector_rank=256, selector_top_k=16,
                  target_layer_ids=[5, 15, 25, 35, 45]):
        raise ValueError("unsupported DFlash2 feature/selector configuration")
    source = checkpoint / "model.safetensors"
    with source.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != WEIGHT_SHA:
        raise ValueError("draft checkpoint SHA-256 mismatch")
    out.mkdir(parents=True, exist_ok=True)
    # Stream the original tensor bytes; the C++ loader uses the exact shapes and offsets.
    with source.open("rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
        start = 8 + header_len
        lines, offset = ["STRATA_DFLASH2 1", f"source {digest}"], 0
        temp = out / "weights.bin.tmp"
        with temp.open("wb") as dst:
            for name, spec in sorted(header.items()):
                if name == "__metadata__":
                    continue
                if spec["dtype"] != "BF16":
                    raise ValueError(f"unexpected dtype for {name}")
                lo, hi = spec["data_offsets"]
                f.seek(start + lo)
                size = hi - lo
                data = f.read(size)
                if len(data) != size:
                    raise ValueError("truncated checkpoint")
                dst.write(data)
                shape = spec["shape"]
                lines.append("tensor " + " ".join(map(str, [name, offset, size, len(shape), *shape])))
                offset += size
            dst.flush()
            os.fsync(dst.fileno())
        if offset != 1206131200:
            raise ValueError("unexpected draft parameter size")
        temp.replace(out / "weights.bin")
        (out / "manifest.txt.tmp").write_text("\n".join(lines) + "\n")
        (out / "manifest.txt.tmp").replace(out / "manifest.txt")
    receipt = dict(checkpoint_sha256=digest, parameter_bytes=offset,
                   config=cfg, format="STRATA_DFLASH2 1")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(prepare(a.checkpoint, a.out), indent=2))
