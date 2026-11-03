"""Stream a checkpoint's BF16 PLE to a standalone Q4_0 or IQ4_NL GGUF.

Uses the supplied GGML library's canonical encoders. No experts or source files
are changed. Memory is bounded by the worker count and chunk size. A .part file
is published only after source/output hashes and sampled round trips pass.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time

import numpy as np

GGML_REVISION = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
CODEBOOK = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                     1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)
FORMATS = {"Q4_0": 2, "IQ4_NL": 20}


def save_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def digest_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = limit
        while remaining is None or remaining > 0:
            block = f.read(8 << 20 if remaining is None else min(8 << 20, remaining))
            if not block:
                if remaining:
                    raise ValueError("short file while hashing")
                break
            h.update(block)
            if remaining is not None:
                remaining -= len(block)
    return h


def source_table(model):
    model = Path(model)
    config = json.loads((model / "config.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    names = []
    for name, filename in index.items():
        match = re.search(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$", name)
        if match:
            names.append((int(match[1]), name, filename))
    names.sort()
    count = config["text_config"]["split_ngram_parts"]
    if [x[0] for x in names] != list(range(count)) or len({x[2] for x in names}) != 1:
        raise ValueError("PLE must be numerically ordered shards in one source file")
    path = (model / names[0][2]).resolve()
    with path.open("rb") as f:
        length_bytes = f.read(8)
        length = struct.unpack("<Q", length_bytes)[0]
        if length > 128 << 20:
            raise ValueError("oversized safetensors header")
        header_bytes = f.read(length)
        header = json.loads(header_bytes)
    cursor, rows, pieces = 0, 0, []
    for number, name, _ in names:
        tensor = header[name]
        begin, end = tensor["data_offsets"]
        shape = tensor["shape"]
        if tensor["dtype"] != "BF16" or len(shape) != 2 or shape[1] != 160:
            raise ValueError("expected BF16 PLE rows of width 160")
        if begin != cursor or end - begin != shape[0] * 320:
            raise ValueError("non-contiguous PLE payload")
        pieces.append({"number": number, "first_row": rows, "rows": shape[0]})
        rows += shape[0]
        cursor = end
    if len(header) - ("__metadata__" in header) != len(names):
        raise ValueError("source shard also contains non-PLE tensors")
    prefix = length_bytes + header_bytes
    if path.stat().st_size != len(prefix) + cursor:
        raise ValueError("source length disagrees with tensor offsets")
    return {"path": str(path), "rows": rows, "prefix": prefix, "pieces": pieces,
            "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}


class Encoder:
    def __init__(self, library):
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.quantize_row_q4_0_ref.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        self.lib.quantize_row_q4_0_ref.restype = None
        self.lib.quantize_iq4_nl.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int64, C.c_void_p]
        self.lib.quantize_iq4_nl.restype = C.c_size_t
        self.lib.quantize_row_iq4_nl_ref.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        self.lib.quantize_row_iq4_nl_ref.restype = None
        for name in ["dequantize_row_q4_0", "dequantize_row_iq4_nl"]:
            fn = getattr(self.lib, name)
            fn.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
            fn.restype = None

    def quantize(self, x, fmt, fast=False):
        x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 160)
        if not np.isfinite(x).all():
            raise ValueError("non-finite source values")
        out = np.empty((x.shape[0], 90), dtype=np.uint8)
        if fmt == "Q4_0":
            self.lib.quantize_row_q4_0_ref(x.ctypes.data, out.ctypes.data, x.size)
        elif fast:
            self.lib.quantize_row_iq4_nl_ref(x.ctypes.data, out.ctypes.data, x.size)
        else:
            n = self.lib.quantize_iq4_nl(x.ctypes.data, out.ctypes.data, x.shape[0], 160, None)
            if n != out.size:
                raise ValueError("GGML encoder returned an unexpected byte count")
        return out

    def dequantize(self, packed, fmt):
        packed = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1, 90)
        result = np.empty((packed.shape[0], 160), dtype=np.float32)
        fn = self.lib.dequantize_row_q4_0 if fmt == "Q4_0" else self.lib.dequantize_row_iq4_nl
        fn(packed.ctypes.data, result.ctypes.data, result.size)
        return result


def bf16(raw):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(-1, 160)


def independent_decode(raw, fmt):
    blocks = np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1, 18)
    scales = blocks[:, :2].copy().view("<f2").astype(np.float32)
    codes = np.concatenate([blocks[:, 2:] & 15, blocks[:, 2:] >> 4], axis=1)
    values = codes.astype(np.float32) - 8 if fmt == "Q4_0" else CODEBOOK[codes]
    return (scales * values).reshape(-1, 160)


def sample_rows(table, n):
    rng = np.random.default_rng(20261103)
    rows = set([0, table["rows"] - 1])
    for piece in table["pieces"]:
        rows.update([piece["first_row"], piece["first_row"] + piece["rows"] - 1])
        for offset in rng.integers(0, piece["rows"], max(1, n // len(table["pieces"]))):
            rows.add(piece["first_row"] + int(offset))
    return sorted(rows)


def read_rows(table, ids):
    result = []
    with open(table["path"], "rb") as f:
        for row in ids:
            f.seek(len(table["prefix"]) + row * 320)
            raw = f.read(320)
            if len(raw) != 320:
                raise ValueError("short source row")
            result.append(raw)
    return bf16(b"".join(result))


def metrics(x, decoded):
    original = x.astype(np.float64)
    restored = decoded.astype(np.float64)
    norm = np.linalg.norm(original, axis=1)
    error = np.linalg.norm(restored - original, axis=1)
    valid = norm > 0
    relative = error[valid] / norm[valid]
    ratio = np.linalg.norm(restored[valid], axis=1) / norm[valid]
    return {"relative_rmse": float(np.sqrt(np.sum((original-restored)**2) / np.sum(original**2))),
            "row_relative_l2_p50": float(np.median(relative)),
            "row_relative_l2_p95": float(np.quantile(relative, .95)),
            "row_norm_ratio_mean": float(ratio.mean()),
            "row_norm_ratio_p50": float(np.median(ratio)),
            "nonfinite_decoded": int((~np.isfinite(decoded)).sum())}


def self_check(encoder):
    rng = np.random.default_rng(52)
    fixtures = np.concatenate([np.zeros((2, 160), np.float32), rng.normal(size=(64, 160)).astype(np.float32),
                               np.full((2, 160), 0.125, np.float32), np.full((2, 160), -0.125, np.float32)])
    fixtures[3, ::7] *= 50
    for fmt in FORMATS:
        raw = encoder.quantize(fixtures, fmt)
        actual = encoder.dequantize(raw, fmt)
        expected = independent_decode(raw, fmt)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        if not np.isfinite(actual).all():
            raise ValueError("GGML dequantizer initialization or finite-value check failed")
    return {"fixtures": int(fixtures.shape[0]), "formats": list(FORMATS), "independent_decoder_bitwise": True}


def compare(table, encoder, n):
    ids = sample_rows(table, n)
    x = read_rows(table, ids)
    result = {"sample_rows": len(ids), "sample_values": int(x.size), "seed": 20261103, "formats": {}}
    for fmt, fast in [("Q4_0", False), ("IQ4_NL", False), ("IQ4_NL_REF_FAST", True)]:
        code = "IQ4_NL" if fast else fmt
        start = time.monotonic()
        raw = encoder.quantize(x, code, fast=fast)
        seconds = time.monotonic() - start
        decoded = independent_decode(raw, code)
        reference = encoder.dequantize(raw, code)
        np.testing.assert_array_equal(decoded.view(np.uint32), reference.view(np.uint32))
        result["formats"][fmt] = metrics(x, decoded) | {"encode_seconds": seconds,
                                                     "values_per_second_single_thread": x.size / seconds}
    return result


def gguf_header(table, fmt, source_sha):
    def string(value):
        b = value.encode()
        return struct.pack("<Q", len(b)) + b
    metadata = {
        "general.architecture": "strata-ple",
        "general.name": "Qwen PLE " + fmt + " from BF16",
        "strata.ple.source.file": Path(table["path"]).name,
        "strata.ple.source.sha256": source_sha,
        "strata.ple.quantizer.ggml_revision": GGML_REVISION,
        "strata.ple.quantizer": "quantize_iq4_nl; ntry=7; no imatrix" if fmt == "IQ4_NL" else "quantize_row_q4_0_ref",
    }
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata))
    for key, value in metadata.items():
        head += string(key) + struct.pack("<I", 8) + string(value)
    head += string("per_layer_token_embd.weight") + struct.pack("<IQQIQ", 2, 160, table["rows"], FORMATS[fmt], 0)
    return head + b"\0" * (-len(head) % 32)


def convert(args, table, encoder, comparison, checks):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part")
    progress = out.with_suffix(out.suffix + ".progress.json")
    report = out.with_suffix(out.suffix + ".json")
    if out.exists() or report.exists():
        raise FileExistsError("refusing to replace an existing published conversion")
    header = gguf_header(table, args.format, args.source_sha256)
    expected = len(header) + table["rows"] * 90
    free = os.statvfs(out.parent).f_bavail * os.statvfs(out.parent).f_frsize
    if free < expected + (15 << 30):
        raise RuntimeError("not enough free space for output plus 15 GiB reserve")
    if part.exists() or progress.exists():
        raise FileExistsError("partial conversion exists; preserve it and select a fresh --out")
    state = {"state": "converting", "format": args.format, "source": table["path"], "output": str(out),
             "total_rows": table["rows"], "completed_rows": 0, "expected_output_bytes": expected,
             "workers": args.workers, "chunk_rows": args.chunk_rows, "started": time.time()}
    save_json(progress, state)
    source_hash = hashlib.sha256(table["prefix"])
    output_hash = hashlib.sha256(header)
    begun = time.monotonic()

    def encode(raw):
        return encoder.quantize(bf16(raw), args.format).tobytes()

    try:
        with open(table["path"], "rb") as source, open(part, "xb") as target, ThreadPoolExecutor(max_workers=args.workers) as pool:
            source.seek(len(table["prefix"]))
            target.write(header)
            pending = deque()

            def consume():
                raw, rows, future = pending.popleft()
                packed = future.result()
                if len(packed) != rows * 90:
                    raise ValueError("incorrect encoded chunk size")
                source_hash.update(raw)
                target.write(packed)
                output_hash.update(packed)
                state["completed_rows"] += rows
                state["elapsed_seconds"] = round(time.monotonic() - begun, 2)
                state["output_bytes_written"] = len(header) + state["completed_rows"] * 90
                state["free_bytes"] = os.statvfs(out.parent).f_bavail * os.statvfs(out.parent).f_frsize
                if state["free_bytes"] < (15 << 30):
                    raise RuntimeError("free space fell below 15 GiB")
                save_json(progress, state)

            for first in range(0, table["rows"], args.chunk_rows):
                rows = min(args.chunk_rows, table["rows"] - first)
                raw = source.read(rows * 320)
                if len(raw) != rows * 320:
                    raise ValueError("short source read")
                pending.append((raw, rows, pool.submit(encode, raw)))
                if len(pending) >= args.workers * 2:
                    consume()
            while pending:
                consume()
            target.flush()
            os.fsync(target.fileno())
        state["state"] = "verifying"
        save_json(progress, state)
        if source_hash.hexdigest() != args.source_sha256:
            raise ValueError("source SHA256 mismatch; output will not be published")
        source_stat = Path(table["path"]).stat()
        if source_stat.st_size != table["size"] or source_stat.st_mtime_ns != table["mtime_ns"]:
            raise ValueError("source file changed during conversion")
        if part.stat().st_size != expected:
            raise ValueError("incorrect output file size")
        if digest_file(part).hexdigest() != output_hash.hexdigest():
            raise ValueError("output readback SHA256 mismatch")
        ids = sample_rows(table, 4096)
        originals = read_rows(table, ids)
        packed_rows = []
        with open(part, "rb") as target:
            for row in ids:
                target.seek(len(header) + row * 90)
                packed_rows.append(target.read(90))
        packed = np.frombuffer(b"".join(packed_rows), dtype=np.uint8).reshape(-1, 90)
        requantized = encoder.quantize(originals, args.format)
        np.testing.assert_array_equal(packed, requantized)
        decoded = independent_decode(packed, args.format)
        np.testing.assert_array_equal(decoded.view(np.uint32), encoder.dequantize(packed, args.format).view(np.uint32))
        result = {"schema": 1, "format": args.format, "ggml_revision": GGML_REVISION,
                  "library_sha256": digest_file(args.libggml).hexdigest(), "source": table["path"],
                  "source_bytes": table["size"], "source_sha256": source_hash.hexdigest(), "output": str(out),
                  "output_bytes": expected, "output_sha256": output_hash.hexdigest(), "header_bytes": len(header),
                  "rows": table["rows"], "columns": 160, "row_bytes": 90, "effective_bits_per_weight": 4.5,
                  "comparison": comparison, "self_check": checks, "sampled_roundtrip_rows": len(ids),
                  "sample_metrics": metrics(originals, decoded), "output_readback_hash_verified": True,
                  "sampled_bytes_match_ggml_encoder": True, "independent_decode_matches_ggml_bitwise": True,
                  "elapsed_seconds": time.monotonic() - begun, "completed": time.time(),
                  "validation_scope": "weight encoding and table lookup; full-model inference quality not evaluated"}
        part.replace(out)
        save_json(report, result)
        state.update(state="complete", output_sha256=result["output_sha256"], report=str(report))
        save_json(progress, state)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        state.update(state="failed", error=repr(error))
        save_json(progress, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--libggml", required=True, help="libggml-base.so from the pinned GGML revision")
    parser.add_argument("--out", help="destination GGUF; required unless --compare-only")
    parser.add_argument("--format", choices=FORMATS, default="IQ4_NL")
    parser.add_argument("--source-sha256", help="expected SHA256 of the original safetensors file")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--sample-rows", type=int, default=16384)
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--comparison-out")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 1024 <= args.chunk_rows <= 262144 or not 128 <= args.sample_rows <= 131072:
        parser.error("invalid worker, chunk or sample limits")
    if not args.compare_only and (not args.out or not re.fullmatch(r"[0-9a-f]{64}", args.source_sha256 or "")):
        parser.error("conversion requires --out and a 64-character --source-sha256")
    table = source_table(args.model)
    encoder = Encoder(args.libggml)
    checks = self_check(encoder)
    comparison = compare(table, encoder, args.sample_rows)
    if args.comparison_out:
        save_json(args.comparison_out, comparison)
    print(json.dumps({"comparison": comparison, "self_check": checks}), flush=True)
    if not args.compare_only:
        convert(args, table, encoder, comparison, checks)


if __name__ == "__main__":
    main()
