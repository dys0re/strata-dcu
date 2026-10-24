"""Import symmetric AutoRound GPTQ INT4 without re-quantizing its weights.

Expert codes are losslessly reblocked as Q4_0 (same FP16 scale, zero point 8).
Unquantized projections remain BF16/F32. PLE stays in the original Safetensors
file: a tiny descriptor supplies its validated byte span to Strata's disk reader.

Requires the engine's pinned llama.cpp converter, torch, numpy and safetensors.
No model file is modified. Prepared files are bound to the source inventory.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import shutil
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nvfp4_local_pack import Checkpoint, _VTensor, _VGguf


def q4_0_reblock(qweight, qzeros, scales, group_size=128):
    """GPTQ [K/8,N] words -> Q4_0 [N,K/32,18] bytes, preserving every weight."""
    qw = np.asarray(qweight)
    qz = np.asarray(qzeros)
    s = np.asarray(scales)
    if qw.ndim != 2 or qw.dtype.kind not in "iu" or qw.dtype.itemsize != 4:
        raise ValueError("qweight must be an int32 matrix")
    k, n = qw.shape[0] * 8, qw.shape[1]
    if group_size < 32 or group_size % 32 or k % group_size or n % 8:
        raise ValueError("only whole symmetric GPTQ groups divisible by 32 are supported")
    if qz.shape != (k // group_size, n // 8) or qz.dtype.kind not in "iu" or qz.dtype.itemsize != 4:
        raise ValueError("qzeros shape/dtype disagrees with qweight")
    # auto_gptq stores zero point minus one. Refuse other/asymmetric zeros,
    # rather than silently change the interpretation to make them fit Q4_0.
    if not np.all(qz.view(np.uint32) == np.uint32(0x77777777)):
        raise ValueError("expected GPTQ symmetric zero point 8 (packed qzeros 0x77777777)")
    if s.dtype != np.dtype("<f2") or s.shape != (k // group_size, n) or not np.isfinite(s).all():
        raise ValueError("scales must be finite FP16 [K/group_size,N]")
    codes = ((qw.view(np.uint32)[:, None, :] >>
              np.arange(0, 32, 4, dtype=np.uint32)[None, :, None]) & 15).astype(np.uint8)
    codes = codes.reshape(k, n).T.reshape(n, k // 32, 32)
    out = np.empty((n, k // 32, 18), dtype=np.uint8)
    repeated = np.repeat(s.T, group_size // 32, axis=1).copy()
    out[:, :, :2] = repeated.view(np.uint8).reshape(n, k // 32, 2)
    out[:, :, 2:] = codes[:, :, :16] | (codes[:, :, 16:] << 4)
    return out


def decode_gptq_reference(qweight, qzeros, scales, group_size=128):
    """Independent coordinate-wise reference, including GPTQ's zero offset."""
    qw = np.asarray(qweight).view(np.uint32)
    qz = np.asarray(qzeros).view(np.uint32)
    k, n = qw.shape[0] * 8, qw.shape[1]
    i = np.arange(k)
    j = np.arange(n)
    codes = (qw[i // 8, :] >> ((i % 8) * 4)[:, None]) & 15
    zeros = ((qz[:, j // 8] >> ((j % 8) * 4)[None, :]) & 15) + 1
    return ((codes.astype(np.int16) - zeros[i // group_size].astype(np.int16)) *
            np.asarray(scales, np.float32)[i // group_size]).T


def decode_q4_0_reference(blocks):
    b = np.asarray(blocks)
    n, nb, _ = b.shape
    scale = b[:, :, :2].copy().view("<f2").reshape(n, nb).astype(np.float32)
    code = np.concatenate((b[:, :, 2:] & 15, b[:, :, 2:] >> 4), axis=2).astype(np.int16)
    return ((code - 8) * scale[:, :, None]).reshape(n, nb * 32)


class AutoRoundCheckpoint(Checkpoint):
    def __init__(self, folder):
        super().__init__(folder)
        self.config = json.loads((self.dir / "config.json").read_text())
        q = self.config.get("quantization_config", {})
        if self.config.get("model_type") != "qwen4_exp":
            raise ValueError("this importer supports Qwen4Exp checkpoints only")
        if (q.get("quant_method") != "auto-round" or q.get("bits") != 4 or
                q.get("packing_format") != "auto_round:auto_gptq" or q.get("sym") is not True):
            raise ValueError("expected symmetric INT4 auto-round:auto_gptq")
        self.group_size = q["group_size"]
        t = self.config["text_config"]
        self.layers, self.experts = t["num_hidden_layers"], t["num_experts"]
        self.n_embd, self.n_ff = t["hidden_size"], t["moe_intermediate_size"]
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        for name, fn in index.items():
            if name not in self.meta or self.meta[name][0] != fn:
                raise ValueError("index/header disagreement: " + name)
        for fn in set(index.values()):
            end = max(v[4] for v in self.meta.values() if v[0] == fn)
            if (self.dir / fn).stat().st_size != end:
                raise ValueError("truncated or trailing data in " + fn)

    def projection(self, layer, expert, role):
        base = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{role}_proj."
        if self.dtype(base + "qweight") != "I32" or self.dtype(base + "qzeros") != "I32":
            raise ValueError("non-int32 GPTQ codes: " + base)
        if self.dtype(base + "scales") != "F16":
            raise ValueError("non-FP16 GPTQ scale: " + base)
        return (self.array(base + "qweight", "<i4"), self.array(base + "qzeros", "<i4"),
                self.array(base + "scales", "<f2"))

    def inventory(self):
        result = []
        for fn in sorted({v[0] for v in self.meta.values()}):
            st = (self.dir / fn).stat()
            result.append({"name": fn, "size": st.st_size, "mtime_ns": st.st_mtime_ns})
        return {"model": str(self.dir.resolve()), "files": result,
                "config_sha256": hashlib.sha256((self.dir / "config.json").read_bytes()).hexdigest(),
                "index_sha256": hashlib.sha256((self.dir / "model.safetensors.index.json").read_bytes()).hexdigest()}


def converter_path(path):
    path = Path(path).resolve()
    if not (path / "conversion" / "qwen4exp.py").is_file():
        raise ValueError("llama.cpp converter is missing: " + str(path))
    sys.path.insert(0, str(path))
    sys.path.insert(0, str(path / "gguf-py"))
    os.environ["STRATA_GGUF_PY"] = str(path / "gguf-py")


def convert_dense(ck, outfile):
    """Reuse pinned architecture transforms, but neither dequantize nor quantize weights."""
    import gguf
    from conversion.base import ModelBase
    from conversion.qwen4exp import Qwen4ExpTextModel
    import iq_pack

    orig_index = ModelBase.index_tensors
    orig_hparams = ModelBase.load_hparams
    orig_force = Qwen4ExpTextModel.tensor_force_quant

    def index(self, remote_hf_model_id=None):
        return {k: v for k, v in orig_index(self, remote_hf_model_id).items()
                if ".mlp.experts." not in k and ".ngram_embedding.shard_" not in k
                and not k.startswith("mtp.")}

    def hparams(folder, is_mistral):
        h = orig_hparams(folder, is_mistral)
        h.pop("quantization_config", None)
        if "text_config" in h:
            h["text_config"].pop("quantization_config", None)
        return h

    def force(self, name, new_name, bid, n_dims):
        form = iq_pack.form_of(new_name)
        if form:
            return getattr(gguf.GGMLQuantizationType, form)
        if n_dims >= 2:
            return gguf.GGMLQuantizationType.BF16
        return orig_force(self, name, new_name, bid, n_dims)

    ModelBase.index_tensors = index
    ModelBase.load_hparams = staticmethod(hparams)
    Qwen4ExpTextModel.tensor_force_quant = force
    try:
        model = Qwen4ExpTextModel(ck.dir, gguf.LlamaFileType.MOSTLY_BF16, Path(outfile))
        # The source's PLE is deliberately not written; retain its row-width metadata.
        model._ple_row_dim = 160
        model.write()
    finally:
        ModelBase.index_tensors = orig_index
        ModelBase.load_hparams = staticmethod(orig_hparams)
        Qwen4ExpTextModel.tensor_force_quant = orig_force


def write_ple_descriptor(ck, outfile):
    import gguf
    pieces = []
    for name, info in ck.meta.items():
        match = re.search(r"\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$", name)
        if match:
            fn, dtype, shape, a, b = info
            if dtype != "BF16" or len(shape) != 2 or shape[1] != 160 or b - a != shape[0] * 320:
                raise ValueError("invalid BF16 PLE shard " + name)
            pieces.append((int(match[1]), fn, shape[0], a, b))
    pieces.sort()
    want = ck.config["text_config"]["split_ngram_parts"]
    if len(pieces) != want or [p[0] for p in pieces] != list(range(want)):
        raise ValueError("missing/duplicate PLE shard numbers")
    if len({p[1] for p in pieces}) != 1 or any(a[4] != b[3] for a, b in zip(pieces, pieces[1:])):
        raise ValueError("PLE shards are not contiguous in numeric order; no implicit reorder/copy")
    source = (ck.dir / pieces[0][1]).resolve()
    w = gguf.GGUFWriter(str(outfile), "strata-ple")
    w.add_string("strata.ple.external.path", str(source))
    w.add_uint64("strata.ple.external.offset", pieces[0][3])
    w.add_uint64("strata.ple.external.rows", sum(p[2] for p in pieces))
    w.add_uint64("strata.ple.external.file_size", source.stat().st_size)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()
    print(f"PLE descriptor: {sum(p[2] for p in pieces)} BF16 rows, source {source}, no table copied", flush=True)


def attach_experts(model, ck, workers):
    """Supply iq_pack with Q4_0 expert tensors, one layer/role resident at a time."""
    source = model.paths[0]
    cache = {}
    geom = {"gate": (ck.n_embd, ck.n_ff), "up": (ck.n_embd, ck.n_ff), "down": (ck.n_ff, ck.n_embd)}
    for layer in range(ck.layers):
        for role, (k, n) in geom.items():
            name = f"blk.{layer}.ffn_{role}_exps.weight"
            tensor = _VTensor(name, [k, n, ck.experts], 2, "Q4_0", ck.experts * n * (k // 32) * 18)
            model.where[name] = (_VGguf(), tensor, None, source)
    original = type(model).bytes

    def data(name):
        match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name)
        if not match:
            return original(model, name)
        key = (int(match[1]), match[2])
        if key not in cache:
            cache.clear()
            t0 = time.monotonic()

            def one(expert):
                return q4_0_reblock(*ck.projection(key[0], expert, key[1]), ck.group_size).reshape(-1)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                values = list(ex.map(one, range(ck.experts)))
            cache[key] = np.stack(values).reshape(-1)
            print(f"AutoRound layer {key[0]} {key[1]}: exact INT4 reblock in {time.monotonic()-t0:.1f}s", flush=True)
        return cache[key]

    model.bytes = data


def pack_experts(ck, dense, out, workers):
    import iq_pack
    original = iq_pack.Model

    class Model(original):
        def __init__(self, first):
            super().__init__(first)
            attach_experts(self, ck, workers)

    iq_pack.Model = Model
    old_argv = sys.argv
    try:
        sys.argv = ["iq_pack.py", "--gguf", str(dense), "--out", str(out), "--experts-bin"]
        rc = iq_pack.main()
        if rc:
            raise RuntimeError(f"iq_pack failed: {rc}")
    finally:
        iq_pack.Model = original
        sys.argv = old_argv


def verify(ck, count):
    # Spread across layers, roles and expert ids, including first/last.
    out = []
    for i in range(count):
        layer = i * (ck.layers - 1) // max(1, count - 1)
        expert = (i * 137) % ck.experts
        role = ("gate", "up", "down")[i % 3]
        arrays = ck.projection(layer, expert, role)
        packed = q4_0_reblock(*arrays, ck.group_size)
        expected = decode_gptq_reference(*arrays, ck.group_size)
        actual = decode_q4_0_reference(packed)
        if not np.array_equal(expected.view(np.uint32), actual.view(np.uint32)):
            raise AssertionError(f"reblock changed decoded weights: layer {layer}, expert {expert}, {role}")
        out.append({"layer": layer, "expert": expert, "role": role, "weights": int(expected.size),
                    "bit_exact": True})
        print("verified", out[-1], flush=True)
    return out


def extract_bf16_mtp(ck, folder):
    """Losslessly fuse local per-expert BF16 tensors into mtp_pack's input ABI."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = []
    prefix = "mtp.layers.0.mlp.experts."
    experts = {f"{prefix}{e}.{role}_proj.weight"
               for e in range(ck.experts) for role in ("gate", "up", "down")}
    actual = {k for k in ck.meta if k.startswith("mtp.") and ".experts." in k}
    if actual != experts:
        raise ValueError("MTP must contain one complete layer of separate BF16 experts")
    dense = sorted(k for k in ck.meta if k.startswith("mtp.") and ".experts." not in k)
    if not dense or any(ck.dtype(k) != "BF16" for k in dense + sorted(experts)):
        raise ValueError("local MTP tensors must all be BF16")
    for name in dense:
        blob = ck.raw(name).tobytes()
        filename = name + ".bin"
        (folder / filename).write_bytes(blob)
        manifest.append({"name": name, "dtype": "BF16", "shape": ck.shape(name),
                         "file": filename, "sha256": hashlib.sha256(blob).hexdigest()})
    for suffix, roles in (("gate_up_proj", ("gate", "up")), ("down_proj", ("down",))):
        name = prefix + suffix
        filename = name + ".bin"
        digest = hashlib.sha256()
        rows, cols = (2 * ck.n_ff, ck.n_embd) if suffix == "gate_up_proj" else (ck.n_embd, ck.n_ff)
        with (folder / filename).open("wb") as f:
            for e in range(ck.experts):
                for role in roles:
                    key = f"{prefix}{e}.{role}_proj.weight"
                    expected = [ck.n_embd, ck.n_ff] if role == "down" else [ck.n_ff, ck.n_embd]
                    if ck.shape(key) != expected:
                        raise ValueError("unexpected MTP shape: " + key)
                    blob = ck.raw(key).tobytes()
                    if len(blob) != expected[0] * expected[1] * 2:
                        raise ValueError("truncated BF16 MTP tensor: " + key)
                    f.write(blob)
                    digest.update(blob)
        manifest.append({"name": name, "dtype": "BF16", "shape": [ck.experts, rows, cols],
                         "file": filename, "sha256": digest.hexdigest()})
        print("MTP fused:", name, [ck.experts, rows, cols], flush=True)
    (folder / "mtp-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check_mtp(ck, out, runtime=None):
    directory = Path(runtime).resolve() if runtime else out / "mtp"
    if not (directory / "ready.json").is_file():
        raise ValueError("MTP cache missing; run autoround_pack.py mtp with the same --model and --out first")
    ready = json.loads((directory / "ready.json").read_text())
    if ready.get("recipe") != 1 or ready.get("source") != ck.inventory():
        raise ValueError("MTP cache does not match the checkpoint or preparation recipe")
    if set(ready.get("files", {})) != {"experts.bin", "dense.bin", "dense.txt", "draft_vocab.bin"}:
        raise ValueError("incomplete MTP runtime manifest")
    for name, info in ready["files"].items():
        st = (directory / name).stat()
        if st.st_size != info["bytes"] or st.st_mtime_ns != info["mtime_ns"]:
            raise ValueError("MTP runtime file changed: " + name)
    return directory


def prepare_mtp(ck, out):
    if (ck.n_embd, ck.n_ff, ck.experts) != (2560, 640, 512):
        raise ValueError("the current MTP runtime requires hidden=2560, FF=640, experts=512")
    if (out / "mtp").exists():
        check_mtp(ck, out)
        return
    identity = ck.inventory()
    with tempfile.TemporaryDirectory(prefix=".mtp-", dir=out) as temporary:
        stage = Path(temporary)
        manifest = extract_bf16_mtp(ck, stage / "src")
        gguf = stage / "mtp-q2_0.gguf"
        subprocess.run([sys.executable, str(HERE / "mtp_pack.py"), "--src", str(stage / "src"),
                        "--experts", "q2_0", "--out", str(gguf)], check=True)
        runtime = stage / "rt"
        subprocess.run([sys.executable, str(HERE / "mtp_rt.py"), "--gguf", str(gguf),
                        "--out", str(runtime)], check=True)
        shutil.copy2(HERE.parent / "data" / "draft_vocab.bin", runtime / "draft_vocab.bin")
        shutil.copy2(stage / "src" / "mtp-manifest.json", runtime / "source-tensors.json")
        shutil.copy2(str(gguf) + ".report.json", runtime / "quantization-report.json")
        if ck.inventory() != identity:
            raise ValueError("checkpoint changed while preparing MTP")
        files = {}
        for name in ("experts.bin", "dense.bin", "dense.txt", "draft_vocab.bin"):
            path = runtime / name
            st = path.stat()
            files[name] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (runtime / "ready.json").write_text(json.dumps({"recipe": 1, "source": identity,
            "draft_experts": "q2_0", "draft_projections": "q8_0", "source_tensors": len(manifest),
            "main_weights_modified": False, "files": files}, indent=2) + "\n")
        runtime.rename(out / "mtp")
    print("Prepared independent MTP cache:", out / "mtp", flush=True)


def run_serve_once(cmd, env, stderr, ids, max_new):
    """Adapt one raw --serve request to this entry point's output-ID contract."""
    lines, generated = [], []
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=stderr, text=True, env=env, bufsize=1)
    def read():
        line = p.stdout.readline()
        if line:
            lines.append(line)
        return line
    try:
        while True:
            line = read()
            if not line or line.startswith("ERR "):
                raise RuntimeError("DFlash2 engine exited before READY")
            if line.startswith("READY "):
                break
        p.stdin.write(f"GEN {max_new} ckpt=0 temperature=0 " + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        while True:
            line = read()
            if not line or line.startswith("ERR "):
                raise RuntimeError("DFlash2 generation failed; see engine logs")
            if line.startswith("T "):
                generated.append(int(line.split()[1]))
            if line.startswith("DONE "):
                fields = line.split()
                if int(fields[1]) != len(generated) or int(fields[8]) != 0:
                    raise RuntimeError("DFlash2 DONE/token count or nonfinite status mismatch")
                break
        p.stdin.write("QUIT\n"); p.stdin.flush()
        lines.append(p.stdout.read())
        code = p.wait(timeout=30)
        # Preserve raw serve output and normalize actual emitted IDs for the existing report reader.
        lines.append("\noutput : " + " ".join(map(str, generated)) + "\n")
        return subprocess.CompletedProcess(cmd, code, "".join(lines))
    finally:
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()


def run_model(ck, out, a):
    machine_variant = {}
    if (os.environ.get("STRATA_DCU_MODEL_DEFAULTS") == "1" and
            not a.dflash2 and a.mtp_mode == "adaptive"):
        config_path = Path(os.environ["STRATA_DCU_MODEL_CONFIG"])
        machine_variant = json.loads(config_path.read_text())
        if (machine_variant.get("schema") != 1 or
                not isinstance(machine_variant.get("dense_gguf"), str) or
                (machine_variant.get("mtp_runtime") is not None and not isinstance(machine_variant["mtp_runtime"], str)) or
                not isinstance(machine_variant.get("hc_q8"), bool) or
                not isinstance(machine_variant.get("draft_head_proxy"), bool)):
            raise ValueError("invalid DCU model-default configuration")
        a.dense_gguf = a.dense_gguf or machine_variant["dense_gguf"]
        a.mtp_runtime = a.mtp_runtime or machine_variant.get("mtp_runtime")
    (out / "generation.json").unlink(missing_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ck.dir, local_files_only=True)
    tokenizer.chat_template = (ck.dir / "chat_template.jinja").read_text()
    messages = [{"role": "user", "content": a.prompt}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    generation = ck.dir / "generation_config.json"
    eos = json.loads(generation.read_text()).get("eos_token_id") if generation.is_file() else None
    if eos is None:
        eos = tokenizer.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(not isinstance(v, int) or v < 0 for v in eos):
        raise ValueError("checkpoint has no valid end-of-turn token ids")
    if len(ids) + a.max_new > a.context:
        raise ValueError("prompt plus output exceeds --context")
    token_file = out / "prompt.ids"
    token_file.write_text(",".join(map(str, ids)) + "\n")
    dense_file = Path(a.dense_gguf).resolve() if a.dense_gguf else out / "dense.gguf"
    cmd = [str(Path(a.engine).resolve()), "--native", str(dense_file),
           "--pack", str(out / "pack"), "--ple-gguf", str(out / "ple.gguf"),
           "--tokens-file", str(token_file), "--max-new", str(a.max_new),
           "--eos-ids", ",".join(map(str, eos)),
           # Native packs require --spec >= 2, but this entry has no drafter.
           # Run only the real token, rather than verifying an unused zero draft.
           "--max-context", str(a.context), "--prefill", "32", "--spec", "2", "--mtp-max-t", "1",
           "--expert-profile", str(HERE.parent / "data" / "expert-profile.bin"),
           "--expert-cache", str(a.expert_cache if a.expert_cache is not None else (8300 if a.dflash2 else 8500)), "--mmap-experts", "--resident-experts" if a.resident_pin else "--resident-cpu-experts",
           "--pool-workers", str(a.pool_workers), "--vram-reserve-mib", "4096",
           "--kv", "int8", "--suffix-draft", str(a.suffix_draft),
           "--adapt-every", str(a.adapt_every), "--adapt-swaps", str(a.adapt_swaps),
           "--pcie-mode", a.pcie_mode,
           "--ple-io", "direct", "--ple-row-cache", "65536"]
    cmd += ["--pcie-frac", str(a.pcie_frac)]
    if a.mtp_mode != "off" and (not a.dflash2 or a.dflash_mtp_prefix):
        runtime = check_mtp(ck, out, a.mtp_runtime)
        cmd[cmd.index("--spec") + 1] = str(min(8, a.mtp_max_t + (2 if a.suffix_draft else 0)))
        cmd[cmd.index("--mtp-max-t") + 1] = str(a.mtp_max_t)
        cmd += ["--mtp", str(runtime)]
        if a.mtp_mode == "adaptive":
            cmd += ["--mtp-adaptive"]
    native_cmd = list(cmd)
    if a.dflash2:
        cmd[cmd.index("--spec") + 1] = str(a.dflash_max_t)
        cmd[cmd.index("--mtp-max-t") + 1] = str(a.mtp_max_t if a.dflash_mtp_prefix else a.dflash_max_t)
        cmd += ["--serve", "--prompt-cache", "0", "--conversation-cache-mib", "0"]
        native_cmd = list(cmd)
        if a.dflash_backend == "native":
            from dflash2_pack import prepare, WEIGHT_SHA
            draft_pack = out / "dflash2-native"
            receipt = draft_pack / "receipt.json"
            ready = receipt.is_file() and (draft_pack / "manifest.txt").is_file() and (draft_pack / "weights.bin").is_file()
            if ready:
                ready = (json.loads(receipt.read_text()).get("checkpoint_sha256") == WEIGHT_SHA and
                         (draft_pack / "weights.bin").stat().st_size == 1206131200)
            if not ready:
                prepare(a.dflash2, draft_pack)
            cmd += ["--dflash2", str(draft_pack)]
            if a.dflash_mtp_prefix:
                cmd += ["--dflash-mtp-prefix",str(a.dflash_mtp_prefix)]
            if not a.dflash_fixed:
                cmd += ["--dflash-adaptive"]
            if a.dflash_eager:
                cmd += ["--dflash-eager"]
            native_cmd = list(cmd)
        else:
            cmd = [sys.executable, str(HERE / "strata_dflash2.py"), "--checkpoint", str(Path(a.dflash2).resolve()),
                   "--model", str(ck.dir), "--stats-file", str(out / "dflash2-worker.json")]+(
                   [] if a.dflash_eager else ["--graph"])+([] if a.dflash_fixed else ["--adaptive"])+["--"]+cmd
    (out / "run-command.json").write_text(json.dumps(native_cmd, indent=2) + "\n")
    if a.dflash2:
        (out / "dflash2-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    print("Starting AutoRound model;", len(ids), "prompt tokens", flush=True)
    env = dict(os.environ, STRATA_MAX_WEIGHT_VRAM_FRACTION="0.5")
    if machine_variant:
        env.setdefault("STRATA_HC_Q8", str(int(machine_variant["hc_q8"])))
        env.setdefault("STRATA_DRAFT_HEAD_PROXY", str(int(machine_variant["draft_head_proxy"])))
    if a.dense_gguf:
        env["STRATA_DENSE_Q8_VARIANT"] = "1"
    # gfx936 measurements: staged passes strict HC parity and improves decode.
    env.setdefault("STRATA_HC_SPLIT", "2")
    # The machine-specific DCU wrapper opts into the combination measured with enhanced MTP. Other entrypoints
    # and other draft modes keep scheduling opt-in; a caller's individual 0/1 override always wins.
    if env.get("STRATA_DCU_SCHEDULING_DEFAULTS") == "1" and not a.dflash2 and a.mtp_mode == "adaptive":
        env.setdefault("STRATA_VERIFY_HIT_PLAN", "1")
        env.setdefault("STRATA_CPU_EXPERT_PIPELINE", "1")
        env.setdefault("STRATA_CPU_EARLY_RESULT", "1")
    if a.dflash2:
        env.update(STRATA_ROUTE_TAIL_SKIP="0", STRATA_QFUSE="0", STRATA_VERIFY_QDEDUP="0", STRATA_QFUSE_HC="0")
        env.setdefault("STRATA_BF16_Q8_VEC4", "1")
        env.setdefault("STRATA_BF16_Q8_COLS", "4")
    with (out / "engine.stderr.log").open("w") as stderr:
        if a.dflash2:
            result = run_serve_once(cmd, env, stderr, ids, a.max_new)
        else:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=stderr, text=True, env=env)
    (out / "engine.stdout.log").write_text(result.stdout)
    print(result.stdout, end="", flush=True)
    if result.returncode:
        print((out / "engine.stderr.log").read_text()[-5000:], file=sys.stderr)
        return result.returncode
    errors = (out / "engine.stderr.log").read_text()
    if "larger than launch bounds" in errors or "Launch params" in errors:
        raise RuntimeError("DTK rejected a kernel launch; generated text is invalid (see engine.stderr.log)")
    usage_lines = re.findall(r"^strata weight usage:.*$", errors, re.MULTILINE)
    if not usage_lines:
        raise RuntimeError("engine did not report/enforce the 50% weight VRAM budget; rebuild the engine")
    usage = {k: float(v) if "." in v else int(v)
             for k, v in re.findall(r"(\w+)=([0-9.]+)", usage_lines[-1])}
    if usage["limit"] != usage["total"] // 2 or usage["weights"] > usage["limit"]:
        raise RuntimeError("engine weights exceed the 50% weight VRAM budget")
    if a.dflash2 and usage.get("external_draft", 0) + usage.get("native_draft", 0) <= 0:
        raise RuntimeError("engine did not account DFlash2 GPU weights")
    match = re.search(r"^output\s*:(.*)$", result.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("engine returned no output token ids")
    generated = [int(v) for v in match[1].split()]
    raw_text = tokenizer.decode(generated, skip_special_tokens=False)
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    report = {"prompt": a.prompt, "prompt_ids": ids, "generated_ids": generated,
              "model_variants": {"dense_gguf": str(dense_file), "mtp_runtime": a.mtp_runtime,
                                 "hc_q8": env.get("STRATA_HC_Q8") == "1",
                                 "draft_head_proxy": env.get("STRATA_DRAFT_HEAD_PROXY") == "1"},
              "text": decoded, "raw_text": raw_text, "engine_exit": result.returncode, "weight_vram": usage,
              "expert_execution": {"resident_pin": a.resident_pin, "adapt_every": a.adapt_every, "adapt_swaps": a.adapt_swaps,
                                   "pcie_mode": "dma-staged" if int(env.get("STRATA_PCIE_STAGED_MAX") or "0") > 0 else a.pcie_mode,
                                   "pcie_frac": a.pcie_frac},
              "suffix_draft": a.suffix_draft,
              "scheduling_options": {"gpu_hit_plan": env.get("STRATA_VERIFY_HIT_PLAN") == "1",
                                     "cpu_expert_pipeline": env.get("STRATA_CPU_EXPERT_PIPELINE") == "1",
                                     "early_result": env.get("STRATA_CPU_EARLY_RESULT") == "1",
                                     "staged_experts_max": max(0,min(16,int(env.get("STRATA_PCIE_STAGED_MAX") or "0"))),
                                     "shared_stream": env.get("STRATA_SH_STREAM", "auto"),
                                     "pool_idle_us": env.get("STRATA_POOL_IDLE_US")},
              "kernel_options": {"hc_split": env.get("STRATA_HC_SPLIT", "auto"), "cpu_q4_mt": env.get("STRATA_CPU_Q4_MT") == "1",
                                 "dflash_small_gemv": env.get("STRATA_DFLASH_SMALL_GEMV") == "1"},
              "finish_reason": "eos" if generated and generated[-1] in eos else "length"}
    if a.dflash2:
        report.update(draft_method="mtp+dflash2" if a.dflash_mtp_prefix else "dflash2", draft_backend=a.dflash_backend,
                      draft_graph=not a.dflash_eager, draft_adaptive=not a.dflash_fixed,
                      draft_max_t=a.dflash_max_t)
        if a.dflash_mtp_prefix:
            report.update(joint_mtp_prefix=a.dflash_mtp_prefix,mtp_mode=a.mtp_mode,mtp_max_t=a.mtp_max_t)
        if a.dflash_backend == "ipc":
            report["draft_worker"] = json.loads((out / "dflash2-worker.json").read_text())
    elif a.mtp_mode != "off":
        report.update(mtp_mode=a.mtp_mode, mtp_max_t=a.mtp_max_t)
    budget = re.search(r"^mtp budget:.*$", errors, re.MULTILINE)
    if budget:
        report["mtp_budget"] = budget.group(0)
        print(budget.group(0), flush=True)
    completed = out / "generation.json.tmp"
    completed.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    completed.replace(out / "generation.json")
    print("GENERATED TEXT:", decoded, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["verify", "convert", "ple", "pack", "mtp", "prepare", "run"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--llama", default=str(HERE.parent / "third_party" / "llama.cpp"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verify-experts", type=int, default=12)
    ap.add_argument("--engine", default=str(HERE.parent / "build" / "strata"))
    ap.add_argument("--dense-gguf", help="optional prepared non-expert weight variant; expert pack stays fixed")
    ap.add_argument("--mtp-runtime", help="optional compatible prepared MTP runtime, with source-bound ready.json")
    ap.add_argument("--prompt", default="Reply with exactly OK.")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--pool-workers", type=int, default=12)
    ap.add_argument("--expert-cache", type=int, default=None, help="fixed GPU expert slots; still subject to the 50%% total weight budget")
    ap.add_argument("--pcie-mode", choices=["auto", "kernel", "dma", "direct"], default="auto", help="auto uses the GPU copy kernel; dma uses the copy engine")
    ap.add_argument("--resident-pin", action=argparse.BooleanOptionalAction, default=None, help="page-lock resident CPU experts for PCIe offload; automatic with explicit nonzero pcie-frac")
    ap.add_argument("--adapt-every", type=int, default=0, help="cache exchange interval in verify windows; 0 disables, upstream 4")
    ap.add_argument("--adapt-swaps", type=int, default=96, help="maximum exchanges per interval; upstream 96")
    ap.add_argument("--pcie-frac", type=float, default=0, help="fraction of cache misses sent to GPU; -1 engine auto, 0 CPU only")
    ap.add_argument("--suffix-draft", type=int, default=0, help="minimum prompt lookup match; 0 disables, upstream 3")
    ap.add_argument("--dflash2", help="trained Flash-Next DFlash2 checkpoint; native GPU execution with shared target head/embedding")
    ap.add_argument("--dflash-backend", choices=["native", "ipc"], default="native")
    ap.add_argument("--dflash-max-t", type=int, choices=range(2,9), default=8, help="anchor plus 1..7 draft proposals")
    ap.add_argument("--dflash-mtp-prefix",type=int,choices=range(4),default=0,
                    help="joint mode: MTP prefix of up to N tokens and DFlash2 continuation (0 disables)")
    ap.add_argument("--dflash-fixed", action="store_true", help="always use the full requested draft window")
    ap.add_argument("--dflash-eager", action="store_true", help="disable draft GPU graphs for diagnostics")
    ap.add_argument("--mtp-mode", choices=["off", "fixed", "adaptive"], default="adaptive")
    ap.add_argument("--mtp-max-t", type=int, default=5, help="verification cap, including the real token (2..8)")
    a = ap.parse_args()
    if a.resident_pin is None:
        a.resident_pin = a.pcie_frac is not None and a.pcie_frac != 0
    if not 1 <= a.workers <= 16 or a.verify_experts < 1:
        ap.error("workers must be 1..16 and verify-experts positive")
    if a.max_new < 1 or a.context < 1 or a.pool_workers < 1:
        ap.error("max-new, context and pool-workers must be positive")
    if a.expert_cache is not None and a.expert_cache < 1:
        ap.error("expert-cache must be positive")
    if a.adapt_every < 0 or a.adapt_swaps < 0 or a.suffix_draft < 0:
        ap.error("adapt-every, adapt-swaps and suffix-draft must be nonnegative")
    if a.pcie_frac is not None and not (a.pcie_frac == -1 or 0 <= a.pcie_frac <= 1):
        ap.error("pcie-frac must be -1 (auto) or 0..1")
    if not 2 <= a.mtp_max_t <= 8:
        ap.error("mtp-max-t must be 2..8")
    if a.dflash_mtp_prefix and (not a.dflash2 or a.dflash_backend!="native" or a.mtp_mode=="off" or
                               a.dflash_mtp_prefix>a.dflash_max_t-2):
        ap.error("joint mode needs native --dflash2, MTP enabled, and room for a DFlash2 tail")
    if a.mtp_runtime and (a.mtp_mode == "off" or (a.dflash2 and not a.dflash_mtp_prefix)):
        ap.error("mtp-runtime requires active MTP")
    if a.dflash2 and a.dflash_backend == "ipc" and (a.context > 512 or (not a.dflash_eager and a.dflash_max_t > 4)):
        ap.error("DFlash2 IPC supports context <=512 and graph --dflash-max-t <=4; use native for full windows")
    converter_path(a.llama)
    ck = AutoRoundCheckpoint(Path(a.model).resolve())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    identity = ck.inventory()
    identity_file = out / "source.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != identity:
        raise ValueError("source checkpoint changed; use a new output directory")
    if a.command == "run":
        if not identity_file.is_file() or not (out / "pack" / "native_experts.txt").is_file():
            raise ValueError("run requires a completed prepare in --out")
        return run_model(ck, out, a)
    identity_file.write_text(json.dumps(identity, indent=2) + "\n")
    if a.command in ("verify", "prepare"):
        report = verify(ck, a.verify_experts)
        (out / "weight-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if a.command in ("ple", "prepare"):
        write_ple_descriptor(ck, out / "ple.gguf")
    if a.command in ("convert", "prepare") and not (out / "dense.done").is_file():
        tmp = out / "dense.tmp.gguf"
        convert_dense(ck, tmp)
        tmp.replace(out / "dense.gguf")
        (out / "dense.done").write_text("BF16/F32 projections; experts and PLE excluded\n")
    if a.command in ("pack", "prepare"):
        pack_experts(ck, out / "dense.gguf", out / "pack", a.workers)
    if a.command == "mtp" or (a.command == "prepare" and a.mtp_mode != "off"):
        prepare_mtp(ck, out)
    print("AutoRound importer completed:", a.command, out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
