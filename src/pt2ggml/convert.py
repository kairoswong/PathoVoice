"""Convert .pt checkpoints to the ggml .bin format.

Usage:
    python -m src.pt2ggml.convert models/asr/checkpoints/tiny.pt -o models/asr/ggml-tiny-ft.bin
    whisper-quantize.exe models/asr/ggml-tiny-ft.bin models/asr/ggml-tiny-ft-q5_1.bin q5_1
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import ggml_format as gf

# Head counts per (encoder layers, hidden size). Not derivable from shapes.
KNOWN_HEADS = {(4, 384): 6, (6, 512): 8, (12, 768): 12, (24, 1024): 16, (32, 1280): 20}


# -----------------------------------
# Checkpoint loading


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint and normalise it to {model_state_dict, dims}."""
    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError(f"expected a dict, got {type(ckpt)}")

    for key in ("model_state_dict", "state_dict", "model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return {"model_state_dict": ckpt[key], "dims": ckpt.get("dims")}

    if any(k.startswith(("encoder.", "decoder.")) for k in ckpt):
        return {"model_state_dict": ckpt, "dims": None}

    raise KeyError(f"no model_state_dict, and keys do not look like whisper: {list(ckpt)[:10]}")


def _to_numpy(tensor) -> np.ndarray:
    """torch.Tensor to numpy, keeping the original precision."""
    result = tensor.detach().cpu() if hasattr(tensor, "detach") else tensor
    if not hasattr(result, "numpy"):
        return np.asarray(result)

    if str(getattr(result, "dtype", "")).endswith(("bfloat16", "float16")):
        return result.float().numpy()  # no numpy equivalent
    return result.numpy()


# -----------------------------------
# Hparams inference


def infer_dims(state: dict[str, np.ndarray]) -> dict:
    """Derive hparams from tensor shapes.

    Needed when dims is missing, e.g. a training script called asdict() on a
    NamedTuple dims and swallowed the TypeError.
    """
    tok_emb = state["decoder.token_embedding.weight"]    # (n_vocab, n_text_state)
    conv1 = state["encoder.conv1.weight"]                # (n_audio_state, n_mels, 3)
    pos_enc = state["encoder.positional_embedding"]      # (n_audio_ctx, n_audio_state)
    pos_dec = state["decoder.positional_embedding"]      # (n_text_ctx, n_text_state)

    n_audio_layer = _count_layers(state, "encoder.")
    n_audio_state = int(conv1.shape[0])
    n_heads = KNOWN_HEADS.get((n_audio_layer, n_audio_state))
    if n_heads is None:
        raise ValueError(
            f"cannot infer head count for {n_audio_layer} layers, "
            f"hidden {n_audio_state}; provide dims in the checkpoint"
        )

    return {
        "n_vocab": int(tok_emb.shape[0]),
        "n_audio_ctx": int(pos_enc.shape[0]),
        "n_audio_state": n_audio_state,
        "n_audio_head": n_heads,
        "n_audio_layer": n_audio_layer,
        "n_text_ctx": int(pos_dec.shape[0]),
        "n_text_state": int(tok_emb.shape[1]),
        "n_text_head": n_heads,
        "n_text_layer": _count_layers(state, "decoder."),
        "n_mels": int(conv1.shape[1]),
    }


def _count_layers(state: dict[str, np.ndarray], prefix: str) -> int:
    """Count transformer blocks from per-block LayerNorm weights."""
    return sum(
        1
        for k in state
        if k.startswith(prefix + "blocks.") and k.endswith("mlp_ln.weight") and "cross_attn" not in k
    )


# -----------------------------------
# Validation


def validate_state(state: dict[str, np.ndarray], hp: gf.HParams, whisper_src: Path | None = None) -> list[str]:
    """Check the state dict against hparams. Returns problems; empty means OK."""
    problems: list[str] = []

    # whisper.cpp reuses d_te for embedding lookup and logits, so an untied
    # proj.weight would be silently ignored.
    if "decoder.proj.weight" in state:
        if not np.array_equal(state["decoder.proj.weight"], state["decoder.token_embedding.weight"]):
            problems.append("decoder.proj.weight is not tied to token_embedding; whisper.cpp would ignore it")

    expected = [
        ("decoder.token_embedding.weight", (hp.n_vocab, hp.n_text_state)),
        ("decoder.positional_embedding", (hp.n_text_ctx, hp.n_text_state)),
        ("encoder.positional_embedding", (hp.n_audio_ctx, hp.n_audio_state)),
        ("encoder.conv1.weight", (hp.n_audio_state, hp.n_mels, 3)),
        ("encoder.conv2.weight", (hp.n_audio_state, hp.n_audio_state, 3)),
    ]
    for name, shape in expected:
        if name not in state:
            problems.append(f"missing tensor {name}")
        elif tuple(state[name].shape) != tuple(shape):
            problems.append(f"{name} has shape {tuple(state[name].shape)}, expected {tuple(shape)}")

    for label, prefix, declared in (
        ("encoder", "encoder.", hp.n_audio_layer),
        ("decoder", "decoder.", hp.n_text_layer),
    ):
        actual = _count_layers(state, prefix)
        if actual != declared:
            problems.append(f"{label} has {actual} layers, hparams declare {declared}")

    if whisper_src is not None:
        allowed = build_allowed_names(hp, whisper_src)
        if allowed:
            for name in sorted(set(state) - allowed):
                problems.append(f"unknown tensor for whisper.cpp: {name}")
            for name in sorted(allowed - set(state)):
                problems.append(f"tensor expected by whisper.cpp is missing: {name}")

    return problems


def build_allowed_names(hp: gf.HParams, whisper_src: Path) -> set[str]:
    """Expand the tensor name table in whisper-arch.h into concrete names."""
    arch = whisper_src / "src" / "whisper-arch.h"
    if not arch.is_file():
        return set()

    templates = set(re.findall(r'"((?:encoder|decoder)\.[^"]*)"', arch.read_text(encoding="utf-8")))
    names: set[str] = set()
    for tpl in templates:
        if "%d" not in tpl:
            names.add(tpl)
            continue
        layers = hp.n_audio_layer if tpl.startswith("encoder.") else hp.n_text_layer
        names.update(tpl % i for i in range(layers))
    return names


# -----------------------------------
# Assets


def find_assets(explicit: Path | None) -> Path:
    """Locate the whisper assets directory (mel_filters.npz and *.tiktoken)."""
    candidates = [explicit] if explicit is not None else []
    try:
        import whisper

        candidates.append(Path(whisper.__file__).parent / "assets")
    except ImportError:
        pass

    candidates += [c / "whisper" / "assets" for c in list(candidates)]
    candidates += [c / "assets" for c in list(candidates)]

    for candidate in candidates:
        if candidate and (candidate / "mel_filters.npz").is_file():
            return candidate

    raise FileNotFoundError(
        "whisper assets not found; pip install openai-whisper or pass --assets "
        "(the directory must contain mel_filters.npz)"
    )


def load_mel_filters(assets: Path, n_mels: int) -> np.ndarray:
    with np.load(assets / "mel_filters.npz") as data:
        key = f"mel_{n_mels}"
        if key not in data:
            raise KeyError(f"mel_filters.npz has no {key} (available: {list(data.keys())})")
        return np.asarray(data[key], dtype=np.float32)


def load_vocab(assets: Path, multilingual: bool) -> dict[bytes, int]:
    """Read a tiktoken vocabulary as {token_bytes: rank}."""
    name = "multilingual.tiktoken" if multilingual else "gpt2.tiktoken"
    path = assets / name
    if not path.is_file():
        raise FileNotFoundError(f"vocabulary not found: {path}")

    tokens: dict[bytes, int] = {}
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                token_b64, rank = line.split()
                tokens[base64.b64decode(token_b64)] = int(rank)

    if sorted(tokens.values()) != list(range(len(tokens))):
        raise ValueError(f"{path} ranks are not a contiguous 0..N-1 range")
    return tokens


# -----------------------------------
# Conversion


def convert(
    pt_path: Path,
    out_path: Path,
    *,
    assets_dir: Path | None = None,
    use_f32: bool = False,
    whisper_src: Path | None = None,
    force: bool = False,
) -> dict:
    """Convert a checkpoint and return summary statistics."""
    ckpt = load_checkpoint(pt_path)
    state_raw = ckpt["model_state_dict"]

    inferred = False
    try:
        hp = gf.HParams.from_any(ckpt.get("dims"))
    except (ValueError, KeyError, TypeError) as exc:
        print(f"[warn] dims unusable ({exc}); inferring from tensor shapes")
        hp = gf.HParams.from_any(infer_dims({k: _to_numpy(v) for k, v in state_raw.items()}))
        inferred = True

    state: dict[str, np.ndarray] = {}
    for name, tensor in state_raw.items():
        arr = _to_numpy(tensor).squeeze()  # matches the official script
        state[name] = arr.reshape(1) if arr.ndim == 0 else arr

    problems = validate_state(state, hp, whisper_src)
    for problem in problems:
        print(f"[error] {problem}")
    if problems:
        raise SystemExit("validation failed, aborted before writing")

    assets = find_assets(assets_dir)
    filters = load_mel_filters(assets, hp.n_mels)
    vocab = load_vocab(assets, hp.is_multilingual)

    print(f"checkpoint      : {pt_path}")
    print(f"output          : {out_path}")
    print(f"dims source     : {'inferred' if inferred else 'checkpoint'}")
    print(f"model type      : {hp.model_type()}  multilingual={hp.is_multilingual}")
    print(f"assets          : {assets}")
    print(f"mel filters     : {filters.shape} {filters.dtype}")
    print(f"vocab           : {len(vocab)} tokens")
    print(f"tensors         : {len(state)}")
    print(f"precision       : {'f32' if use_f32 else 'f16 (small tensors stay f32)'}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite")

    # Write to a temp file and rename, so a failure never leaves a partial model.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    use_f16 = not use_f32
    n_f16 = n_f32 = 0
    with tmp_path.open("wb") as f:
        gf.write_header(f, hp, use_f16=use_f16)
        gf.write_mel_filters(f, filters)
        gf.write_vocab(f, vocab)
        for name, arr in state.items():
            written_type, _ = gf.write_tensor(f, name, arr, use_f16=use_f16)
            n_f16 += written_type == "f16"
            n_f32 += written_type != "f16"
    tmp_path.replace(out_path)

    size = out_path.stat().st_size
    print(f"written         : {size:,} bytes ({size / 1024 / 1024:.2f} MiB)")
    print(f"tensor types    : f16={n_f16}, f32={n_f32}")

    hp2, filters2, vocab2, tensors2 = gf.read_model(out_path)
    assert hp2 == hp, f"hparams mismatch after read-back: {hp2} != {hp}"
    assert filters2.shape == filters.shape and np.allclose(filters2, filters)
    assert len(vocab2) == len(vocab)
    assert len(tensors2) == len(state), f"tensor count {len(tensors2)} != {len(state)}"
    print("read-back check : OK")

    return {
        "hparams": hp.to_dict(),
        "n_tensors": len(state),
        "n_tokens": len(vocab),
        "size_bytes": size,
        "f16": n_f16,
        "f32": n_f32,
        "inferred_dims": inferred,
    }


# -----------------------------------
# Bin self-check


def describe_bin(path: Path) -> int:
    """Check a .bin for structural integrity. Returns 0 on success, 1 on failure.

    Works without whisper.cpp, so it can catch a quantizer that exits 0 but
    writes a malformed layout.
    """
    problems = gf.check_model_file(path)
    print(f"{path.name}  ({path.stat().st_size:,} bytes)")

    if problems:
        for problem in problems:
            print(f"  [FAIL] {problem}")
        return 1

    hp, filters, tokens, tensors = gf.read_model(path)
    counts = Counter(gf.type_name(t.ttype) for t in tensors)
    print(f"  [ OK ] {hp.model_type()}  n_vocab={hp.n_vocab}  mel={filters.shape[0]}x{filters.shape[1]}")
    print(f"  tensors={len(tensors)}  vocab={len(tokens)}  types: "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


# -----------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Whisper .pt to whisper.cpp ggml .bin, and .bin self-checks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pt", type=Path, help="input .pt checkpoint, or a .bin to check")
    parser.add_argument("-o", "--out", type=Path, help="output .bin path")
    parser.add_argument("--assets", type=Path, help="whisper assets directory (auto-detected)")
    parser.add_argument("--f32", action="store_true", help="write f32 instead of f16")
    parser.add_argument(
        "--whisper-src",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "whisper.cpp",
        help="whisper.cpp source tree, used to validate tensor names",
    )
    parser.add_argument("--check", action="store_true", help="check only, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    args = parser.parse_args(argv)

    src_path: Path = args.pt
    if not src_path.is_file():
        print(f"[error] not found: {src_path}")
        return 1

    if src_path.suffix.lower() == ".bin":
        if not args.check:
            print("[error] .bin input is only supported with --check")
            return 1
        return describe_bin(src_path)

    out_path = args.out or src_path.with_name(
        f"ggml-{src_path.stem}{'-f32' if args.f32 else ''}.bin"
    )
    whisper_src = args.whisper_src if args.whisper_src.is_dir() else None

    if args.check:
        ckpt = load_checkpoint(src_path)
        state = {k: _to_numpy(v) for k, v in ckpt["model_state_dict"].items()}
        try:
            hp = gf.HParams.from_any(ckpt.get("dims"))
            origin = "checkpoint"
        except (ValueError, KeyError, TypeError):
            hp = gf.HParams.from_any(infer_dims(state))
            origin = "inferred"

        problems = validate_state(state, hp, whisper_src)
        print(f"dims ({origin}): {hp.to_dict()}")
        print(f"model type: {hp.model_type()}, multilingual={hp.is_multilingual}")
        print(f"tensors: {len(state)}")

        if problems:
            print("\nproblems:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print("\nOK, ready to convert.")
        return 0

    convert(
        src_path,
        out_path,
        assets_dir=args.assets,
        use_f32=args.f32,
        whisper_src=whisper_src,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
