"""
Layout:
    magic      int32           0x67676d6c
    hparams    int32 x 10      order defined by HParams
    ftype      int32           0 = f32, 1 = f16
    mel        int32 n_mel, int32 n_fft, float32[n_mel * n_fft]
    vocab      int32 n_tokens, then n_tokens x (int32 len, uint8 bytes[len])
    tensors    until EOF:
                   int32 n_dims, int32 name_len, int32 ttype
                   int32 ne[n_dims]    reversed: ggml is column-major
                   uint8 name[name_len]
                   data

"""
from __future__ import annotations

import struct
from dataclasses import asdict, astuple, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

GGML_MAGIC = 0x67676D6C
TENSOR_HEADER = 12
TYPE_F32 = 0
TYPE_F16 = 1

TYPE_NAMES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_k",
    11: "q3_k",
    12: "q4_k",
    13: "q5_k",
    14: "q6_k",
    15: "q8_k",
    16: "iq2_xxs",
    17: "iq2_xs",
    18: "iq3_xxs",
    19: "iq1_s",
    20: "iq4_nl",
    21: "iq3_s",
    22: "iq2_s",
    23: "iq4_xs",
    24: "i8",
    25: "i16",
    26: "i32",
    27: "i64",
    28: "f64",
    29: "iq1_m",
    30: "bf16",
    34: "tq1_0",
    35: "tq2_0",
    39: "mxfp4",
    40: "nvfp4",
    41: "q1_0",
    42: "q2_0",
}

TYPE_BLOCK = {
    0: (1, 4),      # f32
    1: (1, 2),      # f16
    2: (32, 18),    # q4_0
    3: (32, 20),    # q4_1
    6: (32, 22),    # q5_0
    7: (32, 24),    # q5_1
    8: (32, 34),    # q8_0
    9: (32, 36),    # q8_1
    10: (256, 84),  # q2_k
    11: (256, 110), # q3_k
    12: (256, 144), # q4_k
    13: (256, 176), # q5_k
    14: (256, 210), # q6_k
    15: (256, 292), # q8_k
    30: (1, 2),     # bf16
    34: (256, 54),  # tq1_0
    35: (256, 66),  # tq2_0
    41: (128, 18),  # q1_0
    42: (64, 18),   # q2_0
}

def type_name(ttype: int) -> str:
    # Enum value to printable name.
    return TYPE_NAMES.get(ttype, f"unknown_{ttype}")

def tensor_nbytes(ttype: int, nelements: int) -> int:
    # Tensor size in bytes according to the ggml block layout.
    if ttype not in TYPE_BLOCK:
        raise ValueError(f"unknown ggml type {type_name(ttype)}, cannot size tensor")
    blck, size = TYPE_BLOCK[ttype]
    if nelements % blck != 0:
        raise ValueError(
            f"{type_name(ttype)} needs element count divisible by {blck}, got {nelements}"
        )
    return nelements // blck * size


@dataclass(frozen=True)
class HParams:
    n_vocab: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int
    n_mels: int

    # Alternative field names seen in HuggingFace-style configs.
    ALIASES = {
        "n_vocab": ("n_vocab", "vocab_size"),
        "n_audio_ctx": ("n_audio_ctx", "max_source_positions", "n_audio_context"),
        "n_audio_state": ("n_audio_state", "d_model", "n_audio_state_size"),
        "n_audio_head": ("n_audio_head", "encoder_attention_heads", "n_audio_heads"),
        "n_audio_layer": ("n_audio_layer", "encoder_layers", "n_audio_layers"),
        "n_text_ctx": ("n_text_ctx", "max_length", "max_target_positions", "n_text_context"),
        "n_text_state": ("n_text_state", "d_model", "n_text_state_size"),
        "n_text_head": ("n_text_head", "decoder_attention_heads", "n_text_heads"),
        "n_text_layer": ("n_text_layer", "decoder_layers", "n_text_layers"),
        "n_mels": ("n_mels", "num_mel_bins"),
    }

    @classmethod
    def from_any(cls, dims) -> "HParams":
        """Accept a dict, NamedTuple or dataclass."""
        if dims is None:
            raise ValueError("dims is None: checkpoint has no hyperparameters")

        if hasattr(dims, "_asdict"):
            dims = dims._asdict()
        elif hasattr(dims, "__dict__"):
            dims = vars(dims)
        elif not isinstance(dims, dict):
            raise TypeError(f"cannot interpret dims of type {type(dims)}")

        values = {}
        for field_name, aliases in cls.ALIASES.items():
            for key in aliases:
                if key in dims and dims[key] is not None:
                    values[field_name] = int(dims[key])
                    break
            else:
                raise KeyError(f"dims missing field {field_name} (tried {aliases})")

        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_multilingual(self) -> bool:
        return self.n_vocab >= 51865

    def model_type(self) -> str:
        return {
            4: "tiny",
            6: "base",
            12: "small",
            24: "medium",
            32: "large",
        }.get(self.n_audio_layer, f"unknown(n_audio_layer={self.n_audio_layer})")


def _w_i32(f: BinaryIO, value: int) -> None:
    f.write(struct.pack("<i", int(value)))


def write_header(f: BinaryIO, hp: HParams, use_f16: bool) -> None:
    _w_i32(f, GGML_MAGIC)
    for value in astuple(hp):
        _w_i32(f, value)
    _w_i32(f, 1 if use_f16 else 0)


def write_mel_filters(f: BinaryIO, filters: np.ndarray) -> None:
    """filters: (n_mel, n_fft) float32, row-major."""
    if filters.ndim != 2:
        raise ValueError(f"mel filters must be 2-D, got {filters.shape}")
    _w_i32(f, filters.shape[0])
    _w_i32(f, filters.shape[1])
    f.write(np.ascontiguousarray(filters, dtype="<f4").tobytes())


def write_vocab(f: BinaryIO, tokens: dict[bytes, int]) -> None:
    _w_i32(f, len(tokens))
    for token in tokens:
        if not isinstance(token, (bytes, bytearray)):
            raise TypeError(f"token must be bytes, got {type(token)}")
        _w_i32(f, len(token))
        f.write(bytes(token))


def write_tensor(
    f: BinaryIO,
    name: str,
    data: np.ndarray,
    use_f16: bool,
) -> tuple[str, tuple]:
    if data.ndim == 0:
        data = data.reshape(1)

    # PyTorch stores conv bias as (n,); ggml expects (n, 1).
    if name in ("encoder.conv1.bias", "encoder.conv2.bias"):
        data = data.reshape(data.shape[0], 1)

    n_dims = data.ndim
    force_f32 = (
        n_dims < 2
        or name in (
            "encoder.conv1.bias",
            "encoder.conv2.bias",
            "encoder.positional_embedding",
            "decoder.positional_embedding",
        )
    )

    if use_f16 and not force_f32:
        ttype = TYPE_F16
        payload = np.ascontiguousarray(data, dtype="<f2").tobytes()
    else:
        ttype = TYPE_F32
        payload = np.ascontiguousarray(data, dtype="<f4").tobytes()

    name_bytes = name.encode("utf-8")
    _w_i32(f, n_dims)
    _w_i32(f, len(name_bytes))
    _w_i32(f, ttype)

    for i in range(n_dims):
        _w_i32(f, data.shape[n_dims - 1 - i])
    f.write(name_bytes)
    f.write(payload)

    return type_name(ttype), tuple(data.shape)


# -----------------------------------
# Reading (for self-checks)

def _r_i32(f: BinaryIO) -> int:
    return struct.unpack("<i", f.read(4))[0]


def _r_i32_n(f: BinaryIO, n: int) -> list[int]:
    return list(struct.unpack(f"<{n}i", f.read(4 * n)))


@dataclass
class TensorInfo:
    name: str
    shape: tuple
    ttype: int
    nbytes: int


def read_model(path: str | Path) -> tuple[HParams, np.ndarray, dict[bytes, int], list[TensorInfo]]:
    path = Path(path)
    with path.open("rb") as f:
        magic = _r_i32(f)
        if magic != GGML_MAGIC:
            raise ValueError(f"magic mismatch: {magic:#x}")

        hp = HParams(*_r_i32_n(f, len(HParams.__dataclass_fields__)))
        _r_i32(f)  # ftype, unused here

        n_mel, n_fft = _r_i32_n(f, 2)
        filters = np.frombuffer(f.read(n_mel * n_fft * 4), dtype="<f4").reshape(n_mel, n_fft)

        tokens: dict[bytes, int] = {}
        for _ in range(_r_i32(f)):
            tokens[f.read(_r_i32(f))] = len(tokens)

        tensors: list[TensorInfo] = []
        while len(head := f.read(TENSOR_HEADER)) == TENSOR_HEADER:
            n_dims, name_len, ttype = struct.unpack("<iii", head)
            ne = _r_i32_n(f, n_dims)
            name = f.read(name_len).decode("utf-8")

            nbytes = tensor_nbytes(ttype, int(np.prod(ne)))
            f.seek(nbytes, 1)
            tensors.append(TensorInfo(name=name, shape=tuple(ne), ttype=ttype, nbytes=nbytes))

    return hp, filters, tokens, tensors


def check_model_file(path: str | Path, *, expect_tensors: int | None = None) -> list[str]:
    # Structural integrity check.
    path = Path(path)
    try:
        _hp, filters, tokens, tensors = read_model(path)
    except ValueError as exc:
        return [f"parse failed: {exc}"]
    except OSError as exc:
        return [f"cannot read: {exc}"]

    problems: list[str] = []

    if expect_tensors is not None and len(tensors) != expect_tensors:
        problems.append(f"tensor count {len(tensors)} != expected {expect_tensors}")

    overhead = (
        4                                           # magic
        + 4 * len(HParams.__dataclass_fields__)     # hparams
        + 4                                         # ftype
        + 8 + filters.nbytes                        # mel dims + data
        + 4                                         # n_tokens
        + sum(4 + len(tok) for tok in tokens)       # vocab entries
        + sum(TENSOR_HEADER + 4 * len(t.shape) + len(t.name) for t in tensors)
    )

    actual = path.stat().st_size
    expected = sum(t.nbytes for t in tensors) + overhead
    if actual != expected:
        problems.append(
            f"size {actual:,} != expected {expected:,} (diff {actual - expected:+,} bytes)"
        )

    return problems

# -----------------------------------
