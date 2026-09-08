"""A minimal GGUF header reader.

Only the key/value block at the head of the file is parsed, and values the
caller did not ask for are skipped with seek() rather than decoded. That
matters: a text model carries `tokenizer.ggml.tokens`, an array of ~150k
strings, and decoding it for every file in a cache scan is the difference
between milliseconds and seconds per model.

Tensor information is never touched, so this stays fast on 30GB files.

The value-type numbering is the GGUF spec's, as implemented by llama.cpp's
gguf-py (MIT).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import struct

GGUF_MAGIC = b"GGUF"

# GGUF value types.
UINT8, INT8, UINT16, INT16 = 0, 1, 2, 3
UINT32, INT32, FLOAT32, BOOL = 4, 5, 6, 7
STRING, ARRAY = 8, 9
UINT64, INT64, FLOAT64 = 10, 11, 12

_FORMAT = {
    UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h",
    UINT32: "<I", INT32: "<i", FLOAT32: "<f", BOOL: "<?",
    UINT64: "<Q", INT64: "<q", FLOAT64: "<d",
}
_SIZE = {t: struct.calcsize(f) for t, f in _FORMAT.items()}

# Arrays longer than this are skipped rather than decoded, even when wanted.
MAX_ARRAY = 256


class GGUFError(Exception):
    """The file is not a GGUF file, or its header could not be parsed."""


def _read(f, count):
    data = f.read(count)
    if len(data) != count:
        raise GGUFError("unexpected end of file")
    return data


def _read_scalar(f, vtype):
    return struct.unpack(_FORMAT[vtype], _read(f, _SIZE[vtype]))[0]


def _read_string(f):
    length = _read_scalar(f, UINT64)
    return _read(f, length).decode("utf-8", "replace")


def _skip_value(f, vtype):
    if vtype in _SIZE:
        f.seek(_SIZE[vtype], 1)
    elif vtype == STRING:
        f.seek(_read_scalar(f, UINT64), 1)
    elif vtype == ARRAY:
        item_type = _read_scalar(f, UINT32)
        count = _read_scalar(f, UINT64)
        _skip_array_items(f, item_type, count)
    else:
        raise GGUFError(f"unknown value type {vtype}")


def _skip_array_items(f, item_type, count):
    if item_type in _SIZE:
        f.seek(_SIZE[item_type] * count, 1)
    elif item_type == STRING:
        _skip_string_array(f, count)
    else:
        raise GGUFError(f"unsupported array item type {item_type}")


def _skip_string_array(f, count):
    """Step over `count` length-prefixed strings.

    Strings can only be skipped by walking them, and a text model's
    `tokenizer.ggml.tokens` holds ~150k of them. Doing that with one seek per
    string is slow twice over: seeking discards the read buffer, so every
    length prefix becomes its own tiny read. Pulling the region in in 1MB
    blocks and scanning it in memory keeps the reads sequential.
    """
    block = 1 << 20
    buf = b""
    pos = 0
    for _ in range(count):
        if len(buf) - pos < 8:
            buf = buf[pos:] + f.read(block)
            pos = 0
            if len(buf) < 8:
                raise GGUFError("unexpected end of file")
        (length,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        if len(buf) - pos < length:
            # A string longer than what is buffered; give up on the buffer.
            f.seek(length - (len(buf) - pos), 1)
            buf = b""
            pos = 0
        else:
            pos += length
    # Put the file back where the array actually ended.
    f.seek(pos - len(buf), 1)


def _read_value(f, vtype):
    if vtype in _SIZE:
        return _read_scalar(f, vtype)
    if vtype == STRING:
        return _read_string(f)
    if vtype == ARRAY:
        item_type = _read_scalar(f, UINT32)
        count = _read_scalar(f, UINT64)
        if count > MAX_ARRAY:
            _skip_array_items(f, item_type, count)
            return None
        if item_type == ARRAY:
            raise GGUFError("nested arrays are not supported")
        return [_read_value(f, item_type) for _ in range(count)]
    raise GGUFError(f"unknown value type {vtype}")


def wanted_by_default(key):
    """Keys worth decoding: identity, quantisation, projector and dimensions."""
    return (
        key.startswith("general.")
        or key.startswith("clip.")
        or key.endswith(".embedding_length")
        or key.endswith(".block_count")
    )


def read_metadata(path, wanted=wanted_by_default):
    """Return the decoded key/value pairs of `path` that `wanted` accepts."""
    with open(path, "rb", buffering=1 << 20) as f:
        if _read(f, 4) != GGUF_MAGIC:
            raise GGUFError("not a GGUF file")
        version = _read_scalar(f, UINT32)
        if version not in (2, 3):
            # v1 counted with 32-bit fields; llama.cpp dropped it long ago.
            raise GGUFError(f"unsupported GGUF version {version}")
        _read_scalar(f, UINT64)  # tensor count
        kv_count = _read_scalar(f, UINT64)

        metadata = {}
        for _ in range(kv_count):
            key = _read_string(f)
            vtype = _read_scalar(f, UINT32)
            if wanted(key):
                metadata[key] = _read_value(f, vtype)
            else:
                _skip_value(f, vtype)
        return metadata


# llama_ftype, the values `general.file_type` holds. Gaps are formats that
# llama.cpp has removed.
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
}

_QUANT_IN_NAME = re.compile(
    r"(?<![A-Za-z0-9])(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|BF16|F16|F32|TQ\d_\d)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def quant_label(file_type, name=""):
    """A human quantisation label, from the metadata or else the file name."""
    if file_type is not None and file_type in FILE_TYPES:
        return FILE_TYPES[file_type]
    match = _QUANT_IN_NAME.search(Path(name).stem)
    if match:
        return match.group(1).upper()
    if file_type is not None:
        return f"ftype{file_type}"
    return "?"


def human_size(size):
    if not size:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return "?"


@dataclass
class GGUFInfo:
    """What one GGUF file is, as far as its header says."""

    path: Path
    size: int = 0
    kind: str = "unknown"  # "model", "mmproj" or "unknown"
    arch: str = ""  # general.architecture, e.g. gemma4, qwen35, clip
    name: str = ""  # general.name
    quant: str = "?"
    n_embd: int = None  # {arch}.embedding_length, the text side
    proj_dim: int = None  # clip.*.projection_dim, the projector output
    projector_type: str = ""  # e.g. gemma4v, qwen3vl_merger
    modalities: tuple = ()
    error: str = ""

    @property
    def is_projector(self):
        return self.kind == "mmproj"

    @property
    def readable(self):
        return not self.error


def _first(metadata, *keys):
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return None


def inspect(path):
    """Read one GGUF file's header. Never raises; check `.error` instead."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    try:
        metadata = read_metadata(path)
    except (GGUFError, OSError) as exc:
        return GGUFInfo(path=path, size=size, quant=quant_label(None, path.name),
                        error=str(exc))

    arch = metadata.get("general.architecture", "") or ""
    kind = metadata.get("general.type") or ("mmproj" if arch == "clip" else "model")
    # Older conversions predate general.type; clip.* keys still give it away.
    if kind != "mmproj" and any(k.startswith("clip.") for k in metadata):
        kind = "mmproj"

    modalities = []
    if metadata.get("clip.has_vision_encoder") or "clip.vision.projection_dim" in metadata:
        modalities.append("vision")
    if metadata.get("clip.has_audio_encoder") or "clip.audio.projection_dim" in metadata:
        modalities.append("audio")

    return GGUFInfo(
        path=path,
        size=size,
        kind=kind,
        arch=arch,
        name=metadata.get("general.name", "") or "",
        quant=quant_label(metadata.get("general.file_type"), path.name),
        n_embd=metadata.get(f"{arch}.embedding_length"),
        # Which of these two keys is used depends on the converter, and both
        # are in the wild: Qwen writes clip.projector_type, Gemma writes
        # clip.vision.projector_type.
        proj_dim=_first(metadata, "clip.vision.projection_dim", "clip.audio.projection_dim"),
        projector_type=_first(metadata, "clip.projector_type",
                              "clip.vision.projector_type",
                              "clip.audio.projector_type") or "",
        modalities=tuple(modalities),
    )
