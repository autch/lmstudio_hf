"""GGUF header reader, exercised against files built here byte by byte."""

import struct
import tempfile
import unittest
from pathlib import Path

from lmshf import gguf


def _u32(value):
    return struct.pack("<I", value)


def _u64(value):
    return struct.pack("<Q", value)


def _string(text):
    raw = text.encode("utf-8")
    return _u64(len(raw)) + raw


def _value(vtype, value):
    if vtype == gguf.STRING:
        return _string(value)
    if vtype == gguf.ARRAY:
        item_type, items = value
        body = b"".join(_value(item_type, item) for item in items)
        return _u32(item_type) + _u64(len(items)) + body
    return struct.pack(gguf._FORMAT[vtype], value)


def build_gguf(pairs, version=3, tensor_count=0, magic=gguf.GGUF_MAGIC):
    """pairs: [(key, value_type, value)] in file order."""
    out = [magic, _u32(version), _u64(tensor_count), _u64(len(pairs))]
    for key, vtype, value in pairs:
        out.append(_string(key) + _u32(vtype) + _value(vtype, value))
    return b"".join(out)


class TempGGUF:
    """Write bytes to a real file; the reader works on paths, not buffers."""

    def __init__(self, data, name="model.gguf"):
        self.data = data
        self.name = name

    def __enter__(self):
        self.dir = tempfile.TemporaryDirectory()
        path = Path(self.dir.name) / self.name
        path.write_bytes(self.data)
        return path

    def __exit__(self, *exc):
        self.dir.cleanup()


class ReadMetadataTest(unittest.TestCase):
    def test_scalars_and_strings(self):
        data = build_gguf([
            ("general.architecture", gguf.STRING, "gemma4"),
            ("general.file_type", gguf.UINT32, 15),
            ("gemma4.embedding_length", gguf.UINT32, 5376),
            ("clip.has_vision_encoder", gguf.BOOL, True),
        ])
        with TempGGUF(data) as path:
            md = gguf.read_metadata(path)
        self.assertEqual(md["general.architecture"], "gemma4")
        self.assertEqual(md["general.file_type"], 15)
        self.assertEqual(md["gemma4.embedding_length"], 5376)
        self.assertIs(md["clip.has_vision_encoder"], True)

    def test_small_arrays_are_decoded(self):
        data = build_gguf([
            ("general.tags", gguf.ARRAY, (gguf.STRING, ["unsloth", "vision"])),
            ("clip.vision.image_mean", gguf.ARRAY, (gguf.FLOAT32, [0.5, 0.5, 0.5])),
        ])
        with TempGGUF(data) as path:
            md = gguf.read_metadata(path)
        self.assertEqual(md["general.tags"], ["unsloth", "vision"])
        self.assertEqual(len(md["clip.vision.image_mean"]), 3)

    def test_huge_wanted_array_is_skipped_not_decoded(self):
        items = [f"tok{i}" for i in range(gguf.MAX_ARRAY + 10)]
        data = build_gguf([
            ("general.tags", gguf.ARRAY, (gguf.STRING, items)),
            ("general.architecture", gguf.STRING, "llama"),
        ])
        with TempGGUF(data) as path:
            md = gguf.read_metadata(path)
        self.assertIsNone(md["general.tags"])
        # Position was kept, so the key after the array still reads.
        self.assertEqual(md["general.architecture"], "llama")

    def test_keys_after_a_vocabulary_sized_array_still_read(self):
        # The regression this guards: skipping a long string array has to
        # leave the file exactly at its end. general.file_type sits after the
        # tokenizer block in real files.
        vocab = [f"token{i}" for i in range(20000)]
        data = build_gguf([
            ("general.architecture", gguf.STRING, "qwen35"),
            ("tokenizer.ggml.tokens", gguf.ARRAY, (gguf.STRING, vocab)),
            ("tokenizer.ggml.token_type", gguf.ARRAY, (gguf.INT32, [1] * 20000)),
            ("general.file_type", gguf.UINT32, 30),
        ])
        with TempGGUF(data) as path:
            md = gguf.read_metadata(path)
        self.assertNotIn("tokenizer.ggml.tokens", md)
        self.assertEqual(md["general.file_type"], 30)

    def test_string_longer_than_the_read_block(self):
        # Forces the branch where a string runs past the buffered block.
        big = "x" * (2 << 20)
        data = build_gguf([
            ("tokenizer.ggml.tokens", gguf.ARRAY, (gguf.STRING, ["a", big, "b"])),
            ("general.file_type", gguf.UINT32, 7),
        ])
        with TempGGUF(data) as path:
            md = gguf.read_metadata(path)
        self.assertEqual(md["general.file_type"], 7)

    def test_unwanted_scalar_types_are_skipped_correctly(self):
        data = build_gguf([
            ("some.u8", gguf.UINT8, 3),
            ("some.i16", gguf.INT16, -2),
            ("some.f64", gguf.FLOAT64, 1.5),
            ("some.i64", gguf.INT64, -9),
            ("general.name", gguf.STRING, "after"),
        ])
        with TempGGUF(data) as path:
            md = gguf.read_metadata(path)
        self.assertEqual(md["general.name"], "after")


class ErrorTest(unittest.TestCase):
    def test_not_a_gguf_file(self):
        with TempGGUF(b"PK\x03\x04rest of a zip") as path:
            with self.assertRaises(gguf.GGUFError):
                gguf.read_metadata(path)

    def test_unsupported_version(self):
        with TempGGUF(build_gguf([], version=1)) as path:
            with self.assertRaises(gguf.GGUFError):
                gguf.read_metadata(path)

    def test_truncated_header(self):
        data = build_gguf([("general.name", gguf.STRING, "cut")])
        with TempGGUF(data[:-4]) as path:
            with self.assertRaises(gguf.GGUFError):
                gguf.read_metadata(path)

    def test_inspect_reports_errors_instead_of_raising(self):
        with TempGGUF(b"not gguf at all") as path:
            info = gguf.inspect(path)
        self.assertFalse(info.readable)
        self.assertTrue(info.error)
        self.assertEqual(info.kind, "unknown")

    def test_inspect_of_a_missing_file(self):
        info = gguf.inspect(Path("no") / "such" / "file.gguf")
        self.assertFalse(info.readable)


class QuantLabelTest(unittest.TestCase):
    def test_from_metadata(self):
        self.assertEqual(gguf.quant_label(15), "Q4_K_M")
        self.assertEqual(gguf.quant_label(30), "IQ4_XS")
        self.assertEqual(gguf.quant_label(32), "BF16")
        self.assertEqual(gguf.quant_label(0), "F32")

    def test_falls_back_to_the_file_name(self):
        self.assertEqual(gguf.quant_label(None, "Model-Q5_K_M.gguf"), "Q5_K_M")
        self.assertEqual(gguf.quant_label(None, "model.q8_0.gguf"), "Q8_0")
        self.assertEqual(gguf.quant_label(None, "mmproj-F16.gguf"), "F16")
        self.assertEqual(gguf.quant_label(None, "Gemma4-12B-BF16.gguf"), "BF16")

    def test_unknown_stays_visible(self):
        self.assertEqual(gguf.quant_label(None, "mmproj-model.gguf"), "?")
        self.assertEqual(gguf.quant_label(99, "mmproj-model.gguf"), "ftype99")

    def test_a_size_label_is_not_mistaken_for_a_quantisation(self):
        self.assertEqual(gguf.quant_label(None, "Qwen3.6-27B-Instruct.gguf"), "?")


class InspectTest(unittest.TestCase):
    def test_text_model(self):
        data = build_gguf([
            ("general.architecture", gguf.STRING, "gemma4"),
            ("general.type", gguf.STRING, "model"),
            ("general.name", gguf.STRING, "Gemma 4 31B"),
            ("gemma4.embedding_length", gguf.UINT32, 5376),
            ("general.file_type", gguf.UINT32, 15),
        ])
        with TempGGUF(data) as path:
            info = gguf.inspect(path)
        self.assertEqual(info.kind, "model")
        self.assertFalse(info.is_projector)
        self.assertEqual(info.arch, "gemma4")
        self.assertEqual(info.n_embd, 5376)
        self.assertEqual(info.quant, "Q4_K_M")
        self.assertEqual(info.modalities, ())

    def test_projector_with_the_gemma_key_spelling(self):
        data = build_gguf([
            ("general.architecture", gguf.STRING, "clip"),
            ("general.type", gguf.STRING, "mmproj"),
            ("general.file_type", gguf.UINT32, 0),
            ("clip.has_vision_encoder", gguf.BOOL, True),
            ("clip.vision.projection_dim", gguf.UINT32, 5376),
            ("clip.vision.projector_type", gguf.STRING, "gemma4v"),
        ])
        with TempGGUF(data, name="mmproj-F32.gguf") as path:
            info = gguf.inspect(path)
        self.assertTrue(info.is_projector)
        self.assertEqual(info.proj_dim, 5376)
        self.assertEqual(info.projector_type, "gemma4v")
        self.assertEqual(info.modalities, ("vision",))

    def test_projector_with_the_qwen_key_spelling(self):
        data = build_gguf([
            ("general.architecture", gguf.STRING, "clip"),
            ("general.type", gguf.STRING, "mmproj"),
            ("clip.has_vision_encoder", gguf.BOOL, True),
            ("clip.vision.projection_dim", gguf.UINT32, 5120),
            ("clip.projector_type", gguf.STRING, "qwen3vl_merger"),
        ])
        with TempGGUF(data) as path:
            info = gguf.inspect(path)
        self.assertEqual(info.projector_type, "qwen3vl_merger")
        self.assertEqual(info.proj_dim, 5120)

    def test_projector_with_audio(self):
        data = build_gguf([
            ("general.architecture", gguf.STRING, "clip"),
            ("general.type", gguf.STRING, "mmproj"),
            ("clip.has_vision_encoder", gguf.BOOL, True),
            ("clip.vision.projection_dim", gguf.UINT32, 3840),
            ("clip.audio.projection_dim", gguf.UINT32, 3840),
            ("clip.vision.projector_type", gguf.STRING, "gemma4uv"),
        ])
        with TempGGUF(data) as path:
            info = gguf.inspect(path)
        self.assertEqual(info.modalities, ("vision", "audio"))

    def test_projector_without_general_type(self):
        # Conversions from before general.type existed.
        data = build_gguf([
            ("general.architecture", gguf.STRING, "clip"),
            ("clip.has_vision_encoder", gguf.BOOL, True),
            ("clip.projector_type", gguf.STRING, "mlp"),
        ])
        with TempGGUF(data, name="llava-projector.gguf") as path:
            info = gguf.inspect(path)
        self.assertTrue(info.is_projector)


if __name__ == "__main__":
    unittest.main()


class HumanSizeTest(unittest.TestCase):
    def test_decimal_units_match_what_lm_studio_shows(self):
        # The two shards of Aratako/Amaterasu-123B-GGUF add up to this, and
        # LM Studio calls it 65.4 GB; 1024-based units would say 60.9.
        self.assertEqual(gguf.human_size(49956119840 + 15478221856), "65.4GB")

    def test_small_sizes(self):
        self.assertEqual(gguf.human_size(0), "?")
        self.assertEqual(gguf.human_size(512), "512B")
        self.assertEqual(gguf.human_size(1500), "1.5KB")
