"""Reading a model directory, and planning what to link into one."""

import tempfile
import unittest
from pathlib import Path

from lmshf import lmstudio
from lmshf.importing import import_plan
from test_mmproj import projector_gguf, text_gguf


def split_shard(part, of, payload=0, **kwargs):
    """A shard of a split model; only the first one carries the header."""
    from lmshf import gguf
    from test_gguf import build_gguf

    if part == 1:
        body = text_gguf(**kwargs)
    else:
        body = build_gguf([("split.no", gguf.UINT16, part - 1),
                           ("split.count", gguf.UINT16, of)])
    return body + b"\0" * payload


class DescribeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "org" / "Model-GGUF"
        self.dir.mkdir(parents=True)

    def test_a_split_model_reports_the_size_of_every_part(self):
        (self.dir / "M-IQ4_XS-00001-of-00002.gguf").write_bytes(split_shard(1, 2, payload=3000))
        (self.dir / "M-IQ4_XS-00002-of-00002.gguf").write_bytes(split_shard(2, 2, payload=1000))
        model = lmstudio.describe("org/Model-GGUF", self.dir)

        self.assertIsNotNone(model.text)
        self.assertIn("00001", model.text.path.name)  # the part with the header
        self.assertEqual(model.extras, [])  # the other part is not a second model
        self.assertEqual(model.size, sum(p.stat().st_size for p in self.dir.glob("*.gguf")))
        self.assertGreater(model.size, model.text.size)

    def test_an_unsplit_model_reports_its_own_size(self):
        path = self.dir / "M-Q4_K_M.gguf"
        path.write_bytes(text_gguf() + b"\0" * 500)
        model = lmstudio.describe("org/Model-GGUF", self.dir)
        self.assertEqual(model.size, path.stat().st_size)

    def test_a_quantisation_subdirectory_is_still_found(self):
        # What LM Studio does with a repository that ships IQ4_XS/, Q5_K_M/ ...
        nested = self.dir / "IQ4_XS"
        nested.mkdir()
        (nested / "M-IQ4_XS.gguf").write_bytes(text_gguf())
        model = lmstudio.describe("org/Model-GGUF", self.dir)

        self.assertIsNotNone(model.text)
        self.assertEqual(model.text.arch, "gemma4")
        self.assertEqual([p.name for p in model.nested], ["M-IQ4_XS.gguf"])

    def test_a_flat_model_is_not_reported_as_nested(self):
        (self.dir / "M-Q4_K_M.gguf").write_bytes(text_gguf())
        self.assertEqual(lmstudio.describe("org/Model-GGUF", self.dir).nested, [])

    def test_an_empty_directory(self):
        model = lmstudio.describe("org/Model-GGUF", self.dir)
        self.assertIsNone(model.text)
        self.assertEqual(model.projectors, [])


class ImportPlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = Path(self.tmp.name) / "snapshot"
        self.snapshot.mkdir(parents=True)

    def plan(self):
        return {name: source for source, name in import_plan(self.snapshot)}

    def test_a_flat_snapshot_is_linked_as_it_is(self):
        (self.snapshot / "M-Q4_K_M.gguf").write_bytes(text_gguf())
        (self.snapshot / "README.md").write_text("hi", encoding="utf-8")
        self.assertEqual(sorted(self.plan()), ["M-Q4_K_M.gguf", "README.md"])

    def test_quantisation_subdirectories_are_flattened(self):
        # Aratako/Amaterasu-123B-GGUF is laid out this way: linking the
        # directory puts the model one level too deep for LM Studio.
        nested = self.snapshot / "IQ4_XS"
        nested.mkdir()
        for part in (1, 2):
            (nested / f"M-IQ4_XS-0000{part}-of-00002.gguf").write_bytes(split_shard(part, 2))
        (self.snapshot / "README.md").write_text("hi", encoding="utf-8")

        plan = self.plan()
        self.assertEqual(sorted(plan), ["M-IQ4_XS-00001-of-00002.gguf",
                                        "M-IQ4_XS-00002-of-00002.gguf", "README.md"])
        self.assertEqual(plan["M-IQ4_XS-00001-of-00002.gguf"].parent, nested)

    def test_identical_names_in_two_directories_are_kept_apart(self):
        for quant in ("IQ4_XS", "Q5_K_M"):
            (self.snapshot / quant).mkdir()
            (self.snapshot / quant / "model.gguf").write_bytes(text_gguf())
        self.assertEqual(sorted(self.plan()), ["IQ4_XS-model.gguf", "Q5_K_M-model.gguf"])

    def test_a_subdirectory_holding_no_gguf_is_left_alone(self):
        # An "original/" or "imatrix/" directory stays one link, as before.
        (self.snapshot / "original").mkdir()
        (self.snapshot / "original" / "weights.safetensors").write_bytes(b"x")
        plan = self.plan()
        self.assertEqual(sorted(plan), ["original"])
        self.assertTrue(plan["original"].is_dir())

    def test_a_top_level_file_is_not_shadowed_by_a_nested_one(self):
        (self.snapshot / "model.gguf").write_bytes(text_gguf())
        (self.snapshot / "IQ4_XS").mkdir()
        (self.snapshot / "IQ4_XS" / "model.gguf").write_bytes(projector_gguf())
        plan = self.plan()
        self.assertEqual(plan["model.gguf"].parent, self.snapshot)
        self.assertEqual(plan["IQ4_XS-model.gguf"].parent, self.snapshot / "IQ4_XS")
