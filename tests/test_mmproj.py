"""Compatibility judgement, link naming, and the attach/detach round trip."""

import tempfile
import unittest
from pathlib import Path

from lmshf import gguf, mmproj
from test_gguf import build_gguf


def text_gguf(arch="gemma4", n_embd=5376, file_type=15):
    return build_gguf([
        ("general.architecture", gguf.STRING, arch),
        ("general.type", gguf.STRING, "model"),
        (f"{arch}.embedding_length", gguf.UINT32, n_embd),
        ("general.file_type", gguf.UINT32, file_type),
    ])


def projector_gguf(projector_type="gemma4v", proj_dim=5376, file_type=0):
    return build_gguf([
        ("general.architecture", gguf.STRING, "clip"),
        ("general.type", gguf.STRING, "mmproj"),
        ("general.file_type", gguf.UINT32, file_type),
        ("clip.has_vision_encoder", gguf.BOOL, True),
        ("clip.vision.projection_dim", gguf.UINT32, proj_dim),
        ("clip.vision.projector_type", gguf.STRING, projector_type),
    ])


class FamilyTest(unittest.TestCase):
    def test_known_families_collapse_versions(self):
        self.assertEqual(mmproj.family("gemma4"), "gemma")
        self.assertEqual(mmproj.family("gemma4v"), "gemma")
        self.assertEqual(mmproj.family("gemma4uv"), "gemma")
        self.assertEqual(mmproj.family("qwen35"), "qwen")
        self.assertEqual(mmproj.family("qwen3vl_merger"), "qwen")
        self.assertEqual(mmproj.family("qwen2vl_merger"), "qwen")

    def test_unknown_families_are_not_judged(self):
        # Deliberate: a pixtral projector rides on a "llama" text model, so
        # calling these different families would reject a working pair.
        self.assertEqual(mmproj.family("pixtral"), "")
        self.assertEqual(mmproj.family("minicpmv"), "")
        self.assertEqual(mmproj.family("mlp"), "")
        self.assertEqual(mmproj.family(""), "")


class CheckTest(unittest.TestCase):
    def _pair(self, text_kwargs=None, proj_kwargs=None):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "text.gguf").write_bytes(text_gguf(**(text_kwargs or {})))
            (tmp / "mmproj.gguf").write_bytes(projector_gguf(**(proj_kwargs or {})))
            return gguf.inspect(tmp / "text.gguf"), gguf.inspect(tmp / "mmproj.gguf")

    def test_matching_pair_is_ok(self):
        text, projector = self._pair()
        result = mmproj.check(text, projector)
        self.assertEqual(result.verdict, mmproj.OK)
        self.assertFalse(result.blocked)

    def test_different_family_is_blocked(self):
        text, projector = self._pair(proj_kwargs={"projector_type": "qwen3vl_merger",
                                                  "proj_dim": 5376})
        result = mmproj.check(text, projector)
        self.assertEqual(result.verdict, mmproj.INCOMPATIBLE)
        self.assertTrue(result.blocked)

    def test_same_family_wrong_size_is_suspect(self):
        # The common mistake: the 12B projector on the 31B model.
        text, projector = self._pair(proj_kwargs={"proj_dim": 3840})
        result = mmproj.check(text, projector)
        self.assertEqual(result.verdict, mmproj.SUSPECT)
        self.assertFalse(result.blocked)
        self.assertIn("3840", result.reason)

    def test_unknown_family_falls_back_to_the_dimensions(self):
        text, projector = self._pair(text_kwargs={"arch": "internlm2", "n_embd": 4096},
                                     proj_kwargs={"projector_type": "internvl",
                                                  "proj_dim": 4096})
        self.assertEqual(mmproj.check(text, projector).verdict, mmproj.OK)

    def test_a_text_model_is_not_a_projector(self):
        text, _ = self._pair()
        self.assertTrue(mmproj.check(text, text).blocked)

    def test_unreadable_text_model_is_suspect(self):
        _, projector = self._pair()
        broken = gguf.inspect(Path("nope.gguf"))
        self.assertEqual(mmproj.check(broken, projector).verdict, mmproj.SUSPECT)


class LinkNameTest(unittest.TestCase):
    def test_carries_the_repository_and_quantisation(self):
        info = gguf.GGUFInfo(path=Path("mmproj-F32.gguf"), quant="F32")
        self.assertEqual(
            mmproj.link_name("unsloth/gemma-4-31B-it-GGUF", info),
            "mmproj-unsloth-gemma-4-31B-it-GGUF-F32.gguf",
        )

    def test_always_contains_the_mmproj_marker(self):
        info = gguf.GGUFInfo(path=Path("visual.gguf"), quant="F16")
        self.assertTrue(mmproj.link_name("org/model", info).startswith("mmproj-"))

    def test_unsafe_characters_and_length(self):
        info = gguf.GGUFInfo(path=Path("p.gguf"), quant="F16")
        name = mmproj.link_name("org name/model:v2 " + "x" * 100, info)
        self.assertNotIn(" ", name)
        self.assertNotIn(":", name)
        self.assertLess(len(name), 90)


class AttachTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.model_dir = root / "models" / "org" / "derivative-GGUF"
        self.model_dir.mkdir(parents=True)
        (self.model_dir / "derivative-Q4_K_M.gguf").write_bytes(text_gguf())
        self.source = root / "cache" / "snapshot"
        self.source.mkdir(parents=True)
        (self.source / "mmproj-F32.gguf").write_bytes(projector_gguf())
        self.projector = gguf.inspect(self.source / "mmproj-F32.gguf")
        self.addCleanup(self.tmp.cleanup)

    def test_attach_links_and_records(self):
        name, method, moved = mmproj.attach(self.model_dir, self.projector, "upstream/base-GGUF")
        self.assertIsNone(moved)
        self.assertIn(method, ("symlink", "junction", "hardlink"))
        linked = self.model_dir / name
        self.assertTrue(linked.exists())
        self.assertIn("mmproj", name)
        # The bytes are the projector's, and it was linked rather than copied.
        self.assertEqual(linked.read_bytes(), (self.source / "mmproj-F32.gguf").read_bytes())

        record = mmproj.read_sidecar(self.model_dir)["mmproj"]
        self.assertEqual(record["source_repo"], "upstream/base-GGUF")
        self.assertEqual(record["source_file"], "mmproj-F32.gguf")
        self.assertEqual(record["proj_dim"], 5376)
        self.assertEqual(record["modalities"], ["vision"])

    def test_the_attached_projector_is_found_by_a_rescan(self):
        mmproj.attach(self.model_dir, self.projector, "upstream/base-GGUF")
        found = mmproj.projectors_in(self.model_dir)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].proj_dim, 5376)

    def test_detach_removes_the_link_and_the_record(self):
        name, _, _ = mmproj.attach(self.model_dir, self.projector, "upstream/base-GGUF")
        self.assertEqual(mmproj.detach(self.model_dir), name)
        self.assertFalse((self.model_dir / name).exists())
        self.assertFalse((self.model_dir / mmproj.SIDECAR).exists())
        # The source is untouched.
        self.assertTrue((self.source / "mmproj-F32.gguf").exists())

    def test_detach_without_a_record_reports_nothing(self):
        self.assertIsNone(mmproj.detach(self.model_dir))

    def test_a_foreign_projector_is_moved_aside(self):
        foreign = self.model_dir / "mmproj-original-F16.gguf"
        foreign.write_bytes(projector_gguf(proj_dim=3840))
        name, _, moved = mmproj.attach(self.model_dir, self.projector, "upstream/base-GGUF")
        self.assertIsNotNone(moved)
        self.assertEqual(moved.name, "mmproj-original-F16.gguf.disabled")
        self.assertFalse(foreign.exists())
        # Exactly one projector is left for LM Studio to find.
        self.assertEqual([p.path.name for p in mmproj.projectors_in(self.model_dir)], [name])

    def test_reattaching_replaces_our_own_link(self):
        first, _, _ = mmproj.attach(self.model_dir, self.projector, "upstream/base-GGUF")
        other = self.source / "mmproj-F16.gguf"
        other.write_bytes(projector_gguf(file_type=1))
        second, _, moved = mmproj.attach(self.model_dir, gguf.inspect(other), "other/repo-GGUF")

        self.assertNotEqual(first, second)
        self.assertIsNone(moved)  # ours is dropped, not kept as .disabled
        self.assertFalse((self.model_dir / first).exists())
        self.assertEqual([p.path.name for p in mmproj.projectors_in(self.model_dir)], [second])
        self.assertEqual(mmproj.read_sidecar(self.model_dir)["mmproj"]["link"], second)

    def test_attach_keeps_other_sidecar_keys(self):
        mmproj.write_sidecar(self.model_dir, {"version": 1, "note": "keep me"})
        mmproj.attach(self.model_dir, self.projector, "upstream/base-GGUF")
        mmproj.detach(self.model_dir)
        record = mmproj.read_sidecar(self.model_dir)
        self.assertEqual(record.get("note"), "keep me")
        self.assertNotIn("mmproj", record)


if __name__ == "__main__":
    unittest.main()
