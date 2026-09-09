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

    def test_unsafe_characters_are_replaced(self):
        info = gguf.GGUFInfo(path=Path("p.gguf"), quant="F16")
        name = mmproj.link_name('org name/model:v2 <x>|y?z*w' + chr(92), info)
        for char in ' <>:"|?*' + chr(92):
            self.assertNotIn(char, name)
        self.assertTrue(name.startswith("mmproj-"))
        self.assertTrue(name.endswith("-F16.gguf"))

    def test_non_ascii_names_are_kept(self):
        # A repository name in kanji or hanzi is a legal file name; mangling
        # it to dashes would throw away which repository the link came from.
        info = gguf.GGUFInfo(path=Path("p.gguf"), quant="F16")
        name = mmproj.link_name(chr(0x6A21) + chr(0x578B) + "/GGUF", info)
        self.assertEqual(name, "mmproj-" + chr(0x6A21) + chr(0x578B) + "-GGUF-F16.gguf")

    def test_the_name_stays_within_filesystem_limits(self):
        info = gguf.GGUFInfo(path=Path("p.gguf"), quant="F16")
        # Three bytes per character in UTF-8, so this repository name is
        # 600 bytes before clipping.
        name = mmproj.link_name(chr(0x6F22) * 200, info)
        self.assertLessEqual(len(name.encode("utf-8")), 145)
        self.assertEqual(name.encode("utf-8").decode("utf-8"), name)  # no split character

    def test_a_name_that_sanitises_away_still_works(self):
        info = gguf.GGUFInfo(path=Path("p.gguf"), quant="F16")
        self.assertEqual(mmproj.link_name("///", info), "mmproj-source-F16.gguf")


class AttachFixture(unittest.TestCase):
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


class AttachTest(AttachFixture):
    def test_attach_links_and_records(self):
        name, method, moved = mmproj.attach(self.model_dir, self.projector, "upstream/base-GGUF")
        self.assertEqual(moved, [])
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
        self.assertEqual([m.name for m in moved], ["mmproj-original-F16.gguf.disabled"])
        self.assertFalse(foreign.exists())
        # Exactly one projector is left for LM Studio to find.
        self.assertEqual([p.path.name for p in mmproj.projectors_in(self.model_dir)], [name])

    def test_reattaching_replaces_our_own_link(self):
        first, _, _ = mmproj.attach(self.model_dir, self.projector, "upstream/base-GGUF")
        other = self.source / "mmproj-F16.gguf"
        other.write_bytes(projector_gguf(file_type=1))
        second, _, moved = mmproj.attach(self.model_dir, gguf.inspect(other), "other/repo-GGUF")

        self.assertNotEqual(first, second)
        self.assertEqual(moved, [])  # ours is dropped, not kept as .disabled
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


class MultipleProjectorsTest(AttachFixture):
    def test_every_stray_projector_is_moved_aside(self):
        for name in ("mmproj-one-F16.gguf", "mmproj-two-F16.gguf"):
            (self.model_dir / name).write_bytes(projector_gguf(proj_dim=3840))
        linked, _, moved = mmproj.attach(self.model_dir, self.projector, "upstream/base-GGUF")
        self.assertEqual(sorted(m.name for m in moved),
                         ["mmproj-one-F16.gguf.disabled", "mmproj-two-F16.gguf.disabled"])
        self.assertEqual([p.path.name for p in mmproj.projectors_in(self.model_dir)], [linked])

    def test_a_second_backup_does_not_overwrite_the_first(self):
        first = self.model_dir / "mmproj-x.gguf"
        first.write_bytes(projector_gguf(proj_dim=3840))
        mmproj.attach(self.model_dir, self.projector, "upstream/base-GGUF")
        first.write_bytes(projector_gguf(proj_dim=2816))
        _, _, moved = mmproj.attach(self.model_dir, self.projector, "other/repo")
        self.assertEqual([m.name for m in moved], ["mmproj-x.gguf.disabled2"])
        self.assertTrue((self.model_dir / "mmproj-x.gguf.disabled").exists())


class NormalisationTest(AttachFixture):
    """macOS stores names decomposed; the same name can come back respelled."""

    # "mmproj-ガ.gguf": composed, then the same name as ka + combining mark.
    NFC = "mmproj-" + chr(0x30AC) + ".gguf"
    NFD = "mmproj-" + chr(0x30AB) + chr(0x3099) + ".gguf"

    def test_the_two_spellings_compare_equal(self):
        self.assertNotEqual(self.NFC, self.NFD)
        self.assertTrue(mmproj.same_name(self.NFC, self.NFD))
        self.assertFalse(mmproj.same_name(self.NFC, "mmproj-other.gguf"))

    def test_our_own_link_is_recognised_through_a_respelling(self):
        # Written decomposed, recorded composed: the tool has to see one file.
        (self.model_dir / self.NFD).write_bytes(projector_gguf())
        mmproj.write_sidecar(self.model_dir, {"version": 1, "mmproj": {"link": self.NFC}})

        _, _, moved = mmproj.attach(self.model_dir, self.projector, "upstream/base-GGUF")
        # Ours, so it is dropped rather than kept as a .disabled stranger.
        self.assertEqual(moved, [])
        self.assertFalse((self.model_dir / (self.NFD + ".disabled")).exists())

    def test_detach_finds_a_respelled_link(self):
        (self.model_dir / self.NFD).write_bytes(projector_gguf())
        mmproj.write_sidecar(self.model_dir, {"version": 1, "mmproj": {"link": self.NFC}})

        self.assertEqual(mmproj.detach(self.model_dir), self.NFC)
        self.assertEqual(mmproj.projectors_in(self.model_dir), [])

    def test_find_named_returns_none_when_it_is_really_absent(self):
        self.assertIsNone(mmproj.find_named(self.model_dir, "mmproj-nowhere.gguf"))


class AvailableProgressTest(unittest.TestCase):
    def test_every_cached_gguf_is_counted_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            snapshot = cache / "hub" / "models--org--M-GGUF" / "snapshots" / "aa"
            snapshot.mkdir(parents=True)
            (snapshot / "M-Q4_K_M.gguf").write_bytes(
                (Path(__file__).parent / "dummy").name.encode())  # not a GGUF
            (snapshot / "mmproj-F16.gguf").write_bytes(projector_gguf())

            from test_lmstudio import Recorder

            recorder = Recorder()
            found = mmproj.available(cache, recorder)

        # A file that cannot be read is still one the scan had to open.
        self.assertEqual(recorder.total, 2)
        self.assertEqual(len(recorder.notes), 2)
        self.assertTrue(all("org/M-GGUF" in note for note in recorder.notes))
        self.assertEqual(len(found), 1)
