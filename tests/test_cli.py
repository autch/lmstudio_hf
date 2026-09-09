"""End to end through main(), against a synthetic cache and models directory."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from lmshf import mmproj
from lmshf.cli import main
from test_lmstudio import split_shard
from test_mmproj import projector_gguf, text_gguf


class CliFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)

        self.cache = root / "hf"
        snapshot = self.cache / "hub" / "models--unsloth--gemma-4-31B-it-GGUF" / "snapshots" / "aa"
        snapshot.mkdir(parents=True)
        (snapshot / "mmproj-F32.gguf").write_bytes(projector_gguf())

        qwen = self.cache / "hub" / "models--unsloth--Qwen3.6-27B-GGUF" / "snapshots" / "bb"
        qwen.mkdir(parents=True)
        (qwen / "mmproj-F32.gguf").write_bytes(
            projector_gguf(projector_type="qwen3vl_merger", proj_dim=5120))

        self.lmstudio = root / "lmstudio"
        self.model_dir = self.lmstudio / "bartowski" / "Novelist-31B-GGUF"
        self.model_dir.mkdir(parents=True)
        (self.model_dir / "Novelist-31B-Q4_K_M.gguf").write_bytes(text_gguf())

        env = {"HF_HOME": str(self.cache), "LMSTUDIO_HOME": str(self.lmstudio)}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def choose(self, *names):
        from lmshf import importing

        def fake_select_many(choices, header, instructions=None, **kwargs):
            picked = [i for i, c in enumerate(choices)
                      if any(name in c.label for name in names)]
            return termui_selection(picked)

        return mock.patch.object(importing, "select_many", side_effect=fake_select_many)

    def run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()


class AttachCommandTest(CliFixture):
    def test_dry_run_changes_nothing(self):
        code, out = self.run_cli("attach-mmproj", "--to", "Novelist-31B-GGUF",
                                 "--from", "gemma-4-31B-it", "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("mmproj-unsloth-gemma-4-31B-it-GGUF-F32.gguf", out)
        self.assertEqual(mmproj.projectors_in(self.model_dir), [])

    def test_attach_then_detach(self):
        code, out = self.run_cli("attach-mmproj", "--to", "Novelist-31B-GGUF",
                                 "--from", "gemma-4-31B-it", "-y")
        self.assertEqual(code, 0)
        self.assertIn("Attached mmproj", out)
        linked = mmproj.projectors_in(self.model_dir)
        self.assertEqual(len(linked), 1)

        code, out = self.run_cli("detach-mmproj", "--from", "Novelist-31B-GGUF")
        self.assertEqual(code, 0)
        self.assertIn("Detached", out)
        self.assertEqual(mmproj.projectors_in(self.model_dir), [])

    def test_incompatible_is_refused_until_forced(self):
        code, out = self.run_cli("attach-mmproj", "--to", "Novelist-31B-GGUF",
                                 "--from", "Qwen3.6-27B", "-y")
        self.assertEqual(code, 1)
        self.assertIn("--force", out)
        self.assertEqual(mmproj.projectors_in(self.model_dir), [])

        code, _ = self.run_cli("attach-mmproj", "--to", "Novelist-31B-GGUF",
                               "--from", "Qwen3.6-27B", "--force", "-y")
        self.assertEqual(code, 0)
        self.assertEqual(len(mmproj.projectors_in(self.model_dir)), 1)

    def test_unknown_target(self):
        code, out = self.run_cli("attach-mmproj", "--to", "nope", "--from", "gemma", "-y")
        self.assertEqual(code, 1)
        self.assertIn("is not in the LM Studio models directory", out)

    def test_ambiguous_target_is_reported(self):
        second = self.lmstudio / "other" / "Novelist-31B-GGUF-clone"
        second.mkdir(parents=True)
        (second / "m.gguf").write_bytes(text_gguf())
        code, out = self.run_cli("attach-mmproj", "--to", "Novelist-31B",
                                 "--from", "gemma", "-y")
        self.assertEqual(code, 1)
        self.assertIn("matches more than one model", out)

    def test_detach_without_an_attachment(self):
        code, out = self.run_cli("detach-mmproj", "--from", "Novelist-31B-GGUF")
        self.assertEqual(code, 1)
        self.assertIn("has not attached", out)


class ReportCommandTest(CliFixture):
    def test_doctor_suggests_a_projector(self):
        code, out = self.run_cli("doctor")
        self.assertEqual(code, 0)
        self.assertIn("bartowski/Novelist-31B-GGUF", out)
        self.assertIn("1 compatible one", out)

    def test_doctor_reports_a_second_projector(self):
        (self.model_dir / "mmproj-a.gguf").write_bytes(projector_gguf())
        (self.model_dir / "mmproj-b.gguf").write_bytes(projector_gguf())
        _, out = self.run_cli("doctor")
        self.assertIn("2 projectors here", out)

    def test_doctor_flags_a_mismatched_projector(self):
        (self.model_dir / "mmproj-small.gguf").write_bytes(projector_gguf(proj_dim=3840))
        _, out = self.run_cli("doctor")
        self.assertIn("size looks wrong", out)

    def test_doctor_flags_an_unreadable_file(self):
        (self.model_dir / "mmproj-broken.gguf").write_bytes(b"truncated")
        _, out = self.run_cli("doctor")
        self.assertIn("cannot be read", out)

    def test_list_shows_the_attachment(self):
        self.run_cli("attach-mmproj", "--to", "Novelist-31B-GGUF",
                     "--from", "gemma-4-31B-it", "-y")
        code, out = self.run_cli("list")
        self.assertEqual(code, 0)
        self.assertIn("attached:", out)
        self.assertIn("unsloth/gemma-4-31B-it-GGUF", out)


if __name__ == "__main__":
    unittest.main()


class ImportCommandTest(CliFixture):
    """The import flow, with the menu answered by a stub."""

    def test_import_links_a_snapshot(self):
        with self.choose("gemma-4-31B-it"):
            code, out = self.run_cli("import")
        self.assertEqual(code, 0)
        self.assertIn("Imported", out)
        imported = self.lmstudio / "unsloth" / "gemma-4-31B-it-GGUF" / "mmproj-F32.gguf"
        self.assertTrue(imported.exists())

    def test_selecting_an_imported_model_removes_it(self):
        with self.choose("gemma-4-31B-it"):
            self.run_cli("import")
        with self.choose("gemma-4-31B-it"):
            code, out = self.run_cli("import")
        self.assertEqual(code, 0)
        self.assertIn("Removed", out)
        self.assertFalse((self.lmstudio / "unsloth" / "gemma-4-31B-it-GGUF").exists())

    def test_cancelling_changes_nothing(self):
        from lmshf import importing
        from lmshf.termui import Selection

        with mock.patch.object(importing, "select_many", return_value=Selection(cancelled=True)):
            code, out = self.run_cli("import")
        self.assertEqual(code, 0)
        self.assertIn("cancelled", out)
        self.assertFalse((self.lmstudio / "unsloth").exists())

    def test_refuses_to_import_into_the_cache_itself(self):
        with mock.patch.dict(os.environ, {"LMSTUDIO_HOME": str(self.cache / "hub")}):
            code, out = self.run_cli("import")
        self.assertIn("Hugging Face", out)


def termui_selection(indices):
    from lmshf.termui import Selection

    return Selection(indices=indices)


class NonAsciiNamesTest(CliFixture):
    """Repository, directory and file names are whatever the publisher chose."""

    KANJI = chr(0x6F22) + chr(0x5B57)  # a directory name in kanji
    HANZI = chr(0x6A21) + chr(0x578B)  # a file name in simplified Chinese

    def setUp(self):
        super().setUp()
        snapshot = (self.cache / "hub" / f"models--{self.KANJI}--Gemma4-GGUF"
                    / "snapshots" / "cc")
        snapshot.mkdir(parents=True)
        self.projector = snapshot / f"{self.HANZI}-mmproj.gguf"
        self.projector.write_bytes(projector_gguf())

        self.kanji_model = self.lmstudio / self.KANJI / f"{self.HANZI}-GGUF"
        self.kanji_model.mkdir(parents=True)
        (self.kanji_model / f"{self.HANZI}-Q4_K_M.gguf").write_bytes(text_gguf())

    def test_a_projector_named_in_hanzi_is_found(self):
        # The file name says nothing about it being a projector; the header does.
        code, out = self.run_cli("attach-mmproj", "--to", self.KANJI,
                                 "--from", self.KANJI, "-y")
        self.assertEqual(code, 0, out)
        linked = mmproj.projectors_in(self.kanji_model)
        self.assertEqual(len(linked), 1)
        self.assertIn("mmproj", linked[0].path.name)
        self.assertIn(self.KANJI, linked[0].path.name)

    def test_the_record_round_trips_through_json(self):
        self.run_cli("attach-mmproj", "--to", self.KANJI, "--from", self.KANJI, "-y")
        record = mmproj.read_sidecar(self.kanji_model)["mmproj"]
        self.assertEqual(record["source_repo"], f"{self.KANJI}/Gemma4-GGUF")
        self.assertEqual(record["source_file"], f"{self.HANZI}-mmproj.gguf")

        code, _ = self.run_cli("detach-mmproj", "--from", self.KANJI)
        self.assertEqual(code, 0)
        self.assertEqual(mmproj.projectors_in(self.kanji_model), [])
        self.assertTrue(self.projector.exists())

    def test_doctor_and_list_report_them(self):
        _, out = self.run_cli("doctor")
        self.assertIn(self.KANJI, out)
        _, out = self.run_cli("list")
        self.assertIn(self.HANZI, out)


class NestedQuantisationRepoTest(CliFixture):
    """Repositories that give each quantisation its own directory."""

    def setUp(self):
        super().setUp()
        snapshot = (self.cache / "hub" / "models--Aratako--Amaterasu-123B-GGUF"
                    / "snapshots" / "dd")
        self.quant_dir = snapshot / "IQ4_XS"
        self.quant_dir.mkdir(parents=True)
        for part in (1, 2):
            (self.quant_dir / f"Amaterasu-123B-IQ4_XS-0000{part}-of-00002.gguf").write_bytes(
                split_shard(part, 2, file_type=30))  # 30 is IQ4_XS
        (snapshot / "README.md").write_text("hi", encoding="utf-8")
        self.target = self.lmstudio / "Aratako" / "Amaterasu-123B-GGUF"

    def test_import_puts_the_files_where_lm_studio_looks(self):
        with self.choose("Amaterasu"):
            code, out = self.run_cli("import")
        self.assertEqual(code, 0, out)
        # Flat, so the model is indexed as Aratako/Amaterasu-123B-GGUF/<file>
        # rather than .../IQ4_XS/<file>, which LM Studio names "iq4_xs".
        self.assertEqual(
            sorted(p.name for p in self.target.glob("*.gguf")),
            ["Amaterasu-123B-IQ4_XS-00001-of-00002.gguf",
             "Amaterasu-123B-IQ4_XS-00002-of-00002.gguf"],
        )
        self.assertFalse((self.target / "IQ4_XS").exists())
        self.assertTrue((self.target / "README.md").exists())

    def test_the_imported_model_reads_back_whole(self):
        with self.choose("Amaterasu"):
            self.run_cli("import")
        code, out = self.run_cli("list")
        self.assertEqual(code, 0)
        self.assertIn("Aratako/Amaterasu-123B-GGUF", out)
        self.assertIn("IQ4_XS", out)

    def test_doctor_flags_a_directory_that_was_not_flattened(self):
        # What a hand-made copy, or an older version of this tool, leaves.
        nested = self.target / "IQ4_XS"
        nested.mkdir(parents=True)
        (nested / "Amaterasu-123B-IQ4_XS-00001-of-00002.gguf").write_bytes(split_shard(1, 2))
        _, out = self.run_cli("doctor")
        self.assertIn("one directory deeper", out)
        self.assertIn("IQ4_XS", out)
