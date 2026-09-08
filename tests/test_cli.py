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
        self.assertIn("見つかりません", out)

    def test_ambiguous_target_is_reported(self):
        second = self.lmstudio / "other" / "Novelist-31B-GGUF-clone"
        second.mkdir(parents=True)
        (second / "m.gguf").write_bytes(text_gguf())
        code, out = self.run_cli("attach-mmproj", "--to", "Novelist-31B",
                                 "--from", "gemma", "-y")
        self.assertEqual(code, 1)
        self.assertIn("複数のモデルに一致", out)

    def test_detach_without_an_attachment(self):
        code, out = self.run_cli("detach-mmproj", "--from", "Novelist-31B-GGUF")
        self.assertEqual(code, 1)
        self.assertIn("ありません", out)


class ReportCommandTest(CliFixture):
    def test_doctor_suggests_a_projector(self):
        code, out = self.run_cli("doctor")
        self.assertEqual(code, 0)
        self.assertIn("bartowski/Novelist-31B-GGUF", out)
        self.assertIn("互換候補が 1 件", out)

    def test_doctor_reports_a_second_projector(self):
        (self.model_dir / "mmproj-a.gguf").write_bytes(projector_gguf())
        (self.model_dir / "mmproj-b.gguf").write_bytes(projector_gguf())
        _, out = self.run_cli("doctor")
        self.assertIn("projector が 2 つあります", out)

    def test_doctor_flags_a_mismatched_projector(self):
        (self.model_dir / "mmproj-small.gguf").write_bytes(projector_gguf(proj_dim=3840))
        _, out = self.run_cli("doctor")
        self.assertIn("次元不一致", out)

    def test_doctor_flags_an_unreadable_file(self):
        (self.model_dir / "mmproj-broken.gguf").write_bytes(b"truncated")
        _, out = self.run_cli("doctor")
        self.assertIn("読めません", out)

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

    def choose(self, *names):
        from lmshf import importing

        def fake_select_many(choices, header, instructions=None, **kwargs):
            picked = [i for i, c in enumerate(choices)
                      if any(name in c.label for name in names)]
            return termui_selection(picked)

        return mock.patch.object(importing, "select_many", side_effect=fake_select_many)

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
