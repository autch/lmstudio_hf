"""Command line entry points."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

from . import gguf, lmstudio, mmproj
from .gguf import human_size
from .hfcache import find_models
from .links import link_into, remove_path
from .lmstudio import existing_models
from .paths import hf_cache_dir, lm_studio_models_dir
from .termui import GLYPH_ARROWS, Choice, select_many, select_one


@dataclass
class ImportCandidate:
    """A cached Hugging Face model and where it would live in LM Studio."""

    model_type: str
    name: str
    imported: bool
    snapshot: Path
    target: Path


def _candidates(found_models, lm_studio_dir):
    existing = existing_models(lm_studio_dir)
    candidates = []
    for model_type, name, snapshot_path in sorted(found_models):
        # Check for the exact model path as it would be created
        target = existing.get(name)
        imported = target is not None
        if target is None:
            target = lm_studio_dir.joinpath(*name.split("/"))
        candidates.append(ImportCandidate(model_type, name, imported, snapshot_path, target))
    return candidates


def _import_model(candidate):
    """Link every file of the snapshot into the LM Studio models directory."""
    candidate.target.mkdir(parents=True, exist_ok=True)
    method = "symlink"
    try:
        for item in candidate.snapshot.iterdir():
            method = link_into(item, candidate.target / item.name)
    except OSError as exc:
        print(f"Failed to import {candidate.name}: {exc}")
        # Don't leave a half-linked model behind for LM Studio to find.
        try:
            remove_path(candidate.target)
        except OSError:
            pass
        return
    print(f"Imported {candidate.name} ({method}ed files)")


def manage_models():
    "Import models from the Hugging Face cache."
    cache_dir = hf_cache_dir()
    lm_studio_dir = lm_studio_models_dir()

    # Importing into the Hugging Face cache itself would have the tool link
    # models on top of their own source files.
    try:
        overlaps = lm_studio_dir.resolve().is_relative_to(cache_dir.resolve())
    except (OSError, ValueError):
        overlaps = False
    if overlaps:
        print(
            f"The LM Studio models directory ({lm_studio_dir}) is inside the "
            f"Hugging Face cache ({cache_dir}).\n"
            "Point LM Studio's models directory somewhere else, or set "
            "LMSTUDIO_HOME, and run again."
        )
        return

    found_models = find_models(cache_dir)
    if not found_models:
        print("No models found in Hugging Face cache")
        return

    candidates = _candidates(found_models, lm_studio_dir)
    choices = [
        Choice(
            label=f"({c.model_type}) {c.name}" + (" (already imported)" if c.imported else ""),
            marked=c.imported,
            value=c,
        )
        for c in candidates
    ]

    result = select_many(
        choices,
        header="lm-studio - Hugging Face Model Manager",
        instructions=None,
    )
    if result.cancelled:
        print("\nImport is cancelled. Do nothing.")
        return
    print("\nImporting models...\n")

    for i in result.indices:
        candidate = candidates[i]
        if candidate.imported:
            # Selecting an already imported model removes it again.
            try:
                remove_path(candidate.target)
            except OSError as exc:
                print(f"Failed to remove {candidate.name}: {exc}")
                continue
            print(f"Removed {candidate.name}")
        else:
            _import_model(candidate)


# --- shared helpers -------------------------------------------------------


def _describe_model(model):
    """The dimmed second line for one LM Studio model row."""
    text = model.text
    if text is None:
        return "テキストモデルなし (projector のみ)"
    arch = text.arch or "?"
    embd = f"n_embd={text.n_embd}" if text.n_embd else "n_embd=?"
    if model.projectors:
        projector = f"mmproj: {model.projectors[0].path.name}"
        if len(model.projectors) > 1:
            projector += f" (他 {len(model.projectors) - 1} 件)"
    else:
        projector = "mmproj: なし"
    return f"{arch}  {embd}  {text.quant}  {human_size(text.size)}  {projector}"


def _describe_projector(candidate, compat=None):
    info = candidate.info
    kinds = "+".join(info.modalities) or "?"
    line = (f"{info.projector_type or '?'}  {info.quant}  "
            f"{human_size(info.size)}  proj_dim={info.proj_dim}  {kinds}")
    if compat is not None and compat.reason:
        # The menu indents every detail line for us.
        line += f"\n{compat.reason}"
    return line


def _resolve_model(models, wanted):
    """Find one model by "org/name", or by any unambiguous part of it."""
    exact = [m for m in models if m.name == wanted]
    if exact:
        return exact[0]
    matches = [m for m in models if wanted.lower() in m.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        print(f"{wanted} は LM Studio のモデルディレクトリに見つかりません。")
    else:
        print(f"{wanted} は複数のモデルに一致します:")
        for m in matches:
            print(f"  {m.name}")
    return None


def _resolve_projector(candidates, repo, file_name):
    matches = [c for c in candidates if repo.lower() in c.repo.lower()]
    if file_name:
        matches = [c for c in matches if c.info.path.name == file_name]
    if not matches:
        print(f"{repo} に projector が見つかりません。")
        return None
    if len(matches) > 1:
        print(f"{repo} には projector が複数あります。--file で選んでください:")
        for c in matches:
            print(f"  {c.info.path.name}  ({c.info.projector_type}, {c.info.quant})")
        return None
    return matches[0]


# --- attach-mmproj --------------------------------------------------------


def _fits(model, candidates):
    """True when this model has no projector but the cache holds one for it."""
    if model.has_projector or model.text is None:
        return False
    return any(mmproj.check(model.text, c.info).verdict == mmproj.OK for c in candidates)


def _choose_target(models, candidates):
    show_all = False
    cursor = 0
    while True:
        rows = models if show_all else [m for m in models if _fits(m, candidates)]
        if not rows:
            print("projector を追加できるモデルがありません。"
                  " a で全件表示するか、doctor で状態を確認してください。")
            show_all = True
            rows = models
        choices = [
            Choice(label=m.name, detail=_describe_model(m), marked=m.has_projector)
            for m in rows
        ]
        shown = "全モデル" if show_all else "projector 未配置かつ互換候補ありのみ"
        result = select_one(
            choices,
            header="lm-studio - attach mmproj  [1/2] 貼り付け先のモデル",
            instructions=f"{GLYPH_ARROWS} で移動, ENTER で決定, a で表示切替, Ctrl+C で中止",
            footer=f"[a] 表示中: {shown} ({len(models)} 件中 {len(rows)} 件)",
            cursor=min(cursor, max(0, len(rows) - 1)),
            extra_keys="a",
        )
        if result.key == "a":
            show_all = not show_all
            cursor = 0
            continue
        if result.cancelled:
            return None
        return rows[result.index]


def _choose_projector(candidates, target):
    checked = [(c, mmproj.check(target.text, c.info)) for c in candidates]
    # Usable ones first, then the merely doubtful, then what cannot work.
    order = {mmproj.OK: 0, mmproj.SUSPECT: 1, mmproj.INCOMPATIBLE: 2}
    checked.sort(key=lambda pair: (order[pair[1].verdict], pair[0].repo))

    allow_all = False
    cursor = 0
    while True:
        choices = [
            Choice(
                label=f"{compat.marker}  {c.label}",
                detail=_describe_projector(c, compat),
                marked=compat.blocked,
                selectable=allow_all or not compat.blocked,
                value=c,
            )
            for c, compat in checked
        ]
        header = ("lm-studio - attach mmproj  [2/2] 使用する projector\n"
                  f"  貼り付け先: {target.name}")
        footer = "[f] 非互換も選択可能にする" if not allow_all else "非互換も選択できます (--force 相当)"
        result = select_one(
            choices,
            header=header,
            instructions=f"{GLYPH_ARROWS} で移動, ENTER で決定, f で非互換も選択可, Ctrl+C で中止",
            footer=footer,
            cursor=cursor,
            extra_keys="f",
        )
        if result.key == "f":
            allow_all = True
            cursor = result.cursor
            continue
        if result.cancelled:
            return None, None
        return checked[result.index]


def _print_plan(target, candidate, compat, name):
    info = candidate.info
    print("\n以下を実行します:\n")
    print(f"  貼り付け先        {target.path}")
    print(f"  作成するリンク名  {name}")
    print(f"  リンク先          {info.path.resolve()}")
    print(f"                    ({human_size(info.size)}, {info.projector_type or '?'},"
          f" {'+'.join(info.modalities) or '?'})")
    if compat.reason:
        print(f"  判定              {compat.marker} {compat.reason}")
    for existing in target.projectors:
        print(f"  既存の projector  {existing.path.name} を退避または削除します")
    print("\n  ※ コピーは発生しません。")
    print("  ※ LM Studio でモデルを一度 Eject して読み込み直してください。")


def attach_command(args):
    cache_dir = hf_cache_dir()
    lm_studio_dir = lm_studio_models_dir()

    models = lmstudio.scan(lm_studio_dir)
    if not models:
        print(f"{lm_studio_dir} にモデルがありません。")
        return 1
    candidates = mmproj.available(cache_dir)
    if not candidates:
        print(f"{cache_dir} に projector (mmproj) が見つかりません。")
        return 1

    if args.to:
        target = _resolve_model(models, args.to)
        if target is None:
            return 1
    else:
        target = _choose_target(models, candidates)
        if target is None:
            print("\n中止しました。")
            return 0

    if args.source:
        candidate = _resolve_projector(candidates, args.source, args.file)
        if candidate is None:
            return 1
        compat = mmproj.check(target.text, candidate.info)
    else:
        candidate, compat = _choose_projector(candidates, target)
        if candidate is None:
            print("\n中止しました。")
            return 0

    if compat.blocked and not args.force:
        print("\n互換性がありません。")
        print(f"  {compat.reason}")
        print("  意図的な場合は --force を付けてください。")
        return 1

    name = mmproj.link_name(candidate.repo, candidate.info)
    _print_plan(target, candidate, compat, name)
    if args.dry_run:
        return 0
    if not args.yes:
        try:
            answer = input("\n続行しますか? [y/N]: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("中止しました。")
            return 0

    try:
        name, method, moved = mmproj.attach(target.path, candidate.info, candidate.repo, compat)
    except OSError as exc:
        print(f"\n失敗しました: {exc}")
        return 1

    if moved:
        print(f"\n既存の projector を {moved.name} に退避しました。")
    print(f"\nAttached mmproj to {target.name} ({method}ed)")
    print(f"  -> {name}")
    print("LM Studio でモデルを Eject して読み込み直してください。")
    return 0


# --- detach-mmproj, doctor, list ------------------------------------------


def detach_command(args):
    models = lmstudio.scan(lm_studio_models_dir())
    target = _resolve_model(models, args.source)
    if target is None:
        return 1

    record = target.attached
    if not record:
        print(f"{target.name} にこのツールが追加した projector はありません。")
        for existing in target.projectors:
            print(f"  {existing.path.name} はこのツール以外が置いたものです。")
        return 1

    try:
        name = mmproj.detach(target.path)
    except OSError as exc:
        print(f"失敗しました: {exc}")
        return 1
    print(f"Detached {name}")
    print(f"  ({record.get('source_repo')} 由来のリンクを削除しました)")
    return 0


def _model_issues(model, candidates):
    """Everything worth telling the user about one model directory."""
    issues = []
    for info in model.projectors + ([model.text] if model.text else []) + model.extras:
        if not info.readable:
            issues.append(f"{info.path.name} を読めません: {info.error}")

    if len(model.projectors) > 1:
        names = ", ".join(p.path.name for p in model.projectors)
        issues.append(f"projector が {len(model.projectors)} つあります: {names}")

    if model.text is None:
        if model.projectors:
            issues.append("テキストモデルがありません (projector のみのフォルダ)")
        return issues

    for projector in model.projectors:
        if not projector.readable:
            continue
        compat = mmproj.check(model.text, projector)
        if compat.verdict != mmproj.OK:
            issues.append(f"{projector.path.name}: {compat.marker} {compat.reason}")

    if not model.projectors:
        fits = [c for c in candidates if mmproj.check(model.text, c.info).verdict == mmproj.OK]
        if fits:
            issues.append(
                f"projector がありません。キャッシュに互換候補が {len(fits)} 件あります "
                f"(attach-mmproj で追加できます)"
            )
    return issues


def doctor_command(args):
    lm_studio_dir = lm_studio_models_dir()
    print(f"{lm_studio_dir} を検査中...")
    models = lmstudio.scan(lm_studio_dir)
    candidates = mmproj.available(hf_cache_dir())
    print(f"{len(models)} モデル, キャッシュ内の projector {len(candidates)} 件\n")

    clean = 0
    for model in models:
        issues = _model_issues(model, candidates)
        if not issues:
            clean += 1
            continue
        print(f"[!] {model.name}")
        for issue in issues:
            print(f"      {issue}")
    print(f"\n[ok] 残り {clean} モデルは問題なし")
    return 0


def list_command(args):
    for model in lmstudio.scan(lm_studio_models_dir()):
        print(model.name)
        print(f"    {_describe_model(model)}")
        record = model.attached
        if record:
            print(f"    attached: {record.get('link')} <- {record.get('source_repo')}")
    return 0


# --- entry point ----------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        prog="lmstudio_hf",
        description="Hugging Face のキャッシュから LM Studio へモデルをリンクする",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("import", help="キャッシュのモデルを選んでインポート/削除する (既定)")

    attach = sub.add_parser("attach-mmproj", help="別リポジトリの mmproj をモデルに追加する")
    attach.add_argument("--to", metavar="MODEL", help="貼り付け先のモデル (org/name)")
    attach.add_argument("--from", dest="source", metavar="REPO", help="projector のあるリポジトリ")
    attach.add_argument("--file", metavar="NAME", help="リポジトリ内に projector が複数ある場合")
    attach.add_argument("--force", action="store_true", help="非互換でも実行する")
    attach.add_argument("--dry-run", action="store_true", help="実行内容を表示するだけ")
    attach.add_argument("-y", "--yes", action="store_true", help="確認を省略する")

    detach = sub.add_parser("detach-mmproj", help="このツールが追加した mmproj を外す")
    detach.add_argument("--from", dest="source", metavar="MODEL", required=True)

    sub.add_parser("doctor", help="LM Studio 側の projector の状態を検査する")
    sub.add_parser("list", help="LM Studio 側のモデルを一覧表示する")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in (None, "import"):
        return manage_models() or 0
    return {
        "attach-mmproj": attach_command,
        "detach-mmproj": detach_command,
        "doctor": doctor_command,
        "list": list_command,
    }[args.command](args)
