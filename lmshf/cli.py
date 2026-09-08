"""Argument parsing and command dispatch."""

from __future__ import annotations

import argparse

from .importing import manage_models
from .projectors import attach_command, detach_command, doctor_command, list_command


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
