"""Argument parsing and command dispatch."""

from __future__ import annotations

import argparse

from .importing import manage_models
from .projectors import attach_command, detach_command, doctor_command, list_command
from .termui import configure_output


def build_parser():
    parser = argparse.ArgumentParser(
        prog="lmstudio_hf",
        description="Link models from the Hugging Face cache into LM Studio",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("import", help="import or remove cached models (default)")

    attach = sub.add_parser("attach-mmproj",
                            help="attach a projector from another repository to a model")
    attach.add_argument("--to", metavar="MODEL", help="the model to attach it to (org/name)")
    attach.add_argument("--from", dest="source", metavar="REPO",
                        help="the repository holding the projector")
    attach.add_argument("--file", metavar="NAME",
                        help="which projector, when the repository holds more than one")
    attach.add_argument("--force", action="store_true",
                        help="attach even when the pair is incompatible")
    attach.add_argument("--dry-run", action="store_true", help="show what would happen and stop")
    attach.add_argument("-y", "--yes", action="store_true", help="skip the confirmation")

    detach = sub.add_parser("detach-mmproj", help="remove a projector this tool attached")
    detach.add_argument("--from", dest="source", metavar="MODEL", required=True)

    sub.add_parser("doctor", help="report projector problems in the models directory")
    sub.add_parser("list", help="show what LM Studio has")
    return parser


def main(argv=None):
    configure_output()
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
