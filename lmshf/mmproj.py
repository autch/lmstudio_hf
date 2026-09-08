"""Putting a projector from one repository next to a model from another.

LM Studio pairs a text model with its vision/audio projector by directory:
the mmproj file has to sit beside the text GGUF, with "mmproj" in its name.
Quantised derivatives are routinely published without one, so the projector
has to come from the repository they were derived from.

Nothing is copied. The projector is linked in, exactly like an import, and a
sidecar records where it came from so it can be detached again.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from . import gguf
from .links import link_into, remove_path

SIDECAR = ".lmstudio_hf.json"
SIDECAR_VERSION = 1

OK = "ok"
SUSPECT = "suspect"
INCOMPATIBLE = "incompatible"

# Families where the text architecture and the projector type are known to
# share a prefix. The list is deliberately short: an unknown family falls
# back to the dimension check, which is the safe direction to be wrong in.
# (Pixtral projectors sit on a "llama" text model, and MiniCPM-V ones on
# "qwen2", so neither can be judged this way.)
KNOWN_FAMILIES = frozenset({"gemma", "qwen", "llama"})

_LEADING_LETTERS = re.compile(r"[a-z]+")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def family(token):
    """The family prefix of an architecture or projector type, if known.

    gemma4 and gemma4v both give "gemma"; qwen35 and qwen3vl_merger both give
    "qwen". Anything not in KNOWN_FAMILIES gives "" (do not judge).
    """
    if not token:
        return ""
    match = _LEADING_LETTERS.match(token.lower())
    name = match.group(0) if match else ""
    return name if name in KNOWN_FAMILIES else ""


@dataclass
class Compat:
    verdict: str
    reason: str = ""

    @property
    def blocked(self):
        return self.verdict == INCOMPATIBLE

    @property
    def marker(self):
        return {OK: "[OK]", SUSPECT: "[? ]", INCOMPATIBLE: "[!!]"}[self.verdict]


def check(text, projector):
    """Judge whether `projector` can serve `text`, from their headers alone.

    This says whether the pair should load, not whether it should work well:
    a derivative that retrained its projector cannot be told apart from one
    that froze it, and neither can a same-family version mismatch.
    """
    if not projector.is_projector:
        return Compat(INCOMPATIBLE, "projector ではありません")
    if text is None or not text.readable:
        return Compat(SUSPECT, "本体の GGUF を読めないため判定できません")

    text_family = family(text.arch)
    proj_family = family(projector.projector_type)
    if text_family and proj_family and text_family != proj_family:
        return Compat(
            INCOMPATIBLE,
            f"アーキ不一致: 本体 {text.arch} に対し projector は {projector.projector_type}",
        )
    if text.n_embd and projector.proj_dim:
        if text.n_embd != projector.proj_dim:
            return Compat(
                SUSPECT,
                f"次元不一致の疑い: proj_dim={projector.proj_dim} != n_embd={text.n_embd}",
            )
        return Compat(OK)
    return Compat(SUSPECT, "次元を確認できないため判定できません")


def link_name(repo, projector):
    """The name to give the linked projector inside the model directory.

    LM Studio keys off the "mmproj" substring, and the source repository is
    worth keeping visible, so the original name is not reused as is.
    """
    stem = _UNSAFE.sub("-", repo.replace("/", "-")).strip("-")
    if len(stem) > 64:
        stem = stem[:64].rstrip("-")
    return f"mmproj-{stem}-{projector.quant}.gguf"


def read_sidecar(model_dir):
    try:
        with open(model_dir / SIDECAR, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_sidecar(model_dir, data):
    with open(model_dir / SIDECAR, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def attached_link(model_dir):
    """The projector this tool attached, if it is still there."""
    record = read_sidecar(model_dir).get("mmproj") or {}
    name = record.get("link")
    if not name:
        return None
    path = model_dir / name
    return path if path.exists() or path.is_symlink() else None


def projectors_in(model_dir):
    """Every GGUF in `model_dir` whose header says it is a projector."""
    found = []
    try:
        entries = sorted(model_dir.glob("*.gguf"))
    except OSError:
        return found
    for path in entries:
        info = gguf.inspect(path)
        if info.is_projector:
            found.append(info)
    return found


def disable(path, ours=False):
    """Move an existing projector aside, or drop it if this tool linked it.

    `ours` comes from the sidecar rather than from the file: a hard link,
    which is what Windows falls back to, is indistinguishable from a plain
    file, so the record is the only reliable way to know we created it.
    """
    if ours or path.is_symlink() or not path.is_file():
        remove_path(path)
        return None
    backup = path.with_name(path.name + ".disabled")
    counter = 1
    while backup.exists():
        counter += 1
        backup = path.with_name(f"{path.name}.disabled{counter}")
    path.rename(backup)
    return backup


def attach(model_dir, projector, repo, compat=None):
    """Link `projector` into `model_dir` and record where it came from.

    Returns (link name, link method, path the previous projector was moved to).
    Any projector already in the directory is cleared first: two of them in
    one directory leaves it undefined which one LM Studio picks up.
    """
    record = read_sidecar(model_dir)
    previous_name = (record.get("mmproj") or {}).get("link")

    moved_aside = None
    for existing in projectors_in(model_dir):
        moved_aside = disable(existing.path, ours=existing.path.name == previous_name)

    name = link_name(repo, projector)
    destination = model_dir / name
    if destination.exists() or destination.is_symlink():
        remove_path(destination)
    method = link_into(projector.path, destination)

    record["version"] = SIDECAR_VERSION
    record["mmproj"] = {
        "link": name,
        "source_repo": repo,
        "source_file": projector.path.name,
        "source_blob": str(projector.path.resolve()),
        "projector_type": projector.projector_type,
        "proj_dim": projector.proj_dim,
        "modalities": list(projector.modalities),
        "verdict": compat.verdict if compat else None,
    }
    write_sidecar(model_dir, record)
    return name, method, moved_aside


def detach(model_dir):
    """Remove the projector this tool attached. Returns its name, or None."""
    record = read_sidecar(model_dir)
    entry = record.get("mmproj") or {}
    name = entry.get("link")
    if not name:
        return None

    path = model_dir / name
    if path.exists() or path.is_symlink():
        remove_path(path)
    record.pop("mmproj", None)
    if set(record) <= {"version"}:
        remove_path(model_dir / SIDECAR)
    else:
        write_sidecar(model_dir, record)
    return name


@dataclass
class Candidate:
    """A projector found in the Hugging Face cache."""

    repo: str
    info: object  # gguf.GGUFInfo

    @property
    def label(self):
        return f"{self.repo} :: {self.info.path.name}"


def available(cache_dir):
    """Every projector in the cache, whatever its file happens to be called."""
    from .hfcache import gguf_files

    found = []
    for repo, path in gguf_files(cache_dir):
        info = gguf.inspect(path)
        if info.is_projector:
            found.append(Candidate(repo, info))
    return found
