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
import unicodedata

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
# Characters no filesystem will take in a name. Everything else is kept:
# a repository name in kanji or hanzi should stay readable in the link.
_UNSAFE_CHARS = frozenset('<>:"/' + chr(92) + "|?*")
# NTFS allows 255 UTF-16 units and ext4 255 bytes per component, and the
# quantisation label and suffix still have to fit after this.
_MAX_STEM_BYTES = 120


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
        return Compat(INCOMPATIBLE, "not a projector")
    if text is None or not text.readable:
        return Compat(SUSPECT, "the model's own GGUF could not be read")

    text_family = family(text.arch)
    proj_family = family(projector.projector_type)
    if text_family and proj_family and text_family != proj_family:
        return Compat(
            INCOMPATIBLE,
            f"architecture mismatch: the model is {text.arch}, "
            f"the projector is {projector.projector_type}",
        )
    if text.n_embd and projector.proj_dim:
        if text.n_embd != projector.proj_dim:
            return Compat(
                SUSPECT,
                f"size looks wrong: proj_dim={projector.proj_dim} != n_embd={text.n_embd}",
            )
        return Compat(OK)
    return Compat(SUSPECT, "the sizes to compare could not be read")


def _sanitise(text):
    """Make `text` usable as one component of a file name."""
    # Whitespace is legal in a file name but tedious to type at a shell, and
    # these names get passed to --file.
    swapped = "".join(
        "-" if ch in _UNSAFE_CHARS or ch.isspace() or ord(ch) < 32 else ch for ch in text
    )
    # Windows also refuses a name that ends in a dot or a space.
    return re.sub("-{2,}", "-", swapped).strip("-. ")


def _clip(text, limit):
    """Cut `text` so its UTF-8 form fits `limit` bytes, on a character boundary."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", "ignore")


def link_name(repo, projector):
    """The name to give the linked projector inside the model directory.

    LM Studio keys off the "mmproj" substring, and the source repository is
    worth keeping visible, so the original name is not reused as is.
    """
    stem = _sanitise(_clip(_sanitise(repo.replace("/", "-")), _MAX_STEM_BYTES))
    return f"mmproj-{stem or 'source'}-{projector.quant}.gguf"


def same_name(one, other):
    """Compare two file names the way a filesystem might not.

    macOS stores names decomposed, so a name written as NFC comes back as
    NFD and a plain == says two spellings of the same Japanese file name are
    different files. That would have the tool treat its own link as a
    stranger's and rename it aside instead of replacing it.
    """
    return (unicodedata.normalize("NFC", one or "")
            == unicodedata.normalize("NFC", other or ""))


def find_named(model_dir, name):
    """The file called `name`, whichever normalisation the filesystem used."""
    direct = model_dir / name
    if direct.exists() or direct.is_symlink():
        return direct
    try:
        entries = list(model_dir.iterdir())
    except OSError:
        return None
    for entry in entries:
        if same_name(entry.name, name):
            return entry
    return None


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
    return find_named(model_dir, name)


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

    Returns (link name, link method, paths any previous projectors were moved
    to). Every projector already in the directory is cleared first: two of
    them leaves it undefined which one LM Studio picks up.
    """
    record = read_sidecar(model_dir)
    previous_name = (record.get("mmproj") or {}).get("link")

    moved_aside = []
    for existing in projectors_in(model_dir):
        backup = disable(existing.path, ours=same_name(existing.path.name, previous_name))
        if backup is not None:
            moved_aside.append(backup)

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

    path = find_named(model_dir, name)
    if path is not None:
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
