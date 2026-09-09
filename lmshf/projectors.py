"""The projector flows: attach, detach, doctor and list."""

from __future__ import annotations

from . import lmstudio, mmproj
from .gguf import human_size
from .paths import hf_cache_dir, lm_studio_models_dir
from .termui import GLYPH_ARROWS, Choice, select_one


def _describe_model(model):
    """The dimmed second line for one LM Studio model row."""
    text = model.text
    if text is None:
        return "projector only, no text model" if model.projectors else "no GGUF here"
    arch = text.arch or "?"
    embd = f"n_embd={text.n_embd}" if text.n_embd else "n_embd=?"
    if model.projectors:
        projector = f"mmproj: {model.projectors[0].path.name}"
        if len(model.projectors) > 1:
            projector += f" (and {len(model.projectors) - 1} more)"
    else:
        projector = "mmproj: none"
    return f"{arch}  {embd}  {text.quant}  {human_size(model.size)}  {projector}"


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
        print(f"{wanted} is not in the LM Studio models directory.")
    else:
        print(f"{wanted} matches more than one model:")
        for m in matches:
            print(f"  {m.name}")
    return None


def _resolve_projector(candidates, repo, file_name):
    matches = [c for c in candidates if repo.lower() in c.repo.lower()]
    if file_name:
        matches = [c for c in matches if c.info.path.name == file_name]
    if not matches:
        print(f"No projector found in {repo}.")
        return None
    if len(matches) > 1:
        print(f"{repo} holds more than one projector; pick one with --file:")
        for c in matches:
            print(f"  {c.info.path.name}  ({c.info.projector_type}, {c.info.quant})")
        return None
    return matches[0]


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
            print("No model here can take a projector."
                  " Showing all of them; doctor explains what each one needs.")
            show_all = True
            rows = models
        choices = [
            Choice(label=m.name, detail=_describe_model(m), marked=m.has_projector)
            for m in rows
        ]
        shown = "all models" if show_all else "models missing a projector the cache can supply"
        result = select_one(
            choices,
            header="lm-studio - attach mmproj  [1/2] the model to attach to",
            instructions=f"{GLYPH_ARROWS} to move, ENTER to choose, a to switch view, "
                         "Ctrl+C to quit",
            footer=f"[a] showing: {shown} ({len(rows)} of {len(models)})",
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
        header = ("lm-studio - attach mmproj  [2/2] the projector to use\n"
                  f"  attaching to: {target.name}")
        footer = ("incompatible projectors can be selected (same as --force)" if allow_all
                  else "[f] allow incompatible projectors to be selected")
        result = select_one(
            choices,
            header=header,
            instructions=f"{GLYPH_ARROWS} to move, ENTER to choose, f to allow incompatible, "
                         "Ctrl+C to quit",
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
    print("\nAbout to do this:\n")
    print(f"  model directory   {target.path}")
    print(f"  link to create    {name}")
    print(f"  pointing at       {info.path.resolve()}")
    print(f"                    ({human_size(info.size)}, {info.projector_type or '?'},"
          f" {'+'.join(info.modalities) or '?'})")
    if compat.reason:
        print(f"  verdict           {compat.marker} {compat.reason}")
    for existing in target.projectors:
        print(f"  in the way        {existing.path.name} will be moved aside or dropped")
    print("\n  Nothing is copied.")
    print("  Eject the model in LM Studio and load it again afterwards.")


def attach_command(args):
    cache_dir = hf_cache_dir()
    lm_studio_dir = lm_studio_models_dir()

    models = lmstudio.scan(lm_studio_dir)
    if not models:
        print(f"No models in {lm_studio_dir}.")
        return 1
    candidates = mmproj.available(cache_dir)
    if not candidates:
        print(f"No projector (mmproj) found in {cache_dir}.")
        return 1

    if args.to:
        target = _resolve_model(models, args.to)
        if target is None:
            return 1
    else:
        target = _choose_target(models, candidates)
        if target is None:
            print("\nCancelled.")
            return 0

    if args.source:
        candidate = _resolve_projector(candidates, args.source, args.file)
        if candidate is None:
            return 1
        compat = mmproj.check(target.text, candidate.info)
    else:
        candidate, compat = _choose_projector(candidates, target)
        if candidate is None:
            print("\nCancelled.")
            return 0

    if compat.blocked and not args.force:
        print("\nThese two are not compatible.")
        print(f"  {compat.reason}")
        print("  Pass --force if you meant it.")
        return 1

    name = mmproj.link_name(candidate.repo, candidate.info)
    _print_plan(target, candidate, compat, name)
    if args.dry_run:
        return 0
    if not args.yes:
        try:
            answer = input("\nGo ahead? [y/N]: ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    try:
        name, method, moved = mmproj.attach(target.path, candidate.info, candidate.repo, compat)
    except OSError as exc:
        print(f"\nFailed: {exc}")
        return 1

    for backup in moved:
        print(f"\nMoved the projector that was there to {backup.name}.")
    print(f"\nAttached mmproj to {target.name} ({method}ed)")
    print(f"  -> {name}")
    print("Eject the model in LM Studio and load it again.")
    return 0


def detach_command(args):
    models = lmstudio.scan(lm_studio_models_dir())
    target = _resolve_model(models, args.source)
    if target is None:
        return 1

    record = target.attached
    if not record:
        print(f"This tool has not attached a projector to {target.name}.")
        for existing in target.projectors:
            print(f"  {existing.path.name} was put there by something else.")
        return 1

    try:
        name = mmproj.detach(target.path)
    except OSError as exc:
        print(f"Failed: {exc}")
        return 1
    print(f"Detached {name}")
    print(f"  (the link came from {record.get('source_repo')})")
    return 0


def _model_issues(model, candidates):
    """Everything worth telling the user about one model directory."""
    issues = []
    for info in model.projectors + ([model.text] if model.text else []) + model.extras:
        if not info.readable:
            issues.append(f"{info.path.name} cannot be read: {info.error}")

    if model.nested:
        where = ", ".join(sorted({p.parent.name for p in model.nested}))
        issues.append(
            f"GGUF files sit one directory deeper than LM Studio expects ({where}); "
            f"LM Studio indexes them under the directory name. Re-import to flatten"
        )

    if len(model.projectors) > 1:
        names = ", ".join(p.path.name for p in model.projectors)
        issues.append(f"{len(model.projectors)} projectors here: {names}")

    if model.text is None:
        if model.projectors:
            issues.append("no text model (a projector-only folder)")
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
            plural = "" if len(fits) == 1 else "s"
            issues.append(
                f"no projector; the cache holds {len(fits)} compatible one{plural} "
                f"(attach-mmproj can link one in)"
            )
    return issues


def doctor_command(args):
    lm_studio_dir = lm_studio_models_dir()
    print(f"Checking {lm_studio_dir} ...")
    models = lmstudio.scan(lm_studio_dir)
    candidates = mmproj.available(hf_cache_dir())
    print(f"{len(models)} models, {len(candidates)} projectors in the cache\n")

    clean = 0
    for model in models:
        issues = _model_issues(model, candidates)
        if not issues:
            clean += 1
            continue
        print(f"[!] {model.name}")
        for issue in issues:
            print(f"      {issue}")
    print(f"\n[ok] the other {clean} models look fine")
    return 0


def list_command(args):
    for model in lmstudio.scan(lm_studio_models_dir()):
        print(model.name)
        print(f"    {_describe_model(model)}")
        record = model.attached
        if record:
            print(f"    attached: {record.get('link')} <- {record.get('source_repo')}")
    return 0
