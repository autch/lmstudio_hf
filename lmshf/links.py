import os
import shutil

from . import IS_WINDOWS


def _is_reparse_point(path):
    """True for a Windows symlink or junction (Path.is_symlink misses junctions)."""
    if not IS_WINDOWS:
        return False
    try:
        return bool(getattr(os.lstat(path), "st_reparse_tag", 0))
    except OSError:
        return False


def remove_path(path):
    """Delete a file, directory, symlink or junction without following it."""
    if path.is_symlink() or _is_reparse_point(path):
        # Never rmtree a link: that would delete the content it points at.
        if os.path.isdir(path):
            os.rmdir(path)
        else:
            path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def link_into(src, dst):
    """Link `src` at `dst`, returning the mechanism used.

    Entries inside a Hugging Face snapshot are themselves symlinks into
    ../../blobs, so resolve them and point straight at the blob.

    Nothing here ever copies: the point of the tool is to avoid a second copy
    of the weights on disk. If no kind of link can be made, this raises.
    """
    target = src.resolve()
    target_is_dir = target.is_dir()
    try:
        os.symlink(target, dst, target_is_directory=target_is_dir)
        return "symlink"
    except OSError as exc:
        # 1314 = ERROR_PRIVILEGE_NOT_HELD, 5 = ERROR_ACCESS_DENIED
        if not IS_WINDOWS or getattr(exc, "winerror", None) not in (5, 1314):
            raise

    # Symlinks on Windows need Developer Mode or an elevated shell. Junctions
    # and hard links need neither, so fall back to those.
    if target_is_dir:
        import _winapi

        _winapi.CreateJunction(str(target), str(dst))
        return "junction"
    try:
        os.link(target, dst)
        return "hardlink"
    except OSError as exc:
        raise OSError(
            f"Cannot link {target} -> {dst}. Enable Developer Mode in Windows "
            "Settings so symlinks can be created, or keep the Hugging Face "
            "cache and the LM Studio models directory on the same NTFS volume "
            f"so hard links can be used. Original error: {exc}"
        ) from exc
