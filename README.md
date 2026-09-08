# LM Studio - Hugging Face Model Manager

A command-line utility to manage models between your Hugging Face cache and LM Studio. This tool makes it easy to import and manage any models you've downloaded from Hugging Face into LM Studio.

## Features

- Interactive model selection interface with keyboard navigation
- Automatic detection of all models in your Hugging Face cache
- Smart handling of model imports via links, never copies
- Support for model removal and re-import
- Terminal-based UI with scrolling for large model lists
- Reads GGUF headers, so quantisation, architecture and projector details are
  shown even for repositories without a `config.json`
- **Attaches a vision/audio projector (`mmproj`) from one repository to a model
  from another**, which is what quantised derivatives usually need
- Checks projector compatibility before linking anything
- `doctor` reports models whose projector is missing, duplicated or mismatched

## Prerequisites

- Python 3.9 or newer
- LM Studio installed
- Hugging Face models downloaded locally

No third-party packages are needed; the tool is standard library only. It works
on macOS, Linux and Windows. See [Windows notes](#windows-notes) for the extra
details that apply there.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/autch/lmstudio_hf.git
cd lmstudio_hf
```

## Usage

```bash
python lmstudio_hf.py                 # import models (the default)
python lmstudio_hf.py attach-mmproj   # attach a projector to a model
python lmstudio_hf.py detach-mmproj --from <model>
python lmstudio_hf.py doctor          # report projector problems
python lmstudio_hf.py list            # show what LM Studio has
```

### Importing models

Running the tool with no arguments lists every model in the Hugging Face cache.
Selecting a model links its snapshot into the LM Studio models directory;
selecting one that is already imported removes it again.

### Attaching a projector from another repository

LM Studio pairs a text model with its projector by directory: the `mmproj` file
has to sit next to the text GGUF, and its name has to contain `mmproj`. People
publishing quantised derivatives frequently leave the projector out, so it has
to be fetched from the repository the derivative came from — at which point it
lands in a different folder and LM Studio shows the model as text-only.

`attach-mmproj` links a projector from any cached repository into any model
directory:

```bash
# pick both ends interactively
python lmstudio_hf.py attach-mmproj

# or name them
python lmstudio_hf.py attach-mmproj \
    --to   bartowski/Some-Gemma-4-Derivative-GGUF \
    --from unsloth/gemma-4-31B-it-GGUF
```

Useful options: `--file NAME` when the source repository holds more than one
projector, `--dry-run` to see what would happen, `-y` to skip the confirmation,
and `--force` to go ahead despite an incompatible verdict.

The projector is linked, never copied. The link is named
`mmproj-<source repo>-<quantisation>.gguf` so that LM Studio recognises it and
so its origin stays visible, and a `.lmstudio_hf.json` sidecar in the model
directory records where it came from. `detach-mmproj` uses that record to
remove exactly what was added.

Names are taken as they come. A repository or file name written in kanji or
hanzi is kept in the link name; only characters a filesystem refuses (and
whitespace, which is tedious to pass to `--file`) are replaced, and the name is
clipped on a character boundary so it fits a filesystem's per-component limit.
Output escapes anything the console's code page cannot encode — a simplified
Chinese file name on a `cp932` console, say — instead of failing part way
through a report.

If the directory already holds a projector it is cleared first — two of them in
one folder leaves it undefined which one LM Studio picks. A projector this tool
did not create is renamed to `*.gguf.disabled` rather than deleted.

### Compatibility checking

Before linking, the headers of both files are compared:

| Verdict | Meaning |
| --- | --- |
| `[OK]` | Same architecture family, and the projector output matches the model's embedding width |
| `[? ]` | Dimensions disagree, or one of them could not be read. Selectable, with a warning |
| `[!!]` | Architecture families differ. Refused unless `--force` (or `f` in the menu) |

This says whether a pair should *load*, not whether it will work *well*. Two
things cannot be detected from metadata: a derivative that retrained its
projector rather than freezing it, and a version mismatch inside one family.

### Navigation controls

- ↑/↓ arrows: move through the list
- SPACE: select/deselect (import screen)
- ENTER: confirm
- `a`: show all models / only those that can take a projector
- `f`: allow incompatible projectors to be selected
- Ctrl+C: cancel

## How it works

1. The tool scans your Hugging Face cache directory (checks `HF_HOME`, `XDG_CACHE_HOME/huggingface`, or `~/.cache/huggingface`)
2. Identifies all downloaded models, reading the model type from `config.json` when one is present and the architecture, quantisation and projector details from GGUF headers when they are not
3. Creates links in the LM Studio models directory (see [Environment Variables](#environment-variables))
4. Shows model type and import status for each model
5. Marks already imported models, so selecting one removes it again

Only the key/value block at the head of a GGUF file is read, and values that
are not needed are skipped rather than decoded, so scanning a cache does not
depend on how large the models are.

## Environment Variables

- `HF_HOME`: Optional. Set this to customize your Hugging Face cache location (highest priority)
- `XDG_CACHE_HOME`: Optional. If set, the tool will look for models in `$XDG_CACHE_HOME/huggingface`
- `LMSTUDIO_HOME`: Optional. Set this to customize your LM Studio models directory

When `LMSTUDIO_HOME` is unset, the models directory is taken from
`downloadsFolder` in `~/.lmstudio/settings.json`. If that file cannot be read,
the default is `%USERPROFILE%\.lmstudio\models` on Windows and
`~/.cache/lm-studio/models` elsewhere.

The tool refuses to run if the LM Studio models directory resolves to somewhere
inside the Hugging Face cache, since it would then link models on top of their
own source files.

## Repository layout

```
lmstudio_hf.py     entry point
lmshf/
  cli.py           argument parsing and command dispatch
  importing.py     the import/remove flow
  projectors.py    the attach, detach, doctor and list flows
  termui.py        keypress input and the selection menu
  paths.py         where the cache and the models directory are
  links.py         symlink / junction / hard link creation and removal
  hfcache.py       walking the Hugging Face cache
  lmstudio.py      reading the LM Studio models directory
  gguf.py          GGUF header reader
  mmproj.py        projector compatibility, attach and detach
tests/             standard library unittest suite
```

## Windows notes

- Creating symbolic links on Windows requires Developer Mode (Settings →
  System → For developers) or an elevated shell. Without it the tool falls back
  to **directory junctions** and **hard links**, which need no special
  privilege.
- Hard links only work within a single NTFS volume. If the Hugging Face cache
  and the LM Studio models directory are on different drives and symbolic links
  are unavailable, the import fails with an explanatory message. Nothing is
  ever copied — avoiding a second copy of the weights is the point of the tool.
- Box-drawing glyphs are replaced with ASCII equivalents when the console code
  page cannot encode them (e.g. `cp932`).

## Tests

```bash
python -m unittest discover -s tests
```

The GGUF tests build files byte by byte rather than relying on models being
present, so the suite runs anywhere.

## Notes

- Models are imported using links, not copies, to save disk space
- Already imported models are marked with "(already imported)" and shown in red
- Nothing is selected when the list opens; selecting an already imported model and
  confirming removes it from LM Studio
- Model types are displayed in parentheses (e.g., `(llama)`, `(bert)`, `(gpt2)`)

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License

[MIT License](LICENSE)
