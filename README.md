# LM Studio - Hugging Face Model Manager

A command-line utility to manage models between your Hugging Face cache and LM Studio. This tool makes it easy to import and manage any models you've downloaded from Hugging Face into LM Studio.

## Features

- Interactive model selection interface with keyboard navigation
- Automatic detection of all models in your Hugging Face cache
- Smart handling of model imports via links, never copies
- Support for model removal and re-import
- Terminal-based UI with scrolling for large model lists
- Shows model type (e.g., llama, bert, gpt2) for easy identification
- Already imported models are clearly marked and can be removed again

## Prerequisites

- Python 3.9 or newer
- LM Studio installed
- Hugging Face models downloaded locally

Works on macOS, Linux and Windows. See [Windows notes](#windows-notes) for the
extra details that apply there.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ivanfioravanti/lmstudio_hf.git
cd lmstudio_hf
```

## Usage

Run the script using Python:

```bash
python lmstudio_hf.py
```

### Using Custom Directories

You can customize the directories using environment variables:

```bash
# Custom Hugging Face cache directory
export HF_HOME="/path/to/huggingface/cache"
python lmstudio_hf.py

# Custom LM Studio models directory
export LMSTUDIO_HOME="/path/to/lmstudio/models"
python lmstudio_hf.py

# Use XDG cache directory
export XDG_CACHE_HOME="/path/to/cache"
python lmstudio_hf.py

# Combine multiple environment variables
export HF_HOME="/custom/hf/cache"
export LMSTUDIO_HOME="/custom/lmstudio/models"
python lmstudio_hf.py
```

### Navigation Controls

- ↑/↓ arrows: Navigate through the model list
- SPACE: Select/deselect a model
- ENTER: Confirm selection and proceed with import
- Ctrl+C: Cancel operation

## How It Works

1. The tool scans your Hugging Face cache directory (checks `HF_HOME`, `XDG_CACHE_HOME/huggingface`, or `~/.cache/huggingface`)
2. Identifies all downloaded models, reading the model type from `config.json` when one is present
3. Creates links in the LM Studio models directory (see [Environment Variables](#environment-variables))
4. Shows model type and import status for each model
5. Marks already imported models, so selecting one removes it again

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