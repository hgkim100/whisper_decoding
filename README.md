# whisper_decoding

Whisper decoding experiments.

## Setup

CPU-only, works on Windows and Ubuntu. No `ffmpeg` required — audio is read
through `soundfile` and (if needed) resampled with `torchaudio`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Ubuntu:  source .venv/bin/activate
pip install -r requirements.txt
```

> On Windows, `pip install torch` from PyPI ships a CPU build by default, so
> the same `requirements.txt` works on machines without a GPU.

## Greedy decoding

```bash
python decode.py path/to/audio.wav
```

Options:

- `--model openai/whisper-tiny` (default; ~150 MB, fits comfortably on CPU)
- `--language en` / `--language ko` — skip for auto-detect
- `--task translate` — translate to English instead of transcribing
- `--max-new-tokens 440`

The first run downloads the model from Hugging Face and caches it under the
default HF cache (`~/.cache/huggingface` on Linux,
`%USERPROFILE%\.cache\huggingface` on Windows).

Input audio is converted to mono 16 kHz internally. WAV and FLAC are
supported out of the box; for other formats, convert first.
