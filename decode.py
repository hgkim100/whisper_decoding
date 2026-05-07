"""Whisper greedy decoding (CPU-friendly, cross-platform).

Loads a Whisper checkpoint via Hugging Face `transformers` and runs greedy
decoding (`num_beams=1`, `do_sample=False`) on a single audio file. Avoids the
`openai-whisper` package on purpose so we don't need an `ffmpeg` install on
Windows.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

TARGET_SR = 16_000


def load_audio(path: str) -> np.ndarray:
    """Read a WAV/FLAC file as float32 mono at 16 kHz."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        import torchaudio  # local import — only needed when resampling

        tensor = torch.from_numpy(audio).unsqueeze(0)
        tensor = torchaudio.functional.resample(tensor, sr, TARGET_SR)
        audio = tensor.squeeze(0).numpy()
    return audio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whisper greedy decoding")
    p.add_argument("audio", help="Path to a WAV or FLAC audio file")
    p.add_argument(
        "--model",
        default="openai/whisper-tiny",
        help="HF model id (default: openai/whisper-tiny — small enough for CPU)",
    )
    p.add_argument(
        "--language",
        default=None,
        help="Language code, e.g. 'en' or 'ko'. Auto-detected if omitted.",
    )
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--max-new-tokens", type=int, default=440)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...", file=sys.stderr)

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    audio = load_audio(args.audio)
    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    generate_kwargs: dict = {
        "num_beams": 1,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.language is not None:
        generate_kwargs["language"] = args.language
    if args.task != "transcribe":
        generate_kwargs["task"] = args.task

    with torch.no_grad():
        generated = model.generate(input_features, **generate_kwargs)

    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
