"""Whisper greedy decoding (CPU-friendly, cross-platform).

Loads a Whisper checkpoint via Hugging Face `transformers` and runs greedy
decoding (`num_beams=1`, `do_sample=False`) on a single audio file. Avoids the
`openai-whisper` package on purpose so we don't need an `ffmpeg` install on
Windows.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

TARGET_SR = 16_000
# Whisper's mel-spectrogram input covers 30 s — anything longer is truncated.
MAX_AUDIO_SECONDS = 30


def load_audio(path: str) -> np.ndarray:
    """Read a WAV/FLAC file as float32 mono at 16 kHz."""
    audio, sr = sf.read(path, dtype="float32")
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
    # Whisper's `max_target_positions` is 448; 440 leaves a small headroom for
    # the forced prefix tokens (<|sot|><|lang|><|task|><|notimestamps|>).
    p.add_argument("--max-new-tokens", type=int, default=440)
    return p.parse_args()


def main() -> int:
    # Force UTF-8 stdout/stderr so non-ASCII transcriptions render correctly
    # on Windows consoles (which default to cp949/cp1252).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    args = parse_args()

    if not os.path.isfile(args.audio):
        sys.exit(f"audio file not found: {args.audio}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...", file=sys.stderr)

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    audio = load_audio(args.audio)
    duration = len(audio) / TARGET_SR
    if duration > MAX_AUDIO_SECONDS:
        print(
            f"warning: audio is {duration:.1f}s; Whisper truncates to "
            f"{MAX_AUDIO_SECONDS}s, only the first chunk will be decoded.",
            file=sys.stderr,
        )

    inputs = processor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.attention_mask.to(device)

    if args.language is None:
        # Surface the auto-detected language so users can sanity-check results.
        # `transformers` >= 5 strips the forced prefix from `generate` output,
        # so we ask the model directly instead of inspecting `generated[0, 1]`.
        with torch.no_grad():
            lang_token_ids = model.detect_language(input_features)
        lang_token = processor.tokenizer.convert_ids_to_tokens(
            lang_token_ids[0].item()
        )
        print(f"Detected language: {lang_token}", file=sys.stderr)

    generate_kwargs: dict = {
        "num_beams": 1,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "attention_mask": attention_mask,
    }
    # `transformers` warns if `language` is set without `task`, so always pair
    # them when either differs from the auto-detect / default path.
    if args.language is not None:
        generate_kwargs["language"] = args.language
        generate_kwargs["task"] = args.task
    elif args.task != "transcribe":
        generate_kwargs["task"] = args.task

    with torch.no_grad():
        generated = model.generate(input_features, **generate_kwargs)

    # `clean_up_tokenization_spaces=True` is destructive for Whisper's BPE
    # tokenizer (it strips spaces around punctuation and around CJK tokens,
    # turning e.g. "음성인식 테스트" into "음성인 식태스트").
    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
