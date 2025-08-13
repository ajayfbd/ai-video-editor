# Transcribe (Whisper)

Turn an audio/video file into a JSON transcript with segments and language info. Uses OpenAI Whisper models.

## CLI

```bash
ai-ve transcribe input.mp4 -o out/transcript.json [--model base|small|medium|large|turbo] [--language en]
```

## Inputs
- input: path to .mp4/.wav/.m4a/.mp3
- options: model (default: medium), language (optional; auto-detect if omitted)

## Outputs
- transcript.json
  - `language`: ISO code
  - `segments`: [{ start, end, text, confidence?, words? }]
  - `text`: full concatenated text

## Example
```bash
ai-ve transcribe tests/data/samples/voice_5s.mp4 -o out/transcript.json --model base
```

## Real smoke test
- File: `tests/real/transcribe/test_transcribe_real.py`
- Run: `RUN_REAL=1 pytest -q -m real -k transcribe`
- Skips automatically if FFmpeg/Whisper not available.

## Notes
- Requires FFmpeg in PATH.
- Downloading a Whisper model happens on first run; prefer `tiny` or `base` for quick tests.
- GPU optional (CUDA). CPU-only works but slower.
