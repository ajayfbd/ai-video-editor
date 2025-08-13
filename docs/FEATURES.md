# Feature Index

This project is organized feature-by-feature so you can run and validate each piece independently. Each feature has a small CLI and a real-data smoke test.

Features:

1. Transcribe (Whisper)
   - CLI: `ai-ve transcribe <input.(mp4|wav)> -o out/transcript.json`
   - Docs: docs/features/transcribe.md
2. Enhance audio (loudness/EQ/noise)
   - CLI: `ai-ve enhance-audio <input.mp4> -o out/audio_enhanced.wav [--target-lufs -14]`
   - Docs: docs/features/enhance_audio.md
3. Suggest B‑roll (from transcript)
   - CLI: `ai-ve suggest-broll transcript.json -o out/broll_plans.json`
   - Docs: docs/features/suggest_broll.md
4. Generate B‑roll assets (charts/slides/animations)
   - CLI: `ai-ve gen-broll broll_plans.json -o out/broll/`
   - Docs: docs/features/gen_broll.md
5. Plan execution (decisions + b‑roll → timeline)
   - CLI: `ai-ve plan-execute --ai-plan ai_plan.json -o out/timeline.json`
   - Docs: docs/features/plan_execute.md
6. Render (timeline or AI plan → MP4)
   - CLI: `ai-ve render --timeline out/timeline.json --videos in.mp4 -o out/final.mp4`
   - Docs: docs/features/render.md
7. Metadata + Thumbnails
   - CLI: `ai-ve metadata transcript.json -o out/metadata.json`, `ai-ve thumbnails context.json -o out/thumbs/`
   - Docs: docs/features/metadata_thumbnails.md

Real tests policy:
- Real tests are optional and live under `tests/real/`. They are skipped by default.
- Run with: `RUN_REAL=1 pytest -q -m real`.
- Each feature has a tiny sample and a smoke test.

Environment:
- FFmpeg must be in PATH.
- movis required for rendering.
- Whisper (openai-whisper + torch) required for transcribe.

See each feature page for contracts (inputs/outputs) and examples.
