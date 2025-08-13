# AI Video Editor – Quick User Guide

This is a short, practical guide. For full docs, see:
- User Guide: docs/user-guide/README.md
- Getting Started: docs/user-guide/getting-started.md
- CLI Reference: docs/user-guide/cli-reference.md

Prereqs
- Python 3.9+
- API keys for Gemini and Imagen (see Getting Started)

Install
- pwsh
  - python -m venv .venv
  - .\.venv\Scripts\Activate.ps1
  - pip install -r requirements.txt
  - python -m ai_video_editor.cli.main init
  - Edit .env with your API keys

Basic Commands
- Status: python -m ai_video_editor.cli.main status
- Process a video (balanced):
  - python -m ai_video_editor.cli.main process video.mp4 --type general --quality high
- Educational workflow (high quality):
  - python -m ai_video_editor.cli.main process lecture.mp4 --type educational --quality high --mode high_quality
- Music workflow (fast):
  - python -m ai_video_editor.cli.main process track.mp4 --type music --quality medium --mode fast

Batch Example
- python -m ai_video_editor.cli.main process *.mp4 --type general --parallel --output .\output

Performance Tips (i7 11th Gen, 32GB RAM)
- Default balanced profile is safe.
- For best quality: use --mode high_quality; expect higher CPU.
- Keep output/temp on SSD (default project structure already does).

Running Tests (fast default)
- python -m pytest -v   # skips slow/performance/acceptance by default
- Include performance: python -m pytest -m "performance or slow" -v --maxfail=1

Troubleshooting
- API keys: ensure .env is set (see Getting Started).
- Timeouts/Hangs: default per-test timeout is 30s; increase with --timeout=NN if needed.
- GPU metrics optional; install/update GPU drivers if enabling CUDA paths.
