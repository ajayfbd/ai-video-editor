# Real Feature Tests

These tests exercise real dependencies (FFmpeg, movis, whisper) and run on tiny media samples. They are skipped by default.

Run with:

```bash
RUN_REAL=1 pytest -q -m real
```

Guidelines:
- Keep samples small (2–10 seconds) in `tests/data/samples/`.
- Mark tests with `@pytest.mark.real`.
- Skip tests gracefully if a dependency is missing.
