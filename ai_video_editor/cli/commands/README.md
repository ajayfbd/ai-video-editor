# CLI Commands Module

This directory contains the modular CLI commands for the `ai-ve` feature tool.

## Structure

```
commands/
├── __init__.py          # Package initialization
├── README.md           # This file
├── utils.py            # Shared utilities (romanization, device detection, etc.)
├── transcribe.py       # Transcribe command with smart defaults
├── subtitles.py        # Convert transcript to SRT/VTT
├── plan_execute.py     # Convert decisions to timeline
└── render.py           # Render MP4 from timeline/plan
```

## Design Principles

### Single Responsibility
Each command module handles one specific feature:
- `transcribe.py`: Audio/video → transcript JSON
- `subtitles.py`: Transcript JSON → SRT/VTT files
- `plan_execute.py`: Editing decisions → execution timeline
- `render.py`: Timeline/plan → final MP4

### Shared Utilities
Common functionality is extracted to `utils.py`:
- JSON loading/saving with proper encoding
- Hindi romanization (Urdu/Devanagari → Latin)
- Audio enhancement fallback (FFmpeg loudnorm)
- Smart device/model selection for low VRAM systems
- Progress formatting and error handling

### Smart Defaults
Commands use intelligent defaults for non-technical users:
- **Transcribe**: faster-whisper + CPU int8 + VAD + Hinglish romanization
- **Device selection**: Auto-detects GPU memory and falls back to CPU
- **Model selection**: Downgrades large models on CPU for performance
- **Enhancement**: Internal engine with FFmpeg fallback

## Adding New Commands

1. Create a new file: `commands/new_command.py`
2. Import click and shared utilities
3. Define the command function with `@click.command("command-name")`
4. Add error handling and logging
5. Register in `../features.py`:
   ```python
   from .commands.new_command import new_command_cmd
   cli.add_command(new_command_cmd)
   ```

## Testing

Each command can be tested independently:
```bash
# Test transcribe module
python -c "from ai_video_editor.cli.commands.transcribe import transcribe_cmd; print('OK')"

# Test utilities
python -c "from ai_video_editor.cli.commands.utils import romanize_hindi_text; print('OK')"
```

## Benefits

- **Maintainability**: Each command is self-contained and focused
- **Testability**: Commands can be unit tested independently
- **Extensibility**: Easy to add new commands without bloating existing code
- **Reusability**: Shared utilities prevent code duplication
- **Performance**: Lazy imports reduce startup time