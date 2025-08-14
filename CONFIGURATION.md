# AI Video Editor Configuration Guide

> **📋 Complete Configuration Reference**: See [`config/README.md`](config/README.md) for comprehensive configuration documentation.

This guide provides quick setup instructions for AI Video Editor configuration.

## Quick Setup

### 1. Choose Your Environment

**Development Setup** (recommended for local development):
```bash
cp config/development.env .env
# Edit .env with your API keys
```

**Production Setup** (for deployment):
```bash
cp config/.env.example .env
# Edit .env with production API keys and settings
```

**Testing Setup** (for automated testing):
```bash
cp config/testing.env .env
# Uses mock APIs by default
```

### 2. Add Your API Keys

Edit the `.env` file:
```bash
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_actual_gemini_key_here
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_actual_imagen_key_here
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id
```

### 3. Verify Setup

```bash
# Check configuration status
python -m ai_video_editor.cli.main status

# Test API connectivity
python tools/dev/test_gemini_access.py

# Run help to see available commands
python -m ai_video_editor.cli.main --help
```

## Configuration Architecture

### Modern Python Standards (Phase 4 Complete ✅)
- **Single Source of Truth**: `pyproject.toml` for all project configuration
- **Centralized Environment Management**: `config/` directory with templates
- **Security First**: No API keys committed, proper gitignore patterns
- **Environment Isolation**: Separate configs for dev/test/prod

### Key Files
- **`pyproject.toml`** - Project dependencies, build settings, tool configuration
- **`.env`** - Runtime environment variables (API keys, settings)
- **`config/`** - Environment templates and configuration documentation

## Common Configuration Tasks

### Change Processing Quality
```bash
# In .env file
AI_VIDEO_EDITOR_DEFAULT_QUALITY=high  # low/medium/high/ultra
```

### Adjust Memory Usage
```bash
# In .env file
AI_VIDEO_EDITOR_MAX_MEMORY_USAGE_GB=4.0  # Reduce for lower-end systems
```

### Enable/Disable GPU
```bash
# In .env file
AI_VIDEO_EDITOR_ENABLE_GPU_ACCELERATION=false  # Disable if GPU issues
```

### Change Whisper Model
```bash
# In .env file
AI_VIDEO_EDITOR_WHISPER_MODEL_SIZE=base  # tiny/base/small/medium/large/large-v3
```

## Troubleshooting

### Quick Fixes
- **Missing API Keys**: Copy appropriate template from `config/` to `.env`
- **Permission Errors**: Check `workspace/` directory permissions
- **Memory Issues**: Reduce `MAX_MEMORY_USAGE_GB` in `.env`
- **GPU Issues**: Set `ENABLE_GPU_ACCELERATION=false` in `.env`

### Get Help
- **Full Documentation**: [`config/README.md`](config/README.md)
- **API Testing**: `python tools/dev/test_gemini_access.py`
- **Configuration Validation**: `python -m ai_video_editor.cli.main status`

## Phase 4 Completion ✅

**Configuration Cleanup and Standardization Complete**
- ✅ Removed redundant `requirements.txt` (use `pyproject.toml`)
- ✅ Secured API keys (templates only, no real keys committed)
- ✅ Consolidated configuration documentation
- ✅ Established single source of truth architecture
- ✅ Created comprehensive environment management system