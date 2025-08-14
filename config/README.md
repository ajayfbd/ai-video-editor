# AI Video Editor Configuration System

This directory contains the centralized configuration system for AI Video Editor, following modern Python project standards with `pyproject.toml` as the single source of truth.

## Configuration Architecture

### Primary Configuration Files
- **`pyproject.toml`** (root) - Single source of truth for project configuration, dependencies, and tool settings
- **`.env`** (root) - Runtime environment variables (API keys, settings)
- **`config/.env.example`** - Template for environment variables
- **`config/development.env`** - Development environment template  
- **`config/testing.env`** - Testing environment template

### Configuration Hierarchy
Settings are loaded in priority order (later overrides earlier):
1. Default values in `ai_video_editor/core/config.py`
2. Environment-specific templates from `config/`
3. Root `.env` file (runtime settings)
4. Command-line arguments

## Quick Setup

### 1. Basic Setup
```bash
# Copy appropriate template to root .env
cp config/.env.example .env

# Edit with your actual API keys
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_actual_key_here
```

### 2. Development Setup
```bash
# Use development-optimized settings
cp config/development.env .env
# Edit with your API keys
```

### 3. Testing Setup
```bash
# Use testing configuration (mock APIs)
cp config/testing.env .env
```

## Configuration Categories

### API Configuration
- `AI_VIDEO_EDITOR_GEMINI_API_KEY` - Gemini API for content analysis
- `AI_VIDEO_EDITOR_IMAGEN_API_KEY` - Imagen API for thumbnail generation  
- `AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT` - Google Cloud project ID

### Processing Settings
- `AI_VIDEO_EDITOR_MAX_MEMORY_USAGE_GB` - Memory limit (default: 8.0)
- `AI_VIDEO_EDITOR_WHISPER_MODEL_SIZE` - Whisper model (tiny/base/small/medium/large/large-v3)
- `AI_VIDEO_EDITOR_DEFAULT_QUALITY` - Output quality (low/medium/high/ultra)
- `AI_VIDEO_EDITOR_MAX_CONCURRENT_PROCESSES` - Parallel processing limit

### Performance Settings  
- `AI_VIDEO_EDITOR_ENABLE_GPU_ACCELERATION` - GPU acceleration toggle
- `AI_VIDEO_EDITOR_VIDEO_PROCESSING_TIMEOUT` - Processing timeout (seconds)
- `AI_VIDEO_EDITOR_API_REQUEST_TIMEOUT` - API request timeout (seconds)
- `AI_VIDEO_EDITOR_MAX_RETRIES` - API retry attempts

### Directory Settings
- `AI_VIDEO_EDITOR_OUTPUT_DIR` - Output directory (default: workspace/outputs)
- `AI_VIDEO_EDITOR_TEMP_DIR` - Temporary files (default: workspace/temp)
- `AI_VIDEO_EDITOR_LOGS_DIR` - Log files (default: workspace/logs)

## Environment-Specific Configuration

### Development Environment
- **Purpose**: Local development with debugging enabled
- **Settings**: Debug mode, verbose logging, reduced resource usage
- **Template**: `config/development.env`
- **Features**: Fast Whisper model, single-threaded processing, medium quality

### Testing Environment  
- **Purpose**: Automated testing and CI/CD
- **Settings**: Mock APIs, minimal resource usage, fast processing
- **Template**: `config/testing.env`
- **Features**: Mock API keys, tiny Whisper model, low quality output

### Production Environment
- **Purpose**: Production deployment with optimal performance
- **Settings**: High performance, full resource utilization, best quality
- **Template**: `config/.env.example`
- **Features**: Large Whisper model, multi-threaded processing, high quality

## Tool Configuration (pyproject.toml)

All development tools are configured in the root `pyproject.toml`:

### Dependencies Management
- **Runtime dependencies**: Listed in `project.dependencies`
- **Development dependencies**: Listed in `project.optional-dependencies.dev`
- **Testing dependencies**: Listed in `project.optional-dependencies.test`

### Code Quality Tools
- **Black**: Code formatting (88 char line length, Python 3.9+)
- **MyPy**: Type checking with strict settings
- **Pytest**: Testing framework with comprehensive markers and coverage

### CLI Entry Points
- **`video-editor`**: Main CLI (`ai_video_editor.cli.main:main`)
- **`ai-ve`**: Feature CLI (`ai_video_editor.cli.features:main`)

## Security Best Practices

### API Key Management
- ✅ **Never commit actual API keys** - Use templates only
- ✅ **Use environment-specific keys** - Different keys for dev/test/prod
- ✅ **Rotate keys regularly** - Update keys periodically
- ✅ **Use least-privilege access** - Minimal required permissions

### Configuration Security
- ✅ **`.env` files are gitignored** - Prevents accidental commits
- ✅ **Templates use placeholder values** - No real credentials in templates
- ✅ **Environment isolation** - Separate configs for each environment
- ✅ **Validation and sanitization** - Config values are validated

## Troubleshooting

### Common Issues
1. **Missing API Keys**: Ensure `.env` exists with valid keys
2. **Permission Errors**: Check workspace directory permissions
3. **Memory Issues**: Reduce `MAX_MEMORY_USAGE_GB` setting
4. **GPU Issues**: Set `ENABLE_GPU_ACCELERATION=false`

### Configuration Validation
```bash
# Check current configuration
python -m ai_video_editor.cli.main status

# Test API connectivity
python tools/dev/test_gemini_access.py

# Validate configuration
python -c "from ai_video_editor.core.config import get_settings; print(get_settings())"
```

### Migration Guide
If upgrading from older versions:
- ✅ **Old `setup.py`** → Now use `pyproject.toml`
- ✅ **Old `requirements.txt`** → Dependencies in `pyproject.toml`
- ✅ **Old `pytest.ini`** → Testing config in `pyproject.toml`
- ✅ **Scattered configs** → Centralized in `config/` directory

## Phase 4 Completion Status

✅ **Configuration Standardization Complete**
- Single source of truth: `pyproject.toml`
- Redundant files removed: `requirements.txt`
- Security enhanced: API key templates sanitized
- Documentation consolidated: Comprehensive configuration guide
- Environment templates organized: Development, testing, production ready