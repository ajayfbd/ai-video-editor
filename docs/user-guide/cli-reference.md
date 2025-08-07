# CLI Reference Guide

Complete reference for the AI Video Editor command-line interface.

## 📋 Table of Contents

1. [**Global Options**](#global-options)
2. [**Commands Overview**](#commands-overview)
3. [**Process Command**](#process-command)
4. [**Status Command**](#status-command)
5. [**Init Command**](#init-command)
6. [**Workflow Commands**](#workflow-commands)
7. [**Examples**](#examples)

## 🌐 Global Options

Available for all commands:

```bash
ai-video-editor [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

### Global Flags

| Option | Description | Default |
|--------|-------------|---------|
| `--debug` | Enable debug logging | `false` |
| `--config PATH` | Path to configuration file | Auto-detect |
| `--version` | Show version information | - |
| `--help` | Show help message | - |

### Examples

```bash
# Enable debug mode
ai-video-editor --debug process video.mp4

# Use custom config
ai-video-editor --config ./my-config.yaml status

# Show version
ai-video-editor --version
```

## 📚 Commands Overview

| Command | Purpose | Usage |
|---------|---------|-------|
| [`process`](#process-command) | Process video files | `process [FILES] [OPTIONS]` |
| [`status`](#status-command) | Check system status | `status` |
| [`init`](#init-command) | Initialize configuration | `init [OPTIONS]` |
| [`analyze`](#analyze-command) | Analyze audio content | `analyze AUDIO_FILE [OPTIONS]` |
| [`enhance`](#enhance-command) | Enhance video quality | `enhance VIDEO_FILE [OPTIONS]` |
| [`workflow`](#workflow-commands) | Manage workflows | `workflow [SUBCOMMAND]` |
| [`test-workflow`](#test-workflow-command) | Test workflow system | `test-workflow [OPTIONS]` |

## 🎬 Process Command

Main command for processing videos through the complete AI pipeline.

### Syntax

```bash
ai-video-editor process INPUT_FILES [OPTIONS]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `INPUT_FILES` | One or more video files to process | `video.mp4` or `*.mp4` |

### Options

#### Content and Quality

| Option | Type | Description | Default | Example |
|--------|------|-------------|---------|---------|
| `--type` | Choice | Content type optimization | `general` | `--type educational` |
| `--quality` | Choice | Output quality level | `high` | `--quality ultra` |
| `--mode` | Choice | Processing mode | `balanced` | `--mode fast` |

**Content Types:**
- `educational`: Tutorials, lectures, explanations
- `music`: Music videos, performances
- `general`: Mixed or general content

**Quality Levels:**
- `low`: Fast processing, basic enhancements
- `medium`: Balanced quality and speed
- `high`: Professional quality (recommended)
- `ultra`: Maximum quality, slower processing

**Processing Modes:**
- `fast`: Prioritizes speed over quality
- `balanced`: Optimal balance (default)
- `high_quality`: Prioritizes quality over speed

#### Output and Performance

| Option | Type | Description | Default | Example |
|--------|------|-------------|---------|---------|
| `--output` | Path | Output directory | `./output` | `--output ./my_project` |
| `--parallel` | Flag | Enable parallel processing | `true` | `--parallel` |
| `--max-memory` | Float | Memory limit in GB | `8.0` | `--max-memory 16` |
| `--timeout` | Integer | Timeout per stage (seconds) | `1800` | `--timeout 3600` |
| `--no-progress` | Flag | Disable progress display | `false` | `--no-progress` |

### Examples

#### Basic Processing

```bash
# Process with defaults
ai-video-editor process video.mp4

# Educational content, high quality
ai-video-editor process lecture.mp4 --type educational --quality high

# Music video, ultra quality
ai-video-editor process song.mp4 --type music --quality ultra
```

#### Advanced Processing

```bash
# High-performance system
ai-video-editor process video.mp4 \
  --type educational \
  --quality ultra \
  --mode high_quality \
  --max-memory 16 \
  --parallel \
  --output ./premium_output

# Memory-constrained system
ai-video-editor process video.mp4 \
  --type general \
  --quality medium \
  --mode fast \
  --max-memory 4 \
  --timeout 600

# Batch processing
ai-video-editor process *.mp4 \
  --type educational \
  --parallel \
  --output ./batch_output
```

## ✅ Status Command

Check system status and configuration.

### Syntax

```bash
ai-video-editor status
```

### Output

The status command displays:
- Overall system readiness
- API key configuration status
- Directory accessibility
- System resource availability
- Configuration warnings/errors

### Example Output

```
┌─────────────────────────────────────────┐
│            AI Video Editor Status       │
├─────────────────┬───────────────────────┤
│ Component       │ Status                │
├─────────────────┼───────────────────────┤
│ Overall         │ ✅ Ready              │
│ Gemini API      │ ✅ Configured         │
│ Imagen API      │ ✅ Configured         │
│ Output Directory│ ✅ Accessible         │
│ System Memory   │ ✅ 16GB Available     │
└─────────────────┴───────────────────────┘
```

## 🔧 Init Command

Initialize configuration file with defaults.

### Syntax

```bash
ai-video-editor init [OPTIONS]
```

### Options

| Option | Type | Description | Default | Example |
|--------|------|-------------|---------|---------|
| `--output` | Path | Configuration file path | `.env` | `--output config.env` |

### Examples

```bash
# Create default .env file
ai-video-editor init

# Create custom config file
ai-video-editor init --output ./config/.env

# Overwrite existing config (with confirmation)
ai-video-editor init --output .env
```

### Generated Configuration

The init command creates a `.env` file with:

```bash
# AI Video Editor Configuration
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_gemini_api_key_here
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_api_key_here
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id_here

# Performance Settings
AI_VIDEO_EDITOR_MAX_MEMORY_USAGE_GB=8
AI_VIDEO_EDITOR_MAX_CONCURRENT_PROCESSES=2
AI_VIDEO_EDITOR_ENABLE_CACHING=true

# Quality Defaults
AI_VIDEO_EDITOR_DEFAULT_QUALITY=high
AI_VIDEO_EDITOR_DEFAULT_MODE=balanced
```

## 🎵 Analyze Command

Analyze audio content and generate transcript.

### Syntax

```bash
ai-video-editor analyze AUDIO_FILE [OPTIONS]
```

### Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `AUDIO_FILE` | Audio file to analyze | `audio.mp3` |

### Options

| Option | Type | Description | Default | Example |
|--------|------|-------------|---------|---------|
| `--output` | Path | Output transcript file | Auto-generated | `--output transcript.txt` |

### Examples

```bash
# Basic audio analysis
ai-video-editor analyze audio.mp3

# Save transcript to file
ai-video-editor analyze lecture.wav --output lecture_transcript.txt
```

## 🎨 Enhance Command

Enhance video quality using AI.

### Syntax

```bash
ai-video-editor enhance INPUT_FILE [OPTIONS]
```

### Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `INPUT_FILE` | Video file to enhance | `video.mp4` |

### Options

| Option | Type | Description | Default | Example |
|--------|------|-------------|---------|---------|
| `--output` | Path | Output file path | Auto-generated | `--output enhanced.mp4` |

### Examples

```bash
# Basic enhancement
ai-video-editor enhance video.mp4

# Custom output path
ai-video-editor enhance low_quality.mp4 --output high_quality.mp4
```

## 🔄 Workflow Commands

Manage and monitor workflow execution.

### Syntax

```bash
ai-video-editor workflow [SUBCOMMAND] [OPTIONS]
```

### Subcommands

#### List Workflows

```bash
ai-video-editor workflow --list
```

Shows all active workflow projects.

#### Check Workflow Status

```bash
ai-video-editor workflow PROJECT_ID [OPTIONS]
```

**Options:**
- `--details`: Show detailed status information

### Examples

```bash
# List all workflows
ai-video-editor workflow --list

# Check specific workflow
ai-video-editor workflow abc123 --details

# Monitor workflow progress
ai-video-editor workflow abc123
```

## 🧪 Test Workflow Command

Test workflow orchestrator functionality.

### Syntax

```bash
ai-video-editor test-workflow [OPTIONS]
```

### Options

| Option | Type | Description | Example |
|--------|------|-------------|---------|
| `--stage` | String | Test specific stage | `--stage audio_processing` |
| `--mock` | Flag | Use mock data | `--mock` |

### Examples

```bash
# Basic workflow test
ai-video-editor test-workflow

# Test specific stage
ai-video-editor test-workflow --stage thumbnail_generation

# Test with mock data
ai-video-editor test-workflow --mock
```

## 📝 Examples

### Complete Workflows

#### Educational Content Pipeline

```bash
# Complete educational video processing
ai-video-editor process educational_video.mp4 \
  --type educational \
  --quality ultra \
  --mode high_quality \
  --output ./educational_output \
  --max-memory 12 \
  --parallel \
  --timeout 2400
```

#### Music Video Pipeline

```bash
# Music video with fast processing
ai-video-editor process music_video.mp4 \
  --type music \
  --quality high \
  --mode balanced \
  --output ./music_output \
  --parallel
```

#### Batch Processing

```bash
# Process multiple videos
ai-video-editor process video1.mp4 video2.mp4 video3.mp4 \
  --type general \
  --quality high \
  --parallel \
  --output ./batch_output \
  --max-memory 16
```

### System Management

#### Setup and Configuration

```bash
# Initial setup
ai-video-editor init
ai-video-editor status

# Custom configuration
ai-video-editor --config ./custom.yaml status
```

#### Monitoring and Debugging

```bash
# Debug mode processing
ai-video-editor --debug process video.mp4 --type educational

# Monitor workflow
ai-video-editor workflow --list
ai-video-editor workflow PROJECT_ID --details
```

## 🆘 Exit Codes

| Code | Meaning | Description |
|------|---------|-------------|
| `0` | Success | Command completed successfully |
| `1` | General Error | Command failed with error |
| `2` | Configuration Error | Invalid configuration or missing API keys |
| `3` | Resource Error | Insufficient system resources |
| `4` | Timeout Error | Processing timeout exceeded |

---

*Complete CLI reference for the AI Video Editor*