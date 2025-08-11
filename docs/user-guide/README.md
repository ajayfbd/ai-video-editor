# AI Video Editor User Guide

Complete guide to using the AI Video Editor for creating professional, engaging video content with AI assistance.

## 🧭 Navigation

**New to AI Video Editor?** Start with the [**Quick Start Guide**](../../quick-start.md) for immediate setup, then return here for comprehensive documentation.

**Looking for specific workflows?** Jump to [**Tutorials**](../../tutorials/README.md) for step-by-step guides.

**Need technical details?** Check the [**Developer Guide**](../../developer/README.md) for architecture and API information.

## 📖 Table of Contents

1. [**Getting Started**](#getting-started) - Installation and first video
2. [**Core Concepts**](#core-concepts) - Understanding the system
3. [**Processing Workflows**](#processing-workflows) - Different use cases
4. [**Content Types**](#content-types) - Optimized processing modes
5. [**Output Management**](#output-management) - Understanding results
6. [**Advanced Features**](#advanced-features) - Power user options
7. [**Performance Tuning**](#performance-tuning) - Optimization tips
8. [**CLI Reference**](#cli-reference) - Complete command documentation
9. [**Configuration**](#configuration) - Advanced settings

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** (Python 3.10+ recommended)
- **8GB+ RAM** (16GB recommended for high-quality processing)
- **Internet connection** for AI services
- **API Keys** for Gemini and Imagen (see [API Setup](#api-setup))

### Quick Installation

#### 1. Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd ai-video-editor

# Install dependencies
pip install -r requirements.txt
```

#### 2. Initialize Configuration

```bash
# Create default configuration
python -m ai_video_editor.cli.main init

# This creates a .env file with default settings
```

#### 3. API Setup

Edit the generated `.env` file and add your API keys:

```bash
# Required API Keys
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_gemini_api_key_here
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_api_key_here
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id_here

# Optional Settings
AI_VIDEO_EDITOR_MAX_MEMORY_USAGE_GB=8
AI_VIDEO_EDITOR_MAX_CONCURRENT_PROCESSES=2
```

##### Getting API Keys

**Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

**Imagen API Key:**
1. Enable the Imagen API in [Google Cloud Console](https://console.cloud.google.com/)
2. Create a service account key
3. Set the key in your `.env` file

#### 4. Verify Installation

```bash
# Check system status
python -m ai_video_editor.cli.main status
```

You should see all components marked as "✅ Ready".

### Process Your First Video

#### Basic Processing

```bash
# Process a video with default settings
python -m ai_video_editor.cli.main process video.mp4
```

#### Educational Content (Recommended)

```bash
# Process educational content with high quality
python -m ai_video_editor.cli.main process lecture.mp4 \
  --type educational \
  --quality high \
  --output ./output
```

#### Music Video

```bash
# Process music content (optimized workflow)
python -m ai_video_editor.cli.main process music_video.mp4 \
  --type music \
  --quality ultra \
  --mode fast
```

### Understanding Your First Output

After processing, you'll find:

```
output/
├── enhanced_video.mp4          # Final edited video
├── thumbnails/                 # Generated thumbnail variations
│   ├── thumbnail_emotional.jpg
│   ├── thumbnail_curiosity.jpg
│   └── thumbnail_authority.jpg
├── metadata/                   # SEO-optimized metadata
│   ├── titles.json
│   ├── descriptions.json
│   └── tags.json
└── analytics/                  # Processing insights
    ├── performance_metrics.json
    └── ai_decisions.json
```

## 🎯 Core Concepts

### AI Director

The **AI Director** is the brain of the system, powered by Google's Gemini API. It:
- Analyzes your video content for key concepts and emotions
- Makes creative editing decisions (cuts, transitions, emphasis)
- Plans B-roll insertion and visual enhancements
- Generates SEO-optimized metadata and thumbnails
- Ensures all outputs work together cohesively

### ContentContext Architecture

All processing modules share a unified **ContentContext** that flows through the pipeline:
- **Input Analysis**: Audio transcription, visual highlights, emotional peaks
- **AI Decisions**: Editing plans, B-roll strategies, metadata approaches
- **Output Generation**: Video composition, thumbnails, metadata packages

This ensures your thumbnail hook text, YouTube title, video cuts, and B-roll all derive from the same creative vision.

### Processing Pipeline

```
Raw Video → Content Analysis → AI Director → Output Generation
    ↓              ↓               ↓              ↓
Audio/Video → Concepts/Emotions → Creative Plan → Final Assets
```

## 🎬 Processing Workflows

### Educational Content Workflow

Perfect for tutorials, lectures, and explanatory videos:

```bash
# Basic educational processing
python -m ai_video_editor.cli.main process lecture.mp4 \
  --type educational \
  --quality high

# Advanced educational with B-roll
python -m ai_video_editor.cli.main process tutorial.mp4 \
  --type educational \
  --quality ultra \
  --mode high_quality \
  --output ./educational_output
```

**Features enabled:**
- Filler word removal and audio cleanup
- Concept-based B-roll generation (charts, animations)
- Educational thumbnail strategies
- SEO optimization for learning keywords

### Music Video Workflow

Optimized for music content and performances:

```bash
# Music video processing
python -m ai_video_editor.cli.main process music_video.mp4 \
  --type music \
  --quality ultra \
  --mode balanced
```

**Features enabled:**
- Beat-synchronized editing
- Visual effect emphasis
- Music-focused thumbnails
- Genre-specific metadata optimization

### General Content Workflow

Balanced processing for mixed or general content:

```bash
# General content processing
python -m ai_video_editor.cli.main process content.mp4 \
  --type general \
  --quality high \
  --parallel
```

**Features enabled:**
- Adaptive content analysis
- Flexible B-roll strategies
- Multi-strategy thumbnails
- Broad keyword optimization

## 📊 Content Types

### Educational (`--type educational`)

**Best for:** Tutorials, lectures, how-to videos, explanations

**Optimizations:**
- Enhanced filler word detection and removal
- Concept-based B-roll generation (charts, formulas, diagrams)
- Educational thumbnail strategies (authority, curiosity)
- Learning-focused SEO keywords
- Explanation segment emphasis

**Example Output:**
- Clean, professional audio
- Animated charts and diagrams
- Authority-building thumbnails
- Educational metadata packages

### Music (`--type music`)

**Best for:** Music videos, performances, concerts, music content

**Optimizations:**
- Beat-synchronized editing decisions
- Visual effect and transition emphasis
- Music-focused thumbnail strategies
- Genre and artist-specific SEO
- Audio quality preservation

**Example Output:**
- Beat-aligned cuts and transitions
- Dynamic visual effects
- Performance-focused thumbnails
- Music discovery metadata

### General (`--type general`)

**Best for:** Mixed content, vlogs, general videos

**Optimizations:**
- Adaptive content analysis
- Flexible editing strategies
- Multi-approach thumbnails
- Broad keyword targeting
- Balanced processing

**Example Output:**
- Content-adaptive editing
- Versatile B-roll integration
- Multi-strategy thumbnails
- Comprehensive metadata

## 📁 Output Management

### Understanding Output Structure

```
output/
├── video/
│   ├── final_video.mp4           # Main edited video
│   ├── enhanced_audio.wav        # Processed audio track
│   └── composition_layers.json   # Movis composition data
├── thumbnails/
│   ├── emotional/                # Emotional strategy thumbnails
│   ├── curiosity/               # Curiosity-driven thumbnails
│   ├── authority/               # Authority-building thumbnails
│   └── recommended.jpg          # AI-recommended thumbnail
├── metadata/
│   ├── titles.json              # Optimized title variations
│   ├── descriptions.json        # SEO descriptions
│   ├── tags.json               # Keyword tags
│   └── synchronized_package.json # Complete metadata package
├── broll/
│   ├── charts/                  # Generated charts and graphs
│   ├── animations/              # Blender animations
│   └── graphics/               # Motion graphics
└── analytics/
    ├── ai_decisions.json        # AI Director decisions
    ├── performance_metrics.json # Processing statistics
    └── content_analysis.json    # Content insights
```

### Thumbnail Variations

The system generates multiple thumbnail strategies:

**Emotional Strategy**
- High-impact emotional expressions
- Bold, attention-grabbing text
- Dynamic backgrounds
- Optimized for high CTR

**Curiosity Strategy**
- Question-based hooks
- Mystery and intrigue elements
- "What happens next" approach
- Optimized for engagement

**Authority Strategy**
- Professional, credible appearance
- Educational and informative tone
- Trust-building elements
- Optimized for expertise positioning

### Metadata Packages

Each video gets comprehensive metadata optimization:

**Titles**
- Multiple variations per strategy
- A/B testing recommendations
- SEO keyword integration
- Platform-specific optimization

**Descriptions**
- Comprehensive, engaging summaries
- Keyword-rich content
- Timestamp integration
- Call-to-action optimization

**Tags**
- Broad and specific keyword mix
- Trending topic integration
- Competitor analysis insights
- Platform algorithm optimization

## ⚙️ Advanced Features

### Parallel Processing

Enable parallel processing for faster results:

```bash
python -m ai_video_editor.cli.main process video.mp4 \
  --parallel \
  --max-memory 16
```

### Custom Output Directory

Organize outputs with custom directories:

```bash
python -m ai_video_editor.cli.main process video.mp4 \
  --output ./projects/my_video \
  --type educational
```

### Workflow Monitoring

Monitor processing progress:

```bash
# List active workflows
python -m ai_video_editor.cli.main workflow --list

# Check specific workflow
python -m ai_video_editor.cli.main workflow PROJECT_ID --details
```

### Batch Processing

Process multiple videos:

```bash
# Process multiple files
python -m ai_video_editor.cli.main process video1.mp4 video2.mp4 video3.mp4 \
  --type educational \
  --parallel
```

## 🚀 Performance Tuning

### Memory Optimization

**For 8GB Systems:**
```bash
python -m ai_video_editor.cli.main process video.mp4 \
  --max-memory 6 \
  --mode fast \
  --quality medium
```

**For 16GB+ Systems:**
```bash
python -m ai_video_editor.cli.main process video.mp4 \
  --max-memory 12 \
  --parallel \
  --quality ultra
```

### Processing Speed

**Fast Processing:**
```bash
python -m ai_video_editor.cli.main process video.mp4 \
  --mode fast \
  --quality medium \
  --timeout 300
```

**High Quality (Slower):**
```bash
python -m ai_video_editor.cli.main process video.mp4 \
  --mode high_quality \
  --quality ultra \
  --timeout 1800
```

### API Cost Optimization

Monitor and optimize API usage:

```bash
# Check API usage in output analytics
cat output/analytics/performance_metrics.json | grep api_cost

# Use caching to reduce costs
export AI_VIDEO_EDITOR_ENABLE_CACHING=true
```

## 🔧 Configuration

### Environment Variables

Key configuration options:

```bash
# API Configuration
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_key
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_key
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project

# Performance Settings
AI_VIDEO_EDITOR_MAX_MEMORY_USAGE_GB=8
AI_VIDEO_EDITOR_MAX_CONCURRENT_PROCESSES=2
AI_VIDEO_EDITOR_ENABLE_CACHING=true

# Quality Settings
AI_VIDEO_EDITOR_DEFAULT_QUALITY=high
AI_VIDEO_EDITOR_DEFAULT_MODE=balanced
```

### Configuration File

Create `config.yaml` for persistent settings:

```yaml
# AI Video Editor Configuration
api:
  gemini_key: ${AI_VIDEO_EDITOR_GEMINI_API_KEY}
  imagen_key: ${AI_VIDEO_EDITOR_IMAGEN_API_KEY}
  project_id: ${AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT}

processing:
  default_quality: high
  default_mode: balanced
  max_memory_gb: 8
  enable_parallel: true
  enable_caching: true

output:
  default_directory: ./output
  preserve_intermediate: false
  generate_analytics: true
```

## 🖥️ CLI Reference

Complete reference for the AI Video Editor command-line interface.

### Global Options

Available for all commands:

```bash
python -m ai_video_editor.cli.main [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

#### Global Flags

| Option | Description | Default |
|--------|-------------|---------|
| `--debug` | Enable debug logging | `false` |
| `--config PATH` | Path to configuration file | Auto-detect |
| `--version` | Show version information | - |
| `--help` | Show help message | - |

### Main Commands

#### Process Command

Main command for processing videos through the complete AI pipeline.

**Syntax:**
```bash
python -m ai_video_editor.cli.main process INPUT_FILES [OPTIONS]
```

**Key Options:**

| Option | Type | Description | Default | Example |
|--------|------|-------------|---------|---------|
| `--type` | Choice | Content type optimization | `general` | `--type educational` |
| `--quality` | Choice | Output quality level | `high` | `--quality ultra` |
| `--mode` | Choice | Processing mode | `balanced` | `--mode fast` |
| `--output` | Path | Output directory | `./output` | `--output ./my_project` |
| `--parallel` | Flag | Enable parallel processing | `true` | `--parallel` |
| `--max-memory` | Float | Memory limit in GB | `8.0` | `--max-memory 16` |
| `--timeout` | Integer | Timeout per stage (seconds) | `1800` | `--timeout 3600` |

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

#### Status Command

Check system status and configuration:

```bash
python -m ai_video_editor.cli.main status
```

#### Init Command

Initialize configuration file with defaults:

```bash
python -m ai_video_editor.cli.main init [--output PATH]
```

#### Workflow Commands

Manage and monitor workflow execution:

```bash
# List all workflows
python -m ai_video_editor.cli.main workflow --list

# Check specific workflow
python -m ai_video_editor.cli.main workflow PROJECT_ID --details
```

### CLI Examples

#### Complete Workflows

**Educational Content Pipeline:**
```bash
python -m ai_video_editor.cli.main process educational_video.mp4 \
  --type educational \
  --quality ultra \
  --mode high_quality \
  --output ./educational_output \
  --max-memory 12 \
  --parallel \
  --timeout 2400
```

**Music Video Pipeline:**
```bash
python -m ai_video_editor.cli.main process music_video.mp4 \
  --type music \
  --quality high \
  --mode balanced \
  --output ./music_output \
  --parallel
```

**Batch Processing:**
```bash
python -m ai_video_editor.cli.main process video1.mp4 video2.mp4 video3.mp4 \
  --type general \
  --quality high \
  --parallel \
  --output ./batch_output \
  --max-memory 16
```

## ⚙️ Configuration

### Environment Variables

Key configuration options:

```bash
# API Configuration
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_key
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_key
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project

# Performance Settings
AI_VIDEO_EDITOR_MAX_MEMORY_USAGE_GB=8
AI_VIDEO_EDITOR_MAX_CONCURRENT_PROCESSES=2
AI_VIDEO_EDITOR_ENABLE_CACHING=true

# Quality Settings
AI_VIDEO_EDITOR_DEFAULT_QUALITY=high
AI_VIDEO_EDITOR_DEFAULT_MODE=balanced
```

### Configuration File

Create `config.yaml` for persistent settings:

```yaml
# AI Video Editor Configuration
api:
  gemini_key: ${AI_VIDEO_EDITOR_GEMINI_API_KEY}
  imagen_key: ${AI_VIDEO_EDITOR_IMAGEN_API_KEY}
  project_id: ${AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT}

processing:
  default_quality: high
  default_mode: balanced
  max_memory_gb: 8
  enable_parallel: true
  enable_caching: true

output:
  default_directory: ./output
  preserve_intermediate: false
  generate_analytics: true
```

### System-Specific Configuration

#### Memory-Constrained Systems (8GB RAM)

```bash
# For systems with limited RAM
python -m ai_video_editor.cli.main process video.mp4 \
  --max-memory 4 \
  --mode fast \
  --quality medium
```

#### High-Performance Systems (16GB+ RAM)

```bash
# For powerful systems
python -m ai_video_editor.cli.main process video.mp4 \
  --max-memory 16 \
  --parallel \
  --quality ultra \
  --mode high_quality
```

### Troubleshooting Configuration

#### Common Issues

**"API key not found"**
- Verify your `.env` file contains the correct API keys
- Check that the file is in the project root directory

**"Out of memory"**
- Reduce `--max-memory` setting
- Use `--mode fast` for lower memory usage
- Process shorter video segments

**"Processing timeout"**
- Increase `--timeout` value
- Check internet connection for API calls
- Try processing with `--mode fast`

#### Getting Help

```bash
# View detailed help
python -m ai_video_editor.cli.main --help

# Check system status
python -m ai_video_editor.cli.main status

# View workflow status
python -m ai_video_editor.cli.main workflow --list
```

## 💡 Tips for Best Results

1. **Use high-quality source videos** (1080p+ recommended)
2. **Ensure clear audio** for better transcription
3. **Choose appropriate content type** for optimization
4. **Monitor system resources** during processing
5. **Start with shorter videos** to test settings

## 📚 Related Documentation

### Next Steps
- **[First Video Tutorial](../tutorials/first-video.md)** - Complete walkthrough for your first video
- **[Workflow Guides](../../tutorials/README.md)** - Content-specific processing tutorials
- **[Understanding Output](../tutorials/understanding-output.md)** - Learn about generated files

### Support Resources
- **[Troubleshooting Guide](../../support/troubleshooting-unified.md)** - Common issues and solutions
- **[FAQ](../../support/faq-unified.md)** - Frequently asked questions
- **[Performance Guide](../../support/performance-unified.md)** - Optimization and tuning

### Developer Resources
- **[API Reference](../developer/api-reference.md)** - Complete API documentation
- **[Architecture Guide](../developer/architecture.md)** - System design and components
- **[Contributing Guide](../developer/contributing.md)** - Development workflow

## 🔗 Quick Links

| I want to... | Go to... |
|---------------|----------|
| **Process my first video** | [First Video Tutorial](../tutorials/first-video.md) |
| **Learn educational workflows** | [Educational Content Guide](../tutorials/workflows/educational-content.md) |
| **Optimize performance** | [Performance Guide](../../support/performance-unified.md) |
| **Fix an issue** | [Troubleshooting](../../support/troubleshooting-unified.md) |
| **Understand the output** | [Understanding Output](../tutorials/understanding-output.md) |
| **Contribute to the project** | [Contributing Guide](../developer/contributing.md) |

---

*Master the AI Video Editor and create amazing content!*