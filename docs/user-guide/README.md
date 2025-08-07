# AI Video Editor User Guide

Complete guide to using the AI Video Editor for creating professional, engaging video content with AI assistance.

## 📖 Table of Contents

1. [**Getting Started**](getting-started.md) - Installation and first video
2. [**Core Concepts**](#core-concepts) - Understanding the system
3. [**Processing Workflows**](#processing-workflows) - Different use cases
4. [**Content Types**](#content-types) - Optimized processing modes
5. [**Output Management**](#output-management) - Understanding results
6. [**Advanced Features**](#advanced-features) - Power user options
7. [**Performance Tuning**](#performance-tuning) - Optimization tips

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

## 📚 Related Documentation

- [**CLI Reference**](cli-reference.md) - Complete command documentation
- [**Configuration Guide**](configuration.md) - Advanced settings
- [**Troubleshooting**](../support/troubleshooting.md) - Common issues
- [**Tutorials**](../tutorials/README.md) - Step-by-step workflows

---

*Master the AI Video Editor and create amazing content!*