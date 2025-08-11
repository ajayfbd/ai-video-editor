# AI Video Editor - Quick Start Guide

Get up and running with the AI Video Editor in under 5 minutes. This guide covers the essential setup and basic usage to process your first video.

The AI Video Editor is an AI-driven content creation system that transforms raw video into professionally edited, engaging, and highly discoverable content packages. At its core, an **AI Director** powered by the Gemini API makes nuanced creative and strategic decisions, driving the entire workflow from video editing to SEO-optimized metadata generation.

## 🚀 5-Minute Setup

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd ai-video-editor
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file with your API keys:
```bash
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_gemini_api_key_here
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_api_key_here
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id_here
```

### 3. Initialize and Verify

Initialize the configuration and check system status:
```bash
python -m ai_video_editor.cli.main init
python -m ai_video_editor.cli.main status
```

### 4. Process Your First Video

Process a video with default settings:
```bash
python -m ai_video_editor.cli.main process video.mp4
```

**That's it!** Your video will be processed with AI-powered editing, professional enhancements, and SEO-optimized metadata.

**Want a complete step-by-step guide?** See the [**First Video Tutorial**](docs/tutorials/first-video.md) for detailed instructions, troubleshooting, and optimization tips.

**Looking for more advanced features?** Browse the [**Complete Documentation**](docs/README.md) or jump to specific [**Workflow Guides**](docs/tutorials/README.md#-complete-workflow-guides).



> **Implementation Status**: The AI Video Editor is currently in active development. Core features like audio processing and workflow orchestration are implemented, while advanced features like B-roll generation and thumbnail creation are planned for future releases.

## 🎯 What You Get

The AI Video Editor transforms your raw video into a complete content package:

- **Enhanced Video**: Professional editing with intelligent cuts and pacing
- **Multiple Thumbnails**: Planned feature for AI-generated thumbnails  
- **SEO Metadata**: Optimized titles, descriptions, and tags for discoverability
- **B-Roll Content**: Planned feature for automated charts and animations

## 📋 Content-Specific Processing

**Educational Content** (tutorials, lectures)
```bash
python -m ai_video_editor.cli.main process lecture.mp4 --type educational
```

**Music Videos** (performances, music content)
```bash
python -m ai_video_editor.cli.main process music.mp4 --type music
```

**General Content** (vlogs, mixed content)
```bash
python -m ai_video_editor.cli.main process content.mp4 --type general
```

**Learn more:** [Complete Workflow Guides](docs/tutorials/README.md#-complete-workflow-guides) | [Advanced Techniques](docs/tutorials/advanced/)

## ⚙️ Quality Settings

Choose the right balance of quality and processing speed:

```bash
# Fast processing (good for previews)
python -m ai_video_editor.cli.main process video.mp4 --quality low --mode fast

# Balanced processing (recommended)
python -m ai_video_editor.cli.main process video.mp4 --quality medium --mode balanced

# High quality (best for final production)
python -m ai_video_editor.cli.main process video.mp4 --quality high --mode high_quality

# Ultra quality (maximum settings)
python -m ai_video_editor.cli.main process video.mp4 --quality ultra --mode high_quality
```

## 🔧 Essential Commands

### Basic Processing
```bash
# Process with specific content type
python -m ai_video_editor.cli.main process video.mp4 --type educational --quality high

# Custom output directory
python -m ai_video_editor.cli.main process video.mp4 --output ./results
```

### Quick Operations
```bash
# Audio enhancement only
python -m ai_video_editor.cli.main process input.mp4 --quality high --output enhanced.mp4

# Content analysis only
python -m ai_video_editor.cli.main process audio.mp3 --type general --output transcript.txt

# Batch processing (parallel by default)
python -m ai_video_editor.cli.main process *.mp4 --output ./batch_results
```

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Close other apps, use `--quality low --mode fast` |
| Slow processing | Check internet connection, use `--quality medium --mode balanced` |
| API errors | Verify API keys and Google Cloud billing |
| Import errors | Run `pip install -r requirements.txt` |

## 📊 Performance Tips

- **Close other applications** to free up memory (8GB+ RAM recommended)
- **Use SSD storage** for faster file access
- **Ensure stable internet** for AI services
- **Start with balanced quality** for best results/speed ratio
- **Monitor system resources** with built-in tools
- **Use batch mode** for multiple videos

## 🧪 Verify Installation

Run the test suite to ensure everything is working:
```bash
# Quick unit tests (fast)
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Full test suite (slower)
python -m pytest tests/ -v
```

## 📚 Next Steps

Once you've processed your first video, explore more advanced features:

### User Documentation
- **[Complete User Guide](docs/user-guide/README.md)** - Comprehensive documentation with installation, CLI reference, and workflows
- **[Tutorials Overview](docs/tutorials/README.md)** - Step-by-step workflows for different content types
- **[Understanding Output](docs/tutorials/understanding-output.md)** - Learn about generated files and optimization

### Developer Resources
- **[API Documentation](docs/developer/api-reference.md)** - Complete API reference and integration guides
- **[Architecture Guide](docs/developer/architecture.md)** - System design and components
- **[Contributing Guide](docs/developer/contributing.md)** - Development guidelines and workflow
- **[Testing Guide](docs/developer/testing.md)** - Testing strategies and best practices

### Support & Troubleshooting
- **[Troubleshooting Guide](docs/support/troubleshooting-unified.md)** - Common issues and solutions
- **[Performance Guide](docs/support/performance-unified.md)** - Optimization and tuning
- **[FAQ](docs/support/faq-unified.md)** - Frequently asked questions
- **[Error Handling](docs/support/error-handling-unified.md)** - Error recovery and debugging

## 🏗️ System Requirements

- **Python**: 3.9+ required
- **Memory**: 8GB+ RAM recommended
- **Storage**: SSD recommended for better performance
- **Internet**: Required for AI services (Gemini, Imagen APIs)
- **GPU**: Optional but supported for acceleration

### Core Libraries
- **Whisper Large-V3**: Speech-to-text conversion
- **OpenCV**: Computer vision and video analysis
- **Gemini API**: Content analysis and keyword research
- **Imagen API**: AI-powered thumbnail backgrounds

## 💡 Pro Tips

1. **Start with educational or general content types** - they're the most versatile
2. **Use balanced quality** for your first few videos to get familiar with the output
3. **Monitor system resources** during processing to optimize performance
4. **Keep your API keys secure** and never commit them to version control
5. **Process shorter segments first** (under 10 minutes) to get familiar with the workflow

---

**Need help?** Check the [troubleshooting guide](docs/support/troubleshooting-unified.md), browse the [FAQ](docs/support/faq-unified.md), or [create an issue](https://github.com/your-repo/issues) with detailed information about your setup and the problem you're experiencing.

**Ready for more?** Continue with the [First Video Tutorial](docs/tutorials/first-video.md) or explore [Advanced Techniques](docs/tutorials/advanced/).