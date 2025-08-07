# Getting Started with AI Video Editor

Welcome to the AI Video Editor! This guide will help you install, configure, and process your first video in under 10 minutes.

## 📋 Prerequisites

- **Python 3.9+** (Python 3.10+ recommended)
- **8GB+ RAM** (16GB recommended for high-quality processing)
- **Internet connection** for AI services
- **API Keys** for Gemini and Imagen (see [API Setup](#api-setup))

## 🚀 Quick Installation

### 1. Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd ai-video-editor

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Configuration

```bash
# Create default configuration
python -m ai_video_editor.cli.main init

# This creates a .env file with default settings
```

### 3. API Setup

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

#### Getting API Keys

**Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

**Imagen API Key:**
1. Enable the Imagen API in [Google Cloud Console](https://console.cloud.google.com/)
2. Create a service account key
3. Set the key in your `.env` file

### 4. Verify Installation

```bash
# Check system status
python -m ai_video_editor.cli.main status
```

You should see all components marked as "✅ Ready".

## 🎬 Process Your First Video

### Basic Processing

```bash
# Process a video with default settings
python -m ai_video_editor.cli.main process video.mp4
```

### Educational Content (Recommended)

```bash
# Process educational content with high quality
python -m ai_video_editor.cli.main process lecture.mp4 \
  --type educational \
  --quality high \
  --output ./output
```

### Music Video

```bash
# Process music content (optimized workflow)
python -m ai_video_editor.cli.main process music_video.mp4 \
  --type music \
  --quality ultra \
  --mode fast
```

## 📊 Understanding the Output

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

## ⚙️ Configuration Options

### Content Types

- **`educational`**: Optimized for tutorials, lectures, explanations
- **`music`**: Optimized for music videos, performances
- **`general`**: Balanced processing for mixed content

### Quality Levels

- **`low`**: Fast processing, basic enhancements
- **`medium`**: Balanced quality and speed
- **`high`**: Professional quality (recommended)
- **`ultra`**: Maximum quality, slower processing

### Processing Modes

- **`fast`**: Prioritizes speed over quality
- **`balanced`**: Optimal balance (default)
- **`high_quality`**: Prioritizes quality over speed

## 🔧 Common Configuration

### Memory-Constrained Systems

```bash
# For systems with limited RAM
python -m ai_video_editor.cli.main process video.mp4 \
  --max-memory 4 \
  --mode fast \
  --quality medium
```

### High-Performance Systems

```bash
# For powerful systems
python -m ai_video_editor.cli.main process video.mp4 \
  --max-memory 16 \
  --parallel \
  --quality ultra \
  --mode high_quality
```

## 🆘 Troubleshooting

### Common Issues

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

### Getting Help

```bash
# View detailed help
python -m ai_video_editor.cli.main --help

# Check system status
python -m ai_video_editor.cli.main status

# View workflow status
python -m ai_video_editor.cli.main workflow --list
```

## 📚 Next Steps

- **[User Guide](README.md)**: Complete feature documentation
- **[CLI Reference](cli-reference.md)**: Detailed command options
- **[Configuration Guide](configuration.md)**: Advanced settings
- **[Tutorials](../tutorials/README.md)**: Step-by-step workflows

## 💡 Tips for Best Results

1. **Use high-quality source videos** (1080p+ recommended)
2. **Ensure clear audio** for better transcription
3. **Choose appropriate content type** for optimization
4. **Monitor system resources** during processing
5. **Start with shorter videos** to test settings

---

*Ready to create amazing content? Let's get started!*