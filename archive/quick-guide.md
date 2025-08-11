# AI Video Editor - Quick Start Guide

## 🚀 Quick Setup

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file with your API keys:
```bash
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_gemini_api_key_here
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_api_key_here
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id_here
```

### 3. Basic Usage
```bash
# Process a single video
python -m ai_video_editor.cli.main process video.mp4

# Process with specific settings
python -m ai_video_editor.cli.main process video.mp4 --type educational --quality high

# Check system status
python -m ai_video_editor.cli.main status
```

**Need detailed guidance?** See the [**Complete Tutorial Collection**](docs/tutorials/README.md) for step-by-step workflows.

## 🎯 Key Features

### AI-Powered Editing
- **AI Director**: Makes creative editing decisions automatically
- **Content Analysis**: Understands your video content and context
- **Smart Cuts**: Removes filler words and optimizes pacing
- **B-Roll Generation**: Creates charts and visual enhancements

### Professional Audio
- **Noise Reduction**: Cleans up background noise
- **Level Adjustment**: Balances audio levels dynamically
- **Synchronization**: Perfect audio-video sync
- **Enhancement**: Professional-grade audio processing

### SEO Optimization
- **Smart Titles**: AI-generated, SEO-optimized titles
- **Descriptions**: Engaging descriptions with trending keywords
- **Thumbnails**: Eye-catching thumbnail concepts
- **Tags**: Relevant tags for maximum discoverability

### Performance Features
- **Batch Processing**: Handle multiple videos efficiently
- **Smart Caching**: Faster processing with intelligent caching
- **Resource Monitoring**: Real-time system resource tracking
- **Cost Optimization**: Efficient API usage to minimize costs

## 📋 Content Types

### Educational Content
```bash
python -m ai_video_editor.cli.main process lecture.mp4 --type educational
```
→ [**Complete Educational Workflow**](docs/tutorials/workflows/educational-content.md)

### Music Videos
```bash
python -m ai_video_editor.cli.main process music.mp4 --type music
```
→ [**Complete Music Video Workflow**](docs/tutorials/workflows/music-videos.md)

### General Content
```bash
python -m ai_video_editor.cli.main process content.mp4 --type general
```
→ [**Complete General Content Workflow**](docs/tutorials/workflows/general-content.md)
- Smart pacing and cuts
- Professional polish
- SEO optimization

## ⚙️ Quality Settings

### Fast Processing
```bash
python -m ai_video_editor.cli.main process video.mp4 --quality fast
```
- Quick turnaround
- Basic enhancements
- Good for previews

### Balanced (Default)
```bash
python -m ai_video_editor.cli.main process video.mp4 --quality balanced
```
- Best quality/speed ratio
- Recommended for most use cases
- Professional results

### High Quality
```bash
python -m ai_video_editor.cli.main process video.mp4 --quality high
```
- Maximum quality output
- Detailed analysis
- Best for final production

## 🔧 Advanced Usage

### Batch Processing
```bash
# Process multiple videos
python -m ai_video_editor.cli.main process *.mp4 --parallel

# Custom output directory
python -m ai_video_editor.cli.main process video.mp4 --output ./results
```

### Audio Enhancement Only
```bash
python -m ai_video_editor.cli.main enhance audio.mp3 --output enhanced.wav
```

### Analysis Only
```bash
python -m ai_video_editor.cli.main analyze video.mp4 --output analysis.json
```

## 📊 Performance Tips

### System Optimization
- **Close other applications** to free up memory
- **Use SSD storage** for faster file access
- **Ensure stable internet** for AI services
- **Monitor resource usage** with built-in tools

### Processing Optimization
- **Use balanced quality** for most content
- **Process shorter segments** for very long videos
- **Enable caching** for repeated processing
- **Use batch mode** for multiple videos

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only (fast)
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Performance tests (slow)
python -m pytest tests/ -m performance -v
```

### Test with Different Markers
```bash
# Skip slow tests (default)
python -m pytest tests/ -m "not slow and not performance"

# Include all tests
python -m pytest tests/ -m ""
```

## 📚 Documentation Links

For more detailed information, see:
- **[Complete User Guide](docs/user-guide/README.md)** - Comprehensive documentation with installation, CLI reference, and workflows
- **[API Documentation](docs/api/README.md)** - Developer API reference
- **[Troubleshooting](docs/support/troubleshooting.md)** - Common issues and solutions

## 🆘 Common Issues

### API Key Issues
- Verify your API keys are correct
- Check your Google Cloud project settings
- Ensure billing is enabled for your project

### Memory Issues
- Close other applications
- Process shorter video segments
- Use fast quality mode for large files

### Performance Issues
- Check internet connection
- Monitor system resources
- Use balanced quality settings

## 💡 Tips for Best Results

1. **Start with balanced quality** - Good results, reasonable processing time
2. **Use appropriate content types** - Educational, music, or general
3. **Ensure good audio quality** - Better input = better output
4. **Monitor system resources** - Don't overload your system
5. **Use batch processing** - More efficient for multiple videos

## 🎯 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Close other apps, use fast quality |
| Slow processing | Check internet, use balanced quality |
| API errors | Verify API keys and billing |
| Import errors | Run `pip install -r requirements.txt` |
| Test failures | See test output for specific issues |

---

**Need more help?** Check the full documentation or create an issue on GitHub.