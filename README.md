# AI Video Editor

An AI-assisted video editing and content optimization system that transforms raw video content into engaging, discoverable content across multiple platforms.

## Features

- **Unified ContentContext Architecture**: All modules share data and insights for deep integration
- **AI-Powered Analysis**: Uses Whisper Large-V3, Gemini API, and Imagen API for intelligent content processing
- **Integrated Output Generation**: Synchronized thumbnail and metadata generation
- **Multi-Platform Optimization**: Optimized for YouTube, Instagram, and other platforms
- **Content-Type Aware Processing**: Specialized workflows for educational, music, and general content

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-video-editor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize configuration:
```bash
python -m ai_video_editor.cli.main init
```

4. Edit the generated `.env` file to add your API keys:
```bash
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_gemini_api_key_here
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_api_key_here
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id_here
```

## Quick Start

1. Check system status:
```bash
python -m ai_video_editor.cli.main status
```

2. Process a video file:
```bash
python -m ai_video_editor.cli.main process video.mp4 --type educational --quality high
```

3. Analyze audio content:
```bash
python -m ai_video_editor.cli.main analyze audio.mp3 --output transcript.txt
```

4. Enhance video quality:
```bash
python -m ai_video_editor.cli.main enhance input.mp4 --output enhanced.mp4
```

## Project Structure

```
ai_video_editor/
├── cli/                    # Command-line interface
├── core/                   # Core functionality
│   ├── config.py          # Configuration management
│   └── exceptions.py      # Custom exceptions
├── modules/                # Processing modules
│   ├── content_analysis/  # Audio/video analysis
│   ├── enhancement/       # Quality enhancement
│   ├── thumbnail_generation/ # Thumbnail creation
│   └── video_processing/  # Video editing
└── utils/                 # Utility functions
    └── logging_config.py  # Logging configuration

tests/
├── unit/                  # Unit tests
└── integration/           # Integration tests
```

## Development

### Running Tests

Run all tests:
```bash
python -m pytest tests/ -v
```

Run unit tests only:
```bash
python -m pytest tests/unit/ -v
```

Run integration tests only:
```bash
python -m pytest tests/integration/ -v
```

### Code Quality

The project uses:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing

### Configuration

The application supports configuration through:
- Environment variables (prefixed with `AI_VIDEO_EDITOR_`)
- `.env` files
- Command-line arguments

Key configuration options:
- `GEMINI_API_KEY`: Required for content analysis
- `IMAGEN_API_KEY`: Required for AI thumbnail generation
- `GOOGLE_CLOUD_PROJECT`: Required for cloud services
- `MAX_MEMORY_USAGE_GB`: Memory limit (default: 8GB)
- `MAX_CONCURRENT_PROCESSES`: Parallel processing limit (default: 2)

## Architecture

The system is built on a **ContentContext-driven architecture** where all modules share a unified data structure that flows through the entire processing pipeline. This enables:

- **Deep Integration**: Thumbnail generation uses the same emotional analysis as metadata creation
- **Consistent Output**: All generated assets are synchronized and aligned
- **Efficient Processing**: Shared insights reduce redundant API calls and processing
- **Error Recovery**: ContentContext preservation enables graceful error handling

## Requirements

- Python 3.9+
- 8GB+ RAM recommended
- GPU acceleration supported (optional)
- Internet connection for AI services

## Core Libraries

- **VideoLab-Pro**: Advanced video processing
- **Whisper Large-V3**: Speech-to-text conversion
- **OpenCV**: Computer vision and video analysis
- **Gemini API**: Content analysis and keyword research
- **Imagen API**: AI-powered thumbnail backgrounds
- **PyVips**: Professional image processing and text rendering

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Support

For issues and questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

## Roadmap

- [x] Project setup and basic infrastructure
- [ ] ContentContext system implementation
- [ ] Audio and video analysis modules
- [ ] AI-powered content analysis
- [ ] Integrated thumbnail and metadata generation
- [ ] Video processing and enhancement
- [ ] Performance optimization and caching
- [ ] Web UI development
- [ ] Advanced analytics and machine learning