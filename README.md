# AI Video Editor

An AI-driven content creation system that transforms raw video into professionally edited, engaging, and highly discoverable content packages. At its core, an **AI Director** powered by the Gemini API makes nuanced creative and strategic decisions, driving the entire workflow from video editing to SEO-optimized metadata generation.

## 🎯 Key Features

- **AI Director**: Gemini-powered creative decisions for professional editing
- **ContentContext Architecture**: Unified data flow ensuring all components work in concert
- **Professional Video Composition**: Movis-based editing with intelligent cuts and transitions
- **Intelligent B-Roll Generation**: Automated charts, animations, and visual enhancements
- **Synchronized Thumbnails & Metadata**: SEO-optimized packages with A/B testing support
- **Multi-Content Optimization**: Specialized workflows for educational, music, and general content
- **Performance Optimized**: Efficient resource usage with intelligent caching

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

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Getting Started](docs/user-guide/getting-started.md)** - Installation and first video
- **[User Guide](docs/user-guide/README.md)** - Complete user documentation
- **[CLI Reference](docs/user-guide/cli-reference.md)** - Command-line interface guide
- **[API Reference](docs/api/README.md)** - Developer API documentation
- **[Tutorials](docs/tutorials/README.md)** - Step-by-step workflows
- **[Architecture Guide](docs/developer/architecture.md)** - System design and components
- **[Troubleshooting](docs/support/troubleshooting.md)** - Common issues and solutions
- **[FAQ](docs/support/faq.md)** - Frequently asked questions
- **[Performance Guide](docs/support/performance.md)** - Optimization and tuning

## 🎬 Quick Examples

### Educational Content
```bash
# Process educational video with all optimizations
python -m ai_video_editor.cli.main process lecture.mp4 \
  --type educational \
  --quality high \
  --output ./educational_output
```

### Music Videos
```bash
# Process music content with beat synchronization
python -m ai_video_editor.cli.main process music_video.mp4 \
  --type music \
  --quality ultra \
  --mode balanced
```

### Batch Processing
```bash
# Process multiple videos efficiently
python -m ai_video_editor.cli.main process *.mp4 \
  --type general \
  --parallel \
  --output ./batch_output
```

## 🏗️ Architecture Overview

The system is built on a **ContentContext-driven architecture** where an AI Director makes all creative decisions, stored in a unified ContentContext that flows through specialized modules:

```
Input Analysis → AI Director Decisions → Asset Generation → Final Output
      ↓                    ↓                    ↓              ↓
  Audio/Video → Creative & Strategic → Thumbnails/Metadata → Complete Package
   Analysis        Decisions            B-roll/Composition
```

## 📊 Project Status

**Overall Completion: 75%** | **Test Coverage: 96.7%** (475/491 tests passing)

- ✅ **Phase 1: Core Processing** - 100% Complete (Audio/Video Analysis)
- ✅ **Phase 2: AI Intelligence** - 100% Complete (AI Director, Metadata, B-roll)  
- ✅ **Phase 3: Output Generation** - 100% Complete (Full video composition pipeline)
- ❌ **Phase 4: Integration & QA** - 75% Complete (Documentation and examples added)

## 🚀 Roadmap

### Completed
- [x] ContentContext system implementation
- [x] Audio and video analysis modules (Whisper, OpenCV)
- [x] AI Director with Gemini API integration
- [x] Intelligent B-roll generation (charts, animations)
- [x] Professional video composition with movis
- [x] Synchronized thumbnail and metadata generation
- [x] Performance optimization and caching
- [x] Comprehensive testing framework (96.7% coverage)
- [x] Complete documentation and examples

### In Progress
- [ ] End-to-end workflow orchestration
- [ ] Advanced performance optimization
- [ ] Production deployment guides

### Future
- [ ] Web UI development
- [ ] Advanced analytics and machine learning
- [ ] Multi-platform distribution tools