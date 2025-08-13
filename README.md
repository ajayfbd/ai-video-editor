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

## Getting Started

**New to AI Video Editor?** Get up and running in 5 minutes with our [**Quick Start Guide**](quick-start.md).

**Need detailed guidance?** Follow the complete [**First Video Tutorial**](docs/tutorials/first-video.md) for step-by-step instructions.

**Looking for comprehensive documentation?** Browse the [**Complete Documentation**](docs/README.md) for detailed guides and API reference.

## 🎙️ Enhanced Transcription System

The AI Video Editor includes a comprehensive **Sanskrit/Hindi transcription system** with:

- **580+ built-in vocabulary terms** (religious, mythological, classical, philosophical)
- **Granular segmentation control** (from natural flow to word-by-word breakdown)
- **Professional output formats** (JSON, SRT, VTT with dual Devanagari/romanized text)
- **Smart model selection** with automatic fallbacks and progress tracking

### Quick Transcription

```bash
# Launch transcription system
transcribe.bat

# Or directly use scripts
cd transcription/scripts
transcribe_hindi.bat "your_video.mp4" "output_name"
```

**📚 Full Documentation**: See [`transcription/README.md`](transcription/README.md) for complete usage guide.

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
    ├── logging_config.py  # Logging configuration
    └── sanskrit_hindi_vocab.py # Comprehensive vocabulary system

transcription/              # Enhanced transcription system
├── scripts/               # Executable transcription scripts
├── docs/                  # Transcription documentation
├── examples/              # Test and example scripts
└── output/                # Transcription outputs

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

For detailed configuration instructions, see the [**User Guide**](docs/user-guide/README.md#-configuration). The application supports environment variables, `.env` files, and command-line arguments for flexible setup.

## Architecture

The system is built on a **ContentContext-driven architecture** where all modules share a unified data structure that flows through the entire processing pipeline. This enables:

- **Deep Integration**: Thumbnail generation uses the same emotional analysis as metadata creation
- **Consistent Output**: All generated assets are synchronized and aligned
- **Efficient Processing**: Shared insights reduce redundant API calls and processing
- **Error Recovery**: ContentContext preservation enables graceful error handling

## Requirements

- Python 3.9+
- 8GB+ RAM recommended  
- Internet connection for AI services

For detailed system requirements and installation instructions, see the [**Quick Start Guide**](quick-start.md).

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
1. Check the [**Complete Documentation**](docs/README.md)
2. Review the [**Troubleshooting Guide**](docs/support/troubleshooting-unified.md)
3. Search existing issues
4. Create a new issue with detailed information

> **Note**: Some documentation files have been consolidated and archived. If you're looking for previously referenced files like `quick-guide.md` or analysis reports, they can be found in the `archive/` directory.

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

### Getting Started
- **[Quick Start Guide](quick-start.md)** - 5-minute setup and first video processing
- **[First Video Tutorial](docs/tutorials/first-video.md)** - Complete step-by-step walkthrough
- **[User Guide](docs/user-guide/README.md)** - Complete user documentation with installation, CLI reference, and workflows

### Developer Resources
- **[API Reference](docs/developer/api-reference.md)** - Complete API documentation and integration guides
- **[Architecture Guide](docs/developer/architecture.md)** - System design and components
- **[Contributing Guide](docs/developer/contributing.md)** - Development workflow and coding standards
- **[Testing Guide](docs/developer/testing.md)** - Testing strategies and best practices

### Tutorials and Workflows
- **[Tutorials Overview](docs/tutorials/README.md)** - Step-by-step workflows for all content types
- **[Educational Content](docs/tutorials/workflows/educational-content.md)** - Specialized workflow for educational videos
- **[Music Videos](docs/tutorials/workflows/music-videos.md)** - Optimized processing for music content
- **[General Content](docs/tutorials/workflows/general-content.md)** - Versatile workflow for mixed content

### Support and Troubleshooting
- **[Troubleshooting Guide](docs/support/troubleshooting-unified.md)** - Common issues and solutions
- **[FAQ](docs/support/faq-unified.md)** - Frequently asked questions
- **[Performance Guide](docs/support/performance-unified.md)** - Optimization and tuning
- **[Error Handling](docs/support/error-handling-unified.md)** - Error recovery and debugging

## 🎬 Usage Examples

For detailed usage examples and workflows, see:
- [**Quick Start Guide**](quick-start.md) - Basic usage and first video processing
- [**Educational Content Workflow**](docs/tutorials/workflows/educational-content.md) - Specialized processing for educational videos
- [**Music Video Workflow**](docs/tutorials/workflows/music-videos.md) - Optimized processing for music content
- [**General Content Workflow**](docs/tutorials/workflows/general-content.md) - Versatile processing for mixed content

## 🏗️ Architecture Overview

The system is built on a **ContentContext-driven architecture** where an AI Director makes all creative decisions, stored in a unified ContentContext that flows through specialized modules:

```
Input Analysis → AI Director Decisions → Asset Generation → Final Output
      ↓                    ↓                    ↓              ↓
  Audio/Video → Creative & Strategic → Thumbnails/Metadata → Complete Package
   Analysis        Decisions            B-roll/Composition
```

## 📊 Project Status

**Overall Completion: 97.4%** | **Test Coverage: 96.7%** (475/491 tests passing)

- ✅ **Phase 1: Core Processing** - 100% Complete (Audio/Video Analysis)
- ✅ **Phase 2: AI Intelligence** - 100% Complete (AI Director, Metadata, B-roll)  
- ✅ **Phase 3: Output Generation** - 100% Complete (Full video composition pipeline)
- ✅ **Phase 4: Integration & QA** - 95% Complete (Documentation consolidation in progress)

> 📋 **Detailed Status**: For comprehensive project analysis, test results, and recent fixes, see [Project Status](docs/support/project-status.md)

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