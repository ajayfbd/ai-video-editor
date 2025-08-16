# Contributing to AI Video Editor

Thank you for your interest in contributing to AI Video Editor! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ai-video-editor.git
   cd ai-video-editor
   ```
3. **Set up development environment**:
   ```bash
   pip install -e ".[dev]"
   cp .env.example .env
   # Edit .env with your API keys
   ```
4. **Run tests** to ensure everything works:
   ```bash
   python -m pytest tests/ -v
   ```

## 🎯 Ways to Contribute

### 🐛 Bug Reports
- Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include detailed reproduction steps
- Provide system information and logs
- Check existing issues first

### ✨ Feature Requests
- Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Explain the use case and benefits
- Consider implementation complexity
- Discuss with maintainers first for large features

### 📝 Documentation
- Fix typos and improve clarity
- Add examples and tutorials
- Update API documentation
- Translate documentation (future)

### 🔧 Code Contributions
- Bug fixes
- Performance improvements
- New features (discuss first)
- Test coverage improvements

## 📋 Development Guidelines

### Code Style
- **Python**: Follow PEP 8 with Black formatting
- **Line Length**: 88 characters (Black default)
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style for all public functions/classes

### Architecture Principles
- **ContentContext Integration**: All modules must use the shared ContentContext
- **Error Handling**: Use ContentContextError base class with context preservation
- **Performance**: Follow memory and processing constraints (see performance-guidelines.md)
- **Testing**: Comprehensive unit tests with mocking for external APIs

### Code Quality Tools
```bash
# Format code
black ai_video_editor/ tests/

# Check linting
flake8 ai_video_editor/ tests/

# Type checking
mypy ai_video_editor/

# Run all quality checks
pre-commit run --all-files
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/                  # Unit tests with mocking
├── integration/           # Integration tests
└── fixtures/             # Test data and fixtures
```

### Writing Tests
- **Unit Tests**: Mock all external dependencies (APIs, file I/O)
- **Integration Tests**: Test data flow between modules
- **Performance Tests**: Benchmark critical operations
- **Coverage**: Aim for 90%+ test coverage

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# With coverage
python -m pytest tests/ --cov=ai_video_editor --cov-report=html
```

## 🏗️ Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Development Process
1. **Write tests first** (TDD approach recommended)
2. **Implement feature** following architecture guidelines
3. **Update documentation** as needed
4. **Run quality checks** before committing

### 3. Commit Guidelines
- **Format**: `type(scope): description`
- **Types**: feat, fix, docs, style, refactor, test, chore
- **Examples**:
  ```
  feat(audio): add Whisper model selection
  fix(thumbnail): resolve memory leak in generation
  docs(api): update ContentContext documentation
  ```

### 4. Pull Request Process
1. **Update your branch** with latest main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
3. **Create Pull Request** using the template
4. **Address review feedback** promptly
5. **Squash commits** if requested

## 📚 Architecture Overview

### ContentContext System
All modules operate on a shared `ContentContext` object that flows through the pipeline:

```python
@dataclass
class ContentContext:
    # Raw Input Data
    video_files: List[str]
    audio_transcript: Transcript
    
    # Analysis Results
    emotional_markers: List[EmotionalPeak]
    key_concepts: List[str]
    
    # Generated Assets
    thumbnail_concepts: List[ThumbnailConcept]
    metadata_variations: List[MetadataSet]
```

### Module Integration
- **Input Processing**: Audio/video analysis and transcription
- **AI Director**: Creative and strategic decisions using Gemini API
- **Output Generation**: Video composition, thumbnails, and metadata

### Error Handling
```python
try:
    with preserve_context_on_error(context, "checkpoint_name"):
        # Risky operation
        result = process_module(context)
except ContentContextError as e:
    # Handle with context preservation
    context = e.context_state
```

## 🔒 Security Guidelines

### API Keys and Secrets
- **Never commit** API keys or secrets
- **Use environment variables** for all sensitive data
- **Template files**: Use `.env.example` for configuration templates
- **Validate inputs**: Sanitize all user inputs

### Dependencies
- **Pin versions** in requirements files
- **Security scanning**: Use `safety check` for known vulnerabilities
- **Regular updates**: Keep dependencies current

## 📖 Documentation Standards

### Code Documentation
- **Docstrings**: All public functions and classes
- **Type hints**: Required for all parameters and return values
- **Examples**: Include usage examples in docstrings

### User Documentation
- **Clear instructions**: Step-by-step guides
- **Code examples**: Working examples for all features
- **Troubleshooting**: Common issues and solutions

## 🎯 Performance Guidelines

### Resource Constraints
- **Memory**: Stay under 16GB peak usage
- **Processing**: Educational content (15+ min) in under 10 minutes
- **API Costs**: Under $2 per project average

### Optimization Strategies
- **Parallel processing**: Use asyncio for independent operations
- **Caching**: Implement intelligent caching for repeated operations
- **Batch operations**: Group similar API calls

## 🤝 Community Guidelines

### Code of Conduct
- **Be respectful** and inclusive
- **Constructive feedback** only
- **Help others** learn and grow
- **Follow project guidelines**

### Communication
- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

## 🏷️ Release Process

### Version Numbering
- **Semantic Versioning**: MAJOR.MINOR.PATCH
- **Breaking changes**: Increment MAJOR
- **New features**: Increment MINOR
- **Bug fixes**: Increment PATCH

### Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Release notes prepared

## 📞 Getting Help

### Resources
- **Documentation**: [docs/](docs/)
- **API Reference**: [docs/developer/api-reference.md](docs/developer/api-reference.md)
- **Troubleshooting**: [docs/support/troubleshooting-unified.md](docs/support/troubleshooting-unified.md)

### Contact
- **Issues**: GitHub Issues for bugs and features
- **Discussions**: GitHub Discussions for questions
- **Security**: See [SECURITY.md](SECURITY.md) for security issues

## 🙏 Recognition

Contributors are recognized in:
- **README.md**: Major contributors
- **CHANGELOG.md**: Release contributions
- **GitHub**: Contributor graphs and statistics

Thank you for contributing to AI Video Editor! 🎬✨