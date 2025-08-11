# Contributing Guide

Comprehensive guide for contributing to the AI Video Editor project, including development workflow, coding standards, and collaboration patterns.

## 📚 Table of Contents

1. [**Development Setup**](#development-setup)
2. [**Collaborative Development Workflow**](#collaborative-development-workflow)
3. [**Coding Standards**](#coding-standards)
4. [**ContentContext Integration**](#contentcontext-integration)
5. [**Testing Requirements**](#testing-requirements)
6. [**Performance Guidelines**](#performance-guidelines)
7. [**Documentation Standards**](#documentation-standards)
8. [**Submission Process**](#submission-process)

## 🚀 Development Setup

### Prerequisites

- **Python 3.11+** with pip and virtual environment support
- **Git** for version control
- **API Keys** for Gemini and Imagen services
- **Development Tools**: pytest, black, flake8, mypy

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-org/ai-video-editor.git
cd ai-video-editor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### API Configuration

```bash
# Required API Keys in .env file
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_gemini_api_key_here
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_api_key_here
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id_here

# Optional configuration
AI_VIDEO_EDITOR_LOG_LEVEL=INFO
AI_VIDEO_EDITOR_CACHE_DIR=./cache
AI_VIDEO_EDITOR_OUTPUT_DIR=./output
```

### Verification

```bash
# Test environment setup
python test_gemini_access.py
python -m pytest tests/unit/test_config.py -v

# Verify ContentContext integration
python -c "from ai_video_editor.core.content_context import ContentContext; print('Setup successful!')"
```

## 🤝 Collaborative Development Workflow

### Team Structure and Roles

The AI Video Editor project uses an orchestrated multi-AI development approach:

**Kiro (Orchestrator & Architect)**
- Architectural guardian and quality assurance
- Creates detailed task specifications following architectural patterns
- Reviews implementations for ContentContext integration compliance
- Ensures alignment with AI Video Editor goals and requirements

**Gemini Flash 2.5 (Development Implementation)**
- Heavy code generation and development tasks
- Implements code following detailed specifications
- Generates comprehensive unit tests with mocking strategies
- Handles iterative refinements based on feedback

**Gemini 2.5 Pro (Production AI Processing)**
- AI Director decisions and content analysis
- Video processing and quality assessment
- Content understanding and emotional analysis
- **Critical**: Always use Pro for production to ensure best reasoning quality

### Workflow Process

#### Phase 1: Task Specification
1. **Read Current Context**: Load architectural patterns from `.kiro/shared_memory.json`
2. **Extract Task Details**: Parse requirements from `.kiro/specs/ai-video-editor/tasks.md`
3. **Create Detailed Specification**: Include ContentContext integration, error handling, testing requirements
4. **Store Specification**: Update shared memory with task specification
5. **Delegate Implementation**: Always delegate to appropriate AI system

#### Phase 2: Implementation
1. **Research Phase**: Use available tools to find current best practices
2. **Read Architectural Context**: Load patterns and requirements from shared memory
3. **Generate Implementation**: Create production-ready code following specifications
4. **Include Comprehensive Testing**: Unit tests with proper mocking
5. **Store Results**: Update shared memory with implementation for review

#### Phase 3: Review and Integration
1. **Architectural Compliance Check**: Verify ContentContext integration patterns
2. **Goal Alignment Validation**: Ensure implementation advances project objectives
3. **Integration Assessment**: Confirm compatibility with existing modules
4. **Performance Evaluation**: Validate against resource constraints
5. **Quality Standards Review**: Check code quality, testing, and documentation

### Shared Memory System

```json
{
  "architectural_patterns": {
    "contentcontext_integration": "...",
    "error_handling": "...",
    "testing_strategy": "..."
  },
  "current_task_execution": {
    "task_number": "4.1",
    "specification": "...",
    "implementation": "...",
    "status": "in_progress"
  },
  "collaboration_history": {
    "completed_tasks": [],
    "lessons_learned": [],
    "optimization_insights": []
  }
}
```

### Model Selection Guidelines

- **Development Tasks**: Use Gemini Flash 2.5 for speed and efficiency
- **Production AI Processing**: Always use Gemini 2.5 Pro for best quality reasoning
- **Code Generation**: Flash 2.5 is sufficient for implementation tasks
- **AI Director Decisions**: Pro 2.5 required for production video processing

## 📝 Coding Standards

### Code Style

**Python Style Guide**
- Follow PEP 8 with line length of 88 characters (Black default)
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public functions and classes
- Use meaningful variable and function names

**Code Formatting**
```bash
# Format code with Black
black ai_video_editor/ tests/

# Check style with flake8
flake8 ai_video_editor/ tests/

# Type checking with mypy
mypy ai_video_editor/
```

### Code Structure

**Module Organization**
```
ai_video_editor/
├── core/                    # Core system components
│   ├── content_context.py   # ContentContext definition
│   ├── workflow_orchestrator.py
│   └── exceptions.py
├── modules/                 # Processing modules
│   ├── content_analysis/
│   ├── intelligence/
│   └── output_generation/
├── utils/                   # Utility functions
└── cli/                     # Command-line interface
```

**Class Design Patterns**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class ContentContext:
    """Central data structure for all processing."""
    project_id: str
    video_files: List[str]
    # ... other fields

class BaseProcessor(ABC):
    """Base class for all processing modules."""
    
    @abstractmethod
    async def process(self, context: ContentContext) -> ContentContext:
        """Process the content context."""
        pass
    
    def validate_input(self, context: ContentContext) -> bool:
        """Validate input context."""
        return True
```

### Error Handling Standards

**Exception Hierarchy**
```python
class ContentContextError(Exception):
    """Base exception for ContentContext-related errors."""
    def __init__(self, message: str, context_state: Optional[ContentContext] = None):
        super().__init__(message)
        self.context_state = context_state
        self.recovery_checkpoint = None

class APIIntegrationError(ContentContextError):
    """Raised when external API calls fail."""
    pass

class ResourceConstraintError(ContentContextError):
    """Raised when system resources are insufficient."""
    pass
```

**Error Handling Pattern**
```python
from contextlib import contextmanager

@contextmanager
def preserve_context_on_error(context: ContentContext, checkpoint_name: str):
    try:
        context_manager.save_checkpoint(context, checkpoint_name)
        yield context
    except Exception as e:
        restored_context = context_manager.load_checkpoint(context.project_id, checkpoint_name)
        raise ContentContextError(f"Processing failed: {str(e)}", restored_context)
```

## 🏗️ ContentContext Integration

### Core Principle

All modules must operate on the shared ContentContext object that flows through the entire pipeline. This ensures deep integration and prevents data silos.

### Integration Requirements

**Mandatory Patterns**
1. **ContentContext Flow**: All processing functions must accept and return ContentContext
2. **Data Preservation**: Never lose or corrupt existing ContentContext data
3. **State Tracking**: Update processing metrics and status information
4. **Error Context**: Preserve ContentContext state during error scenarios

**Implementation Example**
```python
async def process_audio_analysis(context: ContentContext) -> ContentContext:
    """Process audio analysis while preserving ContentContext integrity."""
    
    # Validate input
    if not context.video_files:
        raise ContentContextError("No video files provided", context)
    
    try:
        # Save checkpoint
        with preserve_context_on_error(context, "before_audio_analysis"):
            # Perform audio analysis
            audio_result = await analyze_audio(context.video_files[0])
            
            # Update context with results
            context.audio_analysis = audio_result
            context.key_concepts.extend(audio_result.concepts)
            context.emotional_markers.extend(audio_result.emotional_peaks)
            
            # Update processing metrics
            context.processing_metrics.add_stage_completion("audio_analysis")
            
            return context
            
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        raise ContentContextError(f"Audio analysis failed: {e}", context)
```

### Data Synchronization

**Thumbnail-Metadata Synchronization**
- Thumbnail hook text derived from same emotional analysis used for titles
- Visual concepts identified for thumbnails inform metadata descriptions
- Trending keywords influence both thumbnail text overlays and metadata tags

**Shared Research Layer**
- Keyword research performed once and shared across all output generation
- Competitor analysis informs both thumbnail concepts and metadata strategies
- Trend analysis cached and reused across multiple generation requests

## 🧪 Testing Requirements

### Testing Philosophy

**Unit Testing with Comprehensive Mocking**
- Focus on unit tests with sophisticated mocking strategies
- Mock external APIs (Gemini, Whisper, OpenCV) with realistic responses
- Ensure fast test execution and reliable results
- Maintain 90%+ test coverage

### Test Structure

**Required Test Categories**
```python
class TestModuleName:
    """Test suite for module functionality."""
    
    def test_contentcontext_integration(self, sample_context):
        """Test ContentContext integration and data flow."""
        result = module_function(sample_context)
        assert isinstance(result, ContentContext)
        assert result.project_id == sample_context.project_id
    
    def test_error_handling(self, sample_context):
        """Test error handling and context preservation."""
        with pytest.raises(ContentContextError) as exc_info:
            module_function_with_error(sample_context)
        assert exc_info.value.context_state is not None
    
    def test_performance_constraints(self, sample_context):
        """Test performance meets requirements."""
        start_time = time.time()
        result = module_function(sample_context)
        processing_time = time.time() - start_time
        assert processing_time < MAX_PROCESSING_TIME
```

**Mock Patterns**
```python
@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client with realistic responses."""
    client = Mock()
    client.generate_content.return_value = Mock(
        text=json.dumps({
            "content_analysis": {"key_concepts": ["test", "example"]},
            "trending_keywords": ["test keyword", "example phrase"]
        })
    )
    return client

@pytest.fixture
def sample_context():
    """Sample ContentContext for testing."""
    return ContentContext(
    project_id="example_project",
    video_files=["video.mp4"],
    content_type=ContentType.GENERAL,
    user_preferences=UserPreferences()
)
```

### Test Execution

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=ai_video_editor --cov-report=html

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest -m "not performance"  # Skip slow tests
```

## ⚡ Performance Guidelines

### Resource Management

**Memory Constraints**
- ContentContext size limit: 500MB per project
- Module memory usage: < 2GB per module
- Peak system memory: < 16GB
- Implement explicit cleanup after processing stages

**Processing Time Targets**
- Educational content (15+ min): < 10 minutes processing
- Music videos (5-6 min): < 5 minutes processing
- General content (3 min): < 3 minutes processing

**API Cost Optimization**
- Batch requests when possible
- Implement response caching
- Use minimal token counts
- Target < $2 per project

### Performance Monitoring

```python
@performance_monitor
def process_module(context: ContentContext) -> ContentContext:
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # Module processing logic
    result = actual_processing(context)
    
    # Record metrics
    processing_time = time.time() - start_time
    memory_used = psutil.Process().memory_info().rss - start_memory
    
    context.processing_metrics.add_module_metrics(
        module_name=__name__,
        processing_time=processing_time,
        memory_used=memory_used
    )
    
    return result
```

## 📚 Documentation Standards

### Code Documentation

**Docstring Format**
```python
def process_video_analysis(context: ContentContext, enable_face_detection: bool = True) -> ContentContext:
    """
    Analyze video content and extract visual highlights.
    
    This function processes video files to identify visual highlights, scene changes,
    and facial expressions. Results are integrated into the ContentContext for use
    by downstream modules.
    
    Args:
        context: ContentContext containing video files and processing state
        enable_face_detection: Whether to enable facial expression analysis
        
    Returns:
        Updated ContentContext with video analysis results
        
    Raises:
        ContentContextError: If video analysis fails or context is invalid
        ResourceConstraintError: If insufficient memory or processing power
        
    Example:
        >>> context = ContentContext(project_id="test", video_files=["video.mp4"])
        >>> result = process_video_analysis(context)
        >>> assert len(result.visual_highlights) > 0
    """
```

**Type Hints**
```python
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    success: bool
    processing_time: float
    memory_used: int
    error_message: Optional[str] = None

async def process_content(
    context: ContentContext,
    options: Dict[str, Any],
    timeout: Optional[float] = None
) -> ProcessingResult:
    """Process content with specified options."""
    pass
```

### API Documentation

**Module Documentation**
- Include comprehensive API reference
- Provide usage examples for all public functions
- Document integration patterns and requirements
- Include performance characteristics and constraints

**Architecture Documentation**
- Maintain up-to-date architecture diagrams
- Document data flow patterns
- Explain integration requirements
- Include extension points and customization options

## 🔄 Submission Process

### Pre-submission Checklist

**Code Quality**
- [ ] Code follows style guidelines (Black, flake8, mypy)
- [ ] All functions have type hints and docstrings
- [ ] ContentContext integration implemented correctly
- [ ] Error handling follows established patterns

**Testing**
- [ ] Unit tests written with 90%+ coverage
- [ ] Integration tests for ContentContext flow
- [ ] Performance tests meet target requirements
- [ ] All tests pass locally

**Documentation**
- [ ] API documentation updated
- [ ] Usage examples provided
- [ ] Architecture documentation updated if needed
- [ ] CHANGELOG.md updated with changes

### Pull Request Process

**PR Template**
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## ContentContext Integration
- [ ] Preserves ContentContext integrity
- [ ] Follows established data flow patterns
- [ ] Implements proper error handling

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests meet requirements
- [ ] All tests pass

## Performance Impact
- Memory usage: [impact description]
- Processing time: [impact description]
- API costs: [impact description]

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

**Review Process**
1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Architectural Review**: Verify ContentContext integration and patterns
3. **Performance Review**: Check resource usage and processing time
4. **Code Review**: Human review for logic, clarity, and maintainability
5. **Integration Testing**: Verify compatibility with existing modules

### Continuous Integration

**GitHub Actions Workflow**
```yaml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          python -m pytest --cov=ai_video_editor --cov-report=xml
      - name: Check code style
        run: |
          black --check ai_video_editor/ tests/
          flake8 ai_video_editor/ tests/
          mypy ai_video_editor/
      - name: Performance regression check
        run: |
          python scripts/performance_regression_check.py
```

---

*This contributing guide ensures consistent, high-quality contributions that maintain the architectural integrity and performance standards of the AI Video Editor system.*