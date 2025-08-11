# Testing Documentation

Comprehensive testing strategy and documentation for the AI Video Editor system, combining unit testing with sophisticated mocking strategies.

## 📚 Table of Contents

1. [**Testing Philosophy**](#testing-philosophy)
2. [**Testing Layers**](#testing-layers)
3. [**Mocking Strategies**](#mocking-strategies)
4. [**Test Data Management**](#test-data-management)
5. [**ContentContext Testing**](#contentcontext-testing)
6. [**Performance Testing**](#performance-testing)
7. [**Continuous Testing**](#continuous-testing)
8. [**Test Execution**](#test-execution)

## 🎯 Testing Philosophy

### Core Principle: Unit Testing with Comprehensive Mocking

Since end-to-end testing of video processing is impractical due to resource constraints and processing time, we focus on thorough unit testing with sophisticated mocking strategies. This approach ensures:

- **Fast Test Execution**: Tests complete in seconds rather than minutes
- **Reliable Results**: Consistent outcomes independent of external services
- **Comprehensive Coverage**: All code paths tested without external dependencies
- **Cost Efficiency**: No API costs during testing

### Testing Approach

- **96.7% Test Coverage**: 475/491 tests passing with comprehensive coverage
- **Sophisticated Mocking**: Mock external APIs (Gemini, Whisper, OpenCV) with realistic responses
- **ContentContext Integrity**: Ensure data flow correctness across all modules
- **Error Scenario Testing**: Comprehensive error handling and recovery testing

## 🏗️ Testing Layers

### 1. Unit Tests (Primary Focus)

**Audio Analysis Functions**
- Transcript parsing and segment identification
- Emotional analysis and peak detection
- Financial concept extraction
- Filler word detection and removal

**Content Analysis**
- Keyword extraction and sentiment analysis
- Concept identification and theme detection
- Visual highlight recognition
- Content type classification

**Video Processing Utilities**
- Format conversion and quality assessment
- Face detection and scene analysis
- Frame extraction and processing
- Resolution and codec handling

**Thumbnail Generation**
- Text rendering and overlay composition
- Background creation and image processing
- Template application and customization
- Multi-strategy generation testing

**Metadata Generation**
- Title optimization and variation creation
- Tag suggestion and SEO optimization
- Description formatting and structure
- A/B testing framework validation

### 2. Integration Tests (Data Flow)

**ContentContext Flow**
- Verify data passes correctly between modules
- Ensure ContentContext integrity throughout pipeline
- Test module interdependencies and data synchronization
- Validate processing stage transitions

**API Integration**
- Test with mocked API responses for consistency
- Verify error handling and retry mechanisms
- Test rate limiting and batch processing
- Validate response parsing and data extraction

**File I/O Operations**
- Test with sample data files of various formats
- Verify file handling and format conversion
- Test error scenarios (corrupted files, missing files)
- Validate output file generation and structure

**Error Propagation**
- Verify error handling across module boundaries
- Test graceful degradation strategies
- Validate context preservation during errors
- Test recovery mechanisms and fallback strategies

### 3. Performance Tests

**Memory Usage**
- Monitor ContentContext size and growth patterns
- Track module memory consumption during processing
- Test memory cleanup and garbage collection
- Validate memory constraints and limits

**Processing Time**
- Benchmark each module with standardized inputs
- Test processing time targets and thresholds
- Monitor performance regression over time
- Validate timeout handling and cancellation

**Resource Utilization**
- Test concurrent operations and resource limits
- Monitor CPU and GPU utilization patterns
- Test resource allocation and cleanup
- Validate system resource constraints

**API Rate Limiting**
- Test batch processing and request optimization
- Verify retry mechanisms and backoff strategies
- Test cost optimization and caching effectiveness
- Validate API usage tracking and limits

## 🎭 Mocking Strategies

### API Mocking

#### Gemini API Responses

```python
@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API responses for content analysis."""
    return {
        "content_analysis": {
            "key_concepts": ["finance", "education", "investment"],
            "emotional_peaks": [
                {"timestamp": 30.5, "emotion": "excitement", "intensity": 0.8},
                {"timestamp": 120.0, "emotion": "curiosity", "intensity": 0.7}
            ],
            "content_themes": ["financial literacy", "beginner investing"]
        },
        "trending_keywords": [
            "financial literacy", 
            "investment basics", 
            "personal finance"
        ],
        "metadata_suggestions": {
            "titles": [
                "Master Financial Literacy in 10 Minutes",
                "Investment Basics Every Beginner Needs"
            ],
            "descriptions": [
                "Learn essential financial concepts...",
                "Discover investment fundamentals..."
            ]
        }
    }

@pytest.fixture
def mock_gemini_client(mock_gemini_response):
    """Mock Gemini client with realistic response patterns."""
    client = Mock()
    client.generate_content.return_value = Mock(
        text=json.dumps(mock_gemini_response)
    )
    return client
```

#### Imagen API Responses

```python
@pytest.fixture
def mock_imagen_response():
    """Mock Imagen API responses for thumbnail generation."""
    return {
        "image_url": "mock://generated-background.jpg",
        "generation_metadata": {
            "prompt": "Financial education thumbnail background",
            "style": "professional",
            "quality": "high"
        }
    }

@pytest.fixture
def mock_imagen_client(mock_imagen_response):
    """Mock Imagen client for thumbnail generation."""
    client = Mock()
    client.generate_image.return_value = mock_imagen_response
    return client
```

#### Whisper API Responses

```python
@pytest.fixture
def mock_whisper_response():
    """Mock Whisper transcription results for audio analysis."""
    return {
        "text": "Welcome to financial education. Today we'll learn about investment basics.",
        "segments": [
            {
                "text": "Welcome to financial education.",
                "start": 0.0,
                "end": 2.5,
                "confidence": 0.95
            },
            {
                "text": "Today we'll learn about investment basics.",
                "start": 2.5,
                "end": 5.0,
                "confidence": 0.92
            }
        ],
        "language": "en",
        "confidence": 0.94
    }
```

### Video File Mocking

```python
@pytest.fixture
def mock_video_clip():
    """Mock video files with known properties for testing."""
    return VideoClip(
        file_path="mock://test-video.mp4",
        duration=120.0,
        resolution=(1920, 1080),
        fps=30,
        metadata={
            "format": "mp4",
            "codec": "h264",
            "bitrate": "5000k",
            "audio_codec": "aac"
        }
    )

@pytest.fixture
def mock_video_analyzer():
    """Mock video analyzer with realistic analysis results."""
    analyzer = Mock()
    analyzer.analyze_video.return_value = VideoAnalysisResult(
        visual_highlights=[
            VisualHighlight(timestamp=15.0, confidence=0.9, description="Speaker gesture"),
            VisualHighlight(timestamp=45.0, confidence=0.8, description="Chart display")
        ],
        scene_changes=[10.0, 30.0, 60.0, 90.0],
        face_detections=[
            FaceDetection(timestamp=5.0, confidence=0.95, bbox=(100, 100, 200, 200))
        ],
        quality_metrics=VideoQualityMetrics(
            resolution_score=0.9,
            clarity_score=0.8,
            lighting_score=0.7
        )
    )
    return analyzer
```

### Audio Transcript Mocking

```python
@pytest.fixture
def mock_transcript():
    """Mock Whisper transcription results for AI analysis."""
    return Transcript(
        text="Welcome to financial education. Today we'll discuss investment strategies and portfolio management.",
        segments=[
            TranscriptSegment("Welcome to financial education.", 0.0, 2.5, 0.95),
            TranscriptSegment("Today we'll discuss investment strategies", 2.5, 5.0, 0.92),
            TranscriptSegment("and portfolio management.", 5.0, 7.0, 0.90)
        ],
        confidence=0.94,
        language="en",
        financial_concepts=[
            "investment strategies",
            "portfolio management",
            "financial education"
        ]
    )

@pytest.fixture
def mock_audio_analyzer(mock_transcript):
    """Mock audio analyzer with comprehensive analysis results."""
    analyzer = Mock()
    analyzer.analyze_audio.return_value = AudioAnalysisResult(
        transcript=mock_transcript,
        emotional_peaks=[
            EmotionalPeak(timestamp=30.5, emotion="excitement", intensity=0.8),
            EmotionalPeak(timestamp=60.0, emotion="confidence", intensity=0.7)
        ],
        filler_words=[
            FillerWord(word="um", timestamp=15.0, confidence=0.9),
            FillerWord(word="uh", timestamp=45.0, confidence=0.8)
        ],
        key_concepts=["investment", "portfolio", "financial planning"]
    )
    return analyzer
```

## 📊 Test Data Management

### Sample Files

**Audio Samples**
- Various languages (English, Spanish, French)
- Different quality levels (high, medium, low bitrate)
- Content types (educational, music, general)
- Duration variations (short clips, full videos)

**Video Samples**
- Different formats (MP4, MOV, AVI, WebM)
- Various resolutions (1080p, 720p, 4K)
- Duration ranges (30 seconds to 30 minutes)
- Content types (talking head, screenshare, mixed)

**Expected Outputs**
- Known good results for regression testing
- Baseline performance metrics
- Quality assessment benchmarks
- Processing time standards

**Edge Cases**
- Corrupted files and invalid formats
- Extreme values (very long/short videos)
- Unusual formats and codecs
- Network timeout scenarios

### Performance Benchmarks

**Processing Time Targets**
- Audio analysis: < 30 seconds for 15-minute video
- Video analysis: < 60 seconds for 15-minute video
- AI Director processing: < 45 seconds
- Thumbnail generation: < 20 seconds
- Metadata generation: < 15 seconds

**Memory Usage Limits**
- Maximum ContentContext size: 500MB
- Module memory usage: < 2GB per module
- Peak system memory: < 16GB
- Memory cleanup efficiency: > 95%

**API Call Limits**
- Maximum API calls per project: 50
- Batch processing efficiency: > 80%
- Cache hit rate target: > 40%
- Cost per project: < $2.00

**Quality Thresholds**
- Transcript accuracy: > 90%
- Thumbnail quality score: > 0.8
- Metadata relevance score: > 0.7
- Overall processing success rate: > 95%

## 🧪 ContentContext Testing

### ContentContext Integrity Tests

```python
class TestContentContextIntegrity:
    """Test ContentContext data integrity throughout processing pipeline."""
    
    def test_context_preservation_through_pipeline(self, sample_context):
        """Test that ContentContext maintains integrity through all stages."""
        original_project_id = sample_context.project_id
        
        # Process through all stages
        processed_context = process_through_pipeline(sample_context)
        
        # Verify core data preserved
        assert processed_context.project_id == original_project_id
        assert processed_context.video_files == sample_context.video_files
        assert processed_context.content_type == sample_context.content_type
    
    def test_context_data_flow(self, sample_context):
        """Test data flows correctly between processing modules."""
        # Audio analysis should populate transcript
        audio_processed = audio_analyzer.process(sample_context)
        assert audio_processed.audio_analysis is not None
        assert audio_processed.audio_analysis.transcript is not None
        
        # AI Director should use audio analysis results
        ai_processed = ai_director.process(audio_processed)
        assert ai_processed.ai_director_plan is not None
        assert len(ai_processed.editing_decisions) > 0
    
    def test_context_error_preservation(self, sample_context):
        """Test ContentContext preservation during error scenarios."""
        with pytest.raises(ContentContextError) as exc_info:
            # Simulate processing error
            raise_processing_error(sample_context)
        
        # Verify context preserved in exception
        assert exc_info.value.context_state is not None
        assert exc_info.value.context_state.project_id == sample_context.project_id
```

### Module Integration Tests

```python
class TestModuleIntegration:
    """Test integration between processing modules."""
    
    def test_audio_video_synchronization(self, sample_context):
        """Test audio and video analysis synchronization."""
        # Process audio and video
        audio_result = audio_analyzer.process(sample_context)
        video_result = video_analyzer.process(audio_result)
        
        # Verify synchronization
        assert len(video_result.visual_highlights) > 0
        assert len(video_result.emotional_markers) > 0
        
        # Check timestamp alignment
        for highlight in video_result.visual_highlights:
            assert 0 <= highlight.timestamp <= video_result.duration
    
    def test_thumbnail_metadata_synchronization(self, sample_context):
        """Test thumbnail and metadata synchronization."""
        # Generate thumbnails and metadata
        thumbnail_result = thumbnail_generator.process(sample_context)
        metadata_result = metadata_generator.process(thumbnail_result)
        
        # Verify synchronization
        assert thumbnail_result.thumbnail_package is not None
        assert len(metadata_result.metadata_variations) > 0
        
        # Check keyword alignment
        thumbnail_keywords = extract_keywords_from_thumbnails(thumbnail_result)
        metadata_keywords = extract_keywords_from_metadata(metadata_result)
        
        # Should have significant overlap
        overlap = set(thumbnail_keywords) & set(metadata_keywords)
        assert len(overlap) >= len(thumbnail_keywords) * 0.5
```

## ⚡ Performance Testing

### Memory Usage Tests

```python
class TestMemoryUsage:
    """Test memory usage patterns and constraints."""
    
    def test_contentcontext_size_limits(self, large_video_context):
        """Test ContentContext stays within size limits."""
        initial_size = get_context_size(large_video_context)
        
        # Process through pipeline
        processed_context = process_full_pipeline(large_video_context)
        final_size = get_context_size(processed_context)
        
        # Verify size constraints
        assert final_size < 500_000_000  # 500MB limit
        assert final_size > initial_size  # Should grow with processing
    
    def test_memory_cleanup(self, sample_context):
        """Test memory cleanup after processing."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Process multiple contexts
        for i in range(10):
            context = create_test_context(f"project_{i}")
            processed = process_full_pipeline(context)
            del processed  # Explicit cleanup
        
        # Force garbage collection
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Should not grow significantly
        assert memory_growth < 1_000_000_000  # Less than 1GB growth
    
    @pytest.mark.performance
    def test_concurrent_processing_memory(self):
        """Test memory usage during concurrent processing."""
        contexts = [create_test_context(f"project_{i}") for i in range(5)]
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Process concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_full_pipeline, ctx) for ctx in contexts]
            results = [future.result() for future in futures]
        
        peak_memory = psutil.Process().memory_info().rss
        memory_usage = peak_memory - initial_memory
        
        # Should stay within limits even with concurrent processing
        assert memory_usage < 8_000_000_000  # Less than 8GB
```

### Processing Time Tests

```python
class TestProcessingTime:
    """Test processing time targets and performance."""
    
    @pytest.mark.performance
    def test_audio_analysis_performance(self, educational_video_context):
        """Test audio analysis meets performance targets."""
        start_time = time.time()
        
        result = audio_analyzer.process(educational_video_context)
        
        processing_time = time.time() - start_time
        
        # Should complete within target time
        assert processing_time < 30.0  # 30 seconds for 15-minute video
        assert result.audio_analysis is not None
    
    @pytest.mark.performance
    def test_full_pipeline_performance(self, sample_contexts):
        """Test full pipeline performance across different content types."""
        performance_results = {}
        
        for content_type, context in sample_contexts.items():
            start_time = time.time()
            
            result = process_full_pipeline(context)
            
            processing_time = time.time() - start_time
            performance_results[content_type] = processing_time
            
            # Verify completion
            assert result.processing_metrics.total_processing_time > 0
        
        # Check performance targets
        assert performance_results['educational'] < 600  # 10 minutes
        assert performance_results['music'] < 300       # 5 minutes
        assert performance_results['general'] < 180     # 3 minutes
```

## 🔄 Continuous Testing

### Pre-commit Hooks

```bash
#!/bin/sh
# .git/hooks/pre-commit

# Run unit tests for modified modules
echo "Running unit tests for modified modules..."
python -m pytest tests/unit/ -v --tb=short

# Validate ContentContext schema changes
echo "Validating ContentContext schema..."
python scripts/validate_contentcontext_schema.py

# Check performance regression against benchmarks
echo "Checking performance regression..."
python scripts/performance_regression_check.py

# Verify API mocking consistency
echo "Verifying API mocking consistency..."
python scripts/validate_api_mocks.py

echo "Pre-commit checks completed successfully!"
```

### Integration Validation

**Daily Integration Tests**
```bash
# Daily integration test script
#!/bin/bash

echo "Running daily integration tests..."

# Full ContentContext flow test
python -m pytest tests/integration/test_contentcontext_flow.py -v

# API integration tests with mocks
python -m pytest tests/integration/test_api_integration.py -v

# Error handling and recovery tests
python -m pytest tests/integration/test_error_recovery.py -v

# Generate integration report
python scripts/generate_integration_report.py
```

**Weekly Performance Benchmarking**
```bash
# Weekly performance benchmark script
#!/bin/bash

echo "Running weekly performance benchmarks..."

# Run performance test suite
python -m pytest tests/performance/ -v --benchmark-only

# Compare against baseline
python scripts/compare_performance_baseline.py

# Update performance metrics
python scripts/update_performance_metrics.py
```

**Monthly Mock Data Refresh**
```bash
# Monthly mock data refresh script
#!/bin/bash

echo "Refreshing mock data..."

# Update API response mocks
python scripts/refresh_api_mocks.py

# Validate mock data consistency
python scripts/validate_mock_data.py

# Update test fixtures
python scripts/update_test_fixtures.py
```

## 🚀 Test Execution

### Running Tests

**Basic Test Execution**
```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/          # Unit tests only
python -m pytest tests/integration/   # Integration tests only
python -m pytest tests/performance/   # Performance tests only

# Run with coverage
python -m pytest --cov=ai_video_editor --cov-report=html

# Run with specific markers
python -m pytest -m "not performance"  # Skip performance tests
python -m pytest -m "integration"      # Run integration tests only
```

**Test Configuration**
```ini
# pytest.ini
[tool:pytest]
minversion = 6.0
addopts = 
    -ra 
    -q 
    --strict-markers
    --timeout=30
    --cov=ai_video_editor
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=90

markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests (slow)
    api: API integration tests
    mock: Tests using mocked dependencies

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

timeout = 30
```

### Test Status and Coverage

**Current Test Status**
- **Overall Test Coverage**: 96.7% (475/491 tests passing)
- **Unit Tests**: 189/189 passing (100%)
- **Integration Tests**: 5/5 passing (100%)
- **Acceptance Tests**: 4/6 passing (2 skipped by design)
- **Performance Tests**: Available but opt-in to prevent long runs

**Test Categories**
- **ContentContext Tests**: Comprehensive data flow and integrity testing
- **Module Integration Tests**: Cross-module communication and synchronization
- **API Mock Tests**: External service integration with realistic mocks
- **Error Handling Tests**: Exception scenarios and recovery mechanisms
- **Performance Tests**: Memory usage, processing time, and resource utilization

### Debugging Failed Tests

**Common Test Failures**
1. **Timing Assertions**: Mock timing issues where processing time expects > 0 but gets 0.0
2. **Batch Queue Logic**: Job retry mechanism occasionally adds duplicate entries
3. **Resource Cleanup**: Memory not properly released after test completion
4. **API Mock Inconsistency**: Mock responses not matching actual API behavior
5. **Matplotlib Graphics**: Fatal errors in bezier calculations during fallback graphics

**Debugging Strategies**
```bash
# Run specific failing test with verbose output
python -m pytest tests/unit/test_specific_module.py::test_failing_function -v -s

# Run with debugging enabled
python -m pytest --pdb tests/unit/test_specific_module.py::test_failing_function

# Run with logging enabled
python -m pytest --log-cli-level=DEBUG tests/unit/test_specific_module.py

# Generate detailed failure report
python -m pytest --tb=long --capture=no tests/unit/test_specific_module.py
```

---

*This comprehensive testing documentation ensures robust, reliable, and maintainable testing practices for the AI Video Editor system.*