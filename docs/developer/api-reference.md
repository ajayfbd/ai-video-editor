# API Reference

Complete API documentation for the AI Video Editor system, providing comprehensive reference for developers and integrators.

## 📚 Table of Contents

1. [**Core Architecture**](#core-architecture)
2. [**ContentContext API**](#contentcontext-api)
3. [**Processing Modules**](#processing-modules)
4. [**Workflow Orchestrator**](#workflow-orchestrator)
5. [**Configuration API**](#configuration-api)
6. [**Error Handling**](#error-handling)
7. [**Performance Monitoring**](#performance-monitoring)
8. [**Extension Points**](#extension-points)

## 🏗️ Core Architecture

### ContentContext System

The ContentContext is the central data structure that flows through all processing modules, ensuring unified data flow and preventing data silos:

```python
from ai_video_editor.core.content_context import ContentContext, ContentType

# Create a new content context
context = ContentContext(
    project_id="example_project",
    video_files=["video.mp4"],
    content_type=ContentType.GENERAL,
    user_preferences=UserPreferences()
)
```

### Processing Pipeline

```python
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator

# Create orchestrator
orchestrator = WorkflowOrchestrator(config=workflow_config)

# Process video
result = await orchestrator.process_video(
    input_files=["video.mp4"],
    project_settings=project_settings
)
```

### Integration Patterns

All modules must operate on the shared ContentContext object following these patterns:

- **Thumbnail-Metadata Synchronization**: Thumbnail hook text derived from same emotional analysis used for YouTube titles
- **Shared Research Layer**: Keyword research performed once and shared across all output generation
- **Parallel Processing**: Thumbnail generation and metadata research run in parallel after content analysis

## 📊 ContentContext API

### Core Data Structure

```python
@dataclass
class ContentContext:
    # Project Information
    project_id: str
    video_files: List[str]
    content_type: ContentType
    user_preferences: UserPreferences
    
    # Raw Input Data
    audio_transcript: Transcript
    video_metadata: VideoMetadata
    
    # Analysis Results
    audio_analysis: Optional[AudioAnalysisResult] = None
    visual_highlights: List[VisualHighlight] = field(default_factory=list)
    emotional_markers: List[EmotionalPeak] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    content_themes: List[str] = field(default_factory=list)
    
    # Intelligence Layer
    trending_keywords: TrendingKeywords
    competitor_insights: CompetitorAnalysis
    engagement_predictions: EngagementMetrics
    
    # AI Director Decisions
    ai_director_plan: Optional[AIDirectorPlan] = None
    editing_decisions: List[EditingDecision] = field(default_factory=list)
    broll_plans: List[BRollPlan] = field(default_factory=list)
    metadata_strategy: Optional[MetadataStrategy] = None
    
    # Generated Assets
    thumbnail_concepts: List[ThumbnailConcept]
    metadata_variations: List[MetadataSet]
    generated_thumbnails: List[Thumbnail]
    thumbnail_package: Optional[ThumbnailPackage] = None
    generated_broll: List[BRollAsset] = field(default_factory=list)
    
    # Performance Data
    processing_metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)
    cost_tracking: CostMetrics
    error_history: List[ProcessingError] = field(default_factory=list)
    checkpoints: Dict[str, Any] = field(default_factory=dict)
```

### Key Methods

```python
# Audio Analysis
context.set_audio_analysis(audio_result: AudioAnalysisResult)
audio_analysis = context.get_audio_analysis()

# Visual Highlights
context.add_visual_highlight(highlight: VisualHighlight)
best_highlights = context.get_best_visual_highlights(count=5)

# Emotional Analysis
context.add_emotional_peak(peak: EmotionalPeak)
top_peaks = context.get_top_emotional_peaks(count=3)

# AI Director Integration
context.set_ai_director_plan(plan: AIDirectorPlan)
editing_decisions = context.get_editing_decisions()

# Metrics and Performance
context.add_processing_metric(module: str, metric: ProcessingMetric)
metrics = context.get_processing_summary()
```

## 🎬 Processing Modules

### Input Processing Module

Transforms raw media into structured, analyzable data.

#### Audio Analysis

```python
from ai_video_editor.core.audio_integration import AudioAnalyzer

# Initialize analyzer
analyzer = AudioAnalyzer(
    model_size="medium",
    enable_filler_detection=True
)

# Analyze audio
audio_result = await analyzer.analyze_audio(
    audio_file="audio.wav",
    context=content_context
)

# Access results
transcript = audio_result.transcript_text
segments = audio_result.segments
concepts = audio_result.financial_concepts
```

#### Video Analysis

```python
from ai_video_editor.core.content_context import ContentContext

# Initialize analyzer
analyzer = VideoAnalyzer(
    enable_face_detection=True,
    enable_scene_detection=True
)

# Analyze video
video_result = await analyzer.analyze_video(
    video_file="video.mp4",
    context=content_context
)

# Access results
highlights = video_result.visual_highlights
scenes = video_result.scene_changes
faces = video_result.face_detections
```

### Intelligence Layer Module

AI-driven creative and strategic decision making.

#### AI Director

```python
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator

# Initialize AI Director
director = WorkflowOrchestrator(
    gemini_client=gemini_client,
    enable_broll_analysis=True
)

# Generate editing plan
plan = await director.create_editing_plan(context)

# Access decisions
editing_decisions = plan.editing_decisions
broll_opportunities = plan.broll_opportunities
metadata_strategy = plan.metadata_strategy
```

#### Trend Analysis

```python
from ai_video_editor.modules.intelligence.trend_analyzer import TrendAnalyzer

# Initialize analyzer
analyzer = TrendAnalyzer(
    gemini_client=gemini_client,
    cache_manager=cache_manager
)

# Analyze trends
trends = await analyzer.analyze_trends(
    topic="financial education",
    content_type="educational"
)

# Access results
keywords = trends.trending_keywords
competitor_data = trends.competitor_insights
```

### Output Generation Module

Executes AI Director plans and generates final assets.

#### Video Composition

```python
from ai_video_editor.modules.video_processing.composer import VideoComposer

# Initialize composer
composer = VideoComposer(
    output_dir="./output",
    quality_profile="high"
)

# Compose video
final_video = await composer.compose_video(
    context=context,
    ai_director_plan=plan
)

# Access results
video_path = final_video.output_path
composition_data = final_video.composition_metadata
```

#### Thumbnail Generation

```python
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator

# Initialize generator
generator = ThumbnailGenerator(
    gemini_client=gemini_client,
    imagen_client=imagen_client,
    cache_manager=cache_manager
)

# Generate thumbnails
thumbnail_package = await generator.generate_thumbnail_package(context)

# Access results
variations = thumbnail_package.variations
recommended = thumbnail_package.recommended_variation
sync_data = thumbnail_package.synchronized_metadata
```

#### Metadata Generation

```python
from ai_video_editor.modules.metadata_generation.generator import MetadataGenerator

# Initialize generator
generator = MetadataGenerator(
    gemini_client=gemini_client,
    trend_analyzer=trend_analyzer
)

# Generate metadata
metadata_package = await generator.generate_metadata_package(context)

# Access results
title_variations = metadata_package.title_variations
descriptions = metadata_package.descriptions
tags = metadata_package.optimized_tags
```

## 🔄 Workflow Orchestrator

### Configuration

```python
from ai_video_editor.core.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowConfiguration,
    ProcessingMode
)

# Create configuration
config = WorkflowConfiguration(
    processing_mode=ProcessingMode.HIGH_QUALITY,
    enable_parallel_processing=True,
    max_memory_usage_gb=8.0,
    timeout_per_stage=300,
    enable_progress_display=True,
    output_directory=Path("./output")
)

# Initialize orchestrator
orchestrator = WorkflowOrchestrator(config=config)
```

### Processing Stages

```python
class WorkflowStage(Enum):
    """Workflow processing stages."""
    INITIALIZATION = "initialization"
    AUDIO_PROCESSING = "audio_processing"
    VIDEO_PROCESSING = "video_processing"
    CONTENT_ANALYSIS = "content_analysis"
    INTELLIGENCE_LAYER = "intelligence_layer"
    BROLL_GENERATION = "broll_generation"
    THUMBNAIL_GENERATION = "thumbnail_generation"
    METADATA_GENERATION = "metadata_generation"
    VIDEO_COMPOSITION = "video_composition"
    FINALIZATION = "finalization"
```

### Processing

```python
# Process video
result_context = await orchestrator.process_video(
    input_files=["video.mp4"],
    project_settings=project_settings,
    user_preferences=user_preferences
)

# Monitor progress
status = orchestrator.get_workflow_status()
summary = orchestrator.get_processing_summary()
```

### Workflow Status

```python
# Get current status
status = orchestrator.get_workflow_status()

# Status structure
{
    "project_id": "abc123",
    "current_stage": "thumbnail_generation",
    "progress": 0.75,
    "stages": {
        "audio_processing": {"status": "completed", "duration": 45.2},
        "video_processing": {"status": "completed", "duration": 67.8},
        "intelligence_layer": {"status": "completed", "duration": 23.1},
        "thumbnail_generation": {"status": "running", "progress": 0.6}
    },
    "performance_metrics": {
        "peak_memory_gb": 6.2,
        "current_memory_gb": 4.1,
        "cpu_percent": 78.5
    }
}
```

## ⚙️ Configuration API

### Settings Management

```python
from ai_video_editor.core.config import get_settings, ProjectSettings

# Get current settings
settings = get_settings()

# Create project settings
project_settings = ProjectSettings(
    content_type=ContentType.EDUCATIONAL,
    quality=VideoQuality.HIGH,
    auto_enhance=True,
    enable_b_roll_generation=True,
    enable_thumbnail_generation=True
)
```

### Environment Configuration

```python
from ai_video_editor.core.config import validate_environment

# Validate environment
env_status = validate_environment()

# Check results
if env_status["valid"]:
    print("Environment ready")
else:
    print("Issues found:", env_status["errors"])
```

### API Configuration

```python
# Environment variables required
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_gemini_api_key_here
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_api_key_here
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id_here

# Configuration validation
from ai_video_editor.core.config import ConfigurationManager

config_manager = ConfigurationManager()
config = config_manager.get_configuration()
validation_result = config_manager.validate_configuration(config)
```

## 🚨 Error Handling

### Error Hierarchy

```python
from ai_video_editor.core.exceptions import (
    ContentContextError,
    APIIntegrationError,
    ResourceConstraintError,
    ProcessingTimeoutError
)

class ContentContextError(Exception):
    """Base exception for ContentContext-related errors"""
    def __init__(self, message: str, context_state: Optional[ContentContext] = None):
        super().__init__(message)
        self.context_state = context_state
        self.recovery_checkpoint = None

class APIIntegrationError(ContentContextError):
    """Raised when external API calls fail"""
    pass

class GeminiAPIError(APIIntegrationError):
    """Specific error for Gemini API failures"""
    pass

class ImagenAPIError(APIIntegrationError):
    """Specific error for Imagen API failures"""
    pass

class ResourceConstraintError(ContentContextError):
    """Raised when system resources are insufficient"""
    pass

class MemoryConstraintError(ResourceConstraintError):
    """Raised when memory usage exceeds limits"""
    pass

class ProcessingTimeoutError(ResourceConstraintError):
    """Raised when processing takes too long"""
    pass
```

### ContentContext Error Handling

```python
try:
    result = await process_video(context)
except ContentContextError as e:
    # Access preserved context
    preserved_context = e.context_state
    logger.error(f"Processing failed: {e}")
    
except APIIntegrationError as e:
    # Handle API failures
    logger.warning(f"API error: {e}")
    # Implement fallback strategy
    
except ResourceConstraintError as e:
    # Handle resource constraints
    logger.warning(f"Resource constraint: {e}")
    # Reduce quality or batch size
```

### Graceful Degradation

```python
from ai_video_editor.core.error_recovery import GracefulDegradationManager

# Create degradation manager
degradation_manager = GracefulDegradationManager(context)

try:
    result = await api_call()
except APIError:
    # Apply fallback strategy
    context = degradation_manager.handle_api_failure(context, 'gemini_api')
```

### Recovery Patterns

```python
from ai_video_editor.core.error_recovery import CheckpointManager

# Create checkpoint manager
checkpoint_manager = CheckpointManager("./checkpoints")

# Save checkpoint before risky operation
checkpoint_manager.save_checkpoint(context, "before_ai_processing")

try:
    result = await risky_operation(context)
except Exception as e:
    # Restore from checkpoint
    context = checkpoint_manager.load_checkpoint(
        context.project_id, 
        "before_ai_processing"
    )
```

### Context Preservation Pattern

```python
from contextlib import contextmanager

@contextmanager
def preserve_context_on_error(context: ContentContext, checkpoint_name: str):
    try:
        # Save checkpoint before risky operation
        context_manager.save_checkpoint(context, checkpoint_name)
        yield context
    except Exception as e:
        # Restore from checkpoint on error
        restored_context = context_manager.load_checkpoint(context.project_id, checkpoint_name)
        logger.error(f"Error occurred, restored from checkpoint {checkpoint_name}: {str(e)}")
        raise ContentContextError(f"Processing failed: {str(e)}", restored_context)
```

## 📈 Performance Monitoring

### Processing Metrics

```python
# Access processing metrics
metrics = context.processing_metrics

# Key metrics
processing_time = metrics.total_processing_time
memory_usage = metrics.peak_memory_usage
api_calls = metrics.api_calls_made
cache_hit_rate = metrics.cache_hit_rate
```

### Resource Monitoring

```python
from ai_video_editor.core.performance import ResourceMonitor

# Monitor resources during processing
@ResourceMonitor.monitor_resources
async def process_stage(context: ContentContext):
    # Processing logic here
    return processed_context

# Access monitoring data
monitor_data = ResourceMonitor.get_current_usage()
```

### Performance Targets

- **Educational Content (15+ min)**: Process in under 10 minutes
- **Music Videos (5-6 min)**: Process in under 5 minutes
- **General Content (3 min)**: Process in under 3 minutes
- **Memory Usage**: Stay under 16GB peak usage
- **API Costs**: Under $2 per project on average

### Caching Strategy

```python
from ai_video_editor.core.cache import CacheManager

# Initialize cache manager
cache_manager = CacheManager(
    cache_dir=Path("./cache"),
    max_size_gb=2.0
)

# Cache strategies
cache_strategies = {
    'api_responses': TTLCache(maxsize=1000, ttl=3600),  # 1 hour
    'processed_audio': LRUCache(maxsize=100),
    'generated_thumbnails': LRUCache(maxsize=500),
    'trend_data': TTLCache(maxsize=200, ttl=86400)  # 24 hours
}

# Get or compute with caching
result = await cache_manager.get_or_compute(
    cache_key="trend_analysis_finance",
    compute_func=analyze_trends,
    cache_type="trend_data"
)
```

## 🔌 Extension Points

### Custom Processing Modules

```python
from ai_video_editor.core.base_processor import BaseProcessor

class CustomProcessor(BaseProcessor):
    """Base class for all processing modules."""
    
    @abstractmethod
    async def process(self, context: ContentContext) -> ContentContext:
        """Process the content context."""
        # Custom processing logic
        # Must preserve ContentContext integrity
        return context
    
    def validate_input(self, context: ContentContext) -> bool:
        """Validate input context."""
        return True
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing performance metrics."""
        return {
            'processing_time': self.processing_time,
            'memory_used': self.memory_used,
            'api_calls_made': self.api_calls_made
        }
```

### Custom AI Director Strategies

```python
from ai_video_editor.modules.intelligence.base_director import BaseAIDirector

class CustomAIDirector(BaseAIDirector):
    async def create_editing_plan(self, context: ContentContext) -> AIDirectorPlan:
        # Custom AI Director logic
        return plan
```

### Plugin Architecture

```python
from ai_video_editor.core.plugin_manager import PluginManager

class PluginManager:
    """Manages custom plugins and extensions."""
    
    def __init__(self):
        self.registered_plugins: Dict[str, BaseProcessor] = {}
        self.plugin_hooks: Dict[str, List[Callable]] = {}
    
    def register_plugin(self, name: str, plugin: BaseProcessor):
        """Register a custom processing plugin."""
        self.registered_plugins[name] = plugin
    
    def register_hook(self, hook_name: str, callback: Callable):
        """Register a callback for specific processing events."""
        if hook_name not in self.plugin_hooks:
            self.plugin_hooks[hook_name] = []
        self.plugin_hooks[hook_name].append(callback)
    
    async def execute_hooks(self, hook_name: str, context: ContentContext) -> ContentContext:
        """Execute all registered hooks for an event."""
        if hook_name in self.plugin_hooks:
            for callback in self.plugin_hooks[hook_name]:
                context = await callback(context)
        return context
```

## 🔧 Utility Functions

### Content Type Detection

```python
from ai_video_editor.utils.content_detection import detect_content_type

# Detect content type from video
content_type = await detect_content_type(
    video_file="video.mp4",
    audio_transcript="Welcome to financial education..."
)
```

### Quality Assessment

```python
from ai_video_editor.utils.quality_assessment import assess_video_quality

# Assess video quality
quality_metrics = await assess_video_quality(
    video_file="video.mp4",
    context=content_context
)
```

### Format Conversion

```python
from ai_video_editor.utils.format_conversion import convert_video_format

# Convert video format
converted_path = await convert_video_format(
    input_file="video.mov",
    output_format="mp4",
    quality_profile="high"
)
```

---

*Complete API reference for developers and integrators working with the AI Video Editor system.*
</content>