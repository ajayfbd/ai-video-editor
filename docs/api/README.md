# API Reference

Complete API documentation for the AI Video Editor system.

## 📚 Table of Contents

1. [**Core Architecture**](#core-architecture)
2. [**ContentContext API**](#contentcontext-api)
3. [**Processing Modules**](#processing-modules)
4. [**Workflow Orchestrator**](#workflow-orchestrator)
5. [**Configuration API**](#configuration-api)
6. [**Error Handling**](#error-handling)

## 🏗️ Core Architecture

### ContentContext System

The ContentContext is the central data structure that flows through all processing modules:

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
    
    # Analysis Results
    audio_analysis: Optional[AudioAnalysisResult] = None
    visual_highlights: List[VisualHighlight] = field(default_factory=list)
    emotional_markers: List[EmotionalPeak] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    
    # AI Director Decisions
    ai_director_plan: Optional[AIDirectorPlan] = None
    editing_decisions: List[EditingDecision] = field(default_factory=list)
    broll_plans: List[BRollPlan] = field(default_factory=list)
    
    # Generated Assets
    thumbnail_package: Optional[ThumbnailPackage] = None
    metadata_variations: List[MetadataVariation] = field(default_factory=list)
    
    # Processing Metrics
    processing_metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)
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

### Audio Analysis Module

```python
from ai_video_editor.modules.content_analysis.audio_analyzer import FinancialContentAnalyzer

# Initialize analyzer
analyzer = FinancialContentAnalyzer(
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

### Video Analysis Module

```python
from ai_video_editor.modules.content_analysis.video_analyzer import VideoAnalyzer

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

### AI Director Module

```python
from ai_video_editor.modules.intelligence.ai_director import FinancialVideoEditor

# Initialize AI Director
director = FinancialVideoEditor(
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

### Thumbnail Generation Module

```python
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator

# Initialize generator
generator = ThumbnailGenerator(
    gemini_client=gemini_client,
    cache_manager=cache_manager
)

# Generate thumbnails
thumbnail_package = await generator.generate_thumbnail_package(context)

# Access results
variations = thumbnail_package.variations
recommended = thumbnail_package.recommended_variation
sync_data = thumbnail_package.synchronized_metadata
```

### Video Composition Module

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

## 🚨 Error Handling

### ContentContext Error Handling

```python
from ai_video_editor.core.exceptions import (
    ContentContextError,
    APIIntegrationError,
    ResourceConstraintError
)

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

## 🔌 Extension Points

### Custom Processing Modules

```python
from ai_video_editor.core.base_processor import BaseProcessor

class CustomProcessor(BaseProcessor):
    async def process(self, context: ContentContext) -> ContentContext:
        # Custom processing logic
        # Must preserve ContentContext integrity
        return context
```

### Custom AI Director Strategies

```python
from ai_video_editor.modules.intelligence.base_director import BaseAIDirector

class CustomAIDirector(BaseAIDirector):
    async def create_editing_plan(self, context: ContentContext) -> AIDirectorPlan:
        # Custom AI Director logic
        return plan
```

---

*Complete API reference for developers and integrators*