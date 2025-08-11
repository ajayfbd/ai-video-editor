# Architecture Guide

Comprehensive guide to the AI Video Editor system architecture, design patterns, and implementation details. This document consolidates architectural information from multiple sources to provide a single authoritative reference.

## 🏗️ System Overview

The AI Video Editor is built on a **ContentContext-driven architecture** where all processing modules operate on a shared data structure that flows through the entire pipeline. This ensures deep integration and prevents data silos.

### Core Principles

1. **AI-First Design**: All creative decisions are made by the AI Director
2. **Unified Data Flow**: ContentContext flows through all modules
3. **Error Resilience**: Comprehensive error handling with context preservation
4. **Performance Optimization**: Intelligent caching and resource management
5. **Extensibility**: Modular design for easy extension and customization

## 🎯 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLI Interface Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                  Workflow Orchestrator                          │
│                 (ContentContext Manager)                        │
├─────────────────────────────────────────────────────────────────┤
│  Input Processing  │  Intelligence Layer │  Output Generation   │
│     Module         │      Module         │      Module          │
├─────────────────────────────────────────────────────────────────┤
│  Audio/Video       │  AI Director        │  Video Composition   │
│  Analysis          │  Trend Analysis     │  Thumbnail Generation│
│                    │  Keyword Research   │  Metadata Creation   │
├─────────────────────────────────────────────────────────────────┤
│                    ContentContext Storage                       │
├─────────────────────────────────────────────────────────────────┤
│  Local Libraries   │  Cloud Services     │  Caching Layer       │
│  (movis, OpenCV)   │  (Gemini, Imagen)   │  (Memory/Disk)       │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 ContentContext System

### Core Data Structure

The ContentContext is the central nervous system of the application:

```python
@dataclass
class ContentContext:
    # Project Identification
    project_id: str
    video_files: List[str]
    content_type: ContentType
    user_preferences: UserPreferences
    
    # Input Analysis Results
    audio_analysis: Optional[AudioAnalysisResult] = None
    visual_highlights: List[VisualHighlight] = field(default_factory=list)
    emotional_markers: List[EmotionalPeak] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    content_themes: List[str] = field(default_factory=list)
    
    # AI Director Decisions
    ai_director_plan: Optional[AIDirectorPlan] = None
    editing_decisions: List[EditingDecision] = field(default_factory=list)
    broll_plans: List[BRollPlan] = field(default_factory=list)
    metadata_strategy: Optional[MetadataStrategy] = None
    
    # Generated Assets
    thumbnail_package: Optional[ThumbnailPackage] = None
    metadata_variations: List[MetadataVariation] = field(default_factory=list)
    generated_broll: List[BRollAsset] = field(default_factory=list)
    
    # Processing State
    processing_metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)
    error_history: List[ProcessingError] = field(default_factory=list)
    checkpoints: Dict[str, Any] = field(default_factory=dict)
```

### Data Flow Pattern

```
Input Files → Content Analysis → AI Director → Asset Generation → Final Output
     ↓              ↓               ↓              ↓              ↓
Raw Media → Structured Data → Creative Plan → Generated Assets → Composed Video
     ↓              ↓               ↓              ↓              ↓
ContentContext → ContentContext → ContentContext → ContentContext → Final Package
```

## 🎬 Processing Modules

### 1. Input Processing Module

**Purpose**: Transform raw media into structured, analyzable data.

**Components**:
- **AudioAnalyzer**: Whisper-based transcription and analysis
- **VideoAnalyzer**: OpenCV-based visual analysis
- **ContentAnalyzer**: Multi-modal content understanding

**Key Classes**:

```python
class FinancialContentAnalyzer:
    """Specialized analyzer for financial/educational content."""
    
    async def analyze_audio(self, audio_file: str, context: ContentContext) -> ContentContext:
        # Whisper transcription
        # Filler word detection
        # Financial concept extraction
        # Emotional analysis
        return context

class VideoAnalyzer:
    """Computer vision-based video analysis."""
    
    async def analyze_video(self, video_file: str, context: ContentContext) -> ContentContext:
        # Scene detection
        # Face recognition
        # Visual highlight identification
        # Quality assessment
        return context
```

### 2. Intelligence Layer Module

**Purpose**: AI-driven creative and strategic decision making.

**Components**:
- **AI Director**: Creative editing decisions
- **Trend Analyzer**: SEO and keyword research
- **Metadata Generator**: Content optimization
- **B-Roll Analyzer**: Visual enhancement planning

**Key Classes**:

```python
class FinancialVideoEditor:
    """AI Director specialized for financial/educational content."""
    
    async def create_editing_plan(self, context: ContentContext) -> AIDirectorPlan:
        # Analyze content for editing opportunities
        # Generate cutting and transition decisions
        # Plan B-roll insertion points
        # Create metadata strategy
        return plan

class TrendAnalyzer:
    """Trend analysis and keyword research."""
    
    async def analyze_trends(self, topic: str, content_type: str) -> TrendingKeywords:
        # DDG Search integration
        # Competitor analysis
        # Keyword volume analysis
        # Trend prediction
        return keywords
```

### 3. Output Generation Module

**Purpose**: Execute AI Director plans and generate final assets.

**Components**:
- **VideoComposer**: Professional video editing
- **ThumbnailGenerator**: Multi-strategy thumbnail creation
- **MetadataGenerator**: SEO-optimized content packages
- **BRollGenerator**: Visual enhancement creation

**Key Classes**:

```python
class VideoComposer:
    """Professional video composition using movis."""
    
    async def compose_video(self, context: ContentContext, plan: AIDirectorPlan) -> ComposedVideo:
        # Execute editing decisions
        # Integrate B-roll content
        # Apply audio enhancements
        # Generate final video
        return composed_video

class ThumbnailGenerator:
    """AI-powered thumbnail generation system."""
    
    async def generate_thumbnail_package(self, context: ContentContext) -> ThumbnailPackage:
        # Analyze visual highlights
        # Generate multiple strategies
        # Create thumbnail variations
        # Synchronize with metadata
        return package
```

## 🔄 Workflow Orchestration

### WorkflowOrchestrator

The orchestrator manages the complete processing pipeline:

```python
class WorkflowOrchestrator:
    """Manages the complete video processing workflow."""
    
    def __init__(self, config: WorkflowConfiguration, console: Console = None):
        self.config = config
        self.console = console or Console()
        self.current_context: Optional[ContentContext] = None
        self.processing_stages: Dict[str, WorkflowStage] = {}
        
    async def process_video(
        self, 
        input_files: List[str], 
        project_settings: ProjectSettings,
        user_preferences: Optional[UserPreferences] = None
    ) -> ContentContext:
        """Execute the complete video processing workflow."""
        
        # Initialize context
        context = self._create_context(input_files, project_settings, user_preferences)
        
        # Execute processing stages
        stages = self._get_processing_stages()
        
        for stage in stages:
            context = await self._execute_stage(stage, context)
            
        return context
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

### Stage Execution Pattern

```python
async def _execute_stage(self, stage: WorkflowStage, context: ContentContext) -> ContentContext:
    """Execute a single workflow stage with error handling and recovery."""
    
    stage_name = stage.value
    
    try:
        # Save checkpoint before stage
        if self.config.enable_recovery:
            self._save_checkpoint(context, f"before_{stage_name}")
        
        # Execute stage
        with self._monitor_resources():
            context = await self._run_stage_processor(stage, context)
        
        # Update stage status
        self._update_stage_status(stage_name, "completed")
        
        return context
        
    except Exception as e:
        # Handle stage failure
        context = await self._handle_stage_failure(stage, context, e)
        return context
```

## 🚨 Error Handling Architecture

### Error Hierarchy

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

class ProcessingTimeoutError(ContentContextError):
    """Raised when processing takes too long."""
    pass
```

### Graceful Degradation

```python
class GracefulDegradationManager:
    """Manages fallback strategies when components fail."""
    
    def __init__(self, context: ContentContext):
        self.context = context
        self.fallback_strategies = {
            'gemini_api': self._handle_gemini_failure,
            'imagen_api': self._handle_imagen_failure,
            'whisper_api': self._handle_whisper_failure
        }
    
    def handle_failure(self, component: str, error: Exception) -> ContentContext:
        """Apply appropriate fallback strategy."""
        
        if component in self.fallback_strategies:
            return self.fallback_strategies[component](self.context, error)
        else:
            raise error
```

### Recovery Patterns

```python
class CheckpointManager:
    """Manages processing checkpoints for recovery."""
    
    def save_checkpoint(self, context: ContentContext, checkpoint_name: str) -> bool:
        """Save ContentContext state for recovery."""
        
        checkpoint_data = {
            'context': asdict(context),
            'timestamp': time.time(),
            'stage': checkpoint_name
        }
        
        checkpoint_path = self.storage_path / f"{context.project_id}_{checkpoint_name}.json"
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, project_id: str, checkpoint_name: str) -> Optional[ContentContext]:
        """Load ContentContext from checkpoint."""
        
        checkpoint_path = self.storage_path / f"{project_id}_{checkpoint_name}.json"
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            return ContentContext(**checkpoint_data['context'])
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
```

## 🎨 AI Director Architecture

### Decision Making Process

```python
class AIDirectorPlan:
    """Comprehensive plan created by the AI Director."""
    
    @dataclass
    class EditingDecision:
        decision_id: str
        decision_type: str  # cut, transition, emphasis, etc.
        timestamp: float
        parameters: Dict[str, Any]
        rationale: str
        confidence: float
    
    @dataclass
    class BRollOpportunity:
        timestamp: float
        duration: float
        content_type: str  # chart, animation, graphic
        description: str
        visual_elements: List[str]
        priority: int
    
    @dataclass
    class MetadataStrategy:
        primary_keywords: List[str]
        title_strategies: List[str]
        description_approach: str
        tag_categories: List[str]
        target_audience: str
```

### Prompt Engineering

```python
class PromptTemplates:
    """Structured prompts for AI Director decisions."""
    
    EDITING_ANALYSIS = """
    Analyze this {content_type} video content and create a professional editing plan.
    
    Content Analysis:
    - Key concepts: {key_concepts}
    - Emotional peaks: {emotional_peaks}
    - Visual highlights: {visual_highlights}
    - Duration: {duration} seconds
    
    Create editing decisions for:
    1. Optimal cuts and transitions
    2. Filler word removal points
    3. Emphasis and highlighting
    4. B-roll insertion opportunities
    
    Format as structured JSON with rationale for each decision.
    """
    
    THUMBNAIL_STRATEGY = """
    Create high-CTR thumbnail concepts for this {content_type} content.
    
    Visual Elements Available:
    - Speaker expressions: {facial_expressions}
    - Key moments: {visual_highlights}
    - Emotional peaks: {emotional_context}
    
    Generate 3 thumbnail strategies:
    1. Emotional (high-impact expressions)
    2. Curiosity (question-based hooks)
    3. Authority (professional credibility)
    
    For each strategy, provide hook text and visual composition.
    """
```

## 🎯 Performance Architecture

### Resource Management

```python
class ResourceManager:
    """Manages system resources during processing."""
    
    def __init__(self, max_memory_gb: float, max_concurrent_processes: int):
        self.max_memory_gb = max_memory_gb
        self.max_concurrent_processes = max_concurrent_processes
        self.current_usage = ResourceUsage()
    
    @contextmanager
    def allocate_resources(self, estimated_memory: float, process_count: int):
        """Allocate resources with automatic cleanup."""
        
        # Check availability
        if not self._can_allocate(estimated_memory, process_count):
            raise ResourceConstraintError("Insufficient resources available")
        
        # Allocate
        self._allocate(estimated_memory, process_count)
        
        try:
            yield
        finally:
            # Cleanup
            self._deallocate(estimated_memory, process_count)
```

### Caching Strategy

```python
class CacheManager:
    """Intelligent caching for API responses and processed data."""
    
    def __init__(self, cache_dir: Path, max_size_gb: float = 2.0):
        self.cache_dir = cache_dir
        self.max_size_gb = max_size_gb
        self.cache_strategies = {
            'api_responses': TTLCache(maxsize=1000, ttl=3600),  # 1 hour
            'processed_audio': LRUCache(maxsize=100),
            'generated_thumbnails': LRUCache(maxsize=500),
            'trend_data': TTLCache(maxsize=200, ttl=86400)  # 24 hours
        }
    
    async def get_or_compute(
        self, 
        cache_key: str, 
        compute_func: Callable, 
        cache_type: str = 'default'
    ):
        """Get from cache or compute and store."""
        
        # Try cache first
        cached_result = self._get_from_cache(cache_key, cache_type)
        if cached_result is not None:
            return cached_result
        
        # Compute and cache
        result = await compute_func()
        self._store_in_cache(cache_key, result, cache_type)
        
        return result
```

## 🔌 Extension Points

### Custom Processors

```python
class BaseProcessor(ABC):
    """Base class for all processing modules."""
    
    @abstractmethod
    async def process(self, context: ContentContext) -> ContentContext:
        """Process the content context."""
        pass
    
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

### Plugin Architecture

```python
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

## 📊 Monitoring and Analytics

### Performance Monitoring

```python
class PerformanceMonitor:
    """Monitors system performance during processing."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetric] = []
        self.alert_thresholds = {
            'memory_usage_percent': 85.0,
            'processing_time_seconds': 1800.0,
            'api_error_rate': 0.1
        }
    
    @contextmanager
    def monitor_stage(self, stage_name: str):
        """Monitor a processing stage."""
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            yield
        finally:
            # Record metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            metric = PerformanceMetric(
                stage_name=stage_name,
                processing_time=end_time - start_time,
                memory_used=end_memory - start_memory,
                timestamp=end_time
            )
            
            self.metrics_history.append(metric)
            self._check_alerts(metric)
```

### Analytics Collection

```python
class AnalyticsCollector:
    """Collects analytics data for system optimization."""
    
    def collect_processing_analytics(self, context: ContentContext) -> ProcessingAnalytics:
        """Collect comprehensive processing analytics."""
        
        return ProcessingAnalytics(
            project_id=context.project_id,
            content_type=context.content_type.value,
            processing_duration=context.processing_metrics.total_processing_time,
            memory_peak_usage=context.processing_metrics.memory_peak_usage,
            api_calls_made=context.processing_metrics.api_calls_made,
            cache_hit_rate=context.processing_metrics.cache_hit_rate,
            quality_scores=self._extract_quality_scores(context),
            user_satisfaction_indicators=self._extract_satisfaction_indicators(context)
        )
```

## 🔧 Configuration Architecture

### Hierarchical Configuration

```python
class ConfigurationManager:
    """Manages hierarchical configuration from multiple sources."""
    
    def __init__(self):
        self.config_sources = [
            EnvironmentConfigSource(),
            FileConfigSource(),
            DefaultConfigSource()
        ]
    
    def get_configuration(self) -> Configuration:
        """Get merged configuration from all sources."""
        
        config = Configuration()
        
        # Merge configurations in priority order
        for source in reversed(self.config_sources):
            source_config = source.load_configuration()
            config = self._merge_configurations(config, source_config)
        
        return config
    
    def validate_configuration(self, config: Configuration) -> ValidationResult:
        """Validate configuration completeness and correctness."""
        
        validators = [
            APIKeyValidator(),
            ResourceLimitValidator(),
            PathValidator()
        ]
        
        results = []
        for validator in validators:
            result = validator.validate(config)
            results.append(result)
        
        return ValidationResult.combine(results)
```

## 🚀 Deployment Architecture

### Containerization

```dockerfile
# Dockerfile for AI Video Editor
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Configure environment
ENV PYTHONPATH=/app
ENV AI_VIDEO_EDITOR_CONFIG_PATH=/app/config

# Expose ports and volumes
VOLUME ["/app/input", "/app/output", "/app/config"]
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m ai_video_editor.cli.main status || exit 1

# Entry point
ENTRYPOINT ["python", "-m", "ai_video_editor.cli.main"]
```

### Scalability Considerations

```python
class ScalabilityManager:
    """Manages system scalability and load distribution."""
    
    def __init__(self):
        self.processing_queue = asyncio.Queue()
        self.worker_pool = []
        self.load_balancer = LoadBalancer()
    
    async def scale_workers(self, target_count: int):
        """Scale worker processes based on load."""
        
        current_count = len(self.worker_pool)
        
        if target_count > current_count:
            # Scale up
            for _ in range(target_count - current_count):
                worker = await self._create_worker()
                self.worker_pool.append(worker)
        
        elif target_count < current_count:
            # Scale down
            workers_to_remove = current_count - target_count
            for _ in range(workers_to_remove):
                worker = self.worker_pool.pop()
                await self._shutdown_worker(worker)
```

## 🔗 Integration Patterns

### ContentContext Integration Requirements

All modules in the AI Video Editor must operate on a shared ContentContext object that flows through the entire pipeline. This ensures deep integration and prevents data silos.

#### Key Integration Points

1. **Thumbnail-Metadata Synchronization**
   - Thumbnail hook text MUST be derived from the same emotional analysis used for YouTube titles
   - Visual concepts identified for thumbnails MUST inform metadata descriptions
   - Trending keywords MUST influence both thumbnail text overlays AND metadata tags
   - A/B testing MUST be coordinated between thumbnail and title variations

2. **Shared Research Layer**
   - Keyword research MUST be performed once and shared across all output generation
   - Competitor analysis MUST inform both thumbnail concepts and metadata strategies
   - Trend analysis MUST be cached and reused across multiple generation requests

3. **Parallel Processing Guidelines**
   - Thumbnail generation and metadata research CAN run in parallel after content analysis
   - API calls MUST be batched to minimize costs and latency
   - Shared caching MUST be implemented for repeated operations

#### Data Flow Pattern

```
Input (Raw Video/Audio) → Comprehensive Analysis (Input Processing Module) → AI Director Decisions (Intelligence Layer Module) → Asset Generation & Composition (Output Module)
  ↓                                     ↓                                     ↓
ContentContext (Raw Data) → ContentContext (Analyzed Data) → ContentContext (AI Decisions) → Final Video & Metadata
```

**Key Integration Points:**
- The AI Director's decisions (editing plan, B-roll plan, SEO metadata) are stored directly in the ContentContext
- The Output Generation module reads these plans from the ContentContext to orchestrate `movis` for video composition and motion graphics, and `Blender` for character animations
- Thumbnail generation and metadata creation are driven by the same ContentContext insights, ensuring consistency

## 🚨 Advanced Error Handling Architecture

### ContentContext Preservation Principle

All error handling must preserve ContentContext state to enable recovery and maintain processing continuity.

#### Error Categories and Handling

```python
class ContentContextError(Exception):
    """Base exception for ContentContext-related errors"""
    def __init__(self, message: str, context_state: Optional[ContentContext] = None):
        super().__init__(message)
        self.context_state = context_state
        self.recovery_checkpoint = None

class ContextIntegrityError(ContentContextError):
    """Raised when ContentContext data is corrupted or invalid"""
    pass

# Error handling pattern
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

#### API Integration Error Handling

```python
# Retry pattern with exponential backoff
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(APIIntegrationError)
)
def call_api_with_retry(api_function, context: ContentContext, *args, **kwargs):
    try:
        return api_function(*args, **kwargs)
    except requests.RequestException as e:
        logger.warning(f"API call failed, retrying: {str(e)}")
        raise APIIntegrationError(f"API call failed: {str(e)}", context)
```

#### Resource Monitoring Pattern

```python
def monitor_resources(func):
    @wraps(func)
    def wrapper(context: ContentContext, *args, **kwargs):
        initial_memory = psutil.Process().memory_info().rss
        start_time = time.time()
        
        try:
            result = func(context, *args, **kwargs)
            
            # Check resource usage
            final_memory = psutil.Process().memory_info().rss
            processing_time = time.time() - start_time
            
            if final_memory - initial_memory > 8_000_000_000:  # 8GB
                logger.warning(f"High memory usage in {func.__name__}: {(final_memory - initial_memory) / 1_000_000_000:.2f}GB")
            
            if processing_time > 300:  # 5 minutes
                logger.warning(f"Long processing time in {func.__name__}: {processing_time:.2f}s")
            
            return result
            
        except MemoryError:
            raise MemoryConstraintError(f"Out of memory in {func.__name__}", context)
        except TimeoutError:
            raise ProcessingTimeoutError(f"Processing timeout in {func.__name__}", context)
    
    return wrapper
```

### Graceful Degradation Strategies

#### API Failure Fallbacks

```python
class GracefulDegradationManager:
    def __init__(self, context: ContentContext):
        self.context = context
        self.fallback_strategies = {
            'gemini_api': self._handle_gemini_failure,
            'imagen_api': self._handle_imagen_failure,
            'whisper_api': self._handle_whisper_failure
        }
    
    def _handle_gemini_failure(self, context: ContentContext) -> ContentContext:
        """Fallback for Gemini API failures"""
        logger.warning("Gemini API failed, using cached keyword research")
        
        # Use cached keywords or basic analysis
        if hasattr(context, 'cached_keywords'):
            context.trending_keywords = context.cached_keywords
        else:
            # Generate basic keywords from content concepts
            context.trending_keywords = self._generate_basic_keywords(context.key_concepts)
        
        context.processing_metrics.add_fallback_used('gemini_api')
        return context
    
    def _handle_imagen_failure(self, context: ContentContext) -> ContentContext:
        """Fallback for Imagen API failures"""
        logger.warning("Imagen API failed, using procedural generation")
        
        # Switch to procedural thumbnail generation
        context.thumbnail_generation_strategy = 'procedural_only'
        context.processing_metrics.add_fallback_used('imagen_api')
        return context
```

#### Quality vs Performance Trade-offs

```python
class QualityManager:
    def adjust_for_constraints(self, context: ContentContext, available_memory: int) -> ContentContext:
        """Adjust processing quality based on available resources"""
        
        if available_memory < 4_000_000_000:  # Less than 4GB
            logger.info("Low memory detected, reducing processing quality")
            context.processing_preferences.thumbnail_resolution = (1280, 720)  # Reduce from 1920x1080
            context.processing_preferences.batch_size = 1
            context.processing_preferences.enable_aggressive_caching = True
            
        elif available_memory < 8_000_000_000:  # Less than 8GB
            logger.info("Medium memory detected, using balanced processing")
            context.processing_preferences.batch_size = 2
            context.processing_preferences.parallel_processing = False
            
        return context
```

## 🎯 Performance Architecture Guidelines

### Core Principle: Efficient Resource Management

The AI Video Editor must run efficiently on mid-range hardware (i7 11th gen, 32GB RAM, 2GB GPU) while maintaining high-quality output.

#### Resource Management Strategy

- **ContentContext Size Limit**: Maximum 500MB per project
- **Video Buffer Management**: Stream processing for large files
- **Cache Management**: LRU cache for frequently accessed data
- **Garbage Collection**: Explicit cleanup after each processing stage

#### CPU Optimization

- **Parallel Processing**: Use multiprocessing for independent operations
- **Batch Operations**: Group similar operations to reduce overhead
- **Lazy Loading**: Load data only when needed
- **Efficient Algorithms**: Prefer O(n log n) or better time complexity

#### GPU Utilization

- **OpenCV GPU Acceleration**: Use CUDA when available for video processing
- **Selective GPU Usage**: Reserve GPU for computationally intensive tasks
- **Memory Transfer Optimization**: Minimize CPU-GPU data transfers

### API Cost Optimization

#### Gemini API Efficiency

- **Batch Requests**: Combine multiple analysis requests
- **Response Caching**: Cache results for similar content
- **Request Optimization**: Use minimal token counts for requests
- **Rate Limiting**: Respect API limits to avoid throttling

#### Imagen API Cost Management

- **Hybrid Generation**: Use procedural generation when possible
- **Template Reuse**: Build library of successful background templates
- **Quality Thresholds**: Use AI generation only for high-impact concepts
- **Batch Processing**: Generate multiple variations in single requests

### Performance Targets

- **Educational Content (15+ min)**: Process in under 10 minutes
- **Music Videos (5-6 min)**: Process in under 5 minutes
- **General Content (3 min)**: Process in under 3 minutes
- **Memory Usage**: Stay under 16GB peak usage
- **API Costs**: Under $2 per project on average

### Monitoring Implementation

```python
@performance_monitor
def process_module(context: ContentContext) -> ContentContext:
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # Module processing logic here
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

---

*This comprehensive architecture guide provides the foundation for understanding, extending, and optimizing the AI Video Editor system. It consolidates architectural patterns, integration requirements, error handling strategies, and performance guidelines into a single authoritative reference.*