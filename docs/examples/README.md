# Examples and Code Samples

Comprehensive collection of examples demonstrating AI Video Editor functionality.

## 📚 Example Categories

### 🚀 Getting Started Examples
- [**Basic Processing**](#basic-processing) - Simple video processing
- [**Configuration Setup**](#configuration-setup) - Environment and settings
- [**First Workflow**](#first-workflow) - Complete beginner workflow

### 🎬 Processing Workflows
- [**Educational Content**](#educational-content) - Tutorial and lecture processing
- [**Music Videos**](#music-videos) - Music and performance content
- [**Batch Processing**](#batch-processing) - Multiple video processing

### 🔧 API Integration
- [**Python API Usage**](#python-api-usage) - Programmatic processing
- [**Custom Modules**](#custom-modules) - Extending functionality
- [**Workflow Orchestration**](#workflow-orchestration) - Advanced control

### 🎯 Specialized Features
- [**Thumbnail Generation**](#thumbnail-generation) - Custom thumbnail creation
- [**Metadata Optimization**](#metadata-optimization) - SEO and discovery
- [**B-Roll Integration**](#b-roll-integration) - Visual enhancement

## 🚀 Getting Started Examples

### Basic Processing

**Simple video processing with defaults:**

```bash
# Process with default settings
python -m ai_video_editor.cli.main process video.mp4

# Check results
ls -la output/
```

**Educational content processing:**

```bash
# Optimized for tutorials and lectures
python -m ai_video_editor.cli.main process lecture.mp4 \
  --type educational \
  --quality high \
  --output ./educational_results
```

**Music video processing:**

```bash
# Optimized for music content
python -m ai_video_editor.cli.main process music_video.mp4 \
  --type music \
  --quality ultra \
  --mode balanced
```

### Configuration Setup

**Environment setup:**

```bash
# Initialize configuration
python -m ai_video_editor.cli.main init

# Edit .env file
cat > .env << EOF
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_gemini_key_here
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_key_here
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id
AI_VIDEO_EDITOR_MAX_MEMORY_USAGE_GB=8
AI_VIDEO_EDITOR_ENABLE_CACHING=true
EOF

# Verify setup
python -m ai_video_editor.cli.main status
```

**Custom configuration file:**

```yaml
# config.yaml
api:
  gemini_key: ${AI_VIDEO_EDITOR_GEMINI_API_KEY}
  imagen_key: ${AI_VIDEO_EDITOR_IMAGEN_API_KEY}
  project_id: ${AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT}

processing:
  default_quality: high
  default_mode: balanced
  max_memory_gb: 12
  enable_parallel: true
  enable_caching: true

output:
  default_directory: ./output
  preserve_intermediate: false
  generate_analytics: true

thumbnails:
  strategies: [emotional, curiosity, authority]
  generate_variations: 3
  enable_ai_backgrounds: true

metadata:
  include_timestamps: true
  optimize_for_platform: youtube
  enable_trend_analysis: true
```

### First Workflow

**Complete beginner workflow:**

```bash
#!/bin/bash
# first_workflow.sh - Complete beginner example

echo "🎬 AI Video Editor - First Workflow"
echo "=================================="

# Step 1: Setup
echo "📋 Step 1: Setting up environment..."
python -m ai_video_editor.cli.main init
echo "✅ Configuration created. Please add your API keys to .env"

# Step 2: System check
echo "📋 Step 2: Checking system status..."
python -m ai_video_editor.cli.main status

# Step 3: Process video
echo "📋 Step 3: Processing your first video..."
python -m ai_video_editor.cli.main process "$1" \
  --type educational \
  --quality high \
  --output ./my_first_output \
  --parallel

# Step 4: Review results
echo "📋 Step 4: Reviewing results..."
echo "📁 Output directory: ./my_first_output"
echo "🎬 Final video: ./my_first_output/video/final_video.mp4"
echo "🖼️  Thumbnails: ./my_first_output/thumbnails/"
echo "📊 Analytics: ./my_first_output/analytics/"

echo "✅ First workflow complete!"
```

**Usage:**
```bash
chmod +x first_workflow.sh
./first_workflow.sh my_video.mp4
```

## 🎬 Processing Workflows

### Educational Content

**Complete educational processing:**

```bash
# Educational content with all features
python -m ai_video_editor.cli.main process educational_video.mp4 \
  --type educational \
  --quality ultra \
  --mode high_quality \
  --parallel \
  --max-memory 16 \
  --timeout 2400 \
  --output ./educational_premium
```

**Educational series processing:**

```bash
#!/bin/bash
# process_educational_series.sh

SERIES_NAME="financial_literacy_course"
OUTPUT_BASE="./course_output"

echo "🎓 Processing Educational Series: $SERIES_NAME"

# Create series output directory
mkdir -p "$OUTPUT_BASE"

# Process each video in the series
for video in course_*.mp4; do
    echo "📚 Processing: $video"
    
    # Extract lesson number from filename
    lesson=$(basename "$video" .mp4)
    
    # Process with educational optimization
    python -m ai_video_editor.cli.main process "$video" \
      --type educational \
      --quality high \
      --parallel \
      --output "$OUTPUT_BASE/$lesson" \
      --timeout 1800
    
    echo "✅ Completed: $lesson"
    sleep 30  # Brief pause between videos
done

echo "🎉 Educational series processing complete!"
echo "📁 Results in: $OUTPUT_BASE"
```

### Music Videos

**Music video with beat synchronization:**

```bash
# High-quality music video processing
python -m ai_video_editor.cli.main process music_performance.mp4 \
  --type music \
  --quality ultra \
  --mode balanced \
  --parallel \
  --output ./music_output
```

**Music album processing:**

```bash
#!/bin/bash
# process_music_album.sh

ALBUM_NAME="my_album"
OUTPUT_BASE="./album_output"

echo "🎵 Processing Music Album: $ALBUM_NAME"

mkdir -p "$OUTPUT_BASE"

for video in track_*.mp4; do
    track=$(basename "$video" .mp4)
    
    echo "🎶 Processing: $track"
    
    python -m ai_video_editor.cli.main process "$video" \
      --type music \
      --quality high \
      --mode balanced \
      --output "$OUTPUT_BASE/$track"
    
    echo "✅ Completed: $track"
done

echo "🎉 Album processing complete!"
```

### Batch Processing

**Parallel batch processing:**

```bash
#!/bin/bash
# batch_process.sh - Process multiple videos efficiently

INPUT_DIR="./input_videos"
OUTPUT_DIR="./batch_output"
MAX_PARALLEL=3

echo "🔄 Batch Processing Videos"
echo "========================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to process a single video
process_video() {
    local video="$1"
    local basename=$(basename "$video" .mp4)
    local output_path="$OUTPUT_DIR/$basename"
    
    echo "🎬 Processing: $basename"
    
    python -m ai_video_editor.cli.main process "$video" \
      --type general \
      --quality high \
      --parallel \
      --output "$output_path" \
      --timeout 1800
    
    echo "✅ Completed: $basename"
}

# Export function for parallel execution
export -f process_video
export OUTPUT_DIR

# Process videos in parallel
find "$INPUT_DIR" -name "*.mp4" | \
  xargs -n 1 -P "$MAX_PARALLEL" -I {} bash -c 'process_video "$@"' _ {}

echo "🎉 Batch processing complete!"
echo "📁 Results in: $OUTPUT_DIR"
```

## 🔧 API Integration

### Python API Usage

**Basic programmatic processing:**

```python
import asyncio
from pathlib import Path
from ai_video_editor.core.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowConfiguration,
    ProcessingMode
)
from ai_video_editor.core.config import ProjectSettings, ContentType, VideoQuality

async def process_video_api():
    """Process video using Python API."""
    
    # Configure processing
    config = WorkflowConfiguration(
        processing_mode=ProcessingMode.HIGH_QUALITY,
        enable_parallel_processing=True,
        max_memory_usage_gb=12.0,
        timeout_per_stage=300,
        enable_progress_display=True,
        output_directory=Path("./api_output")
    )
    
    # Project settings
    project_settings = ProjectSettings(
        content_type=ContentType.EDUCATIONAL,
        quality=VideoQuality.HIGH,
        auto_enhance=True,
        enable_b_roll_generation=True,
        enable_thumbnail_generation=True
    )
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(config=config)
    
    try:
        # Process video
        result_context = await orchestrator.process_video(
            input_files=["video.mp4"],
            project_settings=project_settings
        )
        
        # Get results
        summary = orchestrator.get_processing_summary()
        
        print(f"✅ Processing completed!")
        print(f"Project ID: {summary.get('project_id')}")
        print(f"Duration: {summary.get('workflow_duration', 0):.1f}s")
        
        return result_context
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        raise

# Run the processing
if __name__ == "__main__":
    result = asyncio.run(process_video_api())
```

**Advanced API usage with custom settings:**

```python
import asyncio
from ai_video_editor.core.content_context import (
    ContentContext, 
    ContentType, 
    UserPreferences
)
from ai_video_editor.modules.content_analysis.audio_analyzer import FinancialContentAnalyzer
from ai_video_editor.modules.intelligence.ai_director import FinancialVideoEditor
from ai_video_editor.modules.thumbnail_generation.generator import ThumbnailGenerator

async def custom_processing_pipeline():
    """Custom processing pipeline with fine-grained control."""
    
    # Create content context
    context = ContentContext(
        project_id="custom_api_project",
        video_files=["educational_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(
            quality_mode="high",
            batch_size=2,
            parallel_processing=True
        )
    )
    
    # Step 1: Audio Analysis
    print("🎵 Analyzing audio content...")
    audio_analyzer = FinancialContentAnalyzer(
        model_size="medium",
        enable_filler_detection=True
    )
    context = await audio_analyzer.analyze_audio("audio.wav", context)
    
    # Step 2: AI Director Planning
    print("🎬 Creating AI Director plan...")
    ai_director = FinancialVideoEditor(gemini_client)
    plan = await ai_director.create_editing_plan(context)
    context.set_ai_director_plan(plan)
    
    # Step 3: Thumbnail Generation
    print("🖼️ Generating thumbnails...")
    thumbnail_generator = ThumbnailGenerator(gemini_client, cache_manager)
    thumbnail_package = await thumbnail_generator.generate_thumbnail_package(context)
    
    # Step 4: Results
    print("✅ Custom pipeline completed!")
    print(f"Editing decisions: {len(plan.editing_decisions)}")
    print(f"Thumbnail variations: {len(thumbnail_package.variations)}")
    
    return context, thumbnail_package

# Run custom pipeline
if __name__ == "__main__":
    context, thumbnails = asyncio.run(custom_processing_pipeline())
```

### Custom Modules

**Creating a custom processor:**

```python
from ai_video_editor.core.base_processor import BaseProcessor
from ai_video_editor.core.content_context import ContentContext

class CustomContentAnalyzer(BaseProcessor):
    """Custom content analyzer for specialized content types."""
    
    def __init__(self, custom_settings: dict):
        super().__init__()
        self.custom_settings = custom_settings
    
    async def process(self, context: ContentContext) -> ContentContext:
        """Process content with custom analysis."""
        
        # Custom analysis logic
        custom_insights = await self._analyze_custom_content(context)
        
        # Add insights to context
        context.custom_analysis = custom_insights
        
        # Update processing metrics
        context.processing_metrics.add_module_metrics(
            module_name=self.__class__.__name__,
            processing_time=self.processing_time,
            memory_used=self.memory_used
        )
        
        return context
    
    async def _analyze_custom_content(self, context: ContentContext):
        """Implement custom analysis logic."""
        # Your custom analysis here
        return {"custom_metric": 0.85}

# Usage
custom_analyzer = CustomContentAnalyzer({"sensitivity": 0.8})
context = await custom_analyzer.process(context)
```

### Workflow Orchestration

**Custom workflow orchestration:**

```python
import asyncio
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator
from ai_video_editor.core.content_context import ContentContext

class CustomWorkflowOrchestrator(WorkflowOrchestrator):
    """Custom workflow orchestrator with specialized stages."""
    
    async def process_video_custom(self, input_files, project_settings):
        """Custom video processing workflow."""
        
        # Create context
        context = ContentContext(
            project_id=self._generate_project_id(),
            video_files=input_files,
            content_type=project_settings.content_type
        )
        
        # Custom processing stages
        stages = [
            ("custom_analysis", self._custom_analysis_stage),
            ("enhanced_ai_director", self._enhanced_ai_director_stage),
            ("specialized_output", self._specialized_output_stage)
        ]
        
        # Execute stages
        for stage_name, stage_func in stages:
            print(f"🔄 Executing: {stage_name}")
            context = await stage_func(context)
            print(f"✅ Completed: {stage_name}")
        
        return context
    
    async def _custom_analysis_stage(self, context):
        """Custom analysis stage."""
        # Implement custom analysis
        return context
    
    async def _enhanced_ai_director_stage(self, context):
        """Enhanced AI Director stage."""
        # Implement enhanced AI Director logic
        return context
    
    async def _specialized_output_stage(self, context):
        """Specialized output generation."""
        # Implement specialized output generation
        return context

# Usage
custom_orchestrator = CustomWorkflowOrchestrator(config=custom_config)
result = await custom_orchestrator.process_video_custom(
    input_files=["video.mp4"],
    project_settings=project_settings
)
```

## 🎯 Specialized Features

### Thumbnail Generation

**Custom thumbnail generation:**

```python
import asyncio
from ai_video_editor.modules.thumbnail_generation.generator import ThumbnailGenerator
from ai_video_editor.modules.thumbnail_generation.concept_analyzer import ThumbnailConceptAnalyzer

async def generate_custom_thumbnails():
    """Generate thumbnails with custom strategies."""
    
    # Create thumbnail generator
    generator = ThumbnailGenerator(gemini_client, cache_manager)
    
    # Custom concept analysis
    concept_analyzer = ThumbnailConceptAnalyzer(gemini_client)
    
    # Generate concepts
    concepts = await concept_analyzer.analyze_thumbnail_concepts(context)
    
    # Filter for specific strategies
    emotional_concepts = [c for c in concepts if c.strategy == "emotional"]
    authority_concepts = [c for c in concepts if c.strategy == "authority"]
    
    print(f"Generated {len(emotional_concepts)} emotional concepts")
    print(f"Generated {len(authority_concepts)} authority concepts")
    
    # Generate thumbnail package
    thumbnail_package = await generator.generate_thumbnail_package(context)
    
    # Access results
    for variation in thumbnail_package.variations:
        print(f"Thumbnail: {variation.concept.hook_text}")
        print(f"Strategy: {variation.concept.strategy}")
        print(f"Confidence: {variation.confidence_score:.2f}")
        print(f"Estimated CTR: {variation.estimated_ctr:.3f}")
        print()
    
    return thumbnail_package

# Run thumbnail generation
thumbnails = asyncio.run(generate_custom_thumbnails())
```

### Metadata Optimization

**SEO-optimized metadata generation:**

```python
from ai_video_editor.modules.intelligence.metadata_generator import MetadataGenerator
from ai_video_editor.modules.intelligence.trend_analyzer import TrendAnalyzer

async def generate_optimized_metadata():
    """Generate SEO-optimized metadata."""
    
    # Create components
    trend_analyzer = TrendAnalyzer()
    metadata_generator = MetadataGenerator(gemini_client, trend_analyzer)
    
    # Analyze trends
    trending_keywords = await trend_analyzer.analyze_trends(
        topic="financial education",
        content_type="educational"
    )
    
    print(f"Found {len(trending_keywords.keywords)} trending keywords")
    
    # Generate metadata
    metadata_package = await metadata_generator.generate_metadata_package(context)
    
    # Access optimized content
    print("📝 Generated Titles:")
    for title in metadata_package.title_variations[:3]:
        print(f"  • {title['title']} (CTR: {title['estimated_ctr']:.3f})")
    
    print("\n📄 Generated Descriptions:")
    for desc in metadata_package.description_variations[:2]:
        print(f"  • Strategy: {desc['strategy']}")
        print(f"    Length: {len(desc['description'])} chars")
        print(f"    Keywords: {len(desc['keywords'])} included")
    
    print(f"\n🏷️ Generated Tags: {len(metadata_package.tag_suggestions)}")
    print(f"Top tags: {', '.join(metadata_package.tag_suggestions[:5])}")
    
    return metadata_package

# Run metadata generation
metadata = asyncio.run(generate_optimized_metadata())
```

### B-Roll Integration

**Custom B-roll generation:**

```python
from ai_video_editor.modules.video_processing.broll_generation import BRollGenerator
from ai_video_editor.modules.intelligence.graphics_director import AIGraphicsDirector

async def generate_custom_broll():
    """Generate custom B-roll content."""
    
    # Create B-roll generator
    broll_generator = BRollGenerator()
    graphics_director = AIGraphicsDirector(gemini_client)
    
    # Analyze B-roll opportunities
    broll_opportunities = await graphics_director.analyze_broll_opportunities(context)
    
    print(f"Found {len(broll_opportunities)} B-roll opportunities")
    
    # Generate B-roll content
    for opportunity in broll_opportunities:
        print(f"🎬 Generating B-roll for: {opportunity.description}")
        
        if opportunity.content_type == "chart":
            # Generate chart
            chart_path = await broll_generator.generate_chart(
                data=opportunity.data,
                chart_type=opportunity.chart_type,
                style=opportunity.style
            )
            print(f"  📊 Chart generated: {chart_path}")
        
        elif opportunity.content_type == "animation":
            # Generate animation
            animation_path = await broll_generator.generate_animation(
                concept=opportunity.concept,
                duration=opportunity.duration,
                style=opportunity.animation_style
            )
            print(f"  🎞️ Animation generated: {animation_path}")
    
    return broll_opportunities

# Run B-roll generation
broll = asyncio.run(generate_custom_broll())
```

## 📊 Performance Examples

### Resource Monitoring

**Monitor processing resources:**

```python
import psutil
import time
from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator

class ResourceMonitoringOrchestrator(WorkflowOrchestrator):
    """Orchestrator with resource monitoring."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resource_history = []
    
    async def process_video(self, *args, **kwargs):
        """Process video with resource monitoring."""
        
        # Start monitoring
        monitor_task = asyncio.create_task(self._monitor_resources())
        
        try:
            # Process video
            result = await super().process_video(*args, **kwargs)
            
            # Stop monitoring
            monitor_task.cancel()
            
            # Report resource usage
            self._report_resource_usage()
            
            return result
            
        except Exception as e:
            monitor_task.cancel()
            raise
    
    async def _monitor_resources(self):
        """Monitor system resources during processing."""
        while True:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent()
            
            self.resource_history.append({
                'timestamp': time.time(),
                'memory_percent': memory.percent,
                'memory_gb': memory.used / (1024**3),
                'cpu_percent': cpu
            })
            
            await asyncio.sleep(5)  # Monitor every 5 seconds
    
    def _report_resource_usage(self):
        """Report resource usage statistics."""
        if not self.resource_history:
            return
        
        memory_usage = [r['memory_gb'] for r in self.resource_history]
        cpu_usage = [r['cpu_percent'] for r in self.resource_history]
        
        print(f"\n📊 Resource Usage Report:")
        print(f"  Peak Memory: {max(memory_usage):.1f}GB")
        print(f"  Average Memory: {sum(memory_usage)/len(memory_usage):.1f}GB")
        print(f"  Peak CPU: {max(cpu_usage):.1f}%")
        print(f"  Average CPU: {sum(cpu_usage)/len(cpu_usage):.1f}%")

# Usage
monitoring_orchestrator = ResourceMonitoringOrchestrator(config=config)
result = await monitoring_orchestrator.process_video(
    input_files=["video.mp4"],
    project_settings=project_settings
)
```

## 🔍 Testing Examples

**Unit testing with mocks:**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from ai_video_editor.modules.thumbnail_generation.generator import ThumbnailGenerator

@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for testing."""
    client = AsyncMock()
    client.generate_content.return_value = MagicMock(
        content="AMAZING RESULTS!"
    )
    return client

@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing."""
    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = True
    return cache

@pytest.mark.asyncio
async def test_thumbnail_generation(mock_gemini_client, mock_cache_manager):
    """Test thumbnail generation with mocked dependencies."""
    
    # Create generator
    generator = ThumbnailGenerator(mock_gemini_client, mock_cache_manager)
    
    # Create test context
    context = create_test_context()
    
    # Generate thumbnails
    thumbnail_package = await generator.generate_thumbnail_package(context)
    
    # Assertions
    assert thumbnail_package is not None
    assert len(thumbnail_package.variations) > 0
    assert thumbnail_package.recommended_variation is not None
    
    # Verify API calls
    mock_gemini_client.generate_content.assert_called()
```

---

*Explore these examples to master the AI Video Editor and create amazing content!*