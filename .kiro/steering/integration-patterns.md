# Integration Patterns for AI Video Editor

## Core Principle: Unified ContentContext

All modules in the AI Video Editor must operate on a shared ContentContext object that flows through the entire pipeline. This ensures deep integration and prevents data silos.

## ContentContext Structure

```python
@dataclass
class ContentContext:
    # Raw Input Data
    video_files: List[str]
    audio_transcript: Transcript
    video_metadata: VideoMetadata
    
    # Analysis Results
    emotional_markers: List[EmotionalPeak]
    key_concepts: List[str]
    visual_highlights: List[VisualHighlight]
    content_themes: List[str]
    
    # Intelligence Layer
    trending_keywords: TrendingKeywords
    competitor_insights: CompetitorAnalysis
    engagement_predictions: EngagementMetrics
    
    # Generated Assets
    thumbnail_concepts: List[ThumbnailConcept]
    metadata_variations: List[MetadataSet]
    generated_thumbnails: List[Thumbnail]
    
    # Performance Data
    processing_metrics: ProcessingMetrics
    cost_tracking: CostMetrics
```

## Integration Requirements

### 1. Thumbnail-Metadata Synchronization
- Thumbnail hook text MUST be derived from the same emotional analysis used for YouTube titles
- Visual concepts identified for thumbnails MUST inform metadata descriptions
- Trending keywords MUST influence both thumbnail text overlays AND metadata tags
- A/B testing MUST be coordinated between thumbnail and title variations

### 2. Shared Research Layer
- Keyword research MUST be performed once and shared across all output generation
- Competitor analysis MUST inform both thumbnail concepts and metadata strategies
- Trend analysis MUST be cached and reused across multiple generation requests

### 3. Parallel Processing Guidelines
- Thumbnail generation and metadata research CAN run in parallel after content analysis
- API calls MUST be batched to minimize costs and latency
- Shared caching MUST be implemented for repeated operations

## Data Flow Pattern

```
Input (Raw Video/Audio) → Comprehensive Analysis (Input Processing Module) → AI Director Decisions (Intelligence Layer Module) → Asset Generation & Composition (Output Module)
  ↓                                     ↓                                     ↓
ContentContext (Raw Data) → ContentContext (Analyzed Data) → ContentContext (AI Decisions) → Final Video & Metadata

**Key Integration Points:**
- The AI Director's decisions (editing plan, B-roll plan, SEO metadata) are stored directly in the ContentContext.
- The Output Generation module reads these plans from the ContentContext to orchestrate `movis` for video composition and motion graphics, and `Blender` for character animations.
- Thumbnail generation and metadata creation are driven by the same ContentContext insights, ensuring consistency.
```

## Error Handling Integration

All modules MUST implement consistent error handling that preserves ContentContext state and allows for graceful degradation.