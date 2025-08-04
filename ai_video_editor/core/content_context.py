"""
ContentContext System - Core data structure for unified processing pipeline.

This module implements the central ContentContext that flows through all processing
modules, enabling deep integration and shared insights across the entire system.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import json
import uuid


class ContentType(Enum):
    """Content type enumeration for processing optimization."""
    EDUCATIONAL = "educational"
    MUSIC = "music"
    GENERAL = "general"


@dataclass
class EmotionalPeak:
    """Represents an emotional peak identified in content analysis."""
    timestamp: float
    emotion: str  # "excitement", "curiosity", "surprise", etc.
    intensity: float  # 0.0 to 1.0
    confidence: float
    context: str  # What triggered this emotion
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'emotion': self.emotion,
            'intensity': self.intensity,
            'confidence': self.confidence,
            'context': self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalPeak':
        return cls(**data)


@dataclass
class FaceDetection:
    """Face detection data for visual analysis."""
    bbox: List[float]  # [x, y, width, height]
    confidence: float
    expression: Optional[str] = None
    landmarks: Optional[List[List[float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'expression': self.expression,
            'landmarks': self.landmarks
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceDetection':
        return cls(**data)


@dataclass
class VisualHighlight:
    """Visual highlight identified during video analysis."""
    timestamp: float
    description: str
    faces: List[FaceDetection]
    visual_elements: List[str]
    thumbnail_potential: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'description': self.description,
            'faces': [face.to_dict() for face in self.faces],
            'visual_elements': self.visual_elements,
            'thumbnail_potential': self.thumbnail_potential
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualHighlight':
        faces = [FaceDetection.from_dict(face_data) for face_data in data.get('faces', [])]
        return cls(
            timestamp=data['timestamp'],
            description=data['description'],
            faces=faces,
            visual_elements=data['visual_elements'],
            thumbnail_potential=data['thumbnail_potential']
        )


@dataclass
class TrendingKeywords:
    """Trending keywords research results."""
    primary_keywords: List[str]
    long_tail_keywords: List[str]
    trending_hashtags: List[str]
    seasonal_keywords: List[str]
    competitor_keywords: List[str]
    search_volume_data: Dict[str, int]
    research_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_keywords': self.primary_keywords,
            'long_tail_keywords': self.long_tail_keywords,
            'trending_hashtags': self.trending_hashtags,
            'seasonal_keywords': self.seasonal_keywords,
            'competitor_keywords': self.competitor_keywords,
            'search_volume_data': self.search_volume_data,
            'research_timestamp': self.research_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrendingKeywords':
        return cls(
            primary_keywords=data['primary_keywords'],
            long_tail_keywords=data['long_tail_keywords'],
            trending_hashtags=data['trending_hashtags'],
            seasonal_keywords=data['seasonal_keywords'],
            competitor_keywords=data['competitor_keywords'],
            search_volume_data=data['search_volume_data'],
            research_timestamp=datetime.fromisoformat(data['research_timestamp'])
        )


@dataclass
class ProcessingMetrics:
    """Processing performance metrics."""
    total_processing_time: float = 0.0
    module_processing_times: Dict[str, float] = field(default_factory=dict)
    memory_peak_usage: int = 0
    api_calls_made: Dict[str, int] = field(default_factory=dict)
    cache_hit_rate: float = 0.0
    fallbacks_used: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)
    
    def add_module_metrics(self, module_name: str, processing_time: float, memory_used: int):
        """Add processing metrics for a specific module."""
        self.module_processing_times[module_name] = processing_time
        self.total_processing_time += processing_time
        if memory_used > self.memory_peak_usage:
            self.memory_peak_usage = memory_used
    
    def add_api_call(self, service_name: str, count: int = 1):
        """Record API call usage."""
        self.api_calls_made[service_name] = self.api_calls_made.get(service_name, 0) + count
    
    def add_fallback_used(self, fallback_type: str):
        """Record fallback strategy usage."""
        self.fallbacks_used.append(fallback_type)
    
    def add_recovery_action(self, action: str):
        """Record recovery action taken."""
        self.recovery_actions.append(action)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_processing_time': self.total_processing_time,
            'module_processing_times': self.module_processing_times,
            'memory_peak_usage': self.memory_peak_usage,
            'api_calls_made': self.api_calls_made,
            'cache_hit_rate': self.cache_hit_rate,
            'fallbacks_used': self.fallbacks_used,
            'recovery_actions': self.recovery_actions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingMetrics':
        return cls(**data)


@dataclass
class CostMetrics:
    """API cost tracking metrics."""
    gemini_api_cost: float = 0.0
    imagen_api_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_asset: Dict[str, float] = field(default_factory=dict)
    optimization_savings: float = 0.0
    
    def add_cost(self, service: str, cost: float):
        """Add cost for a specific service."""
        if service == 'gemini':
            self.gemini_api_cost += cost
        elif service == 'imagen':
            self.imagen_api_cost += cost
        self.total_cost += cost
    
    def add_asset_cost(self, asset_type: str, cost: float):
        """Track cost per asset type."""
        self.cost_per_asset[asset_type] = self.cost_per_asset.get(asset_type, 0.0) + cost
    
    def add_optimization_savings(self, savings: float):
        """Record cost savings from optimization."""
        self.optimization_savings += savings
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gemini_api_cost': self.gemini_api_cost,
            'imagen_api_cost': self.imagen_api_cost,
            'total_cost': self.total_cost,
            'cost_per_asset': self.cost_per_asset,
            'optimization_savings': self.optimization_savings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CostMetrics':
        return cls(**data)


@dataclass
class UserPreferences:
    """User preferences for processing."""
    quality_mode: str = "balanced"  # "fast", "balanced", "high"
    thumbnail_resolution: tuple = (1920, 1080)
    batch_size: int = 3
    enable_aggressive_caching: bool = False
    parallel_processing: bool = True
    max_api_cost: float = 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'quality_mode': self.quality_mode,
            'thumbnail_resolution': list(self.thumbnail_resolution),
            'batch_size': self.batch_size,
            'enable_aggressive_caching': self.enable_aggressive_caching,
            'parallel_processing': self.parallel_processing,
            'max_api_cost': self.max_api_cost
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        data['thumbnail_resolution'] = tuple(data['thumbnail_resolution'])
        return cls(**data)


@dataclass
class ContentContext:
    """
    Central data structure that flows through all processing modules.
    
    This class serves as the unified context that enables deep integration
    between video processing, thumbnail generation, and metadata optimization.
    """
    
    # Core identification
    project_id: str
    video_files: List[str]
    content_type: ContentType
    user_preferences: UserPreferences
    
    # Input data
    audio_transcript: Optional[str] = None
    video_metadata: Optional[Dict[str, Any]] = None
    
    # Analysis results
    emotional_markers: List[EmotionalPeak] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    visual_highlights: List[VisualHighlight] = field(default_factory=list)
    content_themes: List[str] = field(default_factory=list)
    
    # Intelligence layer
    trending_keywords: Optional[TrendingKeywords] = None
    competitor_insights: Optional[Dict[str, Any]] = None
    engagement_predictions: Optional[Dict[str, Any]] = None
    
    # Generated assets (placeholders for future implementation)
    thumbnail_concepts: List[Dict[str, Any]] = field(default_factory=list)
    generated_thumbnails: List[Dict[str, Any]] = field(default_factory=list)
    metadata_variations: List[Dict[str, Any]] = field(default_factory=list)
    processed_video: Optional[Dict[str, Any]] = None
    
    # Performance tracking
    processing_metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)
    cost_tracking: CostMetrics = field(default_factory=CostMetrics)
    
    # Internal state
    _created_at: datetime = field(default_factory=datetime.now)
    _last_modified: datetime = field(default_factory=datetime.now)
    _processing_stage: str = "initialized"
    _checkpoints: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize project_id if not provided."""
        if not self.project_id:
            self.project_id = str(uuid.uuid4())
    
    def add_emotional_marker(self, timestamp: float, emotion: str, intensity: float, 
                           confidence: float, context: str):
        """Add emotional marker that influences both thumbnails and metadata."""
        marker = EmotionalPeak(timestamp, emotion, intensity, confidence, context)
        self.emotional_markers.append(marker)
        self._update_modified_time()
    
    def add_visual_highlight(self, timestamp: float, description: str, 
                           faces: List[FaceDetection], visual_elements: List[str],
                           thumbnail_potential: float):
        """Add visual highlight that informs thumbnail concepts and metadata descriptions."""
        highlight = VisualHighlight(timestamp, description, faces, visual_elements, thumbnail_potential)
        self.visual_highlights.append(highlight)
        self._update_modified_time()
    
    def get_synchronized_concepts(self) -> List[str]:
        """Get concepts that should be consistent across thumbnails and metadata."""
        concepts = set(self.key_concepts + self.content_themes)
        
        # Add concepts from emotional markers
        for marker in self.emotional_markers:
            if marker.intensity > 0.7:  # High intensity emotions
                concepts.add(f"{marker.emotion}_content")
        
        # Add concepts from visual highlights
        for highlight in self.visual_highlights:
            if highlight.thumbnail_potential > 0.8:  # High thumbnail potential
                concepts.update(highlight.visual_elements)
        
        return list(concepts)
    
    def get_top_emotional_peaks(self, count: int = 3) -> List[EmotionalPeak]:
        """Get top emotional peaks by intensity for thumbnail/metadata generation."""
        return sorted(self.emotional_markers, key=lambda x: x.intensity, reverse=True)[:count]
    
    def get_best_visual_highlights(self, count: int = 5) -> List[VisualHighlight]:
        """Get best visual highlights for thumbnail concepts."""
        return sorted(self.visual_highlights, key=lambda x: x.thumbnail_potential, reverse=True)[:count]
    
    def update_processing_stage(self, stage: str):
        """Update current processing stage."""
        self._processing_stage = stage
        self._update_modified_time()
    
    def add_checkpoint(self, checkpoint_name: str):
        """Add checkpoint for recovery purposes."""
        self._checkpoints.append(checkpoint_name)
        self._update_modified_time()
    
    def _update_modified_time(self):
        """Update last modified timestamp."""
        self._last_modified = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ContentContext to dictionary for serialization."""
        return {
            'project_id': self.project_id,
            'video_files': self.video_files,
            'content_type': self.content_type.value,
            'user_preferences': self.user_preferences.to_dict(),
            'audio_transcript': self.audio_transcript,
            'video_metadata': self.video_metadata,
            'emotional_markers': [marker.to_dict() for marker in self.emotional_markers],
            'key_concepts': self.key_concepts,
            'visual_highlights': [highlight.to_dict() for highlight in self.visual_highlights],
            'content_themes': self.content_themes,
            'trending_keywords': self.trending_keywords.to_dict() if self.trending_keywords else None,
            'competitor_insights': self.competitor_insights,
            'engagement_predictions': self.engagement_predictions,
            'thumbnail_concepts': self.thumbnail_concepts,
            'generated_thumbnails': self.generated_thumbnails,
            'metadata_variations': self.metadata_variations,
            'processed_video': self.processed_video,
            'processing_metrics': self.processing_metrics.to_dict(),
            'cost_tracking': self.cost_tracking.to_dict(),
            '_created_at': self._created_at.isoformat(),
            '_last_modified': self._last_modified.isoformat(),
            '_processing_stage': self._processing_stage,
            '_checkpoints': self._checkpoints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentContext':
        """Create ContentContext from dictionary."""
        # Convert nested objects
        emotional_markers = [EmotionalPeak.from_dict(marker) for marker in data.get('emotional_markers', [])]
        visual_highlights = [VisualHighlight.from_dict(highlight) for highlight in data.get('visual_highlights', [])]
        trending_keywords = TrendingKeywords.from_dict(data['trending_keywords']) if data.get('trending_keywords') else None
        processing_metrics = ProcessingMetrics.from_dict(data.get('processing_metrics', {}))
        cost_tracking = CostMetrics.from_dict(data.get('cost_tracking', {}))
        user_preferences = UserPreferences.from_dict(data.get('user_preferences', {}))
        
        # Create instance
        context = cls(
            project_id=data['project_id'],
            video_files=data['video_files'],
            content_type=ContentType(data['content_type']),
            user_preferences=user_preferences,
            audio_transcript=data.get('audio_transcript'),
            video_metadata=data.get('video_metadata'),
            emotional_markers=emotional_markers,
            key_concepts=data.get('key_concepts', []),
            visual_highlights=visual_highlights,
            content_themes=data.get('content_themes', []),
            trending_keywords=trending_keywords,
            competitor_insights=data.get('competitor_insights'),
            engagement_predictions=data.get('engagement_predictions'),
            thumbnail_concepts=data.get('thumbnail_concepts', []),
            generated_thumbnails=data.get('generated_thumbnails', []),
            metadata_variations=data.get('metadata_variations', []),
            processed_video=data.get('processed_video'),
            processing_metrics=processing_metrics,
            cost_tracking=cost_tracking
        )
        
        # Set internal state
        context._created_at = datetime.fromisoformat(data.get('_created_at', datetime.now().isoformat()))
        context._last_modified = datetime.fromisoformat(data.get('_last_modified', datetime.now().isoformat()))
        context._processing_stage = data.get('_processing_stage', 'initialized')
        context._checkpoints = data.get('_checkpoints', [])
        
        return context
    
    def to_json(self) -> str:
        """Convert ContentContext to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ContentContext':
        """Create ContentContext from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)