"""
Thumbnail Generation Data Models

This module defines the core data structures for the thumbnail generation system,
including concepts, variations, and packages for A/B testing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from ...core.content_context import EmotionalPeak, VisualHighlight


@dataclass
class ThumbnailConcept:
    """Represents a thumbnail concept derived from visual highlights and emotional analysis."""
    concept_id: str
    visual_highlight: VisualHighlight
    emotional_peak: EmotionalPeak
    hook_text: str
    background_style: str
    text_style: Dict[str, Any]
    visual_elements: List[str]
    thumbnail_potential: float
    strategy: str  # "emotional", "curiosity", "authority", "urgency"
    
    def __post_init__(self):
        """Generate concept_id if not provided."""
        if not self.concept_id:
            self.concept_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'concept_id': self.concept_id,
            'visual_highlight': self.visual_highlight.to_dict(),
            'emotional_peak': self.emotional_peak.to_dict(),
            'hook_text': self.hook_text,
            'background_style': self.background_style,
            'text_style': self.text_style,
            'visual_elements': self.visual_elements,
            'thumbnail_potential': self.thumbnail_potential,
            'strategy': self.strategy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThumbnailConcept':
        return cls(
            concept_id=data['concept_id'],
            visual_highlight=VisualHighlight.from_dict(data['visual_highlight']),
            emotional_peak=EmotionalPeak.from_dict(data['emotional_peak']),
            hook_text=data['hook_text'],
            background_style=data['background_style'],
            text_style=data['text_style'],
            visual_elements=data['visual_elements'],
            thumbnail_potential=data['thumbnail_potential'],
            strategy=data['strategy']
        )


@dataclass
class ThumbnailVariation:
    """Represents a generated thumbnail variation for A/B testing."""
    variation_id: str
    concept: ThumbnailConcept
    generated_image_path: str
    generation_method: str  # "ai_generated", "procedural", "template"
    confidence_score: float
    estimated_ctr: float
    visual_appeal_score: float
    text_readability_score: float
    brand_consistency_score: float
    generation_time: float = 0.0
    generation_cost: float = 0.0
    
    def __post_init__(self):
        """Generate variation_id if not provided."""
        if not self.variation_id:
            self.variation_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'variation_id': self.variation_id,
            'concept': self.concept.to_dict(),
            'generated_image_path': self.generated_image_path,
            'generation_method': self.generation_method,
            'confidence_score': self.confidence_score,
            'estimated_ctr': self.estimated_ctr,
            'visual_appeal_score': self.visual_appeal_score,
            'text_readability_score': self.text_readability_score,
            'brand_consistency_score': self.brand_consistency_score,
            'generation_time': self.generation_time,
            'generation_cost': self.generation_cost
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThumbnailVariation':
        return cls(
            variation_id=data['variation_id'],
            concept=ThumbnailConcept.from_dict(data['concept']),
            generated_image_path=data['generated_image_path'],
            generation_method=data['generation_method'],
            confidence_score=data['confidence_score'],
            estimated_ctr=data['estimated_ctr'],
            visual_appeal_score=data['visual_appeal_score'],
            text_readability_score=data['text_readability_score'],
            brand_consistency_score=data['brand_consistency_score'],
            generation_time=data.get('generation_time', 0.0),
            generation_cost=data.get('generation_cost', 0.0)
        )


@dataclass
class ThumbnailPackage:
    """Complete thumbnail package with multiple variations and A/B testing configuration."""
    package_id: str
    variations: List[ThumbnailVariation]
    recommended_variation: str
    generation_timestamp: datetime
    synchronized_metadata: Dict[str, Any]  # Links to metadata variations
    a_b_testing_config: Dict[str, Any]
    performance_predictions: Dict[str, Any]
    total_generation_time: float = 0.0
    total_generation_cost: float = 0.0
    
    def __post_init__(self):
        """Generate package_id if not provided."""
        if not self.package_id:
            self.package_id = str(uuid.uuid4())
    
    def get_recommended_variation(self) -> Optional[ThumbnailVariation]:
        """Get the recommended thumbnail variation."""
        for variation in self.variations:
            if variation.variation_id == self.recommended_variation:
                return variation
        return None
    
    def get_variation_by_strategy(self, strategy: str) -> Optional[ThumbnailVariation]:
        """Get thumbnail variation by strategy type."""
        for variation in self.variations:
            if variation.concept.strategy == strategy:
                return variation
        return None
    
    def get_top_variations(self, count: int = 3) -> List[ThumbnailVariation]:
        """Get top variations by confidence score."""
        return sorted(self.variations, key=lambda x: x.confidence_score, reverse=True)[:count]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'package_id': self.package_id,
            'variations': [var.to_dict() for var in self.variations],
            'recommended_variation': self.recommended_variation,
            'generation_timestamp': self.generation_timestamp.isoformat(),
            'synchronized_metadata': self.synchronized_metadata,
            'a_b_testing_config': self.a_b_testing_config,
            'performance_predictions': self.performance_predictions,
            'total_generation_time': self.total_generation_time,
            'total_generation_cost': self.total_generation_cost
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThumbnailPackage':
        variations = [ThumbnailVariation.from_dict(var) for var in data['variations']]
        return cls(
            package_id=data['package_id'],
            variations=variations,
            recommended_variation=data['recommended_variation'],
            generation_timestamp=datetime.fromisoformat(data['generation_timestamp']),
            synchronized_metadata=data['synchronized_metadata'],
            a_b_testing_config=data['a_b_testing_config'],
            performance_predictions=data['performance_predictions'],
            total_generation_time=data.get('total_generation_time', 0.0),
            total_generation_cost=data.get('total_generation_cost', 0.0)
        )


@dataclass
class ThumbnailGenerationStats:
    """Statistics for thumbnail generation performance tracking."""
    concepts_analyzed: int = 0
    variations_generated: int = 0
    ai_generations: int = 0
    procedural_generations: int = 0
    template_generations: int = 0
    total_processing_time: float = 0.0
    total_api_cost: float = 0.0
    average_confidence_score: float = 0.0
    cache_hit_rate: float = 0.0
    fallbacks_used: List[str] = field(default_factory=list)
    
    def add_generation(self, method: str, processing_time: float, cost: float, confidence: float):
        """Add generation statistics."""
        self.variations_generated += 1
        self.total_processing_time += processing_time
        self.total_api_cost += cost
        
        # Update average confidence score
        if self.variations_generated == 1:
            self.average_confidence_score = confidence
        else:
            self.average_confidence_score = (
                (self.average_confidence_score * (self.variations_generated - 1) + confidence) 
                / self.variations_generated
            )
        
        # Track generation method
        if method == "ai_generated":
            self.ai_generations += 1
        elif method == "procedural":
            self.procedural_generations += 1
        elif method == "template":
            self.template_generations += 1
    
    def add_fallback(self, fallback_type: str):
        """Record fallback usage."""
        self.fallbacks_used.append(fallback_type)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'concepts_analyzed': self.concepts_analyzed,
            'variations_generated': self.variations_generated,
            'ai_generations': self.ai_generations,
            'procedural_generations': self.procedural_generations,
            'template_generations': self.template_generations,
            'total_processing_time': self.total_processing_time,
            'total_api_cost': self.total_api_cost,
            'average_confidence_score': self.average_confidence_score,
            'cache_hit_rate': self.cache_hit_rate,
            'fallbacks_used': self.fallbacks_used
        }