"""
Content Analysis Module - Multi-modal content understanding and analysis.

This module provides comprehensive content analysis capabilities including
audio transcription, video analysis, emotional and engagement analysis,
and multi-modal content understanding for optimal video editing decisions.
"""

from .audio_analyzer import FinancialContentAnalyzer
from .video_analyzer import VideoAnalyzer
from .emotional_analyzer import (
    EmotionalAnalyzer, EmotionType, EmotionalPattern, EngagementMetrics,
    EmotionalAnalysisResult, create_emotional_analyzer
)
from .content_analyzer import (
    ContentAnalyzer,
    MultiModalContentAnalyzer,
    ConceptExtraction,
    ContentTypeDetection,
    MultiModalAnalysisResult,
    create_content_analyzer
)
from .broll_analyzer import FinancialBRollAnalyzer, BRollOpportunity
from .ai_graphics_director import (
    AIGraphicsDirector,
    GraphicsSpecification,
    FinancialGraphicsGenerator,
    EducationalSlideGenerator
)

__all__ = [
    'FinancialContentAnalyzer',
    'VideoAnalyzer',
    'EmotionalAnalyzer',
    'EmotionType',
    'EmotionalPattern',
    'EngagementMetrics',
    'EmotionalAnalysisResult',
    'create_emotional_analyzer',
    'ContentAnalyzer',
    'MultiModalContentAnalyzer',
    'ConceptExtraction',
    'ContentTypeDetection',
    'MultiModalAnalysisResult',
    'create_content_analyzer',
    'FinancialBRollAnalyzer',
    'BRollOpportunity',
    'AIGraphicsDirector',
    'GraphicsSpecification',
    'FinancialGraphicsGenerator',
    'EducationalSlideGenerator'
]