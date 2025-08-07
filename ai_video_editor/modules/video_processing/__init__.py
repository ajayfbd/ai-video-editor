"""
Video processing module for editing and format conversion.

This module provides professional video composition and editing capabilities
using the movis library for high-quality output generation.
"""

from .composer import VideoComposer, CompositionSettings, LayerInfo, CompositionPlan
from .broll_generation import (
    BRollGenerationSystem, 
    GeneratedBRollAsset,
    EnhancedChartGenerator,
    BlenderRenderingPipeline,
    EducationalSlideSystem
)

__all__ = [
    'VideoComposer',
    'CompositionSettings', 
    'LayerInfo',
    'CompositionPlan',
    'BRollGenerationSystem',
    'GeneratedBRollAsset',
    'EnhancedChartGenerator',
    'BlenderRenderingPipeline',
    'EducationalSlideSystem'
]