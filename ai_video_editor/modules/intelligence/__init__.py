"""
Intelligence Layer Module - AI Director and decision-making components.

This module contains the AI Director and related intelligence components
that make creative and strategic decisions for video editing.
"""

from .gemini_client import GeminiClient
from .trend_analyzer import TrendAnalyzer

__all__ = ['GeminiClient', 'TrendAnalyzer']