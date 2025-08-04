#!/usr/bin/env python3
"""
Example usage of FinancialContentAnalyzer for audio transcription and analysis.

This example demonstrates how to use the Whisper integration for financial content analysis.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_video_editor.modules.content_analysis.audio_analyzer import FinancialContentAnalyzer
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.utils.logging_config import setup_logging, get_logger


def main():
    """Demonstrate audio analysis functionality."""
    # Setup logging
    setup_logging("INFO")
    logger = get_logger("ai_video_editor.examples.audio_analysis")
    
    logger.info("Starting audio analysis example")
    
    # Initialize the analyzer with cache manager
    from ai_video_editor.core.cache_manager import CacheManager
    cache_manager = CacheManager(cache_dir="temp/example_cache")
    analyzer = FinancialContentAnalyzer(cache_manager=cache_manager)
    
    # Create a sample ContentContext
    context = ContentContext(
        project_id="example-project",
        video_files=["example_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(quality_mode="balanced")
    )
    
    # Example 1: Demonstrate model loading
    logger.info("Example 1: Loading Whisper models")
    try:
        # Load different model sizes (this will download models if not cached)
        models = ['tiny', 'base', 'medium']
        for model_size in models:
            logger.info(f"Loading {model_size} model...")
            model = analyzer.get_model(model_size)
            logger.info(f"✓ {model_size} model loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
    
    # Example 2: Demonstrate enhanced financial content analysis with filler word detection
    logger.info("Example 2: Enhanced financial content analysis")
    
    # Create a mock transcript for demonstration with filler words
    from ai_video_editor.modules.content_analysis.audio_analyzer import Transcript, TranscriptSegment
    
    sample_segments = [
        TranscriptSegment("Welcome to our financial education series", 0.0, 3.0, 0.95),
        TranscriptSegment("Today we'll, um, explain compound interest and how it works", 3.0, 7.0, 0.92),
        TranscriptSegment("This chart shows the growth over time", 7.0, 10.0, 0.88),
        TranscriptSegment("Um, so basically the returns increase exponentially", 10.0, 14.0, 0.85),
        TranscriptSegment("Let me, like, show you an example with a portfolio", 14.0, 18.0, 0.90),
        TranscriptSegment("Risk tolerance is, you know, important for asset allocation", 18.0, 22.0, 0.87)
    ]
    
    sample_transcript = Transcript(
        text=" ".join([seg.text for seg in sample_segments]),
        segments=sample_segments,
        confidence=0.89,
        language="en",
        processing_time=2.5,
        model_used="medium"
    )
    
    # Analyze the financial content with enhancement
    financial_analysis = analyzer.analyze_financial_content(sample_transcript, enhance_audio=True)
    
    logger.info("Enhanced Financial Analysis Results:")
    logger.info(f"  Concepts mentioned: {financial_analysis.concepts_mentioned}")
    logger.info(f"  Explanation segments: {len(financial_analysis.explanation_segments)}")
    logger.info(f"  Data references: {len(financial_analysis.data_references)}")
    logger.info(f"  Filler word segments detected: {len(financial_analysis.filler_words_detected)}")
    logger.info(f"  Complexity level: {financial_analysis.complexity_level}")
    logger.info(f"  Emotional peaks: {len(financial_analysis.emotional_peaks)}")
    
    # Show audio enhancement results
    if financial_analysis.audio_enhancement:
        enhancement = financial_analysis.audio_enhancement
        logger.info("Audio Enhancement Results:")
        logger.info(f"  Original duration: {enhancement.original_duration:.1f}s")
        logger.info(f"  Enhanced duration: {enhancement.enhanced_duration:.1f}s")
        logger.info(f"  Filler words removed: {enhancement.filler_words_removed}")
        logger.info(f"  Segments modified: {enhancement.segments_modified}")
        logger.info(f"  Quality improvement score: {enhancement.quality_improvement_score:.2f}")
    
    # Show detailed filler word analysis
    if financial_analysis.filler_words_detected:
        logger.info("Detailed Filler Word Analysis:")
        for i, filler_seg in enumerate(financial_analysis.filler_words_detected[:3]):  # Show first 3
            logger.info(f"  Segment {i+1}:")
            logger.info(f"    Original: '{filler_seg.original_text}'")
            logger.info(f"    Cleaned: '{filler_seg.cleaned_text}'")
            logger.info(f"    Fillers detected: {filler_seg.filler_words}")
            logger.info(f"    Should remove: {filler_seg.should_remove}")
    
    # Example 3: ContentContext integration
    logger.info("Example 3: ContentContext integration")
    
    updated_context = analyzer.integrate_with_content_context(
        context, sample_transcript, financial_analysis
    )
    
    logger.info("ContentContext Integration Results:")
    logger.info(f"  Audio transcript length: {len(updated_context.audio_transcript)} characters")
    logger.info(f"  Key concepts: {updated_context.key_concepts}")
    logger.info(f"  Emotional markers: {len(updated_context.emotional_markers)}")
    logger.info(f"  Processing stage: {updated_context._processing_stage}")
    
    # Example 4: Demonstrate cache manager integration
    logger.info("Example 4: Cache manager integration")
    
    # Show cache statistics
    cache_stats = analyzer.cache_manager.get_stats()
    logger.info("Cache Statistics:")
    logger.info(f"  Memory cache size: {cache_stats['memory_cache_size']}")
    logger.info(f"  Cache hits: {cache_stats['hits']}")
    logger.info(f"  Cache misses: {cache_stats['misses']}")
    logger.info(f"  Hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Example 5: Demonstrate complexity assessment
    logger.info("Example 5: Complexity assessment examples")
    
    test_texts = [
        ("Beginner", "Save money in a bank account for emergencies"),
        ("Intermediate", "Diversify your portfolio with asset allocation strategies using compound interest and present value calculations for risk tolerance"),
        ("Advanced", "Use derivatives and options for hedging volatility with Monte Carlo analysis and Black Scholes models")
    ]
    
    for expected_level, text in test_texts:
        assessed_level = analyzer._assess_complexity(text)
        logger.info(f"  {expected_level} text → Assessed as: {assessed_level}")
    
    # Example 6: Batch processing simulation
    logger.info("Example 6: Batch processing simulation")
    
    # Simulate multiple audio files (without actual files)
    mock_files = ["clip1.mp3", "clip2.mp3", "clip3.mp3"]
    
    logger.info(f"Simulating batch processing of {len(mock_files)} files")
    logger.info("In real usage, this would:")
    logger.info("  1. Load audio files")
    logger.info("  2. Transcribe each file using Whisper")
    logger.info("  3. Analyze financial content")
    logger.info("  4. Return comprehensive results")
    
    # Cleanup
    logger.info("Cleaning up models...")
    analyzer.cleanup_models()
    
    logger.info("Audio analysis example completed successfully!")


if __name__ == "__main__":
    main()