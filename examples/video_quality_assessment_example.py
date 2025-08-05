#!/usr/bin/env python3
"""
Video Quality Assessment Example

This example demonstrates the comprehensive video quality assessment functionality
including automatic quality assessment, enhancement recommendations, and performance
benchmarking following the guidelines in .kiro/steering/performance-guidelines.md.
"""

import sys
import os
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_video_editor.modules.content_analysis.video_analyzer import create_video_analyzer
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.utils.cache_manager import CacheManager
from ai_video_editor.utils.performance_benchmarks import create_benchmark_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_video_context(video_path: str, content_type: ContentType = ContentType.EDUCATIONAL) -> ContentContext:
    """Create a sample ContentContext for video quality assessment."""
    return ContentContext(
        project_id="quality_assessment_example",
        video_files=[video_path],
        content_type=content_type,
        user_preferences=UserPreferences(
            quality_mode="high",
            thumbnail_resolution=(1920, 1080),
            batch_size=3,
            parallel_processing=True
        )
    )


def demonstrate_quality_assessment(video_path: str):
    """Demonstrate comprehensive video quality assessment."""
    logger.info("=== Video Quality Assessment Example ===")
    logger.info(f"Analyzing video: {video_path}")
    
    # Create video analyzer with caching
    cache_manager = CacheManager()
    video_analyzer = create_video_analyzer(cache_manager)
    
    # Create content context
    context = create_sample_video_context(video_path, ContentType.EDUCATIONAL)
    
    try:
        # Perform video quality assessment
        logger.info("Starting video quality assessment...")
        result_context = video_analyzer.assess_video_quality(video_path, context)
        
        # Display quality assessment results
        display_quality_results(result_context)
        
        # Display enhancement recommendations
        display_enhancement_recommendations(result_context)
        
        # Display performance metrics
        display_performance_metrics(result_context)
        
        # Demonstrate benchmark analysis
        demonstrate_benchmark_analysis()
        
        return result_context
        
    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        return None


def display_quality_results(context: ContentContext):
    """Display comprehensive quality assessment results."""
    if not context.video_quality_metrics:
        logger.warning("No quality metrics available")
        return
    
    metrics = context.video_quality_metrics
    
    logger.info("\n=== Quality Assessment Results ===")
    
    # Overall quality
    logger.info(f"Overall Quality: {metrics.overall_quality_score:.2f} ({metrics.quality_category.upper()})")
    
    # Resolution assessment
    logger.info(f"\nResolution Analysis:")
    logger.info(f"  - Resolution: {metrics.actual_resolution[0]}x{metrics.actual_resolution[1]}")
    logger.info(f"  - Resolution Score: {metrics.resolution_score:.2f}")
    logger.info(f"  - Category: {metrics.resolution_category}")
    
    # Lighting assessment
    logger.info(f"\nLighting Analysis:")
    logger.info(f"  - Lighting Score: {metrics.lighting_score:.2f}")
    logger.info(f"  - Brightness Mean: {metrics.brightness_mean:.1f}")
    logger.info(f"  - Brightness Std: {metrics.brightness_std:.1f}")
    logger.info(f"  - Exposure Quality: {metrics.exposure_quality}")
    
    # Stability assessment
    logger.info(f"\nStability Analysis:")
    logger.info(f"  - Stability Score: {metrics.stability_score:.2f}")
    logger.info(f"  - Motion Blur Level: {metrics.motion_blur_level:.2f}")
    logger.info(f"  - Camera Shake: {'Yes' if metrics.camera_shake_detected else 'No'}")
    logger.info(f"  - Stability Category: {metrics.stability_category}")
    
    # Color assessment
    logger.info(f"\nColor Analysis:")
    logger.info(f"  - Color Balance Score: {metrics.color_balance_score:.2f}")
    logger.info(f"  - Saturation Level: {metrics.saturation_level:.1f}")
    logger.info(f"  - Contrast Score: {metrics.contrast_score:.2f}")
    logger.info(f"  - Color Temperature: {metrics.color_temperature}")
    
    # Processing info
    logger.info(f"\nProcessing Information:")
    logger.info(f"  - Frames Analyzed: {metrics.frames_analyzed}")
    logger.info(f"  - Assessment Time: {metrics.assessment_time:.2f}s")


def display_enhancement_recommendations(context: ContentContext):
    """Display enhancement recommendations."""
    if not context.video_quality_metrics:
        return
    
    metrics = context.video_quality_metrics
    
    logger.info("\n=== Enhancement Recommendations ===")
    
    # Priority corrections needed
    corrections_needed = []
    if metrics.color_correction_needed:
        corrections_needed.append("Color Correction")
    if metrics.lighting_adjustment_needed:
        corrections_needed.append("Lighting Adjustment")
    if metrics.stabilization_needed:
        corrections_needed.append("Video Stabilization")
    
    if corrections_needed:
        logger.info(f"Priority Corrections Needed: {', '.join(corrections_needed)}")
    else:
        logger.info("No critical corrections needed")
    
    # Specific recommendations
    if metrics.enhancement_recommendations:
        logger.info("\nSpecific Recommendations:")
        for i, recommendation in enumerate(metrics.enhancement_recommendations, 1):
            logger.info(f"  {i}. {recommendation}")
    else:
        logger.info("\nNo specific enhancement recommendations")
    
    # AI Director insights
    ai_insights = context.get_video_quality_insights_for_ai_director()
    if ai_insights:
        logger.info(f"\nAI Director Quality Assessment:")
        logger.info(f"  - Overall Score: {ai_insights['overall_quality_score']:.2f}")
        logger.info(f"  - Quality Category: {ai_insights['quality_category']}")
        
        corrections = ai_insights.get('corrections_needed', {})
        if any(corrections.values()):
            needed = [k.replace('_', ' ').title() for k, v in corrections.items() if v]
            logger.info(f"  - Corrections Needed: {', '.join(needed)}")


def display_performance_metrics(context: ContentContext):
    """Display performance metrics and benchmarking results."""
    logger.info("\n=== Performance Metrics ===")
    
    # Processing metrics
    metrics = context.processing_metrics
    logger.info(f"Total Processing Time: {metrics.total_processing_time:.2f}s")
    
    if 'video_quality_assessment' in metrics.module_processing_times:
        assessment_time = metrics.module_processing_times['video_quality_assessment']
        logger.info(f"Quality Assessment Time: {assessment_time:.2f}s")
    
    logger.info(f"Memory Peak Usage: {metrics.memory_peak_usage / (1024**3):.2f}GB")
    
    # API usage
    if metrics.api_calls_made:
        logger.info(f"API Calls Made: {metrics.api_calls_made}")
    
    # Fallbacks and recovery
    if metrics.fallbacks_used:
        logger.info(f"Fallbacks Used: {metrics.fallbacks_used}")
    
    if metrics.recovery_actions:
        logger.info(f"Recovery Actions: {metrics.recovery_actions}")


def demonstrate_benchmark_analysis():
    """Demonstrate performance benchmark analysis."""
    logger.info("\n=== Performance Benchmark Analysis ===")
    
    try:
        # Load benchmark manager
        benchmark_manager = create_benchmark_manager()
        
        # Get performance statistics
        stats = benchmark_manager.get_performance_statistics()
        
        if stats:
            logger.info(f"Total Benchmarks: {stats['total_benchmarks']}")
            
            if 'processing_time' in stats:
                pt = stats['processing_time']
                logger.info(f"Processing Time - Mean: {pt['mean']:.2f}s, Min: {pt['min']:.2f}s, Max: {pt['max']:.2f}s")
            
            if 'memory_usage_gb' in stats:
                mem = stats['memory_usage_gb']
                logger.info(f"Memory Usage - Mean: {mem['mean']:.2f}GB, Min: {mem['min']:.2f}GB, Max: {mem['max']:.2f}GB")
            
            if 'frames_per_second' in stats:
                fps = stats['frames_per_second']
                logger.info(f"Processing Rate - Mean: {fps['mean']:.1f}fps, Min: {fps['min']:.1f}fps, Max: {fps['max']:.1f}fps")
        
        # Get regression analysis
        regression = benchmark_manager.get_regression_analysis()
        
        if regression.get('status') == 'analysis_complete':
            logger.info(f"\nPerformance Trend Analysis:")
            logger.info(f"  - Performance Trend: {regression['performance_trend'].upper()}")
            logger.info(f"  - Processing Time Change: {regression['processing_time_change_percent']:+.1f}%")
            logger.info(f"  - Memory Usage Change: {regression['memory_usage_change_percent']:+.1f}%")
            logger.info(f"  - FPS Change: {regression['fps_change_percent']:+.1f}%")
        else:
            logger.info(f"Regression Analysis: {regression.get('status', 'unknown')}")
    
    except Exception as e:
        logger.warning(f"Benchmark analysis failed: {e}")


def demonstrate_quality_categories():
    """Demonstrate different quality categories and their characteristics."""
    logger.info("\n=== Quality Categories Guide ===")
    
    categories = {
        "Excellent (0.8-1.0)": [
            "High resolution (1080p+)",
            "Optimal lighting and exposure",
            "Stable footage with minimal shake",
            "Good color balance and contrast",
            "Minimal enhancement needed"
        ],
        "Good (0.6-0.8)": [
            "Decent resolution (720p+)",
            "Acceptable lighting with minor issues",
            "Some camera movement but manageable",
            "Color balance mostly good",
            "Minor enhancements recommended"
        ],
        "Fair (0.4-0.6)": [
            "Lower resolution or quality issues",
            "Lighting problems (under/over exposed)",
            "Noticeable camera shake or blur",
            "Color balance issues",
            "Multiple enhancements needed"
        ],
        "Poor (0.0-0.4)": [
            "Very low resolution",
            "Severe lighting problems",
            "Significant stability issues",
            "Poor color balance and contrast",
            "Extensive enhancement required"
        ]
    }
    
    for category, characteristics in categories.items():
        logger.info(f"\n{category}:")
        for char in characteristics:
            logger.info(f"  • {char}")


def main():
    """Main function to run the video quality assessment example."""
    # Check if video path is provided
    if len(sys.argv) < 2:
        logger.info("Usage: python video_quality_assessment_example.py <video_path>")
        logger.info("\nThis example demonstrates comprehensive video quality assessment.")
        logger.info("If you don't have a video file, the example will show quality categories.")
        
        demonstrate_quality_categories()
        return
    
    video_path = sys.argv[1]
    
    # Check if video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Run quality assessment demonstration
    result_context = demonstrate_quality_assessment(video_path)
    
    if result_context:
        logger.info("\n=== Quality Assessment Complete ===")
        logger.info("The video has been analyzed and quality metrics have been stored.")
        logger.info("Enhancement recommendations are available for the AI Director.")
        logger.info("Performance benchmarks have been recorded for optimization tracking.")
        
        # Show how to use quality insights for AI Director
        if result_context.needs_quality_enhancement():
            priority_enhancements = result_context.get_priority_enhancements()
            logger.info(f"\nPriority enhancements for AI Director: {priority_enhancements}")
        else:
            logger.info("\nVideo quality is sufficient - no major enhancements needed.")
    
    else:
        logger.error("Quality assessment failed - please check the video file and try again.")


if __name__ == "__main__":
    main()