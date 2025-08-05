"""
Video Analysis Example - Demonstrates VideoAnalyzer usage.

This example shows how to use the VideoAnalyzer module to perform
comprehensive video analysis including scene detection, face detection,
and visual element analysis.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_video_editor.modules.content_analysis.video_analyzer import create_video_analyzer
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.utils.cache_manager import CacheManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate video analysis functionality."""
    print("=== AI Video Editor - Video Analysis Example ===\n")
    
    # Create sample video file path (you would use a real video file)
    video_path = "sample_video.mp4"  # Replace with actual video path
    
    if not os.path.exists(video_path):
        print(f"Note: Video file '{video_path}' not found.")
        print("This example demonstrates the API usage with a mock video path.")
        print("In real usage, provide a valid video file path.\n")
    
    try:
        # Initialize cache manager for performance
        cache_manager = CacheManager()
        
        # Create video analyzer
        print("1. Initializing VideoAnalyzer...")
        video_analyzer = create_video_analyzer(cache_manager)
        print("   ✓ VideoAnalyzer initialized successfully\n")
        
        # Create content context
        print("2. Creating ContentContext...")
        context = ContentContext(
            project_id="video_analysis_example",
            video_files=[video_path],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences(
                quality_mode="balanced",
                thumbnail_resolution=(1920, 1080),
                batch_size=3
            )
        )
        print(f"   ✓ ContentContext created for project: {context.project_id}\n")
        
        # Perform video analysis (this would work with a real video file)
        print("3. Performing video analysis...")
        print("   Note: This would analyze the actual video file if it existed.")
        print("   Analysis includes:")
        print("   - Video metadata extraction using ffmpeg")
        print("   - Scene detection using PySceneDetect")
        print("   - Frame-by-frame face detection")
        print("   - Visual element analysis for B-roll opportunities")
        print("   - Quality assessment and thumbnail potential scoring\n")
        
        # In a real scenario, you would call:
        # analyzed_context = video_analyzer.analyze_video(video_path, context)
        
        # Demonstrate batch processing
        print("4. Batch Processing Example:")
        video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
        print(f"   Processing {len(video_files)} video files in batch...")
        print("   Benefits of batch processing:")
        print("   - Optimized memory usage")
        print("   - Shared model initialization")
        print("   - Consolidated error handling")
        print("   - Progress tracking across multiple files\n")
        
        # In a real scenario:
        # batch_context = video_analyzer.analyze_batch(video_files, context)
        
        # Show expected results structure
        print("5. Expected Analysis Results:")
        print("   After analysis, ContentContext would contain:")
        print("   - video_metadata: Duration, FPS, resolution, codec info")
        print("   - visual_highlights: Key frames with high thumbnail potential")
        print("   - Face detection data with expressions and landmarks")
        print("   - Scene boundaries and transitions")
        print("   - Visual elements (text, charts, motion) for B-roll planning")
        print("   - Quality scores and processing metrics\n")
        
        # Demonstrate ContentContext integration
        print("6. ContentContext Integration:")
        print("   The VideoAnalyzer integrates seamlessly with ContentContext:")
        
        # Show how visual highlights would be accessed
        print("   - Access visual highlights: context.visual_highlights")
        print("   - Get best thumbnail candidates: context.get_best_visual_highlights(5)")
        print("   - Retrieve video metadata: context.video_metadata")
        print("   - Check processing metrics: context.processing_metrics")
        
        # Show error handling
        print("\n7. Error Handling:")
        print("   The VideoAnalyzer includes comprehensive error handling:")
        print("   - Graceful degradation when models fail to load")
        print("   - Fallback to Haar cascades if YuNet is unavailable")
        print("   - Memory management for large video files")
        print("   - Recovery from corrupted frames or invalid video formats")
        
        print("\n=== Video Analysis Example Complete ===")
        print("To use with real video files:")
        print("1. Install required dependencies: opencv-python, scenedetect, ffmpeg-python")
        print("2. Provide valid video file paths")
        print("3. The analyzer will process the videos and populate ContentContext")
        print("4. Use the results for thumbnail generation and metadata optimization")
        
    except Exception as e:
        logger.error(f"Video analysis example failed: {e}")
        print(f"\nError: {e}")
        print("This is expected when running without actual video files.")


def demonstrate_api_usage():
    """Demonstrate the VideoAnalyzer API in detail."""
    print("\n=== VideoAnalyzer API Usage Examples ===\n")
    
    # Show initialization options
    print("1. Initialization Options:")
    print("""
    # Basic initialization
    analyzer = create_video_analyzer()
    
    # With cache manager for performance
    cache_manager = CacheManager()
    analyzer = create_video_analyzer(cache_manager)
    
    # Direct instantiation with custom parameters
    analyzer = VideoAnalyzer(cache_manager)
    analyzer.scene_threshold = 25.0  # Adjust scene detection sensitivity
    analyzer.face_confidence_threshold = 0.8  # Higher face detection confidence
    analyzer.thumbnail_sample_rate = 0.5  # Sample every 0.5 seconds
    """)
    
    print("2. Single Video Analysis:")
    print("""
    # Analyze single video
    context = ContentContext(
        project_id="my_project",
        video_files=["my_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences()
    )
    
    analyzed_context = analyzer.analyze_video("my_video.mp4", context)
    
    # Access results
    highlights = analyzed_context.visual_highlights
    metadata = analyzed_context.video_metadata
    processing_time = analyzed_context.processing_metrics.total_processing_time
    """)
    
    print("3. Batch Processing:")
    print("""
    # Process multiple videos
    video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
    batch_context = analyzer.analyze_batch(video_files, context)
    
    # Results are consolidated in the same ContentContext
    all_highlights = batch_context.visual_highlights
    """)
    
    print("4. Working with Results:")
    print("""
    # Get best thumbnail candidates
    best_highlights = context.get_best_visual_highlights(count=5)
    
    # Find faces with specific expressions
    happy_faces = []
    for highlight in context.visual_highlights:
        for face in highlight.faces:
            if face.expression == "happy" and face.confidence > 0.8:
                happy_faces.append(face)
    
    # Get high-quality frames
    quality_frames = [h for h in context.visual_highlights if h.thumbnail_potential > 0.7]
    
    # Access video metadata
    if context.video_metadata:
        duration = context.video_metadata['duration']
        fps = context.video_metadata['fps']
        resolution = (context.video_metadata['width'], context.video_metadata['height'])
    """)
    
    print("5. Integration with Other Modules:")
    print("""
    # The analyzed context flows to other modules
    
    # AI Director uses visual highlights for B-roll planning
    ai_director = AIDirector()
    editing_plan = ai_director.create_editing_plan(analyzed_context)
    
    # Thumbnail generator uses visual highlights
    thumbnail_gen = ThumbnailGenerator()
    thumbnails = thumbnail_gen.generate_thumbnails(analyzed_context)
    
    # Metadata generator considers visual content
    metadata_gen = MetadataGenerator()
    seo_metadata = metadata_gen.generate_metadata(analyzed_context)
    """)


if __name__ == "__main__":
    main()
    demonstrate_api_usage()