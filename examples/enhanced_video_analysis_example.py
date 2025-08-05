#!/usr/bin/env python3
"""
Enhanced Video Analysis Example

This example demonstrates the enhanced Visual Highlight Detection functionality
with Memory integration and confidence scoring for visual elements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_video_editor.modules.content_analysis.video_analyzer import create_video_analyzer
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.utils.cache_manager import CacheManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockMemoryClient:
    """Mock Memory client for demonstration purposes."""
    
    def __init__(self):
        self.stored_patterns = []
    
    def search_nodes(self, query):
        """Mock search for existing patterns."""
        return {'nodes': []}
    
    def create_entities(self, entities):
        """Mock entity creation."""
        self.stored_patterns.extend(entities)
        logger.info(f"Stored {len(entities)} entities in Memory")
        for entity in entities:
            logger.info(f"Entity: {entity['name']} with {len(entity['observations'])} observations")


def demonstrate_enhanced_video_analysis():
    """Demonstrate enhanced video analysis with Memory integration."""
    
    logger.info("=== Enhanced Video Analysis Demonstration ===")
    
    # Create mock Memory client
    memory_client = MockMemoryClient()
    
    # Create cache manager
    cache_manager = CacheManager()
    
    # Create enhanced video analyzer with Memory integration
    analyzer = create_video_analyzer(cache_manager, memory_client)
    
    logger.info("Created enhanced VideoAnalyzer with Memory integration")
    logger.info(f"Element detection weights: {analyzer.element_detection_weights}")
    
    # Create sample content context
    context = ContentContext(
        project_id="demo_project",
        video_files=["demo_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(quality_mode="high")
    )
    
    # Add some mock emotional markers for context
    context.add_emotional_marker(
        timestamp=15.0,
        emotion="excitement",
        intensity=0.8,
        confidence=0.9,
        context="Explaining key financial concept"
    )
    
    context.add_emotional_marker(
        timestamp=45.0,
        emotion="curiosity",
        intensity=0.7,
        confidence=0.85,
        context="Introducing new data visualization"
    )
    
    logger.info(f"Created ContentContext with {len(context.emotional_markers)} emotional markers")
    
    # Demonstrate enhanced visual element detection
    logger.info("\n=== Enhanced Visual Element Detection ===")
    
    # Mock frame for demonstration
    import numpy as np
    mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test text detection
    try:
        # Create mock gray frame for text detection
        gray_frame = np.mean(mock_frame, axis=2).astype(np.uint8)
        text_detection = analyzer._detect_text_regions(gray_frame)
        if text_detection:
            logger.info(f"Text detection: {text_detection.element_type} (confidence: {text_detection.confidence:.2f})")
        else:
            logger.info("No text regions detected in mock frame")
    except Exception as e:
        logger.info(f"Text detection demo skipped: {e}")
    
    # Test chart detection
    try:
        chart_detection = analyzer._detect_chart_elements(mock_frame)
        if chart_detection:
            logger.info(f"Chart detection: {chart_detection.element_type} (confidence: {chart_detection.confidence:.2f})")
        else:
            logger.info("No chart elements detected in mock frame")
    except Exception as e:
        logger.info(f"Chart detection demo skipped: {e}")
    
    # Test motion detection
    try:
        motion_detection = analyzer._detect_motion_elements(mock_frame)
        if motion_detection:
            logger.info(f"Motion detection: {motion_detection.element_type} (confidence: {motion_detection.confidence:.2f})")
        else:
            logger.info("No motion elements detected in mock frame")
    except Exception as e:
        logger.info(f"Motion detection demo skipped: {e}")
    
    # Test color detection
    try:
        # Convert to HSV for color detection
        import cv2
        hsv_frame = cv2.cvtColor(mock_frame, cv2.COLOR_BGR2HSV)
        color_detections = analyzer._detect_dominant_colors(hsv_frame)
        logger.info(f"Color detections: {len(color_detections)} elements found")
        for detection in color_detections[:3]:  # Show first 3
            logger.info(f"  - {detection.element_type} (confidence: {detection.confidence:.2f})")
    except Exception as e:
        logger.info(f"Color detection demo skipped: {e}")
    
    # Demonstrate enhanced thumbnail potential scoring
    logger.info("\n=== Enhanced Thumbnail Potential Scoring ===")
    
    from ai_video_editor.core.content_context import FaceDetection
    from ai_video_editor.modules.content_analysis.video_analyzer import VisualElementDetection
    
    # Create mock face detection
    mock_face = FaceDetection(
        bbox=[100.0, 150.0, 200.0, 250.0],
        confidence=0.85,
        expression="happy"
    )
    
    # Create mock visual elements
    mock_elements = [
        VisualElementDetection(
            element_type="text_overlay",
            confidence=0.8,
            properties={'text_regions_count': 3}
        ),
        VisualElementDetection(
            element_type="data_visualization",
            confidence=0.75,
            properties={'chart_type': 'bar_chart'}
        )
    ]
    
    # Calculate thumbnail potential
    potential = analyzer._calculate_thumbnail_potential(
        faces=[mock_face],
        visual_elements=mock_elements,
        quality_score=0.8,
        context=context
    )
    
    logger.info(f"Thumbnail potential score: {potential:.3f}")
    logger.info(f"Content type: {context.content_type.value}")
    logger.info(f"Face expression bonus applied: {mock_face.expression}")
    logger.info(f"Visual elements considered: {[elem.element_type for elem in mock_elements]}")
    
    # Demonstrate Memory pattern storage
    logger.info("\n=== Memory Pattern Storage ===")
    
    # Create mock frame analyses for pattern storage
    from ai_video_editor.modules.content_analysis.video_analyzer import FrameAnalysis
    
    mock_analyses = [
        FrameAnalysis(
            timestamp=15.0,
            frame_number=450,
            faces=[mock_face],
            visual_elements=mock_elements,
            motion_score=0.6,
            quality_score=0.8,
            thumbnail_potential=potential,
            scene_context="opening_scene",
            emotional_context="excitement_peak"
        )
    ]
    
    # Store patterns in Memory
    analyzer._store_visual_patterns(context, mock_analyses)
    
    # Show what was stored
    logger.info(f"Stored patterns in Memory: {len(memory_client.stored_patterns)} entities")
    for pattern in memory_client.stored_patterns:
        logger.info(f"Pattern observations:")
        for obs in pattern['observations'][:3]:  # Show first 3
            logger.info(f"  - {obs}")
    
    logger.info("\n=== Enhanced Video Analysis Demo Complete ===")
    logger.info("Key improvements demonstrated:")
    logger.info("✓ Confidence scoring for all visual element detection")
    logger.info("✓ Memory integration for pattern learning")
    logger.info("✓ Enhanced thumbnail potential calculation")
    logger.info("✓ Content-type specific adjustments")
    logger.info("✓ Scene and emotional context integration")


if __name__ == "__main__":
    demonstrate_enhanced_video_analysis()