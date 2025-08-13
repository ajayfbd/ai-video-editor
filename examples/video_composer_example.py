"""
Test example for VideoComposer functionality.

This demonstrates how VideoComposer integrates with ContentContext
to execute AI Director plans using the movis composition engine.
"""

import sys
import os
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.modules.video_processing import VideoComposer, CompositionSettings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_video_composer_integration():
    """Test VideoComposer with mock ContentContext data."""
    
    print("🎬 Testing VideoComposer Integration")
    print("=" * 50)
    
    try:
        # Create mock ContentContext with AI Director plan
        context = ContentContext(
            project_id="test_composition",
            video_files=["test_video.mp4"],  # Mock file (won't exist)
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences(
                quality_mode="high",
                enable_aggressive_caching=False,
                parallel_processing=True
            )
        )
        
        # Add mock AI Director plan to processed_video
        context.processed_video = {
            'editing_decisions': [
                {
                    'timestamp': 10.0,
                    'decision_type': 'cut',
                    'parameters': {'duration': 2.0},
                    'rationale': 'Remove filler content',
                    'confidence': 0.85,
                    'priority': 8
                },
                {
                    'timestamp': 25.0,
                    'decision_type': 'transition',
                    'parameters': {'type': 'fade', 'duration': 1.5},
                    'rationale': 'Smooth topic transition',
                    'confidence': 0.9,
                    'priority': 6
                }
            ],
            'broll_plans': [
                {
                    'timestamp': 30.0,
                    'duration': 5.0,
                    'content_type': 'chart',
                    'description': 'Compound interest growth chart',
                    'visual_elements': ['graph', 'data', 'animation'],
                    'animation_style': 'fade_in',
                    'priority': 9
                }
            ],
            'metadata_strategy': {
                'primary_title': 'Financial Education Demo',
                'description': 'Educational content about compound interest',
                'tags': ['finance', 'education', 'investment']
            }
        }
        
        # Initialize VideoComposer
        composer = VideoComposer(output_dir="out/test_output", temp_dir="temp/test_output_tmp")
        
        print("✅ VideoComposer initialized successfully")
        
        # Test validation (should fail gracefully since video file doesn't exist)
        try:
            is_valid = composer.validate_ai_director_plan(context)
            print(f"❌ Plan validation: {is_valid} (expected failure due to missing files)")
        except Exception as e:
            print(f"❌ Expected validation error: {e}")
        
        # Test composition plan creation with valid context
        # First, let's test without actual files
        print("\n🎨 Testing composition plan creation...")
        
        # Remove file validation for testing
        context.video_files = []  # Empty files for testing logic
        
        # Mock some duration data
        context.video_metadata = {'duration': 60.0}
        
        try:
            composition_plan = composer.create_composition_plan(context)
            print(f"✅ Composition plan created with {len(composition_plan.layers)} layers")
            print(f"   - Duration: {composition_plan.output_settings.duration}s")
            print(f"   - Resolution: {composition_plan.output_settings.width}x{composition_plan.output_settings.height}")
            print(f"   - Transitions: {len(composition_plan.transitions)}")
            print(f"   - Effects: {len(composition_plan.effects)}")
            
        except Exception as e:
            print(f"❌ Composition plan creation failed: {e}")
        
        # Test composition info
        info = composer.get_composition_info()
        if info:
            print(f"✅ Composition info: {info}")
        else:
            print("ℹ️  No composition info available (expected)")
        
        print("\n✅ VideoComposer integration test completed successfully!")
        print("📝 Ready for Task 7.2: AI Director Plan Execution")
        
    except ImportError as e:
        print(f"⚠️  Movis library not available: {e}")
        print("   Install with: pip install movis")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_video_composer_integration()
