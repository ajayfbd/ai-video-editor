#!/usr/bin/env python3
"""Test AI Director integration with VideoComposer."""

from ai_video_editor.modules.video_processing.composer import VideoComposer
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
import json

def test_ai_director_integration():
    """Test if AI Director plans can be executed by VideoComposer."""
    
    # Load our AI Director plan
    with open('enhanced_ai_plan.json', 'r', encoding='utf-8') as f:
        ai_plan = json.load(f)

    # Create context with AI plan
    context = ContentContext(
        project_id='test_ai_integration',
        video_files=['media/test/Untitled video.mp4'],
        content_type=ContentType.MUSIC,
        user_preferences=UserPreferences()
    )

    # Add AI Director plan to context
    context.processed_video = ai_plan

    # Check if VideoComposer can validate the plan
    try:
        composer = VideoComposer()
        is_valid = composer.validate_ai_director_plan(context)
        print(f'AI Director plan validation: {is_valid}')
        
        if is_valid:
            print('✅ AI Director plan is ready for video composition!')
            print(f'Editing decisions: {len(ai_plan["editing_decisions"])}')
            print(f'B-roll plans: {len(ai_plan["broll_plans"])}')
            print('VideoComposer can execute this plan with movis!')
            
            # Test composition plan creation
            comp_plan = composer.create_composition_plan(context)
            print(f'Composition plan created with {len(comp_plan.layers)} layers')
            print(f'Total duration: {comp_plan.output_settings.duration:.2f}s')
            print(f'Output resolution: {comp_plan.output_settings.width}x{comp_plan.output_settings.height}')
            print(f'Transitions: {len(comp_plan.transitions)}')
            print(f'Effects: {len(comp_plan.effects)}')
            
            # Show layer details
            for i, layer in enumerate(comp_plan.layers[:3]):  # Show first 3 layers
                print(f'Layer {i+1}: {layer.layer_type} ({layer.start_time:.1f}s - {layer.end_time:.1f}s)')
            
        else:
            print('❌ AI Director plan needs refinement')
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_ai_director_integration()