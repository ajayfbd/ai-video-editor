#!/usr/bin/env python3
"""Test complete AI Director to movis video editing workflow."""

from ai_video_editor.modules.video_processing.composer import VideoComposer
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
import json

def test_full_ai_workflow():
    """Test complete workflow from AI Director plan to video composition."""
    
    print("🎬 Testing Complete AI Director → movis Workflow")
    print("=" * 50)
    
    # Load our AI Director plan
    with open('enhanced_ai_plan.json', 'r', encoding='utf-8') as f:
        ai_plan = json.load(f)

    # Create context with AI plan
    context = ContentContext(
        project_id='full_workflow_test',
        video_files=['media/test/Untitled video.mp4'],
        content_type=ContentType.MUSIC,
        user_preferences=UserPreferences()
    )

    # Add AI Director plan to context
    context.processed_video = ai_plan

    try:
        # Initialize VideoComposer
        composer = VideoComposer()
        
        # Step 1: Validate AI Director plan
        print("Step 1: Validating AI Director plan...")
        is_valid = composer.validate_ai_director_plan(context)
        print(f"✅ Plan validation: {is_valid}")
        
        if not is_valid:
            print("❌ AI Director plan is invalid")
            return
        
        # Step 2: Create composition plan
        print("\nStep 2: Creating composition plan...")
        comp_plan = composer.create_composition_plan(context)
        print(f"✅ Created composition with {len(comp_plan.layers)} layers")
        print(f"   Duration: {comp_plan.output_settings.duration:.2f}s")
        print(f"   Resolution: {comp_plan.output_settings.width}x{comp_plan.output_settings.height}")
        
        # Step 3: Show AI Director decisions being used
        print("\nStep 3: AI Director decisions being executed...")
        for i, decision in enumerate(ai_plan['editing_decisions'][:3]):
            print(f"   Decision {i+1}: {decision['decision_type']} at {decision['timestamp']:.1f}s")
            print(f"      Reason: {decision['parameters']['reason'][:60]}...")
        
        # Step 4: Show B-roll integration
        print(f"\nStep 4: B-roll integration...")
        for i, broll in enumerate(ai_plan['broll_plans']):
            print(f"   B-roll {i+1}: {broll['content_type']} at {broll['timestamp']:.1f}s")
            print(f"      Description: {broll['description'][:60]}...")
        
        # Step 5: Test plan execution (without actual rendering)
        print(f"\nStep 5: Testing plan execution engine...")
        from ai_video_editor.modules.video_processing.plan_execution import PlanExecutionEngine
        
        execution_engine = PlanExecutionEngine()
        timeline = execution_engine.execute_plan(context)
        
        print(f"✅ Execution timeline created:")
        print(f"   Total operations: {len(timeline.operations)}")
        print(f"   Sync points: {len(timeline.sync_points)}")
        print(f"   Timeline duration: {timeline.total_duration:.2f}s")
        print(f"   Conflicts resolved: {timeline.conflicts_resolved}")
        
        # Step 6: Show how operations map to movis
        print(f"\nStep 6: Video operations ready for movis:")
        for i, op in enumerate(timeline.operations[:5]):  # Show first 5 operations
            print(f"   Op {i+1}: {op.operation_type} on {op.track_type} track ({op.start_time:.1f}s-{op.end_time:.1f}s)")
        
        print(f"\n🎉 SUCCESS: AI Director plan successfully converted to executable video operations!")
        print(f"📊 Summary:")
        print(f"   • {len(ai_plan['editing_decisions'])} AI editing decisions")
        print(f"   • {len(ai_plan['broll_plans'])} B-roll opportunities")
        print(f"   • {len(timeline.operations)} video operations")
        print(f"   • {len(comp_plan.layers)} composition layers")
        print(f"   • Ready for movis rendering!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_full_ai_workflow()
    if success:
        print("\n✨ AI Director → movis integration is FULLY FUNCTIONAL!")
    else:
        print("\n💥 Workflow needs debugging")