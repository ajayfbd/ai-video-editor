"""
AI Director Plan Execution Engine Example

This example demonstrates how to use the AI Director Plan Execution Engine
to translate high-level creative decisions into precise video operations.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_video_editor.modules.video_processing.plan_execution import (
    PlanExecutionEngine,
    ExecutionCoordinator,
    EditingDecisionInterpreter,
    TimelineManager,
    BRollInsertionManager,
    AudioVideoSynchronizer
)
from ai_video_editor.modules.video_processing.composer import VideoComposer
from ai_video_editor.modules.intelligence.ai_director import (
    EditingDecision,
    BRollPlan,
    MetadataStrategy,
    AIDirectorPlan
)
from ai_video_editor.core.content_context import (
    ContentContext,
    ContentType,
    UserPreferences,
    AudioAnalysisResult,
    AudioSegment,
    EmotionalPeak,
    VisualHighlight,
    FaceDetection
)
from ai_video_editor.core.exceptions import ContentContextError


def create_sample_content_context():
    """Create a sample ContentContext with AI Director plan for demonstration."""
    
    # Create user preferences
    user_preferences = UserPreferences(
        quality_mode="high"
    )
    
    # Create sample audio analysis
    audio_segments = [
        AudioSegment(
            text="Welcome to our financial education series.",
            start=0.0,
            end=3.5,
            confidence=0.95,
            financial_concepts=["financial education"]
        ),
        AudioSegment(
            text="Today we'll explore compound interest and how it can grow your investments.",
            start=3.5,
            end=8.2,
            confidence=0.92,
            financial_concepts=["compound interest", "investments"]
        ),
        AudioSegment(
            text="Let's start with a simple example to illustrate the power of compounding.",
            start=8.2,
            end=13.1,
            confidence=0.94,
            financial_concepts=["compounding"]
        ),
        AudioSegment(
            text="If you invest $1000 at 7% annual return, after 10 years you'll have $1967.",
            start=13.1,
            end=19.8,
            confidence=0.91,
            financial_concepts=["annual return"]
        ),
        AudioSegment(
            text="But here's where it gets interesting - the growth accelerates over time.",
            start=19.8,
            end=24.5,
            confidence=0.93
        ),
        AudioSegment(
            text="After 20 years, that same $1000 becomes $3870, and after 30 years, $7612.",
            start=24.5,
            end=31.2,
            confidence=0.89
        ),
        AudioSegment(
            text="This exponential growth is the magic of compound interest.",
            start=31.2,
            end=35.8,
            confidence=0.96,
            financial_concepts=["compound interest"]
        )
    ]
    
    audio_analysis = AudioAnalysisResult(
        transcript_text=" ".join([seg.text for seg in audio_segments]),
        segments=audio_segments,
        overall_confidence=0.93,
        language="en",
        processing_time=2.1,
        model_used="whisper-large-v3",
        financial_concepts=["financial education", "compound interest", "investments", "compounding", "annual return"],
        complexity_level="medium"
    )
    
    # Create emotional peaks
    emotional_markers = [
        EmotionalPeak(
            timestamp=8.2,
            emotion="curiosity",
            intensity=0.8,
            confidence=0.85,
            context="Introduction of compound interest concept"
        ),
        EmotionalPeak(
            timestamp=19.8,
            emotion="excitement",
            intensity=0.9,
            confidence=0.88,
            context="Revealing the accelerating growth pattern"
        ),
        EmotionalPeak(
            timestamp=31.2,
            emotion="amazement",
            intensity=0.85,
            confidence=0.82,
            context="Demonstrating the magic of compound interest"
        )
    ]
    
    # Create visual highlights
    visual_highlights = [
        VisualHighlight(
            timestamp=13.1,
            description="Speaker explaining with hand gestures",
            faces=[FaceDetection([100, 50, 200, 250], 0.95, "engaged")],
            visual_elements=["speaker", "hand_gestures"],
            thumbnail_potential=0.7
        ),
        VisualHighlight(
            timestamp=24.5,
            description="Speaker with excited expression",
            faces=[FaceDetection([120, 60, 180, 230], 0.92, "excited")],
            visual_elements=["speaker", "excited_expression"],
            thumbnail_potential=0.9
        )
    ]
    
    # Create ContentContext
    context = ContentContext(
        project_id="compound_interest_demo",
        video_files=["examples/sample_videos/compound_interest_explanation.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=user_preferences
    )
    
    # Set analysis results
    context.set_audio_analysis(audio_analysis)
    context.emotional_markers = emotional_markers
    context.visual_highlights = visual_highlights
    context.key_concepts = ["compound interest", "investment growth", "financial education", "exponential growth"]
    context.content_themes = ["personal finance", "investment strategy", "wealth building"]
    
    # Create AI Director plan
    ai_director_plan = {
        'editing_decisions': [
            # Cut at natural pause after introduction
            {
                'timestamp': 3.5,
                'decision_type': 'cut',
                'parameters': {
                    'duration': 0.3,
                    'cut_type': 'soft',
                    'fade_duration': 0.1,
                    'preserve_audio': True
                },
                'rationale': 'Natural pause after introduction',
                'confidence': 0.9,
                'priority': 8
            },
            
            # Emphasis on key concept introduction
            {
                'timestamp': 8.2,
                'decision_type': 'emphasis',
                'parameters': {
                    'duration': 2.5,
                    'emphasis_type': 'zoom_in',
                    'intensity': 1.2,
                    'fade_in': 0.3,
                    'fade_out': 0.3
                },
                'rationale': 'Emphasize compound interest concept introduction',
                'confidence': 0.85,
                'priority': 9
            },
            
            # Trim unnecessary pause
            {
                'timestamp': 12.8,
                'decision_type': 'trim',
                'parameters': {
                    'duration': 0.8,
                    'trim_type': 'remove',
                    'smooth_transition': True,
                    'fade_in': 0.2,
                    'fade_out': 0.2
                },
                'rationale': 'Remove unnecessary pause before example',
                'confidence': 0.8,
                'priority': 6
            },
            
            # Transition to exciting revelation
            {
                'timestamp': 19.8,
                'decision_type': 'transition',
                'parameters': {
                    'duration': 1.0,
                    'type': 'crossfade',
                    'easing': 'ease_in_out',
                    'direction': 'forward'
                },
                'rationale': 'Smooth transition to exciting growth revelation',
                'confidence': 0.88,
                'priority': 7
            },
            
            # Final emphasis on the magic concept
            {
                'timestamp': 31.2,
                'decision_type': 'emphasis',
                'parameters': {
                    'duration': 3.0,
                    'emphasis_type': 'highlight',
                    'intensity': 1.3,
                    'fade_in': 0.4,
                    'fade_out': 0.4
                },
                'rationale': 'Highlight the magic of compound interest conclusion',
                'confidence': 0.92,
                'priority': 9
            }
        ],
        
        'broll_plans': [
            # Chart showing initial investment
            {
                'timestamp': 13.1,
                'duration': 6.7,
                'content_type': 'chart',
                'description': 'Bar chart showing $1000 investment growing to $1967 over 10 years',
                'visual_elements': ['bar_chart', 'dollar_amounts', 'timeline'],
                'animation_style': 'slide_up',
                'priority': 8
            },
            
            # Animated growth curve
            {
                'timestamp': 24.5,
                'duration': 6.7,
                'content_type': 'animation',
                'description': 'Animated exponential growth curve showing acceleration over time',
                'visual_elements': ['growth_curve', 'data_points', 'trend_line'],
                'animation_style': 'draw_on',
                'priority': 9
            },
            
            # Concept visualization
            {
                'timestamp': 31.2,
                'duration': 4.6,
                'content_type': 'graphic',
                'description': 'Visual metaphor for compound interest magic - snowball effect',
                'visual_elements': ['snowball', 'growth_spiral', 'magic_sparkles'],
                'animation_style': 'zoom_in',
                'priority': 8
            }
        ],
        
        'metadata_strategy': {
            'primary_title': 'The Magic of Compound Interest: How $1000 Becomes $7612',
            'title_variations': [
                'Compound Interest Explained: Turn $1000 into $7612',
                'The Power of Compound Interest: Real Examples',
                'How Compound Interest Creates Wealth Over Time'
            ],
            'description': 'Learn how compound interest can dramatically grow your investments over time. See real examples of how $1000 can become $7612 through the power of compounding. Perfect for beginners in personal finance and investing.',
            'tags': ['compound interest', 'investing', 'personal finance', 'wealth building', 'financial education', 'investment growth', 'money management', 'financial literacy'],
            'thumbnail_concepts': ['excited speaker with growth chart', 'money growing visualization', 'compound interest formula'],
            'hook_text': 'Turn $1000 into $7612!',
            'target_keywords': ['compound interest', 'investment growth', 'financial education']
        }
    }
    
    # Store AI Director plan in context
    context.processed_video = ai_director_plan
    
    return context


def demonstrate_plan_execution_components():
    """Demonstrate individual components of the plan execution engine."""
    
    print("🎬 AI Director Plan Execution Engine Demo")
    print("=" * 50)
    
    # Create sample context
    context = create_sample_content_context()
    
    print(f"\n📋 Sample Project: {context.project_id}")
    print(f"Content Type: {context.content_type.value}")
    print(f"Video Files: {len(context.video_files)}")
    print(f"Key Concepts: {', '.join(context.key_concepts[:3])}...")
    
    # 1. Demonstrate Editing Decision Interpreter
    print("\n🎯 1. Editing Decision Interpreter")
    print("-" * 30)
    
    interpreter = EditingDecisionInterpreter()
    
    # Extract editing decisions from context
    editing_decisions_data = context.processed_video['editing_decisions']
    editing_decisions = []
    
    for decision_data in editing_decisions_data:
        decision = EditingDecision(
            timestamp=decision_data['timestamp'],
            decision_type=decision_data['decision_type'],
            parameters=decision_data['parameters'],
            rationale=decision_data['rationale'],
            confidence=decision_data['confidence'],
            priority=decision_data['priority']
        )
        editing_decisions.append(decision)
    
    # Interpret decisions
    operations = interpreter.interpret_decisions(editing_decisions, context)
    
    print(f"Interpreted {len(editing_decisions)} AI Director decisions into {len(operations)} track operations:")
    for i, operation in enumerate(operations[:5]):  # Show first 5
        print(f"  {i+1}. {operation.operation_type.upper()} on {operation.track_type} track at {operation.start_time:.1f}s")
        print(f"     Priority: {operation.priority}, Duration: {operation.end_time - operation.start_time:.1f}s")
    
    if len(operations) > 5:
        print(f"  ... and {len(operations) - 5} more operations")
    
    # 2. Demonstrate Timeline Manager
    print("\n⏱️ 2. Timeline Manager")
    print("-" * 20)
    
    timeline_manager = TimelineManager()
    timeline = timeline_manager.create_timeline(operations, 40.0)  # ~40 second video
    
    print(f"Created timeline with:")
    print(f"  • Total Duration: {timeline.total_duration:.1f} seconds")
    print(f"  • Operations: {len(timeline.operations)}")
    print(f"  • Sync Points: {len(timeline.sync_points)}")
    print(f"  • Conflicts Resolved: {timeline.conflicts_resolved}")
    print(f"  • Tracks: {len(timeline.track_mapping)}")
    
    # Show track mapping
    print(f"  • Track Types: {', '.join(set(timeline.track_mapping.values()))}")
    
    # 3. Demonstrate B-roll Insertion Manager
    print("\n🎨 3. B-roll Insertion Manager")
    print("-" * 28)
    
    broll_manager = BRollInsertionManager()
    
    # Extract B-roll plans from context
    broll_plans_data = context.processed_video['broll_plans']
    broll_plans = []
    
    for plan_data in broll_plans_data:
        plan = BRollPlan(
            timestamp=plan_data['timestamp'],
            duration=plan_data['duration'],
            content_type=plan_data['content_type'],
            description=plan_data['description'],
            visual_elements=plan_data['visual_elements'],
            animation_style=plan_data['animation_style'],
            priority=plan_data['priority']
        )
        broll_plans.append(plan)
    
    broll_operations = broll_manager.process_broll_plans(broll_plans, context)
    
    print(f"Processed {len(broll_plans)} B-roll plans into {len(broll_operations)} operations:")
    for i, plan in enumerate(broll_plans):
        print(f"  {i+1}. {plan.content_type.upper()} at {plan.timestamp:.1f}s for {plan.duration:.1f}s")
        print(f"     Description: {plan.description[:50]}...")
        print(f"     Animation: {plan.animation_style}, Priority: {plan.priority}")
    
    # 4. Demonstrate Audio-Video Synchronizer
    print("\n🔄 4. Audio-Video Synchronizer")
    print("-" * 26)
    
    synchronizer = AudioVideoSynchronizer()
    
    # Add B-roll operations to timeline
    all_operations = timeline.operations + broll_operations
    enhanced_timeline = timeline_manager.create_timeline(all_operations, 40.0)
    
    # Apply synchronization
    synchronized_timeline = synchronizer.synchronize_timeline(enhanced_timeline, context)
    
    print(f"Synchronization applied:")
    print(f"  • Sync Adjustments Made: {synchronizer.sync_adjustments}")
    print(f"  • Total Sync Points: {len(synchronized_timeline.sync_points)}")
    print(f"  • Timeline Optimized: {synchronized_timeline.optimization_applied}")
    
    return synchronized_timeline


def demonstrate_full_execution_workflow():
    """Demonstrate the complete execution workflow."""
    
    print("\n🚀 5. Complete Execution Workflow")
    print("-" * 32)
    
    # Create execution engine
    execution_engine = PlanExecutionEngine()
    
    # Create sample context
    context = create_sample_content_context()
    
    print("Executing complete AI Director plan...")
    start_time = time.time()
    
    try:
        # Execute the plan
        final_timeline = execution_engine.execute_plan(context)
        
        execution_time = time.time() - start_time
        
        print(f"✅ Plan execution completed in {execution_time:.2f} seconds!")
        
        # Display results
        print(f"\n📊 Final Timeline Summary:")
        print(f"  • Total Duration: {final_timeline.total_duration:.1f} seconds")
        print(f"  • Total Operations: {len(final_timeline.operations)}")
        print(f"  • Sync Points: {len(final_timeline.sync_points)}")
        print(f"  • Conflicts Resolved: {final_timeline.conflicts_resolved}")
        print(f"  • Optimization Applied: {final_timeline.optimization_applied}")
        
        # Show operation breakdown
        operation_types = {}
        track_types = {}
        
        for operation in final_timeline.operations:
            operation_types[operation.operation_type] = operation_types.get(operation.operation_type, 0) + 1
            track_types[operation.track_type] = track_types.get(operation.track_type, 0) + 1
        
        print(f"\n📈 Operation Breakdown:")
        for op_type, count in operation_types.items():
            print(f"  • {op_type.title()}: {count}")
        
        print(f"\n🎵 Track Breakdown:")
        for track_type, count in track_types.items():
            print(f"  • {track_type.title()}: {count}")
        
        # Show execution statistics
        stats = execution_engine.get_execution_stats()
        print(f"\n📊 Execution Statistics:")
        print(f"  • Total Executions: {stats['total_executions']}")
        print(f"  • Average Operations: {stats['average_operations']:.1f}")
        print(f"  • Average Duration: {stats['average_duration']:.1f}s")
        
        return final_timeline
        
    except Exception as e:
        print(f"❌ Plan execution failed: {str(e)}")
        return None


def demonstrate_video_composer_integration():
    """Demonstrate integration with VideoComposer."""
    
    print("\n🎥 6. VideoComposer Integration")
    print("-" * 28)
    
    try:
        # Note: This would require movis to be installed
        print("Note: VideoComposer integration requires 'movis' library to be installed.")
        print("Install with: pip install movis")
        
        # Create sample context
        context = create_sample_content_context()
        
        print(f"\nVideoComposer would process the AI Director plan to create:")
        print(f"  • Professional video composition using movis")
        print(f"  • Multi-track timeline with {len(context.processed_video['editing_decisions'])} editing operations")
        print(f"  • {len(context.processed_video['broll_plans'])} B-roll overlays")
        print(f"  • Synchronized audio-video output")
        print(f"  • High-quality MP4 export")
        
        # Show what the composition would include
        ai_plan = context.processed_video
        
        print(f"\n🎬 Composition Elements:")
        print(f"  • Main video track with {len([d for d in ai_plan['editing_decisions'] if d['decision_type'] in ['cut', 'trim']])} cuts/trims")
        print(f"  • {len([d for d in ai_plan['editing_decisions'] if d['decision_type'] == 'transition'])} transitions")
        print(f"  • {len([d for d in ai_plan['editing_decisions'] if d['decision_type'] == 'emphasis'])} emphasis effects")
        print(f"  • {len(ai_plan['broll_plans'])} B-roll overlays")
        
        print(f"\n📋 Metadata Package:")
        metadata = ai_plan['metadata_strategy']
        print(f"  • Title: {metadata['primary_title']}")
        print(f"  • Description: {metadata['description'][:80]}...")
        print(f"  • Tags: {', '.join(metadata['tags'][:5])}...")
        print(f"  • Hook Text: {metadata['hook_text']}")
        
    except Exception as e:
        print(f"❌ VideoComposer integration demo failed: {str(e)}")


def save_timeline_to_json(timeline, filename="timeline_output.json"):
    """Save the execution timeline to JSON for inspection."""
    
    if timeline:
        timeline_dict = timeline.to_dict()
        
        output_path = Path("examples") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(timeline_dict, f, indent=2, default=str)
        
        print(f"\n💾 Timeline saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """Run the complete demonstration."""
    
    print("🎬 AI Director Plan Execution Engine")
    print("Complete Demonstration")
    print("=" * 50)
    
    try:
        # Demonstrate individual components
        timeline = demonstrate_plan_execution_components()
        
        # Demonstrate full workflow
        final_timeline = demonstrate_full_execution_workflow()
        
        # Demonstrate VideoComposer integration
        demonstrate_video_composer_integration()
        
        # Save timeline for inspection
        if final_timeline:
            save_timeline_to_json(final_timeline)
        
        print("\n✅ Demonstration completed successfully!")
        print("\nThe AI Director Plan Execution Engine successfully:")
        print("  • Interpreted high-level AI Director decisions")
        print("  • Created precise video operations")
        print("  • Managed multi-track timeline coordination")
        print("  • Resolved operation conflicts")
        print("  • Applied audio-video synchronization")
        print("  • Generated comprehensive execution timeline")
        
        print("\n🎯 Next Steps:")
        print("  • Install 'movis' library for video composition")
        print("  • Run with actual video files for full workflow")
        print("  • Integrate with AI Director for live plan generation")
        print("  • Customize operation parameters for specific content types")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()