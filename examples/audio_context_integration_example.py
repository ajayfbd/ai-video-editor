"""
Audio-ContentContext Integration Example.

This example demonstrates how to integrate audio analysis results from the
FinancialContentAnalyzer into the ContentContext system for downstream processing.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_video_editor.core.content_context import (
    ContentContext, 
    ContentType, 
    UserPreferences,
    AudioSegment,
    AudioAnalysisResult,
    EmotionalPeak
)
from ai_video_editor.core.context_manager import ContextManager
from ai_video_editor.core.audio_integration import (
    convert_financial_analysis_to_audio_result,
    integrate_audio_analysis_to_context,
    extract_audio_insights_for_downstream,
    validate_audio_analysis_integration
)


def create_mock_audio_analysis_data():
    """Create mock audio analysis data for demonstration."""
    
    # Mock transcript data (as would come from Whisper)
    transcript_data = {
        'text': 'Welcome to our financial education series. Today we will discuss investment strategies and portfolio diversification. Um, let me explain compound interest with a simple example. The chart shows how your money grows over time.',
        'language': 'en',
        'confidence': 0.92,
        'processing_time': 12.5,
        'model_used': 'medium',
        'segments': [
            {
                'text': 'Welcome to our financial education series.',
                'start': 0.0,
                'end': 3.2,
                'confidence': 0.95
            },
            {
                'text': 'Today we will discuss investment strategies and portfolio diversification.',
                'start': 3.2,
                'end': 8.1,
                'confidence': 0.94
            },
            {
                'text': 'Um, let me explain compound interest with a simple example.',
                'start': 8.1,
                'end': 12.8,
                'confidence': 0.88
            },
            {
                'text': 'The chart shows how your money grows over time.',
                'start': 12.8,
                'end': 16.5,
                'confidence': 0.93
            }
        ]
    }
    
    # Mock financial analysis results (as would come from FinancialContentAnalyzer)
    financial_analysis = {
        'concepts_mentioned': [
            'investment', 'strategies', 'portfolio', 'diversification', 
            'compound interest', 'money', 'financial education'
        ],
        'explanation_segments': [
            {
                'timestamp': 8.1,
                'text': 'Um, let me explain compound interest with a simple example.',
                'type': 'explanation',
                'confidence': 0.88
            }
        ],
        'data_references': [
            {
                'timestamp': 12.8,
                'text': 'The chart shows how your money grows over time.',
                'requires_visual': True,
                'confidence': 0.93
            }
        ],
        'complexity_level': 'medium',
        'filler_words_detected': [
            {
                'timestamp': 8.1,
                'text': 'Um, let me explain compound interest with a simple example.',
                'original_text': 'Um, let me explain compound interest with a simple example.',
                'filler_words': ['um'],
                'confidence': 0.88,
                'should_remove': True,
                'cleaned_text': 'Let me explain compound interest with a simple example.'
            }
        ],
        'emotional_peaks': [
            {
                'timestamp': 0.0,
                'emotion': 'enthusiasm',
                'intensity': 0.7,
                'confidence': 0.85,
                'context': 'Welcome to our financial education series'
            },
            {
                'timestamp': 8.1,
                'emotion': 'confidence',
                'intensity': 0.8,
                'confidence': 0.9,
                'context': 'explaining compound interest'
            }
        ],
        'audio_enhancement': {
            'original_duration': 16.5,
            'enhanced_duration': 16.0,
            'filler_words_removed': 1,
            'segments_modified': 1,
            'quality_improvement_score': 0.75,
            'processing_time': 2.3
        }
    }
    
    return transcript_data, financial_analysis


def demonstrate_basic_integration():
    """Demonstrate basic audio-ContentContext integration."""
    print("=== Basic Audio-ContentContext Integration ===")
    
    # Create a ContentContext
    context = ContentContext(
        project_id="demo_project",
        video_files=["demo_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(quality_mode="high")
    )
    
    print(f"Created ContentContext: {context.project_id}")
    print(f"Initial processing stage: {context._processing_stage}")
    
    # Get mock audio analysis data
    transcript_data, financial_analysis = create_mock_audio_analysis_data()
    
    # Convert and integrate audio analysis
    audio_result = convert_financial_analysis_to_audio_result(transcript_data, financial_analysis)
    context.set_audio_analysis(audio_result)
    
    print(f"Audio analysis integrated. Processing stage: {context._processing_stage}")
    print(f"Transcript text: {context.audio_analysis.transcript_text[:100]}...")
    print(f"Number of segments: {len(context.audio_analysis.segments)}")
    print(f"Overall confidence: {context.audio_analysis.overall_confidence:.3f}")
    print(f"Financial concepts found: {context.audio_analysis.financial_concepts}")
    print(f"Emotional peaks detected: {len(context.audio_analysis.detected_emotions)}")
    print()


def demonstrate_audio_insights_retrieval():
    """Demonstrate retrieving audio insights for downstream processing."""
    print("=== Audio Insights Retrieval ===")
    
    # Create context with audio analysis
    context = ContentContext(
        project_id="insights_demo",
        video_files=["demo_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences()
    )
    
    transcript_data, financial_analysis = create_mock_audio_analysis_data()
    audio_result = convert_financial_analysis_to_audio_result(transcript_data, financial_analysis)
    context.set_audio_analysis(audio_result)
    
    # Demonstrate various insight retrieval methods
    print("High confidence segments (>0.9):")
    high_conf_segments = context.get_audio_segments_by_confidence(0.9)
    for segment in high_conf_segments:
        print(f"  - {segment.text} (confidence: {segment.confidence:.3f})")
    
    print("\nFinancial concept segments:")
    concept_segments = context.get_financial_concept_segments()
    for segment in concept_segments:
        print(f"  - {segment.text}")
        print(f"    Concepts: {segment.financial_concepts}")
    
    print("\nExplanation segments (B-roll opportunities):")
    explanations = context.get_explanation_segments()
    for explanation in explanations:
        print(f"  - {explanation['text']} (timestamp: {explanation['timestamp']}s)")
    
    print("\nData reference segments (visualization opportunities):")
    data_refs = context.get_data_reference_segments()
    for data_ref in data_refs:
        print(f"  - {data_ref['text']} (timestamp: {data_ref['timestamp']}s)")
    
    print("\nEnhanced transcript (filler words removed):")
    enhanced_transcript = context.get_enhanced_transcript()
    print(f"  {enhanced_transcript}")
    
    print("\nAudio quality metrics:")
    quality_metrics = context.get_audio_quality_metrics()
    for key, value in quality_metrics.items():
        print(f"  {key}: {value}")
    
    print()


def demonstrate_ai_director_insights():
    """Demonstrate extracting insights for AI Director."""
    print("=== AI Director Insights ===")
    
    context = ContentContext(
        project_id="ai_director_demo",
        video_files=["demo_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences()
    )
    
    transcript_data, financial_analysis = create_mock_audio_analysis_data()
    audio_result = convert_financial_analysis_to_audio_result(transcript_data, financial_analysis)
    context.set_audio_analysis(audio_result)
    
    # Get structured insights for AI Director
    ai_insights = context.get_audio_insights_for_ai_director()
    
    print("Structured insights for AI Director:")
    print(f"  Transcript length: {len(ai_insights['transcript'])} characters")
    print(f"  Financial concepts: {ai_insights['financial_concepts']}")
    print(f"  Explanation opportunities: {ai_insights['explanation_opportunities']}")
    print(f"  Data visualization opportunities: {ai_insights['data_visualization_opportunities']}")
    print(f"  Emotional peaks: {len(ai_insights['emotional_peaks'])}")
    print(f"  Content complexity: {ai_insights['complexity_level']}")
    print(f"  High confidence segments: {ai_insights['high_confidence_segments']}")
    print(f"  Total segments: {ai_insights['total_segments']}")
    
    # Extract insights for downstream processing
    downstream_insights = extract_audio_insights_for_downstream(context)
    
    print("\nDownstream processing insights:")
    print(f"  Transcript confidence: {downstream_insights['transcript']['confidence']:.3f}")
    print(f"  Enhancement score: {downstream_insights['quality_metrics']['enhancement_score']:.3f}")
    print(f"  Time saved: {downstream_insights['quality_metrics']['time_saved']:.1f}s")
    print(f"  Concepts for B-roll: {downstream_insights['ai_director_ready']['concepts_for_broll']}")
    print(f"  Emotional hooks: {downstream_insights['ai_director_ready']['emotional_hooks']}")
    
    print()


def demonstrate_serialization_and_checkpoints():
    """Demonstrate serialization and checkpoint system integration."""
    print("=== Serialization and Checkpoint Integration ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create context manager
        context_manager = ContextManager(storage_path=temp_dir)
        
        # Create context with audio analysis
        context = context_manager.create_context(
            video_files=["demo_video.mp4"],
            content_type=ContentType.EDUCATIONAL
        )
        
        transcript_data, financial_analysis = create_mock_audio_analysis_data()
        audio_result = convert_financial_analysis_to_audio_result(transcript_data, financial_analysis)
        context.set_audio_analysis(audio_result)
        
        print(f"Created context: {context.project_id}")
        
        # Save checkpoint after audio analysis
        checkpoint_saved = context_manager.save_checkpoint(context, "audio_analysis_complete")
        print(f"Checkpoint saved: {checkpoint_saved}")
        
        # Test serialization
        context_dict = context.to_dict()
        print(f"Serialized context size: {len(str(context_dict))} characters")
        print(f"Audio analysis included: {'audio_analysis' in context_dict}")
        
        # Test JSON serialization
        json_str = context.to_json()
        print(f"JSON serialization size: {len(json_str)} characters")
        
        # Test deserialization
        restored_context = ContentContext.from_json(json_str)
        print(f"Restored context: {restored_context.project_id}")
        print(f"Audio analysis preserved: {restored_context.audio_analysis is not None}")
        
        if restored_context.audio_analysis:
            print(f"Restored transcript: {restored_context.audio_analysis.transcript_text[:50]}...")
            print(f"Restored segments: {len(restored_context.audio_analysis.segments)}")
            print(f"Restored concepts: {restored_context.audio_analysis.financial_concepts}")
        
        # Test checkpoint loading
        loaded_context = context_manager.load_checkpoint(context.project_id, "audio_analysis_complete")
        print(f"Checkpoint loaded: {loaded_context is not None}")
        
        if loaded_context and loaded_context.audio_analysis:
            print(f"Checkpoint audio analysis: {loaded_context.audio_analysis.transcript_text[:50]}...")
        
        # Validate integration
        validation = validate_audio_analysis_integration(loaded_context)
        print(f"Integration validation: valid={validation['valid']}, score={validation['completeness_score']:.3f}")
        
        if validation['issues']:
            print(f"Validation issues: {validation['issues']}")
        if validation['warnings']:
            print(f"Validation warnings: {validation['warnings']}")
    
    print()


def demonstrate_integration_workflow():
    """Demonstrate complete integration workflow."""
    print("=== Complete Integration Workflow ===")
    
    # Step 1: Create ContentContext
    context = ContentContext(
        project_id="workflow_demo",
        video_files=["demo_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(quality_mode="high")
    )
    
    print("Step 1: ContentContext created")
    print(f"  Project ID: {context.project_id}")
    print(f"  Content type: {context.content_type.value}")
    print(f"  Processing stage: {context._processing_stage}")
    
    # Step 2: Simulate audio analysis (normally done by FinancialContentAnalyzer)
    transcript_data, financial_analysis = create_mock_audio_analysis_data()
    
    print("\nStep 2: Audio analysis completed")
    print(f"  Transcript length: {len(transcript_data['text'])} characters")
    print(f"  Segments: {len(transcript_data['segments'])}")
    print(f"  Financial concepts: {len(financial_analysis['concepts_mentioned'])}")
    print(f"  Emotional peaks: {len(financial_analysis['emotional_peaks'])}")
    
    # Step 3: Integrate audio analysis into ContentContext
    updated_context = integrate_audio_analysis_to_context(
        context, transcript_data, financial_analysis
    )
    
    print("\nStep 3: Audio analysis integrated")
    print(f"  Processing stage: {updated_context._processing_stage}")
    print(f"  Audio analysis available: {updated_context.audio_analysis is not None}")
    print(f"  Emotional markers in context: {len(updated_context.emotional_markers)}")
    print(f"  Key concepts in context: {len(updated_context.key_concepts)}")
    
    # Step 4: Extract insights for downstream modules
    ai_insights = updated_context.get_audio_insights_for_ai_director()
    downstream_insights = extract_audio_insights_for_downstream(updated_context)
    
    print("\nStep 4: Insights extracted for downstream processing")
    print(f"  AI Director insights keys: {list(ai_insights.keys())}")
    print(f"  Downstream insights keys: {list(downstream_insights.keys())}")
    
    # Step 5: Validate integration
    validation = validate_audio_analysis_integration(updated_context)
    
    print("\nStep 5: Integration validation")
    print(f"  Valid: {validation['valid']}")
    print(f"  Completeness score: {validation['completeness_score']:.3f}")
    print(f"  Issues: {len(validation['issues'])}")
    print(f"  Warnings: {len(validation['warnings'])}")
    
    print("\n✅ Integration workflow completed successfully!")
    
    return updated_context


def main():
    """Run all demonstration examples."""
    print("Audio-ContentContext Integration Demonstration")
    print("=" * 50)
    print()
    
    try:
        # Run all demonstrations
        demonstrate_basic_integration()
        demonstrate_audio_insights_retrieval()
        demonstrate_ai_director_insights()
        demonstrate_serialization_and_checkpoints()
        final_context = demonstrate_integration_workflow()
        
        print("🎉 All demonstrations completed successfully!")
        print(f"Final context project ID: {final_context.project_id}")
        print(f"Final processing stage: {final_context._processing_stage}")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()