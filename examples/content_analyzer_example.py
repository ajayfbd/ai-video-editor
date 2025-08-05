#!/usr/bin/env python3
"""
Content Analyzer Example - Multi-Modal Content Understanding

This example demonstrates the MultiModalContentAnalyzer implementation
for task 3.1, showing how it integrates audio, video, and emotional analysis
to provide comprehensive content understanding.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_video_editor.modules.content_analysis.content_analyzer import create_content_analyzer
from ai_video_editor.core.content_context import (
    ContentContext, ContentType, UserPreferences, AudioAnalysisResult, 
    AudioSegment, EmotionalPeak, VisualHighlight, FaceDetection
)


def create_sample_context():
    """Create a sample ContentContext with rich multi-modal data."""
    
    # Create sample audio analysis
    audio_segments = [
        AudioSegment(
            text="Welcome to our financial education series",
            start=0.0,
            end=3.0,
            confidence=0.95,
            financial_concepts=['financial', 'education']
        ),
        AudioSegment(
            text="Today we'll explore compound interest and investment strategies",
            start=3.0,
            end=7.0,
            confidence=0.92,
            financial_concepts=['compound interest', 'investment']
        ),
        AudioSegment(
            text="Let me show you this chart that explains the concept",
            start=7.0,
            end=10.0,
            confidence=0.88,
            financial_concepts=['chart', 'concept']
        )
    ]
    
    audio_analysis = AudioAnalysisResult(
        transcript_text="Welcome to our financial education series. Today we'll explore compound interest and investment strategies. Let me show you this chart that explains the concept.",
        segments=audio_segments,
        overall_confidence=0.92,
        language="en",
        processing_time=2.1,
        model_used="whisper-large",
        financial_concepts=['financial', 'education', 'compound interest', 'investment', 'chart'],
        explanation_segments=[
            {'concept': 'compound interest', 'timestamp': 5.0, 'confidence': 0.9},
            {'concept': 'investment strategies', 'timestamp': 8.5, 'confidence': 0.85}
        ],
        detected_emotions=[
            EmotionalPeak(1.0, 'excitement', 0.8, 0.9, 'introduction'),
            EmotionalPeak(5.0, 'curiosity', 0.7, 0.85, 'concept explanation'),
            EmotionalPeak(8.5, 'understanding', 0.9, 0.88, 'visual demonstration')
        ]
    )
    
    # Create sample visual highlights
    visual_highlights = [
        VisualHighlight(
            timestamp=2.0,
            description="Speaker introducing the topic with enthusiasm",
            faces=[FaceDetection([100, 100, 200, 200], 0.95, 'happy')],
            visual_elements=['talking_head', 'good_lighting'],
            thumbnail_potential=0.85
        ),
        VisualHighlight(
            timestamp=8.0,
            description="Chart showing compound interest growth over time",
            faces=[FaceDetection([50, 50, 150, 150], 0.88, 'focused')],
            visual_elements=['chart', 'data_visualization', 'text_overlay'],
            thumbnail_potential=0.92
        )
    ]
    
    # Create sample emotional markers
    emotional_markers = [
        EmotionalPeak(1.0, 'excitement', 0.8, 0.9, 'introduction'),
        EmotionalPeak(5.0, 'curiosity', 0.7, 0.85, 'concept explanation'),
        EmotionalPeak(8.5, 'understanding', 0.9, 0.88, 'visual demonstration')
    ]
    
    # Create ContentContext
    context = ContentContext(
        project_id="financial-education-demo",
        video_files=["financial_education_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(quality_mode="high")
    )
    
    # Set analysis data
    context.set_audio_analysis(audio_analysis)
    context.visual_highlights = visual_highlights
    context.emotional_markers = emotional_markers
    context.key_concepts = ['financial', 'education', 'compound interest', 'investment']
    
    return context


def demonstrate_content_analysis():
    """Demonstrate the multi-modal content analysis capabilities."""
    
    print("=== Multi-Modal Content Analysis Demo ===\n")
    
    # Create analyzer
    print("1. Creating MultiModalContentAnalyzer...")
    analyzer = create_content_analyzer()
    print("   ✓ Analyzer created successfully\n")
    
    # Create sample context
    print("2. Creating sample ContentContext with multi-modal data...")
    context = create_sample_context()
    print(f"   ✓ Context created for project: {context.project_id}")
    print(f"   ✓ Content type: {context.content_type.value}")
    print(f"   ✓ Audio segments: {len(context.audio_analysis.segments)}")
    print(f"   ✓ Visual highlights: {len(context.visual_highlights)}")
    print(f"   ✓ Emotional markers: {len(context.emotional_markers)}\n")
    
    # Perform content analysis
    print("3. Performing multi-modal content analysis...")
    result_context = analyzer.analyze_content(context)
    print("   ✓ Analysis completed successfully\n")
    
    # Display content type detection
    print("4. Content Type Detection:")
    detection = analyzer.detect_content_type(result_context)
    print(f"   Detected Type: {detection.detected_type.value}")
    print(f"   Confidence: {detection.confidence:.1%}")
    print(f"   Reasoning: {detection.reasoning}")
    if detection.alternative_types:
        print("   Alternative types:")
        for alt_type, confidence in detection.alternative_types:
            print(f"     - {alt_type.value}: {confidence:.1%}")
    print()
    
    # Display concept extraction
    print("5. Concept Extraction:")
    concepts = analyzer.extract_concepts(result_context)
    print(f"   Extracted {len(concepts)} concepts:")
    for i, concept in enumerate(concepts[:5], 1):  # Show top 5
        print(f"   {i}. {concept.concept}")
        print(f"      Confidence: {concept.confidence:.1%}")
        print(f"      Sources: {', '.join(concept.sources)}")
        if concept.timestamp:
            print(f"      Timestamp: {concept.timestamp:.1f}s")
        print()
    
    # Display cross-modal insights
    print("6. Cross-Modal Insights:")
    insights = analyzer._generate_cross_modal_insights(result_context, concepts)
    
    if 'cross_modal_consistency' in insights:
        consistency = insights['cross_modal_consistency']
        print(f"   Consistency Score: {consistency['consistency_score']:.1%}")
        if consistency['common_concepts']:
            print(f"   Common Concepts: {', '.join(consistency['common_concepts'])}")
    
    if 'audio_visual_sync' in insights:
        sync = insights['audio_visual_sync']
        print(f"   Audio-Visual Sync: {sync.get('confidence', 0):.1%}")
        print(f"   Synchronized Moments: {sync.get('synchronized_moments', 0)}")
    print()
    
    # Display engagement predictions
    print("7. Engagement Predictions:")
    predictions = analyzer._predict_engagement(result_context, concepts)
    for metric, value in predictions.items():
        print(f"   {metric.replace('_', ' ').title()}: {value:.1%}")
    print()
    
    # Display processing metrics
    print("8. Processing Metrics:")
    metrics = result_context.processing_metrics
    print(f"   Total Processing Time: {metrics.total_processing_time:.2f}s")
    print(f"   Memory Peak Usage: {metrics.memory_peak_usage} bytes")
    if metrics.module_processing_times:
        print("   Module Processing Times:")
        for module, time in metrics.module_processing_times.items():
            print(f"     - {module}: {time:.2f}s")
    print()
    
    print("=== Demo Completed Successfully ===")
    return result_context


if __name__ == "__main__":
    try:
        result = demonstrate_content_analysis()
        print(f"\nFinal result: ContentContext with {len(result.key_concepts)} key concepts")
        print("Task 3.1 Multi-Modal Content Understanding - COMPLETED ✓")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()