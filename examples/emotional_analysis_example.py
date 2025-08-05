#!/usr/bin/env python3
"""
Emotional Analysis Example - Demonstrating emotional peak detection and engagement analysis.

This example shows how to use the EmotionalAnalyzer to detect emotional peaks
from audio and visual content, calculate engagement metrics, and integrate
with the ContentContext system.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_video_editor.core.content_context import (
    ContentContext, ContentType, UserPreferences, EmotionalPeak,
    VisualHighlight, FaceDetection, AudioAnalysisResult, AudioSegment
)
from ai_video_editor.modules.content_analysis.emotional_analyzer import (
    EmotionalAnalyzer, EmotionType, create_emotional_analyzer
)
from ai_video_editor.utils.cache_manager import CacheManager


def create_sample_audio_analysis():
    """Create sample audio analysis with emotional content."""
    print("Creating sample audio analysis with emotional content...")
    
    # Create segments with various emotional content
    segments = [
        AudioSegment(
            text="Welcome everyone! This is absolutely amazing content we're going to cover today!",
            start=0.0,
            end=3.0,
            confidence=0.95,
            emotional_markers=['excitement', 'enthusiasm']
        ),
        AudioSegment(
            text="Now, be very careful about these risks - they can be quite dangerous if ignored.",
            start=3.0,
            end=6.0,
            confidence=0.92,
            emotional_markers=['concern', 'warning']
        ),
        AudioSegment(
            text="I'm really curious about how this mechanism actually works in practice.",
            start=6.0,
            end=9.0,
            confidence=0.88,
            emotional_markers=['curiosity', 'interest']
        ),
        AudioSegment(
            text="I'm definitely confident that this strategy will work for most people.",
            start=9.0,
            end=12.0,
            confidence=0.90,
            emotional_markers=['confidence', 'assurance']
        ),
        AudioSegment(
            text="What a surprising result! I never expected this outcome at all.",
            start=12.0,
            end=15.0,
            confidence=0.87,
            emotional_markers=['surprise', 'astonishment']
        ),
        AudioSegment(
            text="I'm completely satisfied with how this project turned out.",
            start=15.0,
            end=18.0,
            confidence=0.93,
            emotional_markers=['satisfaction', 'fulfillment']
        )
    ]
    
    return AudioAnalysisResult(
        transcript_text=" ".join([seg.text for seg in segments]),
        segments=segments,
        overall_confidence=0.91,
        language="en",
        processing_time=2.5,
        model_used="whisper-large",
        filler_words_removed=5,
        quality_improvement_score=0.75,
        detected_emotions=[
            EmotionalPeak(1.5, 'excitement', 0.9, 0.95, 'absolutely amazing'),
            EmotionalPeak(4.5, 'concern', 0.8, 0.92, 'be very careful'),
            EmotionalPeak(7.5, 'curiosity', 0.7, 0.88, 'curious about how'),
            EmotionalPeak(10.5, 'confidence', 0.85, 0.90, 'definitely confident'),
            EmotionalPeak(13.5, 'surprise', 0.9, 0.87, 'surprising result'),
            EmotionalPeak(16.5, 'satisfaction', 0.75, 0.93, 'completely satisfied')
        ]
    )


def create_sample_visual_highlights():
    """Create sample visual highlights with emotional cues."""
    print("Creating sample visual highlights with emotional cues...")
    
    return [
        VisualHighlight(
            timestamp=1.5,
            description="Speaker with excited expression, animated hand gestures, bright lighting",
            faces=[FaceDetection([100, 100, 200, 200], 0.95, 'happy', 
                                [[120, 130], [180, 130], [150, 170]])],
            visual_elements=['animated_gesture', 'bright_lighting', 'expressive_face'],
            thumbnail_potential=0.90
        ),
        VisualHighlight(
            timestamp=4.5,
            description="Serious expression with cautionary hand gesture, focused lighting",
            faces=[FaceDetection([110, 110, 190, 190], 0.88, 'focused')],
            visual_elements=['cautionary_gesture', 'serious_expression', 'text_overlay'],
            thumbnail_potential=0.75
        ),
        VisualHighlight(
            timestamp=7.5,
            description="Thoughtful expression with questioning gesture, head slightly tilted",
            faces=[FaceDetection([105, 105, 195, 195], 0.92, 'neutral')],
            visual_elements=['questioning_gesture', 'thoughtful_expression', 'head_tilt'],
            thumbnail_potential=0.70
        ),
        VisualHighlight(
            timestamp=10.5,
            description="Confident posture, direct gaze, assertive gesture",
            faces=[FaceDetection([115, 115, 185, 185], 0.90, 'neutral')],
            visual_elements=['assertive_gesture', 'direct_gaze', 'upright_posture'],
            thumbnail_potential=0.80
        ),
        VisualHighlight(
            timestamp=13.5,
            description="Wide eyes, raised eyebrows, surprised expression",
            faces=[FaceDetection([108, 108, 192, 192], 0.87, 'surprised')],
            visual_elements=['wide_eyes', 'raised_eyebrows', 'surprised_expression'],
            thumbnail_potential=0.85
        ),
        VisualHighlight(
            timestamp=16.5,
            description="Relaxed expression, gentle smile, calm posture",
            faces=[FaceDetection([112, 112, 188, 188], 0.93, 'happy')],
            visual_elements=['gentle_smile', 'relaxed_expression', 'calm_posture'],
            thumbnail_potential=0.78
        )
    ]


def create_sample_content_context():
    """Create a comprehensive ContentContext for emotional analysis."""
    print("Creating sample ContentContext...")
    
    context = ContentContext(
        project_id="emotional-analysis-demo",
        video_files=["emotional_demo_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(
            quality_mode="high",
            thumbnail_resolution=(1920, 1080)
        )
    )
    
    # Add audio analysis
    audio_analysis = create_sample_audio_analysis()
    context.set_audio_analysis(audio_analysis)
    
    # Add visual highlights
    context.visual_highlights = create_sample_visual_highlights()
    
    # Add video metadata
    context.video_metadata = {
        'duration': 20.0,
        'fps': 30.0,
        'width': 1920,
        'height': 1080,
        'codec': 'h264'
    }
    
    # Add some existing key concepts
    context.key_concepts = ['education', 'strategy', 'analysis', 'results']
    
    return context


def demonstrate_emotional_analysis():
    """Demonstrate comprehensive emotional analysis functionality."""
    print("=" * 60)
    print("EMOTIONAL ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create cache manager for performance
    cache_manager = CacheManager(default_ttl=3600)
    
    # Create emotional analyzer
    print("\n1. Creating EmotionalAnalyzer...")
    analyzer = create_emotional_analyzer(cache_manager=cache_manager)
    print(f"   ✓ Analyzer initialized with {len(analyzer.emotional_patterns)} emotion patterns")
    
    # Create sample content
    print("\n2. Creating sample content context...")
    context = create_sample_content_context()
    print(f"   ✓ Context created with {len(context.audio_analysis.segments)} audio segments")
    print(f"   ✓ Context has {len(context.visual_highlights)} visual highlights")
    
    # Perform emotional analysis
    print("\n3. Performing emotional analysis...")
    analyzed_context = analyzer.analyze_emotional_content(context)
    
    print(f"   ✓ Analysis completed in {analyzed_context.processing_metrics.total_processing_time:.2f}s")
    print(f"   ✓ Detected {len(analyzed_context.emotional_markers)} emotional peaks")
    
    # Display detected emotional peaks
    print("\n4. Detected Emotional Peaks:")
    print("-" * 40)
    for i, peak in enumerate(analyzed_context.emotional_markers, 1):
        print(f"   Peak {i}: {peak.emotion.upper()} at {peak.timestamp:.1f}s")
        print(f"           Intensity: {peak.intensity:.2f}, Confidence: {peak.confidence:.2f}")
        print(f"           Context: {peak.context[:60]}...")
        print()
    
    # Display engagement metrics
    if analyzed_context.engagement_predictions:
        print("5. Engagement Metrics:")
        print("-" * 40)
        metrics = analyzed_context.engagement_predictions
        
        print(f"   Overall Engagement Score: {metrics['overall_engagement_score']:.2f}")
        print(f"   Emotional Variety Score:  {metrics['emotional_variety_score']:.2f}")
        print(f"   Peak Intensity Score:     {metrics['peak_intensity_score']:.2f}")
        print(f"   Pacing Score:            {metrics['pacing_score']:.2f}")
        print(f"   Visual Engagement Score: {metrics['visual_engagement_score']:.2f}")
        print(f"   Audio Clarity Score:     {metrics['audio_clarity_score']:.2f}")
    
    # Analyze emotional distribution
    print("\n6. Emotional Distribution Analysis:")
    print("-" * 40)
    emotion_counts = {}
    total_intensity = 0
    
    for peak in analyzed_context.emotional_markers:
        emotion_counts[peak.emotion] = emotion_counts.get(peak.emotion, 0) + 1
        total_intensity += peak.intensity
    
    avg_intensity = total_intensity / len(analyzed_context.emotional_markers) if analyzed_context.emotional_markers else 0
    
    print(f"   Total Emotional Peaks: {len(analyzed_context.emotional_markers)}")
    print(f"   Average Peak Intensity: {avg_intensity:.2f}")
    print(f"   Unique Emotions Detected: {len(emotion_counts)}")
    print("\n   Emotion Distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        percentage = (count / len(analyzed_context.emotional_markers)) * 100
        print(f"     {emotion.capitalize()}: {count} peaks ({percentage:.1f}%)")
    
    # Demonstrate timeline analysis
    print("\n7. Emotional Timeline Analysis:")
    print("-" * 40)
    
    # Sort peaks by timestamp for timeline
    sorted_peaks = sorted(analyzed_context.emotional_markers, key=lambda p: p.timestamp)
    
    print("   Emotional Journey:")
    for i, peak in enumerate(sorted_peaks):
        time_marker = f"{peak.timestamp:.1f}s"
        intensity_bar = "█" * int(peak.intensity * 10)
        print(f"     {time_marker:>6} | {peak.emotion.capitalize():<12} | {intensity_bar} ({peak.intensity:.2f})")
    
    # Analyze pacing
    if len(sorted_peaks) > 1:
        intervals = []
        for i in range(1, len(sorted_peaks)):
            interval = sorted_peaks[i].timestamp - sorted_peaks[i-1].timestamp
            intervals.append(interval)
        
        avg_interval = sum(intervals) / len(intervals)
        print(f"\n   Average time between peaks: {avg_interval:.1f} seconds")
        
        # Identify emotional transitions
        print("\n   Emotional Transitions:")
        for i in range(1, len(sorted_peaks)):
            prev_emotion = sorted_peaks[i-1].emotion
            curr_emotion = sorted_peaks[i].emotion
            if prev_emotion != curr_emotion:
                time_gap = sorted_peaks[i].timestamp - sorted_peaks[i-1].timestamp
                print(f"     {prev_emotion} → {curr_emotion} (after {time_gap:.1f}s)")
    
    return analyzed_context


def demonstrate_cross_modal_analysis(context):
    """Demonstrate cross-modal emotional analysis."""
    print("\n8. Cross-Modal Analysis:")
    print("-" * 40)
    
    # Separate audio and visual peaks
    audio_peaks = [p for p in context.emotional_markers if 'Audio keywords' in p.context]
    visual_peaks = [p for p in context.emotional_markers if 'Visual' in p.context]
    
    print(f"   Audio-derived peaks: {len(audio_peaks)}")
    print(f"   Visual-derived peaks: {len(visual_peaks)}")
    
    # Find synchronized peaks (within 2 seconds)
    synchronized_pairs = []
    for audio_peak in audio_peaks:
        for visual_peak in visual_peaks:
            time_diff = abs(audio_peak.timestamp - visual_peak.timestamp)
            if time_diff <= 2.0:
                synchronized_pairs.append((audio_peak, visual_peak, time_diff))
    
    print(f"   Synchronized audio-visual pairs: {len(synchronized_pairs)}")
    
    if synchronized_pairs:
        print("\n   Synchronized Emotional Moments:")
        for audio_peak, visual_peak, time_diff in synchronized_pairs:
            sync_quality = "Strong" if time_diff < 1.0 else "Moderate"
            emotion_match = "✓" if audio_peak.emotion == visual_peak.emotion else "✗"
            
            print(f"     {audio_peak.timestamp:.1f}s: Audio({audio_peak.emotion}) + Visual({visual_peak.emotion})")
            print(f"              Sync: {sync_quality} ({time_diff:.1f}s gap), Match: {emotion_match}")


def demonstrate_engagement_prediction(context):
    """Demonstrate engagement prediction capabilities."""
    print("\n9. Engagement Prediction Analysis:")
    print("-" * 40)
    
    if not context.engagement_predictions:
        print("   No engagement predictions available")
        return
    
    metrics = context.engagement_predictions
    overall_score = metrics['overall_engagement_score']
    
    # Provide engagement assessment
    if overall_score >= 0.8:
        assessment = "EXCELLENT - Highly engaging content"
    elif overall_score >= 0.6:
        assessment = "GOOD - Well-balanced engagement"
    elif overall_score >= 0.4:
        assessment = "MODERATE - Some engagement elements"
    else:
        assessment = "LOW - Needs improvement"
    
    print(f"   Overall Assessment: {assessment}")
    print(f"   Engagement Score: {overall_score:.2f}/1.00")
    
    # Provide specific recommendations
    print("\n   Recommendations:")
    
    if metrics['emotional_variety_score'] < 0.5:
        print("     • Add more emotional variety to maintain viewer interest")
    
    if metrics['peak_intensity_score'] < 0.6:
        print("     • Increase emotional intensity at key moments")
    
    if metrics['pacing_score'] < 0.5:
        print("     • Improve pacing - distribute emotional peaks more evenly")
    
    if metrics['visual_engagement_score'] < 0.6:
        print("     • Enhance visual elements and facial expressions")
    
    if metrics['audio_clarity_score'] < 0.7:
        print("     • Improve audio quality and reduce filler words")
    
    # Highlight strengths
    strengths = []
    if metrics['emotional_variety_score'] >= 0.7:
        strengths.append("Good emotional variety")
    if metrics['peak_intensity_score'] >= 0.7:
        strengths.append("Strong emotional peaks")
    if metrics['visual_engagement_score'] >= 0.7:
        strengths.append("Engaging visual content")
    if metrics['audio_clarity_score'] >= 0.8:
        strengths.append("Clear audio delivery")
    
    if strengths:
        print(f"\n   Strengths: {', '.join(strengths)}")


def main():
    """Main demonstration function."""
    try:
        # Run the main emotional analysis demonstration
        analyzed_context = demonstrate_emotional_analysis()
        
        # Run additional analysis demonstrations
        demonstrate_cross_modal_analysis(analyzed_context)
        demonstrate_engagement_prediction(analyzed_context)
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("• Multi-modal emotional peak detection")
        print("• Audio and visual emotional cue analysis")
        print("• Engagement metrics calculation")
        print("• Cross-modal correlation analysis")
        print("• Emotional timeline creation")
        print("• ContentContext integration")
        print("\nThe EmotionalAnalyzer successfully detected emotional peaks")
        print("from both audio transcripts and visual highlights, providing")
        print("comprehensive engagement insights for video optimization.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())