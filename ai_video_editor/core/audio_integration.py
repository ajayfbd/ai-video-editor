"""
Audio Integration Utilities - Bridge between audio analyzer and ContentContext.

This module provides utilities to convert audio analysis results from the
FinancialContentAnalyzer into ContentContext-compatible data structures.
"""

from typing import List, Dict, Any, Optional
from ai_video_editor.core.content_context import (
    AudioSegment, 
    AudioAnalysisResult, 
    EmotionalPeak,
    ContentContext
)


def convert_transcript_to_audio_segments(transcript_data: Dict[str, Any]) -> List[AudioSegment]:
    """
    Convert transcript segments to AudioSegment objects.
    
    Args:
        transcript_data: Dictionary containing transcript data from audio analyzer
        
    Returns:
        List of AudioSegment objects
    """
    segments = []
    
    for segment_data in transcript_data.get('segments', []):
        audio_segment = AudioSegment(
            text=segment_data.get('text', ''),
            start=segment_data.get('start', 0.0),
            end=segment_data.get('end', 0.0),
            confidence=segment_data.get('confidence', 0.0),
            language=transcript_data.get('language')
        )
        segments.append(audio_segment)
    
    return segments


def convert_financial_analysis_to_audio_result(
    transcript_data: Dict[str, Any],
    financial_analysis: Dict[str, Any]
) -> AudioAnalysisResult:
    """
    Convert FinancialContentAnalyzer results to AudioAnalysisResult.
    
    Args:
        transcript_data: Transcript data from Whisper
        financial_analysis: Financial analysis results
        
    Returns:
        AudioAnalysisResult object with comprehensive analysis
    """
    # Convert transcript segments
    segments = convert_transcript_to_audio_segments(transcript_data)
    
    # Process filler word information
    filler_segments = financial_analysis.get('filler_words_detected', [])
    filler_lookup = {fw.get('timestamp', 0.0): fw for fw in filler_segments}
    
    # Enhance segments with filler word and concept information
    for segment in segments:
        if segment.start in filler_lookup:
            filler_info = filler_lookup[segment.start]
            segment.filler_words = filler_info.get('filler_words', [])
            segment.cleaned_text = filler_info.get('cleaned_text')
        
        # Add financial concepts found in this segment
        segment_text_lower = segment.text.lower()
        segment.financial_concepts = [
            concept for concept in financial_analysis.get('concepts_mentioned', [])
            if concept.lower() in segment_text_lower
        ]
    
    # Convert emotional peaks
    detected_emotions = []
    for emotion_data in financial_analysis.get('emotional_peaks', []):
        if isinstance(emotion_data, dict):
            emotion = EmotionalPeak(
                timestamp=emotion_data.get('timestamp', 0.0),
                emotion=emotion_data.get('emotion', ''),
                intensity=emotion_data.get('intensity', 0.0),
                confidence=emotion_data.get('confidence', 0.0),
                context=emotion_data.get('context', '')
            )
            detected_emotions.append(emotion)
    
    # Extract enhancement results
    enhancement = financial_analysis.get('audio_enhancement', {})
    
    # Create comprehensive audio analysis result
    audio_result = AudioAnalysisResult(
        transcript_text=transcript_data.get('text', ''),
        segments=segments,
        overall_confidence=transcript_data.get('confidence', 0.0),
        language=transcript_data.get('language', 'unknown'),
        processing_time=transcript_data.get('processing_time', 0.0),
        model_used=transcript_data.get('model_used', ''),
        
        # Enhancement results
        filler_words_removed=enhancement.get('filler_words_removed', 0),
        segments_modified=enhancement.get('segments_modified', 0),
        quality_improvement_score=enhancement.get('quality_improvement_score', 0.0),
        original_duration=enhancement.get('original_duration', 0.0),
        enhanced_duration=enhancement.get('enhanced_duration', 0.0),
        
        # Financial content analysis
        financial_concepts=financial_analysis.get('concepts_mentioned', []),
        explanation_segments=financial_analysis.get('explanation_segments', []),
        data_references=financial_analysis.get('data_references', []),
        complexity_level=financial_analysis.get('complexity_level', 'medium'),
        
        # Emotional analysis
        detected_emotions=detected_emotions,
        engagement_points=[]  # Can be populated later by AI Director
    )
    
    return audio_result


def integrate_audio_analysis_to_context(
    context: ContentContext,
    transcript_data: Dict[str, Any],
    financial_analysis: Dict[str, Any]
) -> ContentContext:
    """
    Integrate audio analysis results into ContentContext.
    
    Args:
        context: ContentContext to update
        transcript_data: Transcript data from Whisper
        financial_analysis: Financial analysis results
        
    Returns:
        Updated ContentContext with audio analysis
    """
    # Convert analysis results to ContentContext format
    audio_result = convert_financial_analysis_to_audio_result(
        transcript_data, financial_analysis
    )
    
    # Set audio analysis in context
    context.set_audio_analysis(audio_result)
    
    # Update processing stage
    context.update_processing_stage("audio_analysis_complete")
    
    return context


def extract_audio_insights_for_downstream(context: ContentContext) -> Dict[str, Any]:
    """
    Extract audio insights in format suitable for downstream processing modules.
    
    Args:
        context: ContentContext with audio analysis
        
    Returns:
        Dictionary with structured audio insights
    """
    if not context.audio_analysis:
        return {}
    
    insights = {
        'transcript': {
            'full_text': context.audio_analysis.transcript_text,
            'enhanced_text': context.get_enhanced_transcript(),
            'language': context.audio_analysis.language,
            'confidence': context.audio_analysis.overall_confidence
        },
        
        'content_analysis': {
            'financial_concepts': context.audio_analysis.financial_concepts,
            'complexity_level': context.audio_analysis.complexity_level,
            'explanation_segments': context.audio_analysis.explanation_segments,
            'data_references': context.audio_analysis.data_references
        },
        
        'emotional_analysis': {
            'peaks': [emotion.to_dict() for emotion in context.audio_analysis.detected_emotions],
            'engagement_points': context.audio_analysis.engagement_points
        },
        
        'quality_metrics': {
            'overall_confidence': context.audio_analysis.overall_confidence,
            'enhancement_score': context.audio_analysis.quality_improvement_score,
            'filler_words_removed': context.audio_analysis.filler_words_removed,
            'time_saved': context.audio_analysis.original_duration - context.audio_analysis.enhanced_duration
        },
        
        'timing_data': {
            'total_segments': len(context.audio_analysis.segments),
            'high_confidence_segments': len(context.get_audio_segments_by_confidence(0.9)),
            'original_duration': context.audio_analysis.original_duration,
            'enhanced_duration': context.audio_analysis.enhanced_duration
        },
        
        'ai_director_ready': {
            'concepts_for_broll': context.audio_analysis.financial_concepts,
            'explanation_opportunities': len(context.audio_analysis.explanation_segments),
            'data_viz_opportunities': len(context.audio_analysis.data_references),
            'emotional_hooks': len([e for e in context.audio_analysis.detected_emotions if e.intensity > 0.7]),
            'content_complexity': context.audio_analysis.complexity_level
        }
    }
    
    return insights


def validate_audio_analysis_integration(context: ContentContext) -> Dict[str, Any]:
    """
    Validate that audio analysis is properly integrated into ContentContext.
    
    Args:
        context: ContentContext to validate
        
    Returns:
        Validation results dictionary
    """
    validation = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'completeness_score': 0.0
    }
    
    if not context.audio_analysis:
        validation['valid'] = False
        validation['issues'].append("No audio analysis data found")
        return validation
    
    # Check required fields
    required_fields = [
        'transcript_text', 'segments', 'overall_confidence', 
        'language', 'processing_time', 'model_used'
    ]
    
    for field in required_fields:
        if not hasattr(context.audio_analysis, field) or getattr(context.audio_analysis, field) is None:
            validation['issues'].append(f"Missing required field: {field}")
            validation['valid'] = False
    
    # Check data quality
    if context.audio_analysis.overall_confidence < 0.5:
        validation['warnings'].append(f"Low overall confidence: {context.audio_analysis.overall_confidence}")
    
    if not context.audio_analysis.segments:
        validation['warnings'].append("No audio segments found")
    
    # Check segment data quality
    invalid_segments = 0
    for segment in context.audio_analysis.segments:
        if segment.start >= segment.end:
            invalid_segments += 1
        if segment.confidence < 0.0 or segment.confidence > 1.0:
            invalid_segments += 1
    
    if invalid_segments > 0:
        validation['warnings'].append(f"{invalid_segments} segments have invalid timing or confidence data")
    
    # Calculate completeness score
    score_components = {
        'has_transcript': 1.0 if context.audio_analysis.transcript_text else 0.0,
        'has_segments': 1.0 if context.audio_analysis.segments else 0.0,
        'has_financial_concepts': 1.0 if context.audio_analysis.financial_concepts else 0.0,
        'has_emotional_analysis': 1.0 if context.audio_analysis.detected_emotions else 0.0,
        'has_enhancement_data': 1.0 if context.audio_analysis.filler_words_removed > 0 else 0.0,
        'good_confidence': 1.0 if context.audio_analysis.overall_confidence >= 0.8 else 0.5
    }
    
    validation['completeness_score'] = sum(score_components.values()) / len(score_components)
    
    return validation