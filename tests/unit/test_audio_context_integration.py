"""
Tests for Audio-ContentContext Integration.

This module tests the integration between audio analysis results and ContentContext,
including serialization/deserialization and checkpoint system compatibility.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch

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
    convert_transcript_to_audio_segments,
    convert_financial_analysis_to_audio_result,
    integrate_audio_analysis_to_context,
    extract_audio_insights_for_downstream,
    validate_audio_analysis_integration
)


class TestAudioSegment:
    """Test AudioSegment data structure."""
    
    def test_audio_segment_creation(self):
        """Test creating AudioSegment with all fields."""
        segment = AudioSegment(
            text="This is a test segment about investment strategies.",
            start=10.5,
            end=15.2,
            confidence=0.95,
            speaker_id="speaker_1",
            language="en",
            filler_words=["um", "uh"],
            cleaned_text="This is a test segment about investment strategies.",
            emotional_markers=["confidence"],
            financial_concepts=["investment", "strategies"]
        )
        
        assert segment.text == "This is a test segment about investment strategies."
        assert segment.start == 10.5
        assert segment.end == 15.2
        assert segment.confidence == 0.95
        assert segment.speaker_id == "speaker_1"
        assert segment.language == "en"
        assert segment.filler_words == ["um", "uh"]
        assert segment.cleaned_text == "This is a test segment about investment strategies."
        assert segment.emotional_markers == ["confidence"]
        assert segment.financial_concepts == ["investment", "strategies"]
    
    def test_audio_segment_serialization(self):
        """Test AudioSegment to_dict and from_dict methods."""
        segment = AudioSegment(
            text="Test segment",
            start=5.0,
            end=10.0,
            confidence=0.8,
            filler_words=["um"],
            financial_concepts=["investment"]
        )
        
        # Test serialization
        segment_dict = segment.to_dict()
        assert segment_dict['text'] == "Test segment"
        assert segment_dict['start'] == 5.0
        assert segment_dict['end'] == 10.0
        assert segment_dict['confidence'] == 0.8
        assert segment_dict['filler_words'] == ["um"]
        assert segment_dict['financial_concepts'] == ["investment"]
        
        # Test deserialization
        restored_segment = AudioSegment.from_dict(segment_dict)
        assert restored_segment.text == segment.text
        assert restored_segment.start == segment.start
        assert restored_segment.end == segment.end
        assert restored_segment.confidence == segment.confidence
        assert restored_segment.filler_words == segment.filler_words
        assert restored_segment.financial_concepts == segment.financial_concepts


class TestAudioAnalysisResult:
    """Test AudioAnalysisResult data structure."""
    
    def test_audio_analysis_result_creation(self):
        """Test creating AudioAnalysisResult with comprehensive data."""
        segments = [
            AudioSegment("First segment", 0.0, 5.0, 0.9),
            AudioSegment("Second segment", 5.0, 10.0, 0.85)
        ]
        
        emotions = [
            EmotionalPeak(2.5, "excitement", 0.8, 0.9, "discussing returns")
        ]
        
        result = AudioAnalysisResult(
            transcript_text="First segment Second segment",
            segments=segments,
            overall_confidence=0.875,
            language="en",
            processing_time=15.2,
            model_used="medium",
            filler_words_removed=3,
            segments_modified=1,
            quality_improvement_score=0.7,
            original_duration=12.0,
            enhanced_duration=10.0,
            financial_concepts=["investment", "returns"],
            explanation_segments=[{"timestamp": 2.0, "text": "explanation"}],
            data_references=[{"timestamp": 7.0, "text": "chart shows"}],
            complexity_level="medium",
            detected_emotions=emotions,
            engagement_points=[{"timestamp": 2.5, "type": "peak"}]
        )
        
        assert result.transcript_text == "First segment Second segment"
        assert len(result.segments) == 2
        assert result.overall_confidence == 0.875
        assert result.language == "en"
        assert result.processing_time == 15.2
        assert result.model_used == "medium"
        assert result.filler_words_removed == 3
        assert result.segments_modified == 1
        assert result.quality_improvement_score == 0.7
        assert result.original_duration == 12.0
        assert result.enhanced_duration == 10.0
        assert result.financial_concepts == ["investment", "returns"]
        assert len(result.explanation_segments) == 1
        assert len(result.data_references) == 1
        assert result.complexity_level == "medium"
        assert len(result.detected_emotions) == 1
        assert len(result.engagement_points) == 1
    
    def test_audio_analysis_result_serialization(self):
        """Test AudioAnalysisResult serialization and deserialization."""
        segments = [AudioSegment("Test", 0.0, 5.0, 0.9)]
        emotions = [EmotionalPeak(2.5, "excitement", 0.8, 0.9, "test")]
        
        result = AudioAnalysisResult(
            transcript_text="Test transcript",
            segments=segments,
            overall_confidence=0.9,
            language="en",
            processing_time=10.0,
            model_used="medium",
            financial_concepts=["investment"],
            detected_emotions=emotions
        )
        
        # Test serialization
        result_dict = result.to_dict()
        assert result_dict['transcript_text'] == "Test transcript"
        assert len(result_dict['segments']) == 1
        assert result_dict['overall_confidence'] == 0.9
        assert result_dict['financial_concepts'] == ["investment"]
        assert len(result_dict['detected_emotions']) == 1
        
        # Test deserialization
        restored_result = AudioAnalysisResult.from_dict(result_dict)
        assert restored_result.transcript_text == result.transcript_text
        assert len(restored_result.segments) == len(result.segments)
        assert restored_result.overall_confidence == result.overall_confidence
        assert restored_result.financial_concepts == result.financial_concepts
        assert len(restored_result.detected_emotions) == len(result.detected_emotions)


class TestContentContextAudioIntegration:
    """Test ContentContext integration with audio analysis."""
    
    def test_content_context_with_audio_analysis(self):
        """Test ContentContext with audio analysis data."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Create audio analysis result
        segments = [AudioSegment("Test segment", 0.0, 5.0, 0.9)]
        emotions = [EmotionalPeak(2.5, "excitement", 0.8, 0.9, "test")]
        
        audio_result = AudioAnalysisResult(
            transcript_text="Test transcript",
            segments=segments,
            overall_confidence=0.9,
            language="en",
            processing_time=10.0,
            model_used="medium",
            financial_concepts=["investment"],
            detected_emotions=emotions
        )
        
        # Set audio analysis
        context.set_audio_analysis(audio_result)
        
        # Verify integration
        assert context.audio_analysis is not None
        assert context.audio_transcript == "Test transcript"  # Backward compatibility
        assert len(context.emotional_markers) == 1  # Emotions added to main list
        assert "investment" in context.key_concepts  # Concepts added to main list
    
    def test_audio_insights_retrieval_methods(self):
        """Test methods for retrieving audio insights."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Create segments with varying confidence
        segments = [
            AudioSegment("High confidence", 0.0, 5.0, 0.95, financial_concepts=["investment"]),
            AudioSegment("Medium confidence", 5.0, 10.0, 0.75),
            AudioSegment("Low confidence", 10.0, 15.0, 0.6)
        ]
        
        audio_result = AudioAnalysisResult(
            transcript_text="High confidence Medium confidence Low confidence",
            segments=segments,
            overall_confidence=0.77,
            language="en",
            processing_time=10.0,
            model_used="medium",
            financial_concepts=["investment"],
            explanation_segments=[{"timestamp": 2.0, "text": "explanation"}],
            data_references=[{"timestamp": 7.0, "text": "chart shows"}]
        )
        
        context.set_audio_analysis(audio_result)
        
        # Test confidence filtering
        high_conf_segments = context.get_audio_segments_by_confidence(0.9)
        assert len(high_conf_segments) == 1
        assert high_conf_segments[0].text == "High confidence"
        
        # Test time range filtering
        early_segments = context.get_audio_segments_by_timerange(0.0, 12.0)
        assert len(early_segments) == 2
        
        # Test financial concept segments
        concept_segments = context.get_financial_concept_segments()
        assert len(concept_segments) == 1
        assert concept_segments[0].text == "High confidence"
        
        # Test explanation segments
        explanations = context.get_explanation_segments()
        assert len(explanations) == 1
        assert explanations[0]["timestamp"] == 2.0
        
        # Test data reference segments
        data_refs = context.get_data_reference_segments()
        assert len(data_refs) == 1
        assert data_refs[0]["timestamp"] == 7.0
        
        # Test enhanced transcript
        enhanced = context.get_enhanced_transcript()
        assert "High confidence" in enhanced
        
        # Test quality metrics
        metrics = context.get_audio_quality_metrics()
        assert metrics['overall_confidence'] == 0.77
        assert metrics['model_used'] == "medium"
        
        # Test AI Director insights
        insights = context.get_audio_insights_for_ai_director()
        assert 'transcript' in insights
        assert 'financial_concepts' in insights
        assert 'explanation_opportunities' in insights
        assert insights['total_segments'] == 3
    
    def test_content_context_serialization_with_audio(self):
        """Test ContentContext serialization with audio analysis data."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add audio analysis
        segments = [AudioSegment("Test", 0.0, 5.0, 0.9)]
        audio_result = AudioAnalysisResult(
            transcript_text="Test transcript",
            segments=segments,
            overall_confidence=0.9,
            language="en",
            processing_time=10.0,
            model_used="medium"
        )
        context.set_audio_analysis(audio_result)
        
        # Test serialization
        context_dict = context.to_dict()
        assert 'audio_analysis' in context_dict
        assert context_dict['audio_analysis'] is not None
        assert context_dict['audio_analysis']['transcript_text'] == "Test transcript"
        
        # Test JSON serialization
        json_str = context.to_json()
        assert '"audio_analysis"' in json_str
        
        # Test deserialization
        restored_context = ContentContext.from_dict(context_dict)
        assert restored_context.audio_analysis is not None
        assert restored_context.audio_analysis.transcript_text == "Test transcript"
        assert len(restored_context.audio_analysis.segments) == 1
        
        # Test JSON deserialization
        json_restored_context = ContentContext.from_json(json_str)
        assert json_restored_context.audio_analysis is not None
        assert json_restored_context.audio_analysis.transcript_text == "Test transcript"


class TestContextManagerIntegration:
    """Test ContextManager integration with audio analysis."""
    
    def test_context_manager_checkpoint_with_audio(self):
        """Test ContextManager checkpoint system with audio analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            context_manager = ContextManager(storage_path=temp_dir)
            
            # Create context with audio analysis
            context = context_manager.create_context(
                video_files=["test.mp4"],
                content_type=ContentType.EDUCATIONAL
            )
            
            # Add audio analysis
            segments = [AudioSegment("Test segment", 0.0, 5.0, 0.9)]
            audio_result = AudioAnalysisResult(
                transcript_text="Test transcript",
                segments=segments,
                overall_confidence=0.9,
                language="en",
                processing_time=10.0,
                model_used="medium",
                financial_concepts=["investment"]
            )
            context.set_audio_analysis(audio_result)
            
            # Save checkpoint
            success = context_manager.save_checkpoint(context, "audio_analysis_complete")
            assert success
            
            # Modify context
            context.audio_analysis.transcript_text = "Modified transcript"
            
            # Load checkpoint
            restored_context = context_manager.load_checkpoint(
                context.project_id, "audio_analysis_complete"
            )
            
            assert restored_context is not None
            assert restored_context.audio_analysis is not None
            assert restored_context.audio_analysis.transcript_text == "Test transcript"  # Original value
            assert len(restored_context.audio_analysis.segments) == 1
            assert "investment" in restored_context.audio_analysis.financial_concepts
    
    def test_context_manager_validation_with_audio(self):
        """Test ContextManager validation with audio analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            context_manager = ContextManager(storage_path=temp_dir)
            
            context = context_manager.create_context(
                video_files=["test.mp4"],
                content_type=ContentType.EDUCATIONAL
            )
            
            # Add valid audio analysis
            segments = [AudioSegment("Test", 0.0, 5.0, 0.9)]
            audio_result = AudioAnalysisResult(
                transcript_text="Test transcript",
                segments=segments,
                overall_confidence=0.9,
                language="en",
                processing_time=10.0,
                model_used="medium"
            )
            context.set_audio_analysis(audio_result)
            
            # Validate context
            validation = context_manager.validate_context(context)
            assert validation['valid']
            assert validation['score'] > 0.7  # Lower threshold due to missing video files in test


class TestAudioIntegrationUtilities:
    """Test audio integration utility functions."""
    
    def test_convert_transcript_to_audio_segments(self):
        """Test converting transcript data to AudioSegment objects."""
        transcript_data = {
            'text': 'Full transcript text',
            'language': 'en',
            'segments': [
                {'text': 'First segment', 'start': 0.0, 'end': 5.0, 'confidence': 0.9},
                {'text': 'Second segment', 'start': 5.0, 'end': 10.0, 'confidence': 0.85}
            ]
        }
        
        segments = convert_transcript_to_audio_segments(transcript_data)
        
        assert len(segments) == 2
        assert segments[0].text == 'First segment'
        assert segments[0].start == 0.0
        assert segments[0].end == 5.0
        assert segments[0].confidence == 0.9
        assert segments[0].language == 'en'
        
        assert segments[1].text == 'Second segment'
        assert segments[1].start == 5.0
        assert segments[1].end == 10.0
        assert segments[1].confidence == 0.85
        assert segments[1].language == 'en'
    
    def test_convert_financial_analysis_to_audio_result(self):
        """Test converting financial analysis to AudioAnalysisResult."""
        transcript_data = {
            'text': 'Investment discussion transcript',
            'language': 'en',
            'confidence': 0.9,
            'processing_time': 15.0,
            'model_used': 'medium',
            'segments': [
                {'text': 'Investment discussion', 'start': 0.0, 'end': 5.0, 'confidence': 0.9}
            ]
        }
        
        financial_analysis = {
            'concepts_mentioned': ['investment', 'portfolio'],
            'explanation_segments': [{'timestamp': 2.0, 'text': 'explanation'}],
            'data_references': [{'timestamp': 3.0, 'text': 'chart'}],
            'complexity_level': 'medium',
            'filler_words_detected': [
                {'timestamp': 0.0, 'filler_words': ['um'], 'cleaned_text': 'Investment discussion'}
            ],
            'emotional_peaks': [
                {'timestamp': 2.5, 'emotion': 'excitement', 'intensity': 0.8, 'confidence': 0.9, 'context': 'returns'}
            ],
            'audio_enhancement': {
                'filler_words_removed': 1,
                'segments_modified': 1,
                'quality_improvement_score': 0.7,
                'original_duration': 6.0,
                'enhanced_duration': 5.0
            }
        }
        
        result = convert_financial_analysis_to_audio_result(transcript_data, financial_analysis)
        
        assert result.transcript_text == 'Investment discussion transcript'
        assert len(result.segments) == 1
        assert result.overall_confidence == 0.9
        assert result.language == 'en'
        assert result.processing_time == 15.0
        assert result.model_used == 'medium'
        
        # Check enhancement data
        assert result.filler_words_removed == 1
        assert result.segments_modified == 1
        assert result.quality_improvement_score == 0.7
        assert result.original_duration == 6.0
        assert result.enhanced_duration == 5.0
        
        # Check financial analysis
        assert result.financial_concepts == ['investment', 'portfolio']
        assert len(result.explanation_segments) == 1
        assert len(result.data_references) == 1
        assert result.complexity_level == 'medium'
        
        # Check emotional analysis
        assert len(result.detected_emotions) == 1
        assert result.detected_emotions[0].emotion == 'excitement'
        
        # Check segment enhancement
        segment = result.segments[0]
        assert segment.filler_words == ['um']
        assert segment.cleaned_text == 'Investment discussion'
        assert 'investment' in segment.financial_concepts
    
    def test_integrate_audio_analysis_to_context(self):
        """Test integrating audio analysis into ContentContext."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        transcript_data = {
            'text': 'Test transcript',
            'language': 'en',
            'confidence': 0.9,
            'processing_time': 10.0,
            'model_used': 'medium',
            'segments': [
                {'text': 'Test segment', 'start': 0.0, 'end': 5.0, 'confidence': 0.9}
            ]
        }
        
        financial_analysis = {
            'concepts_mentioned': ['investment'],
            'explanation_segments': [],
            'data_references': [],
            'complexity_level': 'medium',
            'filler_words_detected': [],
            'emotional_peaks': [],
            'audio_enhancement': {}
        }
        
        updated_context = integrate_audio_analysis_to_context(
            context, transcript_data, financial_analysis
        )
        
        assert updated_context.audio_analysis is not None
        assert updated_context.audio_transcript == 'Test transcript'
        assert 'investment' in updated_context.key_concepts
        assert updated_context._processing_stage == "audio_analysis_complete"
    
    def test_extract_audio_insights_for_downstream(self):
        """Test extracting audio insights for downstream processing."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        segments = [
            AudioSegment("High confidence segment", 0.0, 5.0, 0.95),
            AudioSegment("Low confidence segment", 5.0, 10.0, 0.6)
        ]
        
        emotions = [EmotionalPeak(2.5, "excitement", 0.8, 0.9, "returns")]
        
        audio_result = AudioAnalysisResult(
            transcript_text="High confidence segment Low confidence segment",
            segments=segments,
            overall_confidence=0.775,
            language="en",
            processing_time=10.0,
            model_used="medium",
            financial_concepts=["investment"],
            explanation_segments=[{"timestamp": 2.0, "text": "explanation"}],
            data_references=[{"timestamp": 7.0, "text": "chart"}],
            complexity_level="medium",
            detected_emotions=emotions,
            original_duration=12.0,
            enhanced_duration=10.0
        )
        
        context.set_audio_analysis(audio_result)
        
        insights = extract_audio_insights_for_downstream(context)
        
        assert 'transcript' in insights
        assert insights['transcript']['full_text'] == "High confidence segment Low confidence segment"
        assert insights['transcript']['language'] == "en"
        assert insights['transcript']['confidence'] == 0.775
        
        assert 'content_analysis' in insights
        assert insights['content_analysis']['financial_concepts'] == ["investment"]
        assert insights['content_analysis']['complexity_level'] == "medium"
        
        assert 'emotional_analysis' in insights
        assert len(insights['emotional_analysis']['peaks']) == 1
        
        assert 'quality_metrics' in insights
        assert insights['quality_metrics']['overall_confidence'] == 0.775
        assert insights['quality_metrics']['time_saved'] == 2.0
        
        assert 'timing_data' in insights
        assert insights['timing_data']['total_segments'] == 2
        assert insights['timing_data']['high_confidence_segments'] == 1
        
        assert 'ai_director_ready' in insights
        assert insights['ai_director_ready']['concepts_for_broll'] == ["investment"]
        assert insights['ai_director_ready']['explanation_opportunities'] == 1
        assert insights['ai_director_ready']['data_viz_opportunities'] == 1
        assert insights['ai_director_ready']['emotional_hooks'] == 1
    
    def test_validate_audio_analysis_integration(self):
        """Test validation of audio analysis integration."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Test validation without audio analysis
        validation = validate_audio_analysis_integration(context)
        assert not validation['valid']
        assert "No audio analysis data found" in validation['issues']
        
        # Add valid audio analysis
        segments = [AudioSegment("Test", 0.0, 5.0, 0.9)]
        audio_result = AudioAnalysisResult(
            transcript_text="Test transcript",
            segments=segments,
            overall_confidence=0.9,
            language="en",
            processing_time=10.0,
            model_used="medium",
            financial_concepts=["investment"],
            detected_emotions=[EmotionalPeak(2.5, "excitement", 0.8, 0.9, "test")]
        )
        context.set_audio_analysis(audio_result)
        
        # Test validation with valid audio analysis
        validation = validate_audio_analysis_integration(context)
        assert validation['valid']
        assert len(validation['issues']) == 0
        assert validation['completeness_score'] > 0.8
        
        # Test validation with low confidence
        context.audio_analysis.overall_confidence = 0.3
        validation = validate_audio_analysis_integration(context)
        assert validation['valid']  # Still valid, but with warnings
        assert any("Low overall confidence" in warning for warning in validation['warnings'])


if __name__ == "__main__":
    pytest.main([__file__])