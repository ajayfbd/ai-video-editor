"""
Unit tests for EmotionalAnalyzer - Emotional and Engagement Analysis.

Tests the EmotionalAnalyzer implementation with comprehensive mocking
following the testing strategy outlined in testing-strategy.md.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from typing import List, Dict, Any

from ai_video_editor.core.content_context import (
    ContentContext, ContentType, EmotionalPeak, VisualHighlight, 
    FaceDetection, AudioAnalysisResult, AudioSegment, UserPreferences
)
from ai_video_editor.modules.content_analysis.emotional_analyzer import (
    EmotionalAnalyzer, EmotionType, EmotionalPattern, EngagementMetrics,
    EmotionalAnalysisResult, create_emotional_analyzer
)
from ai_video_editor.utils.cache_manager import CacheManager
from ai_video_editor.utils.error_handling import ContentContextError


@pytest.fixture
def mock_cache_manager():
    """Mock CacheManager for testing."""
    return Mock(spec=CacheManager)


@pytest.fixture
def mock_memory_client():
    """Mock Memory client for testing."""
    memory_client = Mock()
    memory_client.search_nodes.return_value = {'nodes': []}
    memory_client.create_entities.return_value = True
    return memory_client


@pytest.fixture
def sample_audio_analysis():
    """Create sample audio analysis with emotional content."""
    segments = [
        AudioSegment(
            text="This is absolutely amazing returns!",
            start=0.0,
            end=2.0,
            confidence=0.95
        ),
        AudioSegment(
            text="Be very careful about the risks involved",
            start=2.0,
            end=4.0,
            confidence=0.92
        ),
        AudioSegment(
            text="I'm curious about how this actually works",
            start=4.0,
            end=6.0,
            confidence=0.88
        ),
        AudioSegment(
            text="I'm definitely confident in this strategy",
            start=6.0,
            end=8.0,
            confidence=0.90
        )
    ]
    
    return AudioAnalysisResult(
        transcript_text=" ".join([seg.text for seg in segments]),
        segments=segments,
        overall_confidence=0.91,
        language="en",
        processing_time=2.0,
        model_used="whisper-large",
        filler_words_removed=3,
        quality_improvement_score=0.7
    )


@pytest.fixture
def sample_visual_highlights():
    """Create sample visual highlights with emotional cues."""
    return [
        VisualHighlight(
            timestamp=1.0,
            description="Speaker with excited expression and animated gestures",
            faces=[FaceDetection([100, 100, 200, 200], 0.90, 'happy')],
            visual_elements=['animated_gesture', 'bright_lighting'],
            thumbnail_potential=0.85
        ),
        VisualHighlight(
            timestamp=3.0,
            description="Serious expression with cautionary gesture",
            faces=[FaceDetection([120, 120, 180, 180], 0.85, 'focused')],
            visual_elements=['text_overlay', 'data_visualization'],
            thumbnail_potential=0.70
        ),
        VisualHighlight(
            timestamp=5.0,
            description="Thoughtful expression with questioning gesture",
            faces=[FaceDetection([110, 110, 190, 190], 0.88, 'neutral')],
            visual_elements=['chart', 'dynamic_movement'],
            thumbnail_potential=0.75
        )
    ]


@pytest.fixture
def sample_content_context(sample_audio_analysis, sample_visual_highlights):
    """Create sample ContentContext with emotional content."""
    context = ContentContext(
        project_id="test-emotional-project",
        video_files=["emotional_test.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences()
    )
    
    context.set_audio_analysis(sample_audio_analysis)
    context.visual_highlights = sample_visual_highlights
    context.video_metadata = {'duration': 120.0}
    
    return context


@pytest.fixture
def emotional_analyzer(mock_cache_manager, mock_memory_client):
    """Create EmotionalAnalyzer instance for testing."""
    return EmotionalAnalyzer(
        cache_manager=mock_cache_manager,
        memory_client=mock_memory_client
    )


class TestEmotionalAnalyzer:
    """Test cases for EmotionalAnalyzer class."""
    
    def test_initialization(self, mock_cache_manager, mock_memory_client):
        """Test EmotionalAnalyzer initialization."""
        analyzer = EmotionalAnalyzer(
            cache_manager=mock_cache_manager,
            memory_client=mock_memory_client
        )
        
        assert analyzer.cache_manager == mock_cache_manager
        assert analyzer.memory_client == mock_memory_client
        assert len(analyzer.emotional_patterns) == 6  # 6 emotion types
        assert analyzer.min_peak_intensity == 0.5
        assert analyzer.peak_merge_threshold == 3.0
        assert analyzer.engagement_window_size == 10.0
    
    def test_emotional_patterns_initialization(self, emotional_analyzer):
        """Test emotional patterns are properly initialized."""
        patterns = emotional_analyzer.emotional_patterns
        
        # Check all expected emotion types are present
        expected_emotions = [
            EmotionType.EXCITEMENT, EmotionType.CURIOSITY, EmotionType.CONCERN,
            EmotionType.CONFIDENCE, EmotionType.SURPRISE, EmotionType.SATISFACTION
        ]
        
        for emotion in expected_emotions:
            assert emotion in patterns
            pattern = patterns[emotion]
            assert isinstance(pattern, EmotionalPattern)
            assert len(pattern.audio_keywords) > 0
            assert len(pattern.intensity_multipliers) > 0
            assert len(pattern.visual_indicators) > 0
            assert len(pattern.context_patterns) > 0
    
    def test_load_emotional_patterns_from_memory(self, mock_memory_client):
        """Test loading emotional patterns from Memory."""
        # Setup mock memory response
        mock_memory_client.search_nodes.return_value = {
            'nodes': [{
                'name': 'Emotional Analysis Patterns',
                'observations': [
                    'emotion_detection_accuracy excitement accuracy 85%',
                    'engagement_prediction_success high_engagement'
                ]
            }]
        }
        
        analyzer = EmotionalAnalyzer(memory_client=mock_memory_client)
        
        # Verify Memory was queried
        mock_memory_client.search_nodes.assert_called_once_with("emotional analysis patterns")
        
        # Verify patterns were loaded
        assert 'excitement' in analyzer.pattern_weights
        assert analyzer.min_peak_intensity < 0.5  # Should be reduced for high engagement
    
    def test_analyze_emotional_content_success(self, emotional_analyzer, sample_content_context):
        """Test successful emotional content analysis."""
        result_context = emotional_analyzer.analyze_emotional_content(sample_content_context)
        
        # Verify context was updated with emotional markers
        assert len(result_context.emotional_markers) > 0
        
        # Verify engagement predictions were added
        assert result_context.engagement_predictions is not None
        assert 'overall_engagement_score' in result_context.engagement_predictions
        
        # Verify processing metrics were updated
        assert result_context.processing_metrics.total_processing_time >= 0
        
        # Verify Memory was updated
        emotional_analyzer.memory_client.create_entities.assert_called()
    
    def test_detect_audio_emotional_peaks(self, emotional_analyzer, sample_audio_analysis):
        """Test audio emotional peak detection."""
        peaks = emotional_analyzer._detect_audio_emotional_peaks(sample_audio_analysis)
        
        # Should detect multiple emotional peaks
        assert len(peaks) > 0
        
        # Verify peak properties
        for peak in peaks:
            assert isinstance(peak, EmotionalPeak)
            assert peak.timestamp >= 0.0
            assert 0.0 <= peak.intensity <= 1.0
            assert 0.0 <= peak.confidence <= 1.0
            assert peak.emotion in [e.value for e in EmotionType]
            assert "Audio keywords:" in peak.context
        
        # Check what emotions were actually detected
        detected_emotions = [p.emotion for p in peaks]
        
        # Should detect concern from "careful about risks"  
        concern_peaks = [p for p in peaks if p.emotion == EmotionType.CONCERN.value]
        assert len(concern_peaks) > 0, f"Expected concern peaks from 'careful about risks'. Detected: {detected_emotions}"
        
        # Should detect curiosity from "curious about how"
        curiosity_peaks = [p for p in peaks if p.emotion == EmotionType.CURIOSITY.value]
        assert len(curiosity_peaks) > 0, f"Expected curiosity peaks from 'curious about how'. Detected: {detected_emotions}"
        
        # Should detect confidence from "definitely confident"
        confidence_peaks = [p for p in peaks if p.emotion == EmotionType.CONFIDENCE.value]
        assert len(confidence_peaks) > 0, f"Expected confidence peaks from 'definitely confident'. Detected: {detected_emotions}"
        
        # Should have detected multiple different emotions
        unique_emotions = set(detected_emotions)
        assert len(unique_emotions) >= 3, f"Expected at least 3 different emotions, got: {unique_emotions}"
    
    def test_detect_visual_emotional_peaks(self, emotional_analyzer, sample_visual_highlights):
        """Test visual emotional peak detection."""
        peaks = emotional_analyzer._detect_visual_emotional_peaks(sample_visual_highlights)
        
        # Should detect peaks from facial expressions and visual elements
        assert len(peaks) > 0
        
        # Verify peak properties
        for peak in peaks:
            assert isinstance(peak, EmotionalPeak)
            assert peak.timestamp >= 0.0
            assert 0.0 <= peak.intensity <= 1.0
            assert 0.0 <= peak.confidence <= 1.0
            assert "Visual" in peak.context
        
        # Should detect excitement from happy expression
        excitement_peaks = [p for p in peaks if p.emotion == EmotionType.EXCITEMENT.value]
        assert len(excitement_peaks) > 0
        
        # Should detect focus from focused expression
        focus_peaks = [p for p in peaks if p.emotion == EmotionType.FOCUS.value]
        assert len(focus_peaks) > 0
    
    def test_merge_emotional_peaks(self, emotional_analyzer):
        """Test emotional peak merging functionality."""
        # Create peaks that should be merged (same emotion, close in time)
        peaks = [
            EmotionalPeak(1.0, EmotionType.EXCITEMENT.value, 0.8, 0.9, "First peak"),
            EmotionalPeak(2.5, EmotionType.EXCITEMENT.value, 0.7, 0.85, "Second peak"),
            EmotionalPeak(10.0, EmotionType.CURIOSITY.value, 0.6, 0.8, "Different emotion"),
            EmotionalPeak(11.0, EmotionType.CURIOSITY.value, 0.75, 0.82, "Close curiosity peak")
        ]
        
        merged_peaks = emotional_analyzer._merge_emotional_peaks(peaks)
        
        # Should merge close peaks of same emotion
        assert len(merged_peaks) == 2  # Two merged peaks
        
        # Verify merged properties
        excitement_peak = next(p for p in merged_peaks if p.emotion == EmotionType.EXCITEMENT.value)
        assert excitement_peak.intensity == 0.8  # Max of merged intensities
        assert "First peak; Second peak" in excitement_peak.context
        
        curiosity_peak = next(p for p in merged_peaks if p.emotion == EmotionType.CURIOSITY.value)
        assert curiosity_peak.intensity == 0.75  # Max of merged intensities
    
    def test_calculate_engagement_metrics(self, emotional_analyzer, sample_content_context):
        """Test engagement metrics calculation."""
        # Create sample peaks with variety
        peaks = [
            EmotionalPeak(1.0, EmotionType.EXCITEMENT.value, 0.8, 0.9, "Excitement"),
            EmotionalPeak(3.0, EmotionType.CONCERN.value, 0.7, 0.85, "Concern"),
            EmotionalPeak(5.0, EmotionType.CURIOSITY.value, 0.6, 0.8, "Curiosity"),
            EmotionalPeak(7.0, EmotionType.CONFIDENCE.value, 0.75, 0.88, "Confidence")
        ]
        
        metrics = emotional_analyzer._calculate_engagement_metrics(sample_content_context, peaks)
        
        assert isinstance(metrics, EngagementMetrics)
        
        # Verify all metrics are in valid range
        assert 0.0 <= metrics.emotional_variety_score <= 1.0
        assert 0.0 <= metrics.peak_intensity_score <= 1.0
        assert 0.0 <= metrics.pacing_score <= 1.0
        assert 0.0 <= metrics.visual_engagement_score <= 1.0
        assert 0.0 <= metrics.audio_clarity_score <= 1.0
        assert 0.0 <= metrics.overall_engagement_score <= 1.0
        
        # Should have high emotional variety (4 different emotions)
        assert metrics.emotional_variety_score >= 0.8
        
        # Should have good peak intensity (high average)
        assert metrics.peak_intensity_score >= 0.6
    
    def test_calculate_pacing_score(self, emotional_analyzer, sample_content_context):
        """Test pacing score calculation."""
        # Test with well-distributed peaks
        well_distributed_peaks = [
            EmotionalPeak(10.0, EmotionType.EXCITEMENT.value, 0.8, 0.9, "Peak 1"),
            EmotionalPeak(30.0, EmotionType.CONCERN.value, 0.7, 0.85, "Peak 2"),
            EmotionalPeak(50.0, EmotionType.CURIOSITY.value, 0.6, 0.8, "Peak 3"),
            EmotionalPeak(70.0, EmotionType.CONFIDENCE.value, 0.75, 0.88, "Peak 4"),
            EmotionalPeak(90.0, EmotionType.SATISFACTION.value, 0.65, 0.82, "Peak 5")
        ]
        
        pacing_score = emotional_analyzer._calculate_pacing_score(
            well_distributed_peaks, sample_content_context
        )
        
        assert 0.0 <= pacing_score <= 1.0
        assert pacing_score >= 0.5  # Should be good for well-distributed peaks
        
        # Test with poorly distributed peaks (all at beginning)
        poorly_distributed_peaks = [
            EmotionalPeak(1.0, EmotionType.EXCITEMENT.value, 0.8, 0.9, "Peak 1"),
            EmotionalPeak(2.0, EmotionType.CONCERN.value, 0.7, 0.85, "Peak 2"),
            EmotionalPeak(3.0, EmotionType.CURIOSITY.value, 0.6, 0.8, "Peak 3")
        ]
        
        poor_pacing_score = emotional_analyzer._calculate_pacing_score(
            poorly_distributed_peaks, sample_content_context
        )
        
        assert poor_pacing_score < pacing_score  # Should be lower than well-distributed
    
    def test_calculate_visual_engagement_score(self, emotional_analyzer, sample_content_context):
        """Test visual engagement score calculation."""
        score = emotional_analyzer._calculate_visual_engagement_score(sample_content_context)
        
        assert 0.0 <= score <= 1.0
        
        # Should be relatively high due to faces and visual elements
        assert score >= 0.5
        
        # Test with no visual highlights
        empty_context = ContentContext(
            project_id="empty",
            video_files=["test.mp4"],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        empty_score = emotional_analyzer._calculate_visual_engagement_score(empty_context)
        assert empty_score == 0.3  # Low score for no visual analysis
    
    def test_calculate_audio_clarity_score(self, emotional_analyzer, sample_content_context):
        """Test audio clarity score calculation."""
        score = emotional_analyzer._calculate_audio_clarity_score(sample_content_context)
        
        assert 0.0 <= score <= 1.0
        
        # Should be relatively high due to good confidence and enhancement
        assert score >= 0.7
        
        # Test with no audio analysis
        empty_context = ContentContext(
            project_id="empty",
            video_files=["test.mp4"],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        empty_score = emotional_analyzer._calculate_audio_clarity_score(empty_context)
        assert empty_score == 0.5  # Neutral score for no audio analysis
    
    def test_create_emotional_timeline(self, emotional_analyzer, sample_content_context):
        """Test emotional timeline creation."""
        peaks = [
            EmotionalPeak(1.0, EmotionType.EXCITEMENT.value, 0.8, 0.9, "Peak 1"),
            EmotionalPeak(5.0, EmotionType.CONCERN.value, 0.7, 0.85, "Peak 2"),
            EmotionalPeak(10.0, EmotionType.CURIOSITY.value, 0.6, 0.8, "Peak 3")
        ]
        
        timeline = emotional_analyzer._create_emotional_timeline(peaks, sample_content_context)
        
        assert len(timeline) == 3
        
        # Verify timeline structure
        for i, entry in enumerate(timeline):
            assert 'timestamp' in entry
            assert 'emotion' in entry
            assert 'intensity' in entry
            assert 'confidence' in entry
            assert 'context' in entry
            assert 'sequence_position' in entry
            assert 'total_peaks' in entry
            
            assert entry['sequence_position'] == i + 1
            assert entry['total_peaks'] == 3
            
            # Check relative timing
            if i > 0:
                assert 'time_since_previous' in entry
                assert entry['time_since_previous'] > 0
            
            if i < len(timeline) - 1:
                assert 'time_to_next' in entry
                assert entry['time_to_next'] > 0
    
    def test_analyze_cross_modal_correlations(self, emotional_analyzer, sample_content_context):
        """Test cross-modal correlation analysis."""
        # Create peaks with both audio and visual sources
        peaks = [
            EmotionalPeak(1.0, EmotionType.EXCITEMENT.value, 0.8, 0.9, "Audio keywords: amazing"),
            EmotionalPeak(1.2, EmotionType.EXCITEMENT.value, 0.7, 0.85, "Visual expression: happy"),
            EmotionalPeak(3.0, EmotionType.CONCERN.value, 0.6, 0.8, "Audio keywords: careful"),
            EmotionalPeak(3.5, EmotionType.FOCUS.value, 0.65, 0.82, "Visual expression: focused")
        ]
        
        correlations = emotional_analyzer._analyze_cross_modal_correlations(
            sample_content_context, peaks
        )
        
        # Verify correlation structure
        expected_keys = [
            'audio_visual_sync', 'emotion_consistency', 
            'intensity_correlation', 'temporal_alignment'
        ]
        
        for key in expected_keys:
            assert key in correlations
            assert 0.0 <= correlations[key] <= 1.0
        
        # Should have good temporal alignment (peaks are close in time)
        assert correlations['temporal_alignment'] > 0.0
    
    def test_error_handling(self, mock_cache_manager, mock_memory_client):
        """Test error handling in emotional analysis."""
        analyzer = EmotionalAnalyzer(
            cache_manager=mock_cache_manager,
            memory_client=mock_memory_client
        )
        
        # Create context with missing data
        context = ContentContext(
            project_id="test-error",
            video_files=[],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        # Should handle gracefully without crashing
        result = analyzer.analyze_emotional_content(context)
        assert result.project_id == "test-error"
        assert len(result.emotional_markers) == 0  # No peaks detected from empty data
    
    def test_memory_integration(self, emotional_analyzer, sample_content_context):
        """Test Memory integration for storing analysis patterns."""
        emotional_analyzer.analyze_emotional_content(sample_content_context)
        
        # Verify Memory was called to store patterns
        emotional_analyzer.memory_client.create_entities.assert_called()
        
        # Verify the stored data structure
        call_args = emotional_analyzer.memory_client.create_entities.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]['name'] == 'Emotional Analysis Patterns'
        assert call_args[0]['entityType'] == 'analysis_insights'
        assert len(call_args[0]['observations']) > 0
    
    def test_multi_modal_peak_detection_integration(self, emotional_analyzer, sample_content_context):
        """Test integration of multi-modal peak detection."""
        peaks = emotional_analyzer._detect_multi_modal_emotional_peaks(sample_content_context)
        
        # Should detect peaks from both audio and visual sources
        assert len(peaks) > 0
        
        # Verify we have both audio and visual peaks
        audio_peaks = [p for p in peaks if 'Audio keywords' in p.context]
        visual_peaks = [p for p in peaks if 'Visual' in p.context]
        
        # Should have at least some peaks from different sources
        assert len(peaks) > 0, "Should detect some emotional peaks"
        
        # Check if we have visual peaks (which we should from the sample data)
        assert len(visual_peaks) > 0, f"Should detect visual peaks. All peaks: {[(p.context[:50], p.emotion) for p in peaks]}"
        
        # All peaks should meet minimum thresholds
        for peak in peaks:
            assert peak.intensity >= emotional_analyzer.min_peak_intensity
            assert peak.confidence >= 0.6


class TestEmotionalPattern:
    """Test cases for EmotionalPattern dataclass."""
    
    def test_emotional_pattern_creation(self):
        """Test EmotionalPattern creation and serialization."""
        pattern = EmotionalPattern(
            emotion=EmotionType.EXCITEMENT,
            audio_keywords=['amazing', 'fantastic'],
            intensity_multipliers={'amazing': 1.0, 'fantastic': 0.9},
            visual_indicators=['bright_lighting', 'animated_gesture'],
            context_patterns=['achievement', 'success'],
            confidence_threshold=0.7
        )
        
        assert pattern.emotion == EmotionType.EXCITEMENT
        assert len(pattern.audio_keywords) == 2
        assert pattern.confidence_threshold == 0.7
        
        # Test serialization
        pattern_dict = pattern.to_dict()
        assert pattern_dict['emotion'] == 'excitement'
        assert pattern_dict['confidence_threshold'] == 0.7


class TestEngagementMetrics:
    """Test cases for EngagementMetrics dataclass."""
    
    def test_engagement_metrics_creation(self):
        """Test EngagementMetrics creation and serialization."""
        metrics = EngagementMetrics(
            emotional_variety_score=0.8,
            peak_intensity_score=0.7,
            pacing_score=0.6,
            visual_engagement_score=0.75,
            audio_clarity_score=0.85,
            overall_engagement_score=0.73
        )
        
        assert metrics.emotional_variety_score == 0.8
        assert metrics.overall_engagement_score == 0.73
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        assert metrics_dict['emotional_variety_score'] == 0.8
        assert metrics_dict['overall_engagement_score'] == 0.73
        
        # Test deserialization
        restored_metrics = EngagementMetrics.from_dict(metrics_dict)
        assert restored_metrics.emotional_variety_score == 0.8
        assert restored_metrics.overall_engagement_score == 0.73


class TestEmotionalAnalysisResult:
    """Test cases for EmotionalAnalysisResult dataclass."""
    
    def test_emotional_analysis_result_creation(self):
        """Test EmotionalAnalysisResult creation and serialization."""
        peaks = [
            EmotionalPeak(1.0, EmotionType.EXCITEMENT.value, 0.8, 0.9, "Test peak")
        ]
        
        metrics = EngagementMetrics(
            emotional_variety_score=0.8,
            peak_intensity_score=0.7,
            pacing_score=0.6,
            visual_engagement_score=0.75,
            audio_clarity_score=0.85,
            overall_engagement_score=0.73
        )
        
        result = EmotionalAnalysisResult(
            detected_peaks=peaks,
            engagement_metrics=metrics,
            emotional_timeline=[{'timestamp': 1.0, 'emotion': 'excitement'}],
            cross_modal_correlations={'audio_visual_sync': 0.7},
            processing_time=2.5
        )
        
        assert len(result.detected_peaks) == 1
        assert result.processing_time == 2.5
        
        # Test serialization
        result_dict = result.to_dict()
        assert len(result_dict['detected_peaks']) == 1
        assert result_dict['processing_time'] == 2.5
        
        # Test deserialization
        restored_result = EmotionalAnalysisResult.from_dict(result_dict)
        assert len(restored_result.detected_peaks) == 1
        assert restored_result.processing_time == 2.5


def test_create_emotional_analyzer():
    """Test factory function for creating EmotionalAnalyzer."""
    analyzer = create_emotional_analyzer()
    
    assert isinstance(analyzer, EmotionalAnalyzer)
    assert analyzer.cache_manager is None
    assert analyzer.memory_client is None
    
    # Test with parameters
    mock_cache = Mock()
    mock_memory = Mock()
    analyzer_with_params = create_emotional_analyzer(
        cache_manager=mock_cache,
        memory_client=mock_memory
    )
    
    assert analyzer_with_params.cache_manager == mock_cache
    assert analyzer_with_params.memory_client == mock_memory


class TestEmotionTypeEnum:
    """Test cases for EmotionType enumeration."""
    
    def test_emotion_type_values(self):
        """Test EmotionType enum values."""
        assert EmotionType.EXCITEMENT.value == "excitement"
        assert EmotionType.CURIOSITY.value == "curiosity"
        assert EmotionType.CONCERN.value == "concern"
        assert EmotionType.CONFIDENCE.value == "confidence"
        assert EmotionType.SURPRISE.value == "surprise"
        assert EmotionType.SATISFACTION.value == "satisfaction"
        assert EmotionType.ANTICIPATION.value == "anticipation"
        assert EmotionType.FOCUS.value == "focus"
    
    def test_emotion_type_iteration(self):
        """Test EmotionType enum iteration."""
        emotion_values = [emotion.value for emotion in EmotionType]
        
        expected_values = [
            "excitement", "curiosity", "concern", "confidence",
            "surprise", "satisfaction", "anticipation", "focus"
        ]
        
        assert set(emotion_values) == set(expected_values)


if __name__ == '__main__':
    pytest.main([__file__])