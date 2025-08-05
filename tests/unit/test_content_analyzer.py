"""
Unit tests for ContentAnalyzer - Multi-Modal Content Understanding.

Tests the ContentAnalyzer base class and MultiModalContentAnalyzer implementation
with comprehensive mocking for audio and video analysis components.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

from ai_video_editor.core.content_context import (
    ContentContext, ContentType, EmotionalPeak, VisualHighlight, 
    FaceDetection, AudioAnalysisResult, AudioSegment, UserPreferences
)
from ai_video_editor.modules.content_analysis.content_analyzer import (
    ContentAnalyzer, MultiModalContentAnalyzer, ConceptExtraction,
    ContentTypeDetection, MultiModalAnalysisResult, create_content_analyzer
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
def sample_content_context():
    """Create a sample ContentContext for testing."""
    # Create sample audio analysis
    audio_segments = [
        AudioSegment(
            text="Welcome to financial education",
            start=0.0,
            end=2.0,
            confidence=0.95,
            financial_concepts=['financial', 'education']
        ),
        AudioSegment(
            text="Let's discuss compound interest",
            start=2.0,
            end=4.0,
            confidence=0.92,
            financial_concepts=['compound interest']
        )
    ]
    
    audio_analysis = AudioAnalysisResult(
        transcript_text="Welcome to financial education. Let's discuss compound interest.",
        segments=audio_segments,
        overall_confidence=0.93,
        language="en",
        processing_time=1.5,
        model_used="whisper-large",
        financial_concepts=['financial', 'education', 'compound interest'],
        explanation_segments=[
            {'concept': 'compound interest', 'timestamp': 3.0, 'confidence': 0.9}
        ],
        detected_emotions=[
            EmotionalPeak(1.0, 'excitement', 0.8, 0.9, 'introduction')
        ]
    )
    
    # Create sample visual highlights
    visual_highlights = [
        VisualHighlight(
            timestamp=1.5,
            description="Speaker explaining concept with chart",
            faces=[FaceDetection([100, 100, 200, 200], 0.95, 'neutral')],
            visual_elements=['chart', 'text_overlay'],
            thumbnail_potential=0.85
        )
    ]
    
    # Create sample emotional markers
    emotional_markers = [
        EmotionalPeak(1.0, 'excitement', 0.8, 0.9, 'introduction'),
        EmotionalPeak(3.0, 'curiosity', 0.7, 0.85, 'concept explanation')
    ]
    
    context = ContentContext(
        project_id="test-project",
        video_files=["test_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences()
    )
    
    context.set_audio_analysis(audio_analysis)
    context.visual_highlights = visual_highlights
    context.emotional_markers = emotional_markers
    context.key_concepts = ['financial', 'education', 'compound interest']
    
    return context


class TestContentAnalyzer:
    """Test cases for ContentAnalyzer abstract base class."""
    
    def test_content_analyzer_is_abstract(self):
        """Test that ContentAnalyzer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ContentAnalyzer()
    
    def test_load_analysis_patterns_with_memory(self, mock_memory_client):
        """Test loading analysis patterns from Memory."""
        # Setup mock memory response
        mock_memory_client.search_nodes.return_value = {
            'nodes': [{
                'name': 'Content Analysis Patterns',
                'observations': [
                    'educational_content_accuracy 85%',
                    'concept_extraction_success financial_concepts'
                ]
            }]
        }
        
        analyzer = MultiModalContentAnalyzer(memory_client=mock_memory_client)
        
        # Verify Memory was queried
        mock_memory_client.search_nodes.assert_called_once_with("content analysis patterns")
        
        # Verify patterns were loaded
        assert 'educational_weight' in analyzer.analysis_patterns
        assert analyzer.analysis_patterns['financial_concept_threshold'] == 0.7


class TestMultiModalContentAnalyzer:
    """Test cases for MultiModalContentAnalyzer implementation."""
    
    def test_initialization(self, mock_cache_manager, mock_memory_client):
        """Test MultiModalContentAnalyzer initialization."""
        analyzer = MultiModalContentAnalyzer(
            cache_manager=mock_cache_manager,
            memory_client=mock_memory_client
        )
        
        assert analyzer.cache_manager == mock_cache_manager
        assert analyzer.memory_client == mock_memory_client
        assert hasattr(analyzer, 'audio_analyzer')
        assert hasattr(analyzer, 'video_analyzer')
        assert hasattr(analyzer, 'content_type_patterns')
    
    def test_analyze_content_success(self, sample_content_context, mock_cache_manager, mock_memory_client):
        """Test successful content analysis."""
        analyzer = MultiModalContentAnalyzer(
            cache_manager=mock_cache_manager,
            memory_client=mock_memory_client
        )
        
        result_context = analyzer.analyze_content(sample_content_context)
        
        # Verify context was updated
        assert result_context.project_id == sample_content_context.project_id
        assert len(result_context.key_concepts) >= 3
        assert result_context.processing_metrics.total_processing_time > 0
        
        # Verify Memory was updated
        mock_memory_client.create_entities.assert_called()
    
    def test_detect_content_type_educational(self, sample_content_context, mock_cache_manager):
        """Test content type detection for educational content."""
        analyzer = MultiModalContentAnalyzer(cache_manager=mock_cache_manager)
        
        detection = analyzer.detect_content_type(sample_content_context)
        
        assert detection.detected_type == ContentType.EDUCATIONAL
        assert detection.confidence > 0.5
        assert 'financial' in detection.reasoning.lower() or 'education' in detection.reasoning.lower()
    
    def test_extract_concepts_multi_modal(self, sample_content_context, mock_cache_manager):
        """Test concept extraction from multiple modalities."""
        analyzer = MultiModalContentAnalyzer(cache_manager=mock_cache_manager)
        
        concepts = analyzer.extract_concepts(sample_content_context)
        
        # Verify concepts were extracted
        assert len(concepts) > 0
        
        # Verify multi-modal sources
        concept_sources = set()
        for concept in concepts:
            concept_sources.update(concept.sources)
        
        assert 'audio' in concept_sources
        
        # Verify concept confidence
        high_confidence_concepts = [c for c in concepts if c.confidence > 0.7]
        assert len(high_confidence_concepts) > 0
    
    def test_extract_audio_concepts(self, sample_content_context, mock_cache_manager):
        """Test audio concept extraction."""
        analyzer = MultiModalContentAnalyzer(cache_manager=mock_cache_manager)
        
        audio_concepts = analyzer._extract_audio_concepts(sample_content_context.audio_analysis)
        
        # Verify financial concepts were extracted
        concept_names = [concept[0] for concept in audio_concepts]
        assert any('financial' in name or 'education' in name or 'compound interest' in name 
                  for name in concept_names)
        
        # Verify confidence scores
        for concept, confidence, timestamp in audio_concepts:
            assert 0.0 <= confidence <= 1.0
            assert timestamp >= 0.0
    
    def test_extract_visual_concepts(self, sample_content_context, mock_cache_manager):
        """Test visual concept extraction."""
        analyzer = MultiModalContentAnalyzer(cache_manager=mock_cache_manager)
        
        visual_concepts = analyzer._extract_visual_concepts(sample_content_context.visual_highlights)
        
        # Verify visual elements were extracted as concepts
        concept_names = [concept[0] for concept in visual_concepts]
        assert 'chart' in concept_names
        assert 'text_overlay' in concept_names
        
        # Verify timestamps match highlights
        for concept, confidence, timestamp in visual_concepts:
            assert timestamp == 1.5  # From sample data
    
    def test_extract_emotional_concepts(self, sample_content_context, mock_cache_manager):
        """Test emotional concept extraction."""
        analyzer = MultiModalContentAnalyzer(cache_manager=mock_cache_manager)
        
        emotional_concepts = analyzer._extract_emotional_concepts(sample_content_context.emotional_markers)
        
        # Verify emotional concepts were extracted
        concept_names = [concept[0] for concept in emotional_concepts]
        assert any('engaging_content' in name or 'educational_content' in name 
                  for name in emotional_concepts)
    
    def test_generate_cross_modal_insights(self, sample_content_context, mock_cache_manager):
        """Test cross-modal insights generation."""
        analyzer = MultiModalContentAnalyzer(cache_manager=mock_cache_manager)
        
        concepts = analyzer.extract_concepts(sample_content_context)
        insights = analyzer._generate_cross_modal_insights(sample_content_context, concepts)
        
        # Verify insights structure
        assert 'audio_visual_sync' in insights
        assert 'cross_modal_consistency' in insights
        assert 'emotional_visual_correlation' in insights
        
        # Verify consistency metrics
        consistency = insights['cross_modal_consistency']
        assert 'common_concepts' in consistency
        assert 'consistency_score' in consistency
        assert 0.0 <= consistency['consistency_score'] <= 1.0
    
    def test_predict_engagement(self, sample_content_context, mock_cache_manager):
        """Test engagement prediction."""
        analyzer = MultiModalContentAnalyzer(cache_manager=mock_cache_manager)
        
        concepts = analyzer.extract_concepts(sample_content_context)
        predictions = analyzer._predict_engagement(sample_content_context, concepts)
        
        # Verify prediction structure
        assert 'content_clarity' in predictions
        assert 'emotional_engagement' in predictions
        assert 'visual_appeal' in predictions
        assert 'overall_engagement' in predictions
        
        # Verify educational content gets educational_value
        if sample_content_context.content_type == ContentType.EDUCATIONAL:
            assert 'educational_value' in predictions
        
        # Verify all predictions are in valid range
        for key, value in predictions.items():
            assert 0.0 <= value <= 1.0
    
    def test_error_handling(self, mock_cache_manager, mock_memory_client):
        """Test error handling in content analysis."""
        analyzer = MultiModalContentAnalyzer(
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
        result = analyzer.analyze_content(context)
        assert result.project_id == "test-error"
    
    def test_memory_integration(self, sample_content_context, mock_cache_manager, mock_memory_client):
        """Test Memory integration for storing analysis patterns."""
        analyzer = MultiModalContentAnalyzer(
            cache_manager=mock_cache_manager,
            memory_client=mock_memory_client
        )
        
        analyzer.analyze_content(sample_content_context)
        
        # Verify Memory was called to store patterns
        mock_memory_client.create_entities.assert_called()
        
        # Verify the stored data structure
        call_args = mock_memory_client.create_entities.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]['name'] == 'Content Analysis Patterns'
        assert call_args[0]['entityType'] == 'analysis_insights'
        assert len(call_args[0]['observations']) > 0


class TestConceptExtraction:
    """Test cases for ConceptExtraction data class."""
    
    def test_concept_extraction_creation(self):
        """Test ConceptExtraction creation and serialization."""
        concept = ConceptExtraction(
            concept="financial_planning",
            confidence=0.85,
            sources=['audio', 'visual'],
            context="transcript_analysis; visual_analysis",
            timestamp=30.5
        )
        
        assert concept.concept == "financial_planning"
        assert concept.confidence == 0.85
        assert concept.sources == ['audio', 'visual']
        assert concept.timestamp == 30.5
        
        # Test serialization
        concept_dict = concept.to_dict()
        assert concept_dict['concept'] == "financial_planning"
        assert concept_dict['confidence'] == 0.85
        
        # Test deserialization
        restored_concept = ConceptExtraction.from_dict(concept_dict)
        assert restored_concept.concept == concept.concept
        assert restored_concept.confidence == concept.confidence


class TestContentTypeDetection:
    """Test cases for ContentTypeDetection data class."""
    
    def test_content_type_detection_creation(self):
        """Test ContentTypeDetection creation and serialization."""
        detection = ContentTypeDetection(
            detected_type=ContentType.EDUCATIONAL,
            confidence=0.92,
            reasoning="Audio keywords: financial, education; Visual elements: chart",
            alternative_types=[(ContentType.GENERAL, 0.3)]
        )
        
        assert detection.detected_type == ContentType.EDUCATIONAL
        assert detection.confidence == 0.92
        assert len(detection.alternative_types) == 1
        
        # Test serialization
        detection_dict = detection.to_dict()
        assert detection_dict['detected_type'] == 'educational'
        assert detection_dict['confidence'] == 0.92
        
        # Test deserialization
        restored_detection = ContentTypeDetection.from_dict(detection_dict)
        assert restored_detection.detected_type == ContentType.EDUCATIONAL
        assert restored_detection.confidence == 0.92


def test_create_content_analyzer():
    """Test factory function for creating ContentAnalyzer."""
    analyzer = create_content_analyzer()
    
    assert isinstance(analyzer, MultiModalContentAnalyzer)
    assert analyzer.cache_manager is None
    assert analyzer.memory_client is None
    
    # Test with parameters
    mock_cache = Mock()
    mock_memory = Mock()
    analyzer_with_params = create_content_analyzer(
        cache_manager=mock_cache,
        memory_client=mock_memory
    )
    
    assert analyzer_with_params.cache_manager == mock_cache
    assert analyzer_with_params.memory_client == mock_memory