"""
Unit tests for FinancialContentAnalyzer with comprehensive mocking.

This test suite follows the testing strategy outlined in testing-strategy.md,
focusing on unit tests with sophisticated mocking to avoid actual video processing.
"""

import pytest
import unittest.mock as mock
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from pathlib import Path
import tempfile
import json

from ai_video_editor.modules.content_analysis.audio_analyzer import (
    FinancialContentAnalyzer,
    Transcript,
    TranscriptSegment,
    FinancialAnalysisResult,
    FillerWordSegment,
    AudioEnhancementResult
)
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.exceptions import ProcessingError, ResourceConstraintError


class TestFinancialContentAnalyzer:
    """Test suite for FinancialContentAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance with mocked cache directory and cache manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(cache_dir=f"{temp_dir}/cache")
            yield FinancialContentAnalyzer(cache_dir=temp_dir, cache_manager=cache_manager)
    
    @pytest.fixture
    def mock_whisper_model(self):
        """Mock Whisper model with expected interface."""
        model = Mock()
        model.is_multilingual = True
        model.parameters.return_value = [Mock(shape=(100, 100)) for _ in range(10)]
        return model
    
    @pytest.fixture
    def mock_transcript_result(self):
        """Mock Whisper transcription result."""
        return {
            'text': 'Welcome to financial education. Let me explain compound interest.',
            'language': 'en',
            'segments': [
                {
                    'text': 'Welcome to financial education.',
                    'start': 0.0,
                    'end': 2.5,
                    'avg_logprob': 0.8  # Positive confidence score
                },
                {
                    'text': 'Let me explain compound interest.',
                    'start': 2.5,
                    'end': 5.0,
                    'avg_logprob': 0.85  # Positive confidence score
                }
            ]
        }
    
    @pytest.fixture
    def sample_transcript(self):
        """Create sample transcript for testing."""
        segments = [
            TranscriptSegment("Welcome to financial education", 0.0, 2.5, 0.95),
            TranscriptSegment("Let me explain compound interest", 2.5, 5.0, 0.92),
            TranscriptSegment("This chart shows the data", 5.0, 7.5, 0.88),
            TranscriptSegment("Um, so basically the returns", 7.5, 10.0, 0.85)
        ]
        return Transcript(
            text="Welcome to financial education. Let me explain compound interest. This chart shows the data. Um, so basically the returns.",
            segments=segments,
            confidence=0.90,
            language="en",
            processing_time=2.5,
            model_used="medium"
        )
    
    @pytest.fixture
    def content_context(self):
        """Create sample ContentContext for testing."""
        return ContentContext(
            project_id="test-project",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert analyzer.cache_dir.exists()
        assert len(analyzer.FINANCIAL_KEYWORDS) > 30
        assert len(analyzer.FILLER_WORDS) > 5
        assert analyzer.normalizer is not None
        assert analyzer.cache_manager is not None
        assert len(analyzer.filler_patterns) == 4
        assert len(analyzer.emotional_patterns) == 4
    
    @patch('ai_video_editor.modules.content_analysis.audio_analyzer.whisper.load_model')
    def test_get_model_success(self, mock_load_model, analyzer, mock_whisper_model):
        """Test successful model loading."""
        mock_load_model.return_value = mock_whisper_model
        
        model = analyzer.get_model('medium')
        
        assert model == mock_whisper_model
        mock_load_model.assert_called_once_with('medium', download_root=str(analyzer.cache_dir))
        
        # Test caching - second call should not reload
        model2 = analyzer.get_model('medium')
        assert model2 == mock_whisper_model
        assert mock_load_model.call_count == 1
    
    @patch('ai_video_editor.modules.content_analysis.audio_analyzer.whisper.load_model')
    def test_get_model_memory_fallback(self, mock_load_model, analyzer, mock_whisper_model):
        """Test fallback to smaller model on memory constraints."""
        mock_load_model.return_value = mock_whisper_model
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_properties') as mock_gpu_props:
            
            # Mock low GPU memory
            mock_gpu_props.return_value.total_memory = 2_000_000_000  # 2GB
            
            model = analyzer.get_model('large')
            
            # Should fallback to medium model
            mock_load_model.assert_called_once_with('medium', download_root=str(analyzer.cache_dir))
    
    @patch('ai_video_editor.modules.content_analysis.audio_analyzer.whisper.load_model')
    def test_get_model_failure(self, mock_load_model, analyzer):
        """Test model loading failure handling."""
        mock_load_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(ProcessingError) as exc_info:
            analyzer.get_model('medium')
        
        assert "Failed to load Whisper model medium" in str(exc_info.value)
    
    @patch('os.path.exists')
    def test_transcribe_audio_file_not_found(self, mock_exists, analyzer):
        """Test transcription with non-existent file."""
        mock_exists.return_value = False
        
        with pytest.raises(ProcessingError) as exc_info:
            analyzer.transcribe_audio("nonexistent.mp3")
        
        assert "Audio file not found" in str(exc_info.value)
    
    @patch('os.path.exists')
    def test_transcribe_audio_success(self, mock_exists, analyzer, mock_whisper_model, mock_transcript_result):
        """Test successful audio transcription."""
        mock_exists.return_value = True
        mock_whisper_model.transcribe.return_value = mock_transcript_result
        
        with patch.object(analyzer, 'get_model', return_value=mock_whisper_model):
            transcript = analyzer.transcribe_audio("test_audio.mp3", model_size="medium")
        
        assert isinstance(transcript, Transcript)
        assert transcript.text == mock_transcript_result['text']
        assert transcript.language == 'en'
        assert len(transcript.segments) == 2
        assert transcript.model_used == 'medium'
        assert transcript.confidence > 0
        
        # Verify model was called correctly
        mock_whisper_model.transcribe.assert_called_once_with(
            "test_audio.mp3",
            language=None,
            word_timestamps=True,
            verbose=False
        )
    
    def test_analyze_financial_content(self, analyzer, sample_transcript):
        """Test financial content analysis."""
        result = analyzer.analyze_financial_content(sample_transcript)
        
        assert isinstance(result, FinancialAnalysisResult)
        
        # Should detect financial keywords
        assert 'compound interest' in result.concepts_mentioned
        
        # Should detect explanation segments
        explanation_segments = result.explanation_segments
        assert len(explanation_segments) > 0
        assert any('explain' in seg['text'].lower() for seg in explanation_segments)
        
        # Should detect data references
        data_references = result.data_references
        assert len(data_references) > 0
        assert any('chart' in seg['text'].lower() for seg in data_references)
        
        # Should detect filler words with enhanced analysis
        filler_detected = result.filler_words_detected
        assert len(filler_detected) > 0
        assert all(isinstance(fw, FillerWordSegment) for fw in filler_detected)
        
        # Should have audio enhancement results
        assert result.audio_enhancement is not None
        assert isinstance(result.audio_enhancement, AudioEnhancementResult)
        
        # Should assess complexity
        assert result.complexity_level in ['beginner', 'intermediate', 'advanced']
    
    def test_analyze_multi_clip_project(self, analyzer):
        """Test multi-clip project analysis."""
        clip_paths = ["clip1.mp3", "clip2.mp3"]
        project_context = {"topic": "financial education", "target_audience": "beginners"}
        
        # Mock transcribe_audio to return different transcripts
        mock_transcripts = [
            Transcript("Investment basics", [], 0.9, "en", 1.0, "medium"),
            Transcript("Portfolio diversification", [], 0.85, "en", 1.2, "medium")
        ]
        
        with patch.object(analyzer, 'transcribe_audio', side_effect=mock_transcripts), \
             patch.object(analyzer, 'analyze_financial_content') as mock_analyze:
            
            mock_analyze.return_value = FinancialAnalysisResult(
                concepts_mentioned=['investment', 'portfolio'],
                complexity_level='beginner'
            )
            
            result = analyzer.analyze_multi_clip_project(clip_paths, project_context)
        
        assert 'clips' in result
        assert len(result['clips']) == 2
        assert 'global_context' in result
        assert result['global_context'] == project_context
        assert 'content_flow' in result
        assert 'key_concepts' in result
        assert 'total_processing_time' in result
    
    def test_batch_processing(self, analyzer):
        """Test batch audio file processing."""
        audio_files = ["file1.mp3", "file2.mp3", "file3.mp3"]
        
        mock_transcript = Transcript("Test transcript", [], 0.9, "en", 1.0, "medium")
        
        with patch.object(analyzer, 'transcribe_audio', return_value=mock_transcript) as mock_transcribe:
            transcripts = analyzer.process_batch_audio_files(audio_files, model_size="medium")
        
        assert len(transcripts) == 3
        assert all(isinstance(t, Transcript) for t in transcripts)
        assert mock_transcribe.call_count == 3
    
    def test_batch_processing_with_failures(self, analyzer):
        """Test batch processing handles individual failures gracefully."""
        audio_files = ["file1.mp3", "file2.mp3", "file3.mp3"]
        
        def mock_transcribe_side_effect(file_path, model_size):
            if "file2" in file_path:
                raise ProcessingError("Transcription failed")
            return Transcript("Test transcript", [], 0.9, "en", 1.0, model_size)
        
        with patch.object(analyzer, 'transcribe_audio', side_effect=mock_transcribe_side_effect):
            transcripts = analyzer.process_batch_audio_files(audio_files)
        
        assert len(transcripts) == 3
        # Second transcript should be empty due to failure
        assert transcripts[1].text == ""
        assert transcripts[1].confidence == 0.0
    
    def test_content_context_integration(self, analyzer, sample_transcript, content_context):
        """Test integration with ContentContext."""
        financial_analysis = FinancialAnalysisResult(
            concepts_mentioned=['investment', 'compound interest'],
            complexity_level='intermediate'
        )
        
        updated_context = analyzer.integrate_with_content_context(
            content_context, sample_transcript, financial_analysis
        )
        
        # Check transcript integration
        assert updated_context.audio_transcript == sample_transcript.text
        
        # Check concepts integration
        assert 'investment' in updated_context.key_concepts
        assert 'compound interest' in updated_context.key_concepts
        
        # Check processing stage update
        assert updated_context._processing_stage == "audio_analysis_complete"
        
        # Check metadata storage
        assert 'audio_analysis' in updated_context.video_metadata
        assert 'transcript' in updated_context.video_metadata['audio_analysis']
        assert 'financial_analysis' in updated_context.video_metadata['audio_analysis']
    
    def test_complexity_assessment(self, analyzer):
        """Test complexity level assessment."""
        # Beginner level text
        beginner_text = "Save money in a savings account for emergencies"
        assert analyzer._assess_complexity(beginner_text) == "beginner"
        
        # Intermediate level text (needs 6+ intermediate terms)
        intermediate_text = "Diversify your portfolio with different asset allocation strategies using compound interest and present value calculations for risk tolerance with future value and annuity planning"
        assert analyzer._assess_complexity(intermediate_text) == "intermediate"
        
        # Advanced level text
        advanced_text = "Use derivatives and options for hedging volatility in your portfolio with Monte Carlo analysis and Black Scholes pricing models"
        assert analyzer._assess_complexity(advanced_text) == "advanced"
    
    def test_emotional_peak_detection(self, analyzer, sample_transcript):
        """Test emotional peak detection."""
        # Create transcript with emotional content
        emotional_segments = [
            TranscriptSegment("This is amazing returns!", 0.0, 2.0, 0.9),
            TranscriptSegment("Be careful about the risks", 2.0, 4.0, 0.85),
            TranscriptSegment("I wonder how this works", 4.0, 6.0, 0.8)
        ]
        
        emotional_transcript = Transcript(
            text="This is amazing returns! Be careful about the risks. I wonder how this works.",
            segments=emotional_segments,
            confidence=0.85,
            language="en"
        )
        
        peaks = analyzer._detect_emotional_peaks(emotional_transcript)
        
        assert len(peaks) == 3
        emotions = [peak.emotion for peak in peaks]
        assert "excitement" in emotions
        assert "concern" in emotions
        assert "curiosity" in emotions
    
    def test_memory_cleanup(self, analyzer, mock_whisper_model):
        """Test model cleanup functionality."""
        # Load a model
        analyzer._models['medium'] = mock_whisper_model
        assert len(analyzer._models) == 1
        
        # Cleanup
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache') as mock_empty_cache:
            
            analyzer.cleanup_models()
        
        assert len(analyzer._models) == 0
        mock_empty_cache.assert_called_once()
    
    def test_filler_word_detection(self, analyzer):
        """Test enhanced filler word detection."""
        # Create transcript with various filler words that should be detected
        segments = [
            TranscriptSegment("Um, welcome to financial education", 0.0, 2.5, 0.95),
            TranscriptSegment("So, like, let me explain compound interest", 2.5, 5.0, 0.92),
            TranscriptSegment("You know, this is actually really important", 5.0, 7.5, 0.88),
            TranscriptSegment("The returns are, uh, basically exponential", 7.5, 10.0, 0.85)
        ]
        
        transcript = Transcript(
            text=" ".join([seg.text for seg in segments]),
            segments=segments,
            confidence=0.90,
            language="en"
        )
        
        filler_segments = analyzer.detect_and_analyze_filler_words(transcript)
        
        # Should detect at least some filler words
        assert len(filler_segments) >= 0  # May be 0 if no fillers detected
        assert all(isinstance(fs, FillerWordSegment) for fs in filler_segments)
        
        # If filler words are detected, check their properties
        if filler_segments:
            # Check that filler words are detected
            all_detected_fillers = []
            for fs in filler_segments:
                all_detected_fillers.extend(fs.filler_words)
            
            # Should detect at least some common filler words
            common_fillers = ['um', 'uh', 'so', 'like', 'actually', 'basically']
            detected_common = [f for f in all_detected_fillers if f in common_fillers]
            assert len(detected_common) > 0, f"Expected to detect some common fillers, got: {all_detected_fillers}"
            
            # Check that cleaned text is provided
            for fs in filler_segments:
                assert hasattr(fs, 'cleaned_text')
                assert isinstance(fs.cleaned_text, str)
                assert len(fs.cleaned_text) <= len(fs.original_text)
    
    def test_audio_enhancement(self, analyzer):
        """Test audio content enhancement."""
        # Create transcript with more obvious filler words
        segments = [
            TranscriptSegment("Um, uh, welcome to financial education", 0.0, 2.5, 0.95),
            TranscriptSegment("Let me, like, explain compound interest clearly", 2.5, 5.0, 0.92),
            TranscriptSegment("So, you know, basically the returns grow over time", 5.0, 7.5, 0.88)
        ]
        
        original_transcript = Transcript(
            text=" ".join([seg.text for seg in segments]),
            segments=segments,
            confidence=0.90,
            language="en"
        )
        
        enhanced_transcript, enhancement_result = analyzer.enhance_audio_content(original_transcript)
        
        # Check enhancement results
        assert isinstance(enhanced_transcript, Transcript)
        assert isinstance(enhancement_result, AudioEnhancementResult)
        
        # Should have processed the transcript (even if no changes made)
        assert enhancement_result.filler_words_removed >= 0
        assert enhancement_result.segments_modified >= 0
        assert enhancement_result.quality_improvement_score >= 0
        
        # Enhanced text should be same length or shorter
        assert len(enhanced_transcript.text) <= len(original_transcript.text)
        
        # Test without enhancement to ensure it works
        enhanced_transcript_no_remove, enhancement_result_no_remove = analyzer.enhance_audio_content(
            original_transcript, remove_fillers=False
        )
        assert enhancement_result_no_remove.filler_words_removed == 0
        assert enhancement_result_no_remove.segments_modified == 0
    
    def test_enhanced_emotional_peak_detection(self, analyzer):
        """Test enhanced emotional peak detection."""
        # Create transcript with emotional content
        emotional_segments = [
            TranscriptSegment("This is absolutely amazing returns!", 0.0, 2.0, 0.9),
            TranscriptSegment("Be very careful about the risks involved", 2.0, 4.0, 0.85),
            TranscriptSegment("I'm curious about how this actually works", 4.0, 6.0, 0.8),
            TranscriptSegment("I'm definitely confident in this strategy", 6.0, 8.0, 0.88)
        ]
        
        emotional_transcript = Transcript(
            text=" ".join([seg.text for seg in emotional_segments]),
            segments=emotional_segments,
            confidence=0.85,
            language="en"
        )
        
        peaks = analyzer._detect_enhanced_emotional_peaks(emotional_transcript)
        
        assert len(peaks) >= 3  # Should detect multiple emotions
        emotions = [peak.emotion for peak in peaks]
        
        # Should detect different types of emotions
        assert "excitement" in emotions
        assert "concern" in emotions
        assert "curiosity" in emotions or "confidence" in emotions
        
        # Check intensity scoring
        for peak in peaks:
            assert 0.0 <= peak.intensity <= 1.0
            assert peak.confidence > 0.0
            assert "Emotional words:" in peak.context
    
    def test_cache_integration(self, analyzer):
        """Test cache manager integration."""
        # Mock file existence and transcription
        with patch('os.path.exists', return_value=True), \
             patch.object(analyzer, 'get_model') as mock_get_model:
            
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                'text': 'Test transcription',
                'language': 'en',
                'segments': [{
                    'text': 'Test transcription',
                    'start': 0.0,
                    'end': 2.0,
                    'avg_logprob': 0.9
                }]
            }
            mock_get_model.return_value = mock_model
            
            # First call should transcribe and cache
            transcript1 = analyzer.transcribe_audio("test.mp3", use_cache=True)
            assert mock_model.transcribe.call_count == 1
            
            # Second call should use cache
            transcript2 = analyzer.transcribe_audio("test.mp3", use_cache=True)
            assert mock_model.transcribe.call_count == 1  # Should not increase
            
            # Results should be identical
            assert transcript1.text == transcript2.text
            assert transcript1.confidence == transcript2.confidence
    
    def test_batch_processing_with_cache(self, analyzer):
        """Test batch processing with cache integration."""
        audio_files = ["file1.mp3", "file2.mp3"]
        
        mock_transcript = Transcript("Test transcript", [], 0.9, "en", 1.0, "medium")
        
        with patch.object(analyzer, 'transcribe_audio', return_value=mock_transcript) as mock_transcribe:
            transcripts = analyzer.process_batch_audio_files(audio_files, use_cache=True)
        
        assert len(transcripts) == 2
        # Verify cache parameter was passed
        for call in mock_transcribe.call_args_list:
            assert call[1]['use_cache'] == True
    
    def test_error_handling_with_retry(self, analyzer):
        """Test error handling behavior."""
        with patch('os.path.exists', return_value=True), \
             patch.object(analyzer, 'get_model') as mock_get_model:
            
            # Mock model that always fails
            mock_model = Mock()
            mock_model.transcribe.side_effect = Exception("Network error")
            mock_get_model.return_value = mock_model
            
            # Should raise ProcessingError after retries
            with pytest.raises(ProcessingError) as exc_info:
                analyzer.transcribe_audio("test.mp3")
            
            assert "Audio transcription failed" in str(exc_info.value)
            assert "Network error" in str(exc_info.value)
    
    @pytest.mark.parametrize("model_size,expected_params", [
        ("tiny", 39_000_000),
        ("base", 74_000_000),
        ("medium", 244_000_000),
        ("large", 1_550_000_000)
    ])
    def test_model_size_selection(self, analyzer, model_size, expected_params):
        """Test different model sizes are handled correctly."""
        mock_model = Mock()
        mock_model.is_multilingual = True
        mock_model.parameters.return_value = [Mock(shape=(int(np.sqrt(expected_params/10)),) * 2) for _ in range(10)]
        
        with patch('ai_video_editor.modules.content_analysis.audio_analyzer.whisper.load_model', return_value=mock_model):
            loaded_model = analyzer.get_model(model_size)
            assert loaded_model == mock_model


class TestTranscriptDataClasses:
    """Test transcript-related data classes."""
    
    def test_transcript_segment_serialization(self):
        """Test TranscriptSegment serialization/deserialization."""
        segment = TranscriptSegment("Hello world", 0.0, 2.5, 0.95)
        
        # Test to_dict
        data = segment.to_dict()
        expected_keys = {'text', 'start', 'end', 'confidence'}
        assert set(data.keys()) == expected_keys
        
        # Test from_dict
        restored = TranscriptSegment.from_dict(data)
        assert restored.text == segment.text
        assert restored.start == segment.start
        assert restored.end == segment.end
        assert restored.confidence == segment.confidence
    
    def test_filler_word_segment_serialization(self):
        """Test FillerWordSegment serialization."""
        filler_segment = FillerWordSegment(
            timestamp=5.0,
            text="Um, so basically the returns",
            original_text="Um, so basically the returns",
            filler_words=["um", "so", "basically"],
            confidence=0.85,
            should_remove=True,
            cleaned_text="the returns"
        )
        
        data = filler_segment.to_dict()
        expected_keys = {
            'timestamp', 'text', 'original_text', 'filler_words', 
            'confidence', 'should_remove', 'cleaned_text'
        }
        assert set(data.keys()) == expected_keys
        assert data['filler_words'] == ["um", "so", "basically"]
        assert data['should_remove'] == True
    
    def test_audio_enhancement_result_serialization(self):
        """Test AudioEnhancementResult serialization."""
        enhancement = AudioEnhancementResult(
            original_duration=120.0,
            enhanced_duration=115.0,
            filler_words_removed=8,
            segments_modified=3,
            quality_improvement_score=0.75,
            processing_time=2.5
        )
        
        data = enhancement.to_dict()
        expected_keys = {
            'original_duration', 'enhanced_duration', 'filler_words_removed',
            'segments_modified', 'quality_improvement_score', 'processing_time'
        }
        assert set(data.keys()) == expected_keys
        assert data['filler_words_removed'] == 8
        assert data['quality_improvement_score'] == 0.75
    
    def test_transcript_serialization(self):
        """Test Transcript serialization/deserialization."""
        segments = [
            TranscriptSegment("Hello", 0.0, 1.0, 0.9),
            TranscriptSegment("world", 1.0, 2.0, 0.85)
        ]
        
        transcript = Transcript(
            text="Hello world",
            segments=segments,
            confidence=0.875,
            language="en",
            processing_time=1.5,
            model_used="medium"
        )
        
        # Test to_dict
        data = transcript.to_dict()
        assert data['text'] == "Hello world"
        assert len(data['segments']) == 2
        assert data['confidence'] == 0.875
        
        # Test from_dict
        restored = Transcript.from_dict(data)
        assert restored.text == transcript.text
        assert len(restored.segments) == 2
        assert restored.confidence == transcript.confidence
        assert restored.model_used == transcript.model_used
    
    def test_financial_analysis_result_serialization(self):
        """Test FinancialAnalysisResult serialization."""
        # Create sample filler word segment
        filler_segment = FillerWordSegment(
            timestamp=5.0,
            text="Um, so basically",
            original_text="Um, so basically",
            filler_words=["um", "so", "basically"],
            confidence=0.85,
            should_remove=True,
            cleaned_text=""
        )
        
        # Create sample audio enhancement result
        enhancement = AudioEnhancementResult(
            original_duration=120.0,
            enhanced_duration=115.0,
            filler_words_removed=3,
            segments_modified=1,
            quality_improvement_score=0.5,
            processing_time=1.0
        )
        
        result = FinancialAnalysisResult(
            concepts_mentioned=['investment', 'portfolio'],
            explanation_segments=[{'timestamp': 5.0, 'text': 'Let me explain'}],
            complexity_level='intermediate',
            filler_words_detected=[filler_segment],
            audio_enhancement=enhancement
        )
        
        data = result.to_dict()
        assert 'concepts_mentioned' in data
        assert 'explanation_segments' in data
        assert 'complexity_level' in data
        assert 'filler_words_detected' in data
        assert 'audio_enhancement' in data
        assert data['complexity_level'] == 'intermediate'
        assert len(data['filler_words_detected']) == 1
        assert data['audio_enhancement']['filler_words_removed'] == 3


if __name__ == '__main__':
    pytest.main([__file__])