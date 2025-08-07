"""
Unit tests for Audio Enhancement Engine.

Tests the comprehensive audio enhancement system including noise reduction,
dynamic level adjustment, and integration with existing filler word removal.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from ai_video_editor.modules.enhancement.audio_enhancement import (
    AudioEnhancementEngine,
    AudioEnhancementSettings,
    AudioEnhancementResult,
    AudioCleanupPipeline,
    DynamicLevelAdjuster
)
from ai_video_editor.core.content_context import (
    ContentContext, 
    AudioAnalysisResult, 
    AudioSegment,
    EmotionalPeak,
    ContentType,
    UserPreferences
)
from ai_video_editor.core.exceptions import ProcessingError, ContentContextError


class TestAudioEnhancementSettings:
    """Test AudioEnhancementSettings configuration."""
    
    def test_default_settings(self):
        """Test default enhancement settings."""
        settings = AudioEnhancementSettings()
        
        assert settings.noise_reduction_strength == 0.5
        assert settings.enable_dynamic_levels is True
        assert settings.emotional_boost_factor == 1.2
        assert settings.target_lufs == -16.0
        assert settings.enable_eq is True
        
    def test_settings_to_dict(self):
        """Test settings serialization."""
        settings = AudioEnhancementSettings(
            noise_reduction_strength=0.7,
            emotional_boost_factor=1.3
        )
        
        settings_dict = settings.to_dict()
        
        assert settings_dict['noise_reduction_strength'] == 0.7
        assert settings_dict['emotional_boost_factor'] == 1.3
        assert 'enable_dynamic_levels' in settings_dict


class TestAudioCleanupPipeline:
    """Test AudioCleanupPipeline processing."""
    
    @pytest.fixture
    def cleanup_pipeline(self):
        """Create cleanup pipeline for testing."""
        settings = AudioEnhancementSettings()
        return AudioCleanupPipeline(settings)
    
    @pytest.fixture
    def mock_audio_data(self):
        """Create mock audio data for testing."""
        # Generate 1 second of test audio at 44.1kHz
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create test signal with noise
        signal = np.sin(2 * np.pi * 440 * t)  # 440Hz tone
        noise = np.random.normal(0, 0.1, len(signal))  # Add noise
        
        return signal + noise, sample_rate
    
    @patch('ai_video_editor.modules.enhancement.audio_enhancement.AUDIO_LIBS_AVAILABLE', True)
    def test_apply_noise_reduction(self, cleanup_pipeline, mock_audio_data):
        """Test noise reduction application."""
        audio_data, sample_rate = mock_audio_data
        
        # Mock numpy FFT functions
        with patch('numpy.fft.rfft') as mock_rfft, \
             patch('numpy.fft.irfft') as mock_irfft:
            
            # Setup mock FFT responses
            mock_fft = np.random.random(len(audio_data) // 2 + 1) + 1j * np.random.random(len(audio_data) // 2 + 1)
            mock_rfft.return_value = mock_fft
            mock_irfft.return_value = audio_data * 0.9  # Simulate noise reduction
            
            result = cleanup_pipeline.apply_noise_reduction(audio_data, sample_rate)
            
            assert isinstance(result, np.ndarray)
            assert len(result) == len(audio_data)
            mock_rfft.assert_called()
            mock_irfft.assert_called()
    
    @patch('ai_video_editor.modules.enhancement.audio_enhancement.AUDIO_LIBS_AVAILABLE', False)
    def test_noise_reduction_without_libs(self, cleanup_pipeline, mock_audio_data):
        """Test noise reduction when audio libraries are not available."""
        audio_data, sample_rate = mock_audio_data
        
        result = cleanup_pipeline.apply_noise_reduction(audio_data, sample_rate)
        
        # Should return original data unchanged
        np.testing.assert_array_equal(result, audio_data)
    
    def test_apply_eq_filtering(self, cleanup_pipeline, mock_audio_data):
        """Test EQ filtering application."""
        audio_data, sample_rate = mock_audio_data
        
        # Test with EQ disabled
        cleanup_pipeline.settings.enable_eq = False
        result = cleanup_pipeline.apply_eq_filtering(audio_data, sample_rate)
        np.testing.assert_array_equal(result, audio_data)
        
        # Test with EQ enabled
        cleanup_pipeline.settings.enable_eq = True
        with patch('scipy.signal.butter') as mock_butter, \
             patch('scipy.signal.filtfilt') as mock_filtfilt:
            
            # Setup mock filter responses
            mock_butter.return_value = ([1, 0, 0], [1, 0, 0])  # Mock filter coefficients
            mock_filtfilt.return_value = audio_data * 1.1  # Simulate EQ boost
            
            result = cleanup_pipeline.apply_eq_filtering(audio_data, sample_rate)
            
            assert isinstance(result, np.ndarray)
            assert len(result) == len(audio_data)
            mock_butter.assert_called()
            mock_filtfilt.assert_called()
    
    def test_apply_dynamic_range_compression(self, cleanup_pipeline, mock_audio_data):
        """Test dynamic range compression."""
        audio_data, sample_rate = mock_audio_data
        
        result = cleanup_pipeline.apply_dynamic_range_compression(audio_data, sample_rate)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio_data)
        
        # Check that compression was applied (peak levels should be reduced)
        original_peak = np.max(np.abs(audio_data))
        compressed_peak = np.max(np.abs(result))
        
        # Compression should generally reduce peaks or keep them similar
        assert compressed_peak <= original_peak * 1.1  # Allow small tolerance


class TestDynamicLevelAdjuster:
    """Test DynamicLevelAdjuster functionality."""
    
    @pytest.fixture
    def level_adjuster(self):
        """Create level adjuster for testing."""
        settings = AudioEnhancementSettings()
        return DynamicLevelAdjuster(settings)
    
    @pytest.fixture
    def mock_context_with_emotions(self):
        """Create ContentContext with emotional markers."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add emotional markers
        context.emotional_markers = [
            EmotionalPeak(
                timestamp=10.0,
                emotion="excitement",
                intensity=0.8,
                confidence=0.9,
                context="Great explanation"
            ),
            EmotionalPeak(
                timestamp=25.0,
                emotion="curiosity",
                intensity=0.6,
                confidence=0.8,
                context="Interesting concept"
            )
        ]
        
        # Add audio analysis with explanation segments
        audio_analysis = AudioAnalysisResult(
            transcript_text="Test transcript",
            segments=[],
            overall_confidence=0.9,
            language="en",
            processing_time=1.0,
            model_used="test"
        )
        audio_analysis.explanation_segments = [
            {
                'timestamp': 15.0,
                'text': 'Let me explain this concept',
                'type': 'explanation',
                'confidence': 0.9
            }
        ]
        
        context.set_audio_analysis(audio_analysis)
        
        return context
    
    def test_calculate_level_adjustments(self, level_adjuster, mock_context_with_emotions):
        """Test calculation of level adjustments."""
        adjustments = level_adjuster.calculate_level_adjustments(mock_context_with_emotions)
        
        assert len(adjustments) >= 2  # At least emotional and explanation adjustments
        
        # Check emotional boost adjustment
        emotional_adjustments = [adj for adj in adjustments if adj['type'] == 'emotional_boost']
        assert len(emotional_adjustments) >= 1
        
        emotional_adj = emotional_adjustments[0]
        assert emotional_adj['timestamp'] == 10.0
        assert emotional_adj['factor'] == 1.2  # Default emotional boost factor
        assert 'excitement' in emotional_adj['reason']
        
        # Check explanation boost adjustment
        explanation_adjustments = [adj for adj in adjustments if adj['type'] == 'explanation_boost']
        assert len(explanation_adjustments) >= 1
        
        explanation_adj = explanation_adjustments[0]
        assert explanation_adj['timestamp'] == 15.0
        assert explanation_adj['factor'] == 1.1  # Default explanation boost factor
    
    def test_calculate_level_adjustments_no_audio_analysis(self, level_adjuster):
        """Test level adjustment calculation with no audio analysis."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        adjustments = level_adjuster.calculate_level_adjustments(context)
        
        assert adjustments == []
    
    def test_apply_level_adjustments(self, level_adjuster):
        """Test application of level adjustments to audio data."""
        # Create test audio data
        sample_rate = 44100
        duration = 30.0  # 30 seconds
        audio_data = np.random.normal(0, 0.1, int(sample_rate * duration))
        
        # Create test adjustments
        adjustments = [
            {
                'timestamp': 10.0,
                'duration': 2.0,
                'type': 'emotional_boost',
                'factor': 1.5,
                'reason': 'Test boost'
            },
            {
                'timestamp': 20.0,
                'duration': 1.0,
                'type': 'filler_reduction',
                'factor': 0.7,
                'reason': 'Test reduction'
            }
        ]
        
        result = level_adjuster.apply_level_adjustments(audio_data, sample_rate, adjustments)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio_data)
        
        # Check that adjustments were applied at correct timestamps
        boost_start = int(10.0 * sample_rate)
        boost_end = int(12.0 * sample_rate)
        
        # The boosted section should have higher amplitude
        original_boost_section = audio_data[boost_start:boost_end]
        enhanced_boost_section = result[boost_start:boost_end]
        
        # Check that some enhancement was applied (allowing for envelope effects)
        assert np.mean(np.abs(enhanced_boost_section)) >= np.mean(np.abs(original_boost_section)) * 0.9


class TestAudioEnhancementEngine:
    """Test AudioEnhancementEngine main functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def enhancement_engine(self, temp_dir):
        """Create enhancement engine for testing."""
        with patch('ai_video_editor.modules.enhancement.audio_enhancement.AUDIO_LIBS_AVAILABLE', True):
            return AudioEnhancementEngine(output_dir=temp_dir)
    
    @pytest.fixture
    def mock_context_full(self, temp_dir):
        """Create full ContentContext for testing."""
        context = ContentContext(
            project_id="test_project",
            video_files=[str(Path(temp_dir) / "test_video.mp4")],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add comprehensive audio analysis
        segments = [
            AudioSegment(
                text="Welcome to this tutorial",
                start=0.0,
                end=2.0,
                confidence=0.9
            ),
            AudioSegment(
                text="Let me explain the concept",
                start=2.0,
                end=5.0,
                confidence=0.8
            )
        ]
        
        audio_analysis = AudioAnalysisResult(
            transcript_text="Welcome to this tutorial. Let me explain the concept.",
            segments=segments,
            overall_confidence=0.85,
            language="en",
            processing_time=2.0,
            model_used="medium",
            filler_words_removed=2,
            segments_modified=1,
            quality_improvement_score=0.7,
            original_duration=5.0,
            enhanced_duration=4.8
        )
        
        context.set_audio_analysis(audio_analysis)
        
        # Add emotional markers
        context.emotional_markers = [
            EmotionalPeak(
                timestamp=3.0,
                emotion="excitement",
                intensity=0.8,
                confidence=0.9,
                context="Explaining concept"
            )
        ]
        
        return context
    
    @patch('ai_video_editor.modules.enhancement.audio_enhancement.AUDIO_LIBS_AVAILABLE', False)
    def test_init_without_audio_libs(self, temp_dir):
        """Test initialization without audio libraries."""
        with pytest.raises(ImportError, match="Audio processing libraries are required"):
            AudioEnhancementEngine(output_dir=temp_dir)
    
    def test_enhance_audio_no_audio_analysis(self, enhancement_engine):
        """Test enhancement with no audio analysis in context."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        with pytest.raises((ContentContextError, ProcessingError)):
            enhancement_engine.enhance_audio(context)
    
    def test_enhance_audio_no_audio_source(self, enhancement_engine):
        """Test enhancement with no audio source available."""
        context = ContentContext(
            project_id="test_project",
            video_files=[],  # No video files
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add minimal audio analysis
        audio_analysis = AudioAnalysisResult(
            transcript_text="Test",
            segments=[],
            overall_confidence=0.9,
            language="en",
            processing_time=1.0,
            model_used="test"
        )
        context.set_audio_analysis(audio_analysis)
        
        with pytest.raises((ContentContextError, ProcessingError)):
            enhancement_engine.enhance_audio(context)
    
    def test_enhance_audio_success(self, enhancement_engine, mock_context_full):
        """Test successful audio enhancement."""
        with patch('librosa.load') as mock_load, \
             patch('librosa.output.write_wav') as mock_write_wav:
            
            # Mock audio loading
            sample_rate = 44100
            duration = 5.0
            mock_audio_data = np.random.normal(0, 0.1, int(sample_rate * duration))
            mock_load.return_value = (mock_audio_data, sample_rate)
            
            # Mock audio saving
            mock_write_wav.return_value = None
            
            # Create mock video file
            video_path = Path(mock_context_full.video_files[0])
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.touch()
            
            result = enhancement_engine.enhance_audio(mock_context_full)
            
            assert isinstance(result, AudioEnhancementResult)
            assert result.processing_time > 0
            assert result.original_duration == duration
            assert result.enhanced_duration > 0
            assert result.enhanced_audio_path is not None
            
            # Check that audio loading and saving were called
            mock_load.assert_called_once()
            mock_write_wav.assert_called_once()
            
            # Check that context was updated
            assert mock_context_full.processed_video is not None
            assert 'audio_enhancement' in mock_context_full.processed_video
    
    def test_get_enhancement_settings(self, enhancement_engine):
        """Test getting enhancement settings."""
        settings = enhancement_engine.get_enhancement_settings()
        
        assert isinstance(settings, AudioEnhancementSettings)
        assert settings.noise_reduction_strength == 0.5  # Default value
    
    def test_update_enhancement_settings(self, enhancement_engine):
        """Test updating enhancement settings."""
        new_settings = AudioEnhancementSettings(
            noise_reduction_strength=0.8,
            emotional_boost_factor=1.5
        )
        
        enhancement_engine.update_enhancement_settings(new_settings)
        
        updated_settings = enhancement_engine.get_enhancement_settings()
        assert updated_settings.noise_reduction_strength == 0.8
        assert updated_settings.emotional_boost_factor == 1.5
    
    def test_load_audio_data_video_file(self, enhancement_engine):
        """Test loading audio data from video file."""
        with patch('librosa.load') as mock_load:
            mock_audio_data = np.random.normal(0, 0.1, 44100)
            mock_load.return_value = (mock_audio_data, 44100)
            
            audio_data, sample_rate = enhancement_engine._load_audio_data("test_video.mp4")
            
            assert isinstance(audio_data, np.ndarray)
            assert sample_rate == 44100
            mock_load.assert_called_once_with("test_video.mp4", sr=None, mono=True)
    
    def test_load_audio_data_audio_file(self, enhancement_engine):
        """Test loading audio data from audio file."""
        with patch('librosa.load') as mock_load:
            mock_audio_data = np.random.normal(0, 0.1, 44100)
            mock_load.return_value = (mock_audio_data, 44100)
            
            audio_data, sample_rate = enhancement_engine._load_audio_data("test_audio.wav")
            
            assert isinstance(audio_data, np.ndarray)
            assert sample_rate == 44100
            mock_load.assert_called_once_with("test_audio.wav", sr=None, mono=True)
    
    def test_apply_final_normalization(self, enhancement_engine):
        """Test final normalization application."""
        # Create test audio with peaks above limit
        audio_data = np.array([0.5, -0.8, 0.9, -0.7, 0.6])
        
        normalized = enhancement_engine._apply_final_normalization(audio_data)
        
        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(audio_data)
        
        # Check that peaks are within limit
        peak_level = np.max(np.abs(normalized))
        target_peak = 10 ** (-1.0 / 20)  # Default peak limit
        assert peak_level <= target_peak * 1.01  # Small tolerance for floating point
    
    def test_calculate_snr_improvement(self, enhancement_engine):
        """Test SNR improvement calculation."""
        original = np.random.normal(0, 0.2, 1000)
        enhanced = np.random.normal(0, 0.1, 1000)  # Less noise
        
        improvement = enhancement_engine._calculate_snr_improvement(original, enhanced)
        
        assert isinstance(improvement, float)
        assert improvement >= 0.0  # Only positive improvements counted
    
    def test_calculate_loudness_consistency(self, enhancement_engine):
        """Test loudness consistency calculation."""
        # Create audio with varying levels
        audio_data = np.concatenate([
            np.random.normal(0, 0.1, 1000),  # Quiet section
            np.random.normal(0, 0.3, 1000),  # Loud section
            np.random.normal(0, 0.2, 1000)   # Medium section
        ])
        
        consistency = enhancement_engine._calculate_loudness_consistency(audio_data)
        
        assert isinstance(consistency, float)
        assert 0.0 <= consistency <= 1.0
    
    def test_create_sync_points(self, enhancement_engine, mock_context_full):
        """Test sync points creation."""
        level_adjustments = [
            {
                'timestamp': 10.0,
                'type': 'emotional_boost',
                'factor': 1.2,
                'reason': 'Test boost'
            }
        ]
        
        sync_points = enhancement_engine._create_sync_points(mock_context_full, level_adjustments)
        
        assert isinstance(sync_points, list)
        assert len(sync_points) >= 1
        
        # Check emotional peak sync point
        emotional_sync = [sp for sp in sync_points if sp['type'] == 'emotional_peak']
        assert len(emotional_sync) >= 1
        
        emotional_point = emotional_sync[0]
        assert emotional_point['timestamp'] == 3.0
        assert emotional_point['metadata']['emotion'] == 'excitement'
        
        # Check level adjustment sync point
        level_sync = [sp for sp in sync_points if sp['type'] == 'level_adjustment']
        assert len(level_sync) >= 1
        
        level_point = level_sync[0]
        assert level_point['timestamp'] == 10.0
        assert level_point['metadata']['factor'] == 1.2


class TestAudioEnhancementIntegration:
    """Test integration between enhancement components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @patch('ai_video_editor.modules.enhancement.audio_enhancement.AUDIO_LIBS_AVAILABLE', True)
    @patch('librosa.load')
    @patch('librosa.output.write_wav')
    def test_full_enhancement_pipeline(self, mock_write_wav, mock_load, temp_dir):
        """Test complete enhancement pipeline integration."""
        # Setup mocks
        sample_rate = 44100
        duration = 10.0
        mock_audio_data = np.random.normal(0, 0.1, int(sample_rate * duration))
        mock_load.return_value = (mock_audio_data, sample_rate)
        mock_write_wav.return_value = None
        
        # Create enhancement engine
        engine = AudioEnhancementEngine(output_dir=temp_dir)
        
        # Create comprehensive context
        context = ContentContext(
            project_id="integration_test",
            video_files=[str(Path(temp_dir) / "test_video.mp4")],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add audio analysis with all features
        segments = [
            AudioSegment(
                text="Welcome to the tutorial",
                start=0.0,
                end=3.0,
                confidence=0.9,
                filler_words=["um", "uh"],
                cleaned_text="Welcome to the tutorial"
            ),
            AudioSegment(
                text="Let me explain this concept",
                start=3.0,
                end=7.0,
                confidence=0.8,
                financial_concepts=["investment", "portfolio"]
            )
        ]
        
        audio_analysis = AudioAnalysisResult(
            transcript_text="Welcome to the tutorial. Let me explain this concept.",
            segments=segments,
            overall_confidence=0.85,
            language="en",
            processing_time=2.0,
            model_used="medium",
            filler_words_removed=2,
            segments_modified=1,
            quality_improvement_score=0.7,
            original_duration=duration,
            enhanced_duration=duration - 0.5
        )
        
        # Add explanation segments
        audio_analysis.explanation_segments = [
            {
                'timestamp': 4.0,
                'text': 'Let me explain this concept',
                'type': 'explanation',
                'confidence': 0.8
            }
        ]
        
        context.set_audio_analysis(audio_analysis)
        
        # Add emotional markers
        context.emotional_markers = [
            EmotionalPeak(
                timestamp=2.0,
                emotion="excitement",
                intensity=0.9,
                confidence=0.8,
                context="Tutorial introduction"
            ),
            EmotionalPeak(
                timestamp=5.0,
                emotion="curiosity",
                intensity=0.7,
                confidence=0.9,
                context="Concept explanation"
            )
        ]
        
        # Create mock video file
        video_path = Path(context.video_files[0])
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()
        
        # Run enhancement
        result = engine.enhance_audio(context)
        
        # Verify comprehensive results
        assert isinstance(result, AudioEnhancementResult)
        assert result.processing_time > 0
        assert result.original_duration == duration
        assert result.enhanced_duration > 0
        assert result.noise_reduction_applied is True
        assert result.dynamic_adjustments_made >= 2  # Emotional + explanation adjustments
        assert result.snr_improvement >= 0
        assert 0.0 <= result.loudness_consistency_score <= 1.0
        assert result.enhanced_audio_path is not None
        assert len(result.sync_points) >= 2  # Emotional peaks
        assert len(result.level_adjustments) >= 2  # Dynamic adjustments
        
        # Verify context updates
        assert context.processed_video is not None
        assert 'audio_enhancement' in context.processed_video
        assert context.processing_metrics.module_processing_times['audio_enhancement'] > 0
        
        # Verify audio analysis updates
        assert context.audio_analysis.enhanced_duration == result.enhanced_duration
        assert context.audio_analysis.quality_improvement_score > 0
        
        # Verify sync points contain expected data
        emotional_sync_points = [sp for sp in result.sync_points if sp['type'] == 'emotional_peak']
        assert len(emotional_sync_points) == 2
        
        level_adjustment_points = [sp for sp in result.sync_points if sp['type'] == 'level_adjustment']
        assert len(level_adjustment_points) >= 2
        
        # Verify level adjustments
        emotional_adjustments = [adj for adj in result.level_adjustments if adj['type'] == 'emotional_boost']
        explanation_adjustments = [adj for adj in result.level_adjustments if adj['type'] == 'explanation_boost']
        
        assert len(emotional_adjustments) >= 1
        assert len(explanation_adjustments) >= 1
        
        # Check adjustment factors
        for adj in emotional_adjustments:
            assert adj['factor'] == 1.2  # Default emotional boost
        
        for adj in explanation_adjustments:
            assert adj['factor'] == 1.1  # Default explanation boost