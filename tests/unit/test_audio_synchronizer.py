"""
Unit tests for Audio Synchronizer.

Tests the audio-video synchronization system including timing analysis,
sync point management, and movis integration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from ai_video_editor.modules.enhancement.audio_synchronizer import (
    AudioSynchronizer,
    SyncPoint,
    AudioTrackInfo,
    SynchronizationResult,
    TimingAnalyzer
)
from ai_video_editor.modules.enhancement.audio_enhancement import (
    AudioEnhancementResult
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


class TestSyncPoint:
    """Test SyncPoint data structure."""
    
    def test_sync_point_creation(self):
        """Test SyncPoint creation and serialization."""
        sync_point = SyncPoint(
            timestamp=10.5,
            sync_type="cut",
            audio_adjustment=-0.1,
            video_adjustment=0.05,
            priority="high",
            metadata={"decision_id": "cut_001"}
        )
        
        assert sync_point.timestamp == 10.5
        assert sync_point.sync_type == "cut"
        assert sync_point.audio_adjustment == -0.1
        assert sync_point.video_adjustment == 0.05
        assert sync_point.priority == "high"
        assert sync_point.tolerance == 0.033  # Default 1 frame at 30fps
        
        # Test serialization
        sync_dict = sync_point.to_dict()
        assert sync_dict['timestamp'] == 10.5
        assert sync_dict['metadata']['decision_id'] == "cut_001"


class TestAudioTrackInfo:
    """Test AudioTrackInfo data structure."""
    
    def test_audio_track_info_creation(self):
        """Test AudioTrackInfo creation and serialization."""
        track = AudioTrackInfo(
            track_id="main_audio",
            source_path="/path/to/audio.wav",
            start_time=0.0,
            duration=120.0,
            volume=0.8,
            fade_in=0.1,
            fade_out=0.2
        )
        
        assert track.track_id == "main_audio"
        assert track.source_path == "/path/to/audio.wav"
        assert track.duration == 120.0
        assert track.volume == 0.8
        assert track.fade_in == 0.1
        assert track.fade_out == 0.2
        assert track.sync_adjustments == []
        
        # Test serialization
        track_dict = track.to_dict()
        assert track_dict['track_id'] == "main_audio"
        assert track_dict['duration'] == 120.0


class TestTimingAnalyzer:
    """Test TimingAnalyzer functionality."""
    
    @pytest.fixture
    def timing_analyzer(self):
        """Create timing analyzer for testing."""
        return TimingAnalyzer(fps=30.0)
    
    @pytest.fixture
    def mock_context_with_decisions(self):
        """Create ContentContext with editing decisions."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add processed video with editing decisions
        context.processed_video = {
            'editing_decisions': [
                {
                    'decision_id': 'cut_001',
                    'decision_type': 'cut',
                    'timestamp': 15.0,
                    'parameters': {'duration': 2.0},
                    'rationale': 'Remove filler segment'
                },
                {
                    'decision_id': 'transition_001',
                    'decision_type': 'transition',
                    'timestamp': 30.0,
                    'parameters': {'type': 'fade', 'duration': 1.0},
                    'rationale': 'Smooth transition'
                }
            ]
        }
        
        # Add emotional markers
        context.emotional_markers = [
            EmotionalPeak(
                timestamp=10.0,
                emotion="excitement",
                intensity=0.8,
                confidence=0.9,
                context="Key point"
            )
        ]
        
        # Add audio analysis with filler words
        segments = [
            AudioSegment(
                text="This is a test",
                start=0.0,
                end=5.0,
                confidence=0.9
            )
        ]
        
        audio_analysis = AudioAnalysisResult(
            transcript_text="This is a test",
            segments=segments,
            overall_confidence=0.9,
            language="en",
            processing_time=1.0,
            model_used="test"
        )
        
        # Mock filler words detected
        class MockFillerSegment:
            def __init__(self, timestamp, filler_words, should_remove=True):
                self.timestamp = timestamp
                self.filler_words = filler_words
                self.should_remove = should_remove
                self.original_text = "um, this is a test"
                self.cleaned_text = "this is a test"
        
        audio_analysis.filler_words_detected = [
            MockFillerSegment(5.0, ["um", "uh"], True)
        ]
        
        context.set_audio_analysis(audio_analysis)
        
        return context
    
    @pytest.fixture
    def mock_enhancement_result(self):
        """Create mock AudioEnhancementResult."""
        return AudioEnhancementResult(
            processing_time=2.0,
            original_duration=60.0,
            enhanced_duration=58.5,
            noise_reduction_applied=True,
            dynamic_adjustments_made=3,
            peak_reductions=1,
            level_boosts=2,
            snr_improvement=3.5,
            dynamic_range_improvement=2.1,
            loudness_consistency_score=0.8,
            enhanced_audio_path="/path/to/enhanced.wav",
            sync_points=[
                {
                    'timestamp': 10.0,
                    'type': 'emotional_peak',
                    'priority': 'high',
                    'metadata': {'emotion': 'excitement', 'intensity': 0.8}
                },
                {
                    'timestamp': 25.0,
                    'type': 'level_adjustment',
                    'priority': 'medium',
                    'metadata': {'adjustment_type': 'boost', 'factor': 1.2}
                }
            ],
            level_adjustments=[
                {
                    'timestamp': 25.0,
                    'type': 'emotional_boost',
                    'factor': 1.2,
                    'reason': 'Boost for excitement'
                }
            ]
        )
    
    def test_analyze_sync_requirements(self, timing_analyzer, mock_context_with_decisions, 
                                     mock_enhancement_result):
        """Test analysis of synchronization requirements."""
        sync_points = timing_analyzer.analyze_sync_requirements(
            mock_context_with_decisions, mock_enhancement_result
        )
        
        assert len(sync_points) >= 4  # Enhancement + editing + emotional + filler
        
        # Check for different sync point types
        sync_types = [point.sync_type for point in sync_points]
        assert 'emotional_peak' in sync_types
        assert 'level_adjustment' in sync_types
        assert 'cut' in sync_types
        assert 'transition' in sync_types
        assert 'filler_removal' in sync_types
        
        # Verify sync points are sorted by timestamp
        timestamps = [point.timestamp for point in sync_points]
        assert timestamps == sorted(timestamps)
        
        # Check specific sync point details
        cut_points = [p for p in sync_points if p.sync_type == 'cut']
        assert len(cut_points) >= 1
        cut_point = cut_points[0]
        assert cut_point.timestamp == 15.0
        assert cut_point.priority == 'high'
        assert cut_point.metadata['decision_id'] == 'cut_001'
    
    def test_remove_duplicate_sync_points(self, timing_analyzer):
        """Test removal of duplicate sync points."""
        # Create sync points that are too close together
        sync_points = [
            SyncPoint(timestamp=10.0, sync_type="cut", priority="high"),
            SyncPoint(timestamp=10.01, sync_type="cut", priority="medium"),  # Too close
            SyncPoint(timestamp=10.1, sync_type="transition", priority="high"),  # Different type, keep
            SyncPoint(timestamp=15.0, sync_type="cut", priority="low")  # Far enough, keep
        ]
        
        unique_points = timing_analyzer._remove_duplicate_sync_points(sync_points)
        
        # Should keep first cut (higher priority), transition, and distant cut
        assert len(unique_points) == 3
        assert unique_points[0].timestamp == 10.0
        assert unique_points[0].priority == "high"
        assert unique_points[1].timestamp == 10.1
        assert unique_points[1].sync_type == "transition"
        assert unique_points[2].timestamp == 15.0
    
    def test_calculate_timing_adjustments(self, timing_analyzer):
        """Test calculation of timing adjustments."""
        sync_points = [
            SyncPoint(timestamp=10.0, sync_type="cut", priority="high"),
            SyncPoint(timestamp=20.0, sync_type="transition", priority="medium"),
            SyncPoint(timestamp=30.0, sync_type="filler_removal", priority="medium",
                     metadata={'filler_words': ['um', 'uh']})
        ]
        
        adjusted_points = timing_analyzer.calculate_timing_adjustments(
            sync_points, original_duration=60.0, enhanced_duration=58.0
        )
        
        assert len(adjusted_points) == len(sync_points)
        
        # Check that timestamps are scaled
        time_scale = 58.0 / 60.0
        for i, point in enumerate(adjusted_points):
            expected_timestamp = sync_points[i].timestamp * time_scale
            assert abs(point.timestamp - expected_timestamp) < 0.01
        
        # Check cut adjustment (should be frame-accurate)
        cut_point = adjusted_points[0]
        assert cut_point.sync_type == "cut"
        assert cut_point.tolerance == timing_analyzer.frame_duration / 2
        
        # Check filler removal adjustment
        filler_point = adjusted_points[2]
        assert filler_point.sync_type == "filler_removal"
        assert filler_point.audio_adjustment < 0  # Audio is shorter due to filler removal
    
    def test_snap_to_frame(self, timing_analyzer):
        """Test frame snapping functionality."""
        # Test various timestamps
        assert timing_analyzer._snap_to_frame(10.0) == 10.0  # Exact frame
        assert timing_analyzer._snap_to_frame(10.016) == 10.0  # Round down
        assert timing_analyzer._snap_to_frame(10.017) == 10.033333333333333  # Round up
        
        # Test with different frame rate
        analyzer_24fps = TimingAnalyzer(fps=24.0)
        assert abs(analyzer_24fps._snap_to_frame(1.0) - 1.0) < 0.001
        assert abs(analyzer_24fps._snap_to_frame(1.02) - 1.0) < 0.001  # Round to nearest frame
    
    def test_estimate_filler_duration(self, timing_analyzer):
        """Test filler duration estimation."""
        metadata = {'filler_words': ['um', 'uh', 'like']}
        duration = timing_analyzer._estimate_filler_duration(metadata)
        
        # Should be 3 words * 0.3 seconds = 0.9 seconds
        assert duration == 0.9
        
        # Test with no filler words
        empty_metadata = {'filler_words': []}
        assert timing_analyzer._estimate_filler_duration(empty_metadata) == 0.0


class TestAudioSynchronizer:
    """Test AudioSynchronizer main functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def audio_synchronizer(self):
        """Create audio synchronizer for testing."""
        with patch('ai_video_editor.modules.enhancement.audio_synchronizer.MOVIS_AVAILABLE', True):
            return AudioSynchronizer(fps=30.0, sample_rate=48000)
    
    @pytest.fixture
    def mock_context_full(self, temp_dir):
        """Create full ContentContext for testing."""
        context = ContentContext(
            project_id="sync_test",
            video_files=[str(Path(temp_dir) / "test_video.mp4")],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add audio analysis
        segments = [
            AudioSegment(text="Test segment", start=0.0, end=5.0, confidence=0.9)
        ]
        
        audio_analysis = AudioAnalysisResult(
            transcript_text="Test segment",
            segments=segments,
            overall_confidence=0.9,
            language="en",
            processing_time=1.0,
            model_used="test"
        )
        
        context.set_audio_analysis(audio_analysis)
        
        return context
    
    @pytest.fixture
    def mock_enhancement_result_full(self, temp_dir):
        """Create comprehensive AudioEnhancementResult."""
        enhanced_path = Path(temp_dir) / "enhanced_audio.wav"
        enhanced_path.touch()  # Create mock file
        
        return AudioEnhancementResult(
            processing_time=3.0,
            original_duration=30.0,
            enhanced_duration=29.5,
            noise_reduction_applied=True,
            dynamic_adjustments_made=2,
            peak_reductions=1,
            level_boosts=1,
            snr_improvement=4.2,
            dynamic_range_improvement=2.8,
            loudness_consistency_score=0.85,
            enhanced_audio_path=str(enhanced_path),
            sync_points=[
                {
                    'timestamp': 5.0,
                    'type': 'emotional_peak',
                    'priority': 'high',
                    'metadata': {'emotion': 'excitement'}
                }
            ],
            level_adjustments=[
                {
                    'timestamp': 5.0,
                    'type': 'emotional_boost',
                    'factor': 1.3,
                    'reason': 'Excitement boost'
                }
            ]
        )
    
    @patch('ai_video_editor.modules.enhancement.audio_synchronizer.MOVIS_AVAILABLE', False)
    def test_init_without_movis(self):
        """Test initialization without movis library."""
        with pytest.raises(ImportError, match="movis library is required"):
            AudioSynchronizer()
    
    def test_init_with_custom_settings(self):
        """Test initialization with custom settings."""
        with patch('ai_video_editor.modules.enhancement.audio_synchronizer.MOVIS_AVAILABLE', True):
            synchronizer = AudioSynchronizer(fps=24.0, sample_rate=44100)
            
            assert synchronizer.fps == 24.0
            assert synchronizer.sample_rate == 44100
            assert synchronizer.frame_duration == 1.0 / 24.0
    
    def test_validate_synchronization_inputs_no_video(self, audio_synchronizer, 
                                                    mock_enhancement_result_full):
        """Test validation with no video files."""
        context = ContentContext(
            project_id="test",
            video_files=[],  # No video files
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        with pytest.raises(ContentContextError, match="No video files found"):
            audio_synchronizer._validate_synchronization_inputs(context, mock_enhancement_result_full)
    
    def test_validate_synchronization_inputs_no_enhanced_audio(self, audio_synchronizer, 
                                                             mock_context_full):
        """Test validation with no enhanced audio."""
        enhancement_result = AudioEnhancementResult(
            processing_time=1.0,
            original_duration=30.0,
            enhanced_duration=29.5,
            noise_reduction_applied=False,
            dynamic_adjustments_made=0,
            peak_reductions=0,
            level_boosts=0,
            snr_improvement=0.0,
            dynamic_range_improvement=0.0,
            loudness_consistency_score=0.5,
            enhanced_audio_path=None  # No enhanced audio path
        )
        
        with pytest.raises(ContentContextError, match="No enhanced audio path"):
            audio_synchronizer._validate_synchronization_inputs(mock_context_full, enhancement_result)
    
    def test_create_audio_tracks(self, audio_synchronizer, mock_context_full, 
                               mock_enhancement_result_full):
        """Test creation of audio tracks."""
        # Create mock sync points
        sync_points = [
            SyncPoint(timestamp=5.0, sync_type="cut", audio_adjustment=-0.1),
            SyncPoint(timestamp=10.0, sync_type="transition", audio_adjustment=0.0)
        ]
        
        tracks = audio_synchronizer._create_audio_tracks(
            mock_context_full, mock_enhancement_result_full, sync_points
        )
        
        assert len(tracks) >= 1
        
        main_track = tracks[0]
        assert main_track.track_id == "main_audio"
        assert main_track.source_path == mock_enhancement_result_full.enhanced_audio_path
        assert main_track.duration == mock_enhancement_result_full.enhanced_duration
        assert main_track.fade_in == 0.1
        assert main_track.fade_out == 0.1
        
        # Check sync adjustments
        assert len(main_track.sync_adjustments) >= 1
        adjustment = main_track.sync_adjustments[0]
        assert adjustment['timestamp'] == 5.0
        assert adjustment['adjustment'] == -0.1
        assert adjustment['type'] == 'cut'
    
    def test_apply_synchronization_adjustments(self, audio_synchronizer):
        """Test application of synchronization adjustments."""
        # Create test tracks
        tracks = [
            AudioTrackInfo(
                track_id="test_track",
                source_path="/path/to/audio.wav",
                start_time=0.0,
                duration=30.0,
                volume=1.0
            )
        ]
        
        # Create test sync points
        sync_points = [
            SyncPoint(timestamp=5.0, sync_type="cut", audio_adjustment=-0.5),
            SyncPoint(timestamp=15.0, sync_type="filler_removal", audio_adjustment=-0.3)
        ]
        
        adjusted_tracks = audio_synchronizer._apply_synchronization_adjustments(tracks, sync_points)
        
        assert len(adjusted_tracks) == 1
        
        adjusted_track = adjusted_tracks[0]
        assert adjusted_track.track_id == "test_track"
        assert adjusted_track.duration < 30.0  # Should be reduced due to adjustments
        assert adjusted_track.duration == 30.0 - 0.8  # Total adjustment: -0.5 + -0.3
    
    def test_create_movis_audio_layers(self, audio_synchronizer, mock_context_full):
        """Test creation of movis audio layer configurations."""
        # Create test tracks
        tracks = [
            AudioTrackInfo(
                track_id="main_audio",
                source_path="/path/to/enhanced.wav",
                start_time=0.0,
                duration=29.5,
                volume=0.8,
                fade_in=0.1,
                fade_out=0.2,
                sync_adjustments=[
                    {'timestamp': 5.0, 'adjustment': -0.1, 'type': 'cut'}
                ]
            )
        ]
        
        movis_layers = audio_synchronizer._create_movis_audio_layers(tracks, mock_context_full)
        
        assert len(movis_layers) == 1
        
        layer = movis_layers[0]
        assert layer['layer_id'] == 'main_audio'
        assert layer['layer_type'] == 'audio'
        assert layer['source_path'] == '/path/to/enhanced.wav'
        assert layer['duration'] == 29.5
        assert layer['volume'] == 0.8
        assert layer['fade_in'] == 0.1
        assert layer['fade_out'] == 0.2
        assert len(layer['sync_adjustments']) == 1
        
        # Check movis parameters
        movis_params = layer['movis_params']
        assert movis_params['sample_rate'] == 48000
        assert movis_params['channels'] == 2
        assert movis_params['format'] == 'float32'
    
    def test_calculate_synchronization_metrics(self, audio_synchronizer):
        """Test calculation of synchronization metrics."""
        sync_points = [
            SyncPoint(timestamp=5.0, sync_type="cut", 
                     audio_adjustment=-0.01, video_adjustment=0.005),  # Frame accurate
            SyncPoint(timestamp=10.0, sync_type="transition", 
                     audio_adjustment=-0.1, video_adjustment=0.05),   # Not frame accurate
            SyncPoint(timestamp=15.0, sync_type="filler_removal", 
                     audio_adjustment=-0.005, video_adjustment=0.0)   # Frame accurate
        ]
        
        metrics = audio_synchronizer._calculate_synchronization_metrics(sync_points)
        
        assert metrics['max_error'] == 0.15  # Max total adjustment
        assert metrics['avg_error'] > 0.0
        assert metrics['frame_accurate'] == 2  # Two points within frame tolerance
    
    def test_get_composition_settings(self, audio_synchronizer, mock_context_full):
        """Test getting composition settings."""
        settings = audio_synchronizer._get_composition_settings(mock_context_full)
        
        assert settings['fps'] == 30.0
        assert settings['sample_rate'] == 48000
        assert settings['audio_channels'] == 2
        assert settings['audio_format'] == 'float32'
        assert settings['sync_tolerance'] == audio_synchronizer.frame_duration / 2
        assert 'quality_mode' in settings
    
    @patch('movis.Composition')
    @patch('movis.layer.AudioFile')
    @patch('movis.layer.Volume')
    @patch('movis.layer.FadeIn')
    @patch('movis.layer.FadeOut')
    def test_create_synchronized_movis_composition(self, mock_fade_out, mock_fade_in, 
                                                 mock_volume, mock_audio_file, mock_composition,
                                                 audio_synchronizer, mock_context_full):
        """Test creation of synchronized movis composition."""
        # Setup mocks
        mock_composition_instance = Mock()
        mock_composition.return_value = mock_composition_instance
        
        mock_audio_layer = Mock()
        mock_audio_file.return_value = mock_audio_layer
        mock_volume.return_value = lambda x: x
        mock_fade_in.return_value = lambda x: x
        mock_fade_out.return_value = lambda x: x
        
        # Create mock sync result
        audio_synchronizer.sync_result = SynchronizationResult(
            processing_time=2.0,
            sync_points_processed=2,
            adjustments_applied=1,
            max_sync_error=0.1,
            average_sync_error=0.05,
            frame_accurate_points=1,
            audio_tracks=[
                AudioTrackInfo(
                    track_id="main_audio",
                    source_path="/path/to/enhanced.wav",
                    start_time=0.0,
                    duration=30.0,
                    volume=0.8,
                    fade_in=0.1,
                    fade_out=0.2
                )
            ],
            movis_audio_layers=[
                {
                    'layer_id': 'main_audio',
                    'layer_type': 'audio',
                    'source_path': '/path/to/enhanced.wav',
                    'start_time': 0.0,
                    'duration': 30.0,
                    'volume': 0.8,
                    'fade_in': 0.1,
                    'fade_out': 0.2,
                    'sync_adjustments': [],
                    'movis_params': {'sample_rate': 48000, 'channels': 2, 'format': 'float32'}
                }
            ],
            composition_settings={'fps': 30.0, 'sample_rate': 48000}
        )
        
        composition = audio_synchronizer.create_synchronized_movis_composition(mock_context_full)
        
        assert composition is not None
        
        # Verify composition creation
        mock_composition.assert_called_once()
        mock_composition_instance.add_layer.assert_called()
        
        # Verify audio layer creation
        mock_audio_file.assert_called_once_with(
            path='/path/to/enhanced.wav',
            start_time=0.0,
            duration=30.0
        )
    
    def test_synchronize_audio_video_success(self, audio_synchronizer, mock_context_full, 
                                           mock_enhancement_result_full):
        """Test successful audio-video synchronization."""
        # Create mock video file
        video_path = Path(mock_context_full.video_files[0])
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()
        
        result = audio_synchronizer.synchronize_audio_video(
            mock_context_full, mock_enhancement_result_full
        )
        
        assert isinstance(result, SynchronizationResult)
        assert result.processing_time > 0
        assert result.sync_points_processed >= 1
        assert len(result.audio_tracks) >= 1
        assert len(result.movis_audio_layers) >= 1
        
        # Check audio track
        main_track = result.audio_tracks[0]
        assert main_track.track_id == "main_audio"
        assert main_track.source_path == mock_enhancement_result_full.enhanced_audio_path
        assert main_track.duration == mock_enhancement_result_full.enhanced_duration
        
        # Check movis layer
        main_layer = result.movis_audio_layers[0]
        assert main_layer['layer_id'] == 'main_audio'
        assert main_layer['layer_type'] == 'audio'
        
        # Check context updates
        assert mock_context_full.processed_video is not None
        assert 'audio_synchronization' in mock_context_full.processed_video
        assert mock_context_full.processing_metrics.module_processing_times['audio_synchronization'] > 0
    
    def test_get_synchronization_report(self, audio_synchronizer):
        """Test getting synchronization report."""
        # Test with no sync result
        report = audio_synchronizer.get_synchronization_report()
        assert report == {}
        
        # Create mock sync result
        audio_synchronizer.sync_result = SynchronizationResult(
            processing_time=2.5,
            sync_points_processed=5,
            adjustments_applied=3,
            max_sync_error=0.08,
            average_sync_error=0.03,
            frame_accurate_points=4,
            audio_tracks=[],
            composition_settings={'fps': 30.0}
        )
        
        report = audio_synchronizer.get_synchronization_report()
        
        assert 'summary' in report
        assert 'quality_metrics' in report
        assert 'audio_tracks' in report
        assert 'composition_settings' in report
        
        # Check summary
        summary = report['summary']
        assert summary['processing_time'] == 2.5
        assert summary['sync_points_processed'] == 5
        assert summary['adjustments_applied'] == 3
        assert summary['frame_accurate_percentage'] == 80.0  # 4/5 * 100
        
        # Check quality metrics
        quality = report['quality_metrics']
        assert quality['max_sync_error'] == 0.08
        assert quality['average_sync_error'] == 0.03
        assert quality['frame_accurate_points'] == 4


class TestAudioSynchronizerIntegration:
    """Test integration between synchronizer components."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @patch('ai_video_editor.modules.enhancement.audio_synchronizer.MOVIS_AVAILABLE', True)
    def test_full_synchronization_pipeline(self, temp_dir):
        """Test complete synchronization pipeline integration."""
        # Create synchronizer
        synchronizer = AudioSynchronizer(fps=30.0, sample_rate=48000)
        
        # Create comprehensive context
        context = ContentContext(
            project_id="integration_sync_test",
            video_files=[str(Path(temp_dir) / "test_video.mp4")],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add comprehensive audio analysis
        segments = [
            AudioSegment(
                text="Welcome to the tutorial",
                start=0.0,
                end=3.0,
                confidence=0.9
            ),
            AudioSegment(
                text="Let me explain this concept",
                start=3.0,
                end=7.0,
                confidence=0.8
            )
        ]
        
        audio_analysis = AudioAnalysisResult(
            transcript_text="Welcome to the tutorial. Let me explain this concept.",
            segments=segments,
            overall_confidence=0.85,
            language="en",
            processing_time=2.0,
            model_used="medium"
        )
        
        # Add filler words
        class MockFillerSegment:
            def __init__(self, timestamp, filler_words, should_remove=True):
                self.timestamp = timestamp
                self.filler_words = filler_words
                self.should_remove = should_remove
                self.original_text = "um, let me explain"
                self.cleaned_text = "let me explain"
        
        audio_analysis.filler_words_detected = [
            MockFillerSegment(4.0, ["um"], True)
        ]
        
        context.set_audio_analysis(audio_analysis)
        
        # Add editing decisions
        context.processed_video = {
            'editing_decisions': [
                {
                    'decision_id': 'cut_001',
                    'decision_type': 'cut',
                    'timestamp': 5.0,
                    'parameters': {'duration': 1.0},
                    'rationale': 'Remove pause'
                }
            ]
        }
        
        # Add emotional markers
        context.emotional_markers = [
            EmotionalPeak(
                timestamp=2.0,
                emotion="excitement",
                intensity=0.9,
                confidence=0.8,
                context="Tutorial start"
            )
        ]
        
        # Create comprehensive enhancement result
        enhanced_path = Path(temp_dir) / "enhanced_audio.wav"
        enhanced_path.touch()
        
        enhancement_result = AudioEnhancementResult(
            processing_time=3.0,
            original_duration=10.0,
            enhanced_duration=9.5,
            noise_reduction_applied=True,
            dynamic_adjustments_made=3,
            peak_reductions=1,
            level_boosts=2,
            snr_improvement=4.5,
            dynamic_range_improvement=3.2,
            loudness_consistency_score=0.9,
            enhanced_audio_path=str(enhanced_path),
            sync_points=[
                {
                    'timestamp': 2.0,
                    'type': 'emotional_peak',
                    'priority': 'high',
                    'metadata': {'emotion': 'excitement', 'intensity': 0.9}
                },
                {
                    'timestamp': 6.0,
                    'type': 'level_adjustment',
                    'priority': 'medium',
                    'metadata': {'adjustment_type': 'boost', 'factor': 1.2}
                }
            ],
            level_adjustments=[
                {
                    'timestamp': 2.0,
                    'type': 'emotional_boost',
                    'factor': 1.3,
                    'reason': 'Excitement boost'
                },
                {
                    'timestamp': 6.0,
                    'type': 'explanation_boost',
                    'factor': 1.1,
                    'reason': 'Explanation clarity'
                }
            ]
        )
        
        # Create mock video file
        video_path = Path(context.video_files[0])
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()
        
        # Run synchronization
        result = synchronizer.synchronize_audio_video(context, enhancement_result)
        
        # Verify comprehensive results
        assert isinstance(result, SynchronizationResult)
        assert result.processing_time > 0
        assert result.sync_points_processed >= 4  # Enhancement + editing + emotional + filler
        assert result.adjustments_applied >= 1
        assert result.max_sync_error >= 0.0
        assert result.average_sync_error >= 0.0
        assert result.frame_accurate_points >= 0
        
        # Verify audio tracks
        assert len(result.audio_tracks) >= 1
        main_track = result.audio_tracks[0]
        assert main_track.track_id == "main_audio"
        assert main_track.source_path == str(enhanced_path)
        assert main_track.duration == enhancement_result.enhanced_duration
        assert len(main_track.sync_adjustments) >= 1
        
        # Verify movis layers
        assert len(result.movis_audio_layers) >= 1
        main_layer = result.movis_audio_layers[0]
        assert main_layer['layer_id'] == 'main_audio'
        assert main_layer['layer_type'] == 'audio'
        assert main_layer['source_path'] == str(enhanced_path)
        assert main_layer['movis_params']['sample_rate'] == 48000
        
        # Verify composition settings
        settings = result.composition_settings
        assert settings['fps'] == 30.0
        assert settings['sample_rate'] == 48000
        assert settings['audio_channels'] == 2
        
        # Verify context updates
        assert context.processed_video is not None
        assert 'audio_synchronization' in context.processed_video
        assert context.processing_metrics.module_processing_times['audio_synchronization'] > 0
        
        # Test synchronization report
        report = synchronizer.get_synchronization_report()
        assert 'summary' in report
        assert 'quality_metrics' in report
        assert report['summary']['sync_points_processed'] >= 4
        assert report['summary']['frame_accurate_percentage'] >= 0.0