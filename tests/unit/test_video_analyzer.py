"""
Unit tests for VideoAnalyzer module.

Tests video analysis functionality with comprehensive mocking to avoid
actual video processing during tests.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import cv2

from ai_video_editor.modules.content_analysis.video_analyzer import (
    VideoAnalyzer, VideoMetadata, SceneInfo, FrameAnalysis, VisualElementDetection, create_video_analyzer
)
from ai_video_editor.core.content_context import (
    ContentContext, ContentType, UserPreferences, FaceDetection
)
from ai_video_editor.utils.cache_manager import CacheManager


@pytest.fixture
def mock_video_metadata():
    """Mock video metadata for testing."""
    return VideoMetadata(
        duration=120.0,
        fps=30.0,
        width=1920,
        height=1080,
        codec="h264",
        bitrate=5000000,
        format="mp4",
        total_frames=3600
    )


@pytest.fixture
def mock_scene_info():
    """Mock scene information for testing."""
    return [
        SceneInfo(start_time=0.0, end_time=30.0, duration=30.0, scene_id=0, confidence=0.8),
        SceneInfo(start_time=30.0, end_time=60.0, duration=30.0, scene_id=1, confidence=0.9),
        SceneInfo(start_time=60.0, end_time=120.0, duration=60.0, scene_id=2, confidence=0.7)
    ]


@pytest.fixture
def mock_face_detection():
    """Mock face detection result."""
    return FaceDetection(
        bbox=[100.0, 150.0, 200.0, 250.0],
        confidence=0.85,
        expression="happy",
        landmarks=[[120.0, 180.0], [180.0, 180.0], [150.0, 220.0]]
    )


@pytest.fixture
def mock_visual_element():
    """Mock visual element detection."""
    return VisualElementDetection(
        element_type="text_overlay",
        confidence=0.85,
        bbox=[100.0, 200.0, 300.0, 50.0],
        properties={'text_regions_count': 3}
    )


@pytest.fixture
def mock_frame_analysis(mock_face_detection, mock_visual_element):
    """Mock enhanced frame analysis result."""
    return FrameAnalysis(
        timestamp=15.5,
        frame_number=465,
        faces=[mock_face_detection],
        visual_elements=[mock_visual_element],
        motion_score=0.6,
        quality_score=0.8,
        thumbnail_potential=0.75,
        scene_context="scene_1",
        emotional_context="excitement_peak"
    )


@pytest.fixture
def sample_context():
    """Create sample ContentContext for testing."""
    return ContentContext(
        project_id="test_project",
        video_files=["test_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences()
    )


@pytest.fixture
def video_analyzer():
    """Create VideoAnalyzer instance for testing."""
    with patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2'):
        analyzer = VideoAnalyzer()
        return analyzer


@pytest.fixture
def video_analyzer_with_memory():
    """Create VideoAnalyzer instance with mock Memory client for testing."""
    mock_memory = Mock()
    mock_memory.search_nodes.return_value = {'nodes': []}
    mock_memory.create_entities.return_value = None
    
    with patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2'):
        analyzer = VideoAnalyzer(memory_client=mock_memory)
        return analyzer


class TestVideoAnalyzer:
    """Test cases for VideoAnalyzer class."""
    
    def test_initialization(self):
        """Test VideoAnalyzer initialization."""
        with patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2'):
            analyzer = VideoAnalyzer()
            assert analyzer.scene_threshold == 30.0
            assert analyzer.face_confidence_threshold == 0.7
            assert analyzer.thumbnail_sample_rate == 1.0
            assert analyzer.max_faces_per_frame == 10
    
    def test_initialization_with_cache_manager(self):
        """Test VideoAnalyzer initialization with cache manager."""
        cache_manager = Mock(spec=CacheManager)
        with patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2'):
            analyzer = VideoAnalyzer(cache_manager)
            assert analyzer.cache_manager == cache_manager
    
    @patch('ai_video_editor.modules.content_analysis.video_analyzer.ffmpeg')
    def test_extract_video_metadata(self, mock_ffmpeg, video_analyzer):
        """Test video metadata extraction."""
        # Mock ffmpeg probe response
        mock_probe_data = {
            'format': {
                'duration': '120.5',
                'bit_rate': '5000000',
                'format_name': 'mp4,m4a,3gp,3g2,mj2'
            },
            'streams': [{
                'codec_type': 'video',
                'codec_name': 'h264',
                'width': 1920,
                'height': 1080,
                'r_frame_rate': '30/1',
                'nb_frames': '3615'
            }]
        }
        mock_ffmpeg.probe.return_value = mock_probe_data
        
        metadata = video_analyzer._extract_video_metadata("test_video.mp4")
        
        assert metadata.duration == 120.5
        assert metadata.fps == 30.0
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.codec == "h264"
        assert metadata.bitrate == 5000000
        assert metadata.total_frames == 3615
    
    @patch('ai_video_editor.modules.content_analysis.video_analyzer.ffmpeg')
    def test_extract_video_metadata_error_handling(self, mock_ffmpeg, video_analyzer):
        """Test video metadata extraction error handling."""
        mock_ffmpeg.probe.side_effect = Exception("FFmpeg error")
        
        metadata = video_analyzer._extract_video_metadata("invalid_video.mp4")
        
        # Should return default metadata
        assert metadata.duration == 0.0
        assert metadata.fps == 30.0
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.codec == "unknown"
    
    @patch('ai_video_editor.modules.content_analysis.video_analyzer.VideoManager')
    @patch('ai_video_editor.modules.content_analysis.video_analyzer.SceneManager')
    def test_detect_scenes(self, mock_scene_manager_class, mock_video_manager_class, video_analyzer):
        """Test scene detection functionality."""
        # Mock scene detection
        mock_video_manager = Mock()
        mock_scene_manager = Mock()
        mock_video_manager_class.return_value = mock_video_manager
        mock_scene_manager_class.return_value = mock_scene_manager
        
        # Mock scene list with timecode objects that support subtraction
        mock_start_time = Mock()
        mock_start_time.get_seconds.return_value = 0.0
        mock_end_time = Mock()
        mock_end_time.get_seconds.return_value = 30.0
        
        # Mock the subtraction operation for duration calculation
        mock_duration = Mock()
        mock_duration.get_seconds.return_value = 30.0
        mock_end_time.__sub__ = Mock(return_value=mock_duration)
        
        mock_scene_manager.get_scene_list.return_value = [(mock_start_time, mock_end_time)]
        
        scenes = video_analyzer._detect_scenes("test_video.mp4")
        
        assert len(scenes) == 1
        assert scenes[0].start_time == 0.0
        assert scenes[0].end_time == 30.0
        assert scenes[0].duration == 30.0
        assert scenes[0].scene_id == 0
        assert scenes[0].confidence == 0.8
    
    @patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.VideoCapture')
    def test_analyze_frames(self, mock_video_capture, video_analyzer, mock_video_metadata):
        """Test frame analysis functionality."""
        # Mock video capture
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        
        # Mock frame reading
        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, mock_frame),  # First frame
            (True, mock_frame),  # Second frame
            (False, None)        # End of video
        ]
        
        # Mock frame analysis
        with patch.object(video_analyzer, '_analyze_single_frame') as mock_analyze:
            mock_analyze.return_value = FrameAnalysis(
                timestamp=0.0, frame_number=0, faces=[], visual_elements=[],
                motion_score=0.5, quality_score=0.6, thumbnail_potential=0.4
            )
            
            analyses = video_analyzer._analyze_frames("test_video.mp4", mock_video_metadata)
            
            assert len(analyses) == 1  # Only first frame due to sampling
            mock_cap.release.assert_called_once()
    
    def test_analyze_single_frame(self, video_analyzer, mock_visual_element):
        """Test enhanced single frame analysis."""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch.object(video_analyzer, '_detect_faces_in_frame') as mock_faces, \
             patch.object(video_analyzer, '_analyze_visual_elements') as mock_elements, \
             patch.object(video_analyzer, '_calculate_motion_score') as mock_motion, \
             patch.object(video_analyzer, '_calculate_quality_score') as mock_quality, \
             patch.object(video_analyzer, '_calculate_thumbnail_potential') as mock_thumbnail, \
             patch.object(video_analyzer, '_determine_scene_context') as mock_scene, \
             patch.object(video_analyzer, '_determine_emotional_context') as mock_emotion:
            
            mock_faces.return_value = []
            mock_elements.return_value = [mock_visual_element]
            mock_motion.return_value = 0.6
            mock_quality.return_value = 0.8
            mock_thumbnail.return_value = 0.7
            mock_scene.return_value = "scene_1"
            mock_emotion.return_value = "excitement_peak"
            
            analysis = video_analyzer._analyze_single_frame(mock_frame, 10.5, 315)
            
            assert analysis is not None
            assert analysis.timestamp == 10.5
            assert analysis.frame_number == 315
            assert len(analysis.visual_elements) == 1
            assert analysis.visual_elements[0].element_type == "text_overlay"
            assert analysis.motion_score == 0.6
            assert analysis.quality_score == 0.8
            assert analysis.thumbnail_potential == 0.7
            assert analysis.scene_context == "scene_1"
            assert analysis.emotional_context == "excitement_peak"
    
    def test_detect_faces_haar_cascade(self, video_analyzer):
        """Test face detection with Haar cascade."""
        # Mock Haar cascade detector
        mock_detector = Mock()
        mock_detector.detectMultiScale.return_value = [(100, 150, 200, 250)]
        video_analyzer.face_detector = mock_detector
        
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch.object(video_analyzer, '_analyze_expression') as mock_expression:
            mock_expression.return_value = "happy"
            
            faces = video_analyzer._detect_faces_in_frame(mock_frame)
            
            assert len(faces) == 1
            assert faces[0].bbox == [100.0, 150.0, 200.0, 250.0]
            assert faces[0].confidence == 0.8
            assert faces[0].expression == "happy"
    
    def test_analyze_expression(self, video_analyzer):
        """Test facial expression analysis."""
        # Test with bright face (should return "happy")
        bright_face = np.ones((100, 100, 3), dtype=np.uint8) * 150
        expression = video_analyzer._analyze_expression(bright_face)
        assert expression == "happy"
        
        # Test with dark face (should return "focused")
        dark_face = np.ones((100, 100, 3), dtype=np.uint8) * 50
        expression = video_analyzer._analyze_expression(dark_face)
        assert expression == "focused"
        
        # Test with medium face (should return "neutral")
        medium_face = np.ones((100, 100, 3), dtype=np.uint8) * 100
        expression = video_analyzer._analyze_expression(medium_face)
        assert expression == "neutral"
    
    @patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.Canny')
    @patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.findContours')
    def test_has_text_regions(self, mock_find_contours, mock_canny, video_analyzer):
        """Test text region detection."""
        mock_frame = np.zeros((480, 640), dtype=np.uint8)
        mock_canny.return_value = mock_frame
        
        # Mock contours that look like text
        mock_contours = [
            np.array([[50, 100], [150, 100], [150, 120], [50, 120]]),  # Text-like rectangle
            np.array([[200, 150], [300, 150], [300, 170], [200, 170]]),  # Another text-like rectangle
        ] * 3  # Repeat to get 6 contours (above threshold)
        
        mock_find_contours.return_value = (mock_contours, None)
        
        has_text = video_analyzer._has_text_regions(mock_frame)
        assert has_text is True
    
    @patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.HoughLinesP')
    def test_has_chart_elements(self, mock_hough_lines, video_analyzer):
        """Test chart element detection."""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock many lines (indicating chart presence)
        mock_lines = [[[0, 0, 100, 100]]] * 15  # 15 lines
        mock_hough_lines.return_value = mock_lines
        
        has_chart = video_analyzer._has_chart_elements(mock_frame)
        assert has_chart is True
    
    def test_calculate_quality_score(self, video_analyzer):
        """Test frame quality score calculation."""
        # Create a frame with some variation
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        quality_score = video_analyzer._calculate_quality_score(mock_frame)
        
        assert 0.0 <= quality_score <= 1.0
        assert isinstance(quality_score, float)
    
    def test_calculate_thumbnail_potential(self, video_analyzer, mock_face_detection):
        """Test thumbnail potential calculation."""
        faces = [mock_face_detection]
        visual_elements = ["text_overlay", "data_visualization"]
        quality_score = 0.8
        
        potential = video_analyzer._calculate_thumbnail_potential(
            faces, visual_elements, quality_score
        )
        
        assert 0.0 <= potential <= 1.0
        assert potential > 0.5  # Should be high due to face + elements + quality
    
    def test_create_frame_description(self, video_analyzer, mock_frame_analysis):
        """Test frame description creation."""
        description = video_analyzer._create_frame_description(
            mock_frame_analysis, "scene_1"
        )
        
        assert "scene_1" in description
        assert "face" in description.lower()
        assert "happy" in description
        assert "text_overlay" in description
    
    @patch.object(VideoAnalyzer, '_extract_video_metadata')
    @patch.object(VideoAnalyzer, '_detect_scenes')
    @patch.object(VideoAnalyzer, '_analyze_frames')
    @patch.object(VideoAnalyzer, '_create_visual_highlights')
    @patch('time.time')
    def test_analyze_video_integration(self, mock_time, mock_create_highlights, mock_analyze_frames,
                                     mock_detect_scenes, mock_extract_metadata,
                                     video_analyzer, sample_context, mock_video_metadata,
                                     mock_scene_info, mock_frame_analysis):
        """Test complete video analysis integration."""
        # Mock time to simulate processing duration
        mock_time.side_effect = [0.0, 1.5]  # Start time, end time
        
        # Setup mocks
        mock_extract_metadata.return_value = mock_video_metadata
        mock_detect_scenes.return_value = mock_scene_info
        mock_analyze_frames.return_value = [mock_frame_analysis]
        
        mock_highlight = Mock()
        mock_highlight.timestamp = 15.5
        mock_highlight.description = "Test highlight"
        mock_highlight.faces = []
        mock_highlight.visual_elements = ["text_overlay"]
        mock_highlight.thumbnail_potential = 0.75
        mock_create_highlights.return_value = [mock_highlight]
        
        # Run analysis
        result_context = video_analyzer.analyze_video("test_video.mp4", sample_context)
        
        # Verify results
        assert result_context.video_metadata is not None
        assert len(result_context.visual_highlights) == 1
        assert result_context.processing_metrics.total_processing_time == 1.5
    
    def test_analyze_batch(self, video_analyzer, sample_context):
        """Test batch video analysis."""
        video_paths = ["video1.mp4", "video2.mp4"]
        
        with patch.object(video_analyzer, 'analyze_video') as mock_analyze:
            mock_analyze.return_value = sample_context
            
            result_context = video_analyzer.analyze_batch(video_paths, sample_context)
            
            assert mock_analyze.call_count == 2
            assert result_context == sample_context
    
    def test_create_video_analyzer_factory(self):
        """Test factory function for creating VideoAnalyzer."""
        cache_manager = Mock(spec=CacheManager)
        
        with patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2'):
            analyzer = create_video_analyzer(cache_manager)
            
            assert isinstance(analyzer, VideoAnalyzer)
            assert analyzer.cache_manager == cache_manager
    
    def test_create_video_analyzer_with_memory(self):
        """Test factory function with Memory client."""
        cache_manager = Mock(spec=CacheManager)
        memory_client = Mock()
        memory_client.search_nodes.return_value = {'nodes': []}
        
        with patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2'):
            analyzer = create_video_analyzer(cache_manager, memory_client)
            
            assert isinstance(analyzer, VideoAnalyzer)
            assert analyzer.cache_manager == cache_manager
            assert analyzer.memory_client == memory_client


class TestVideoMetadata:
    """Test cases for VideoMetadata dataclass."""
    
    def test_video_metadata_creation(self):
        """Test VideoMetadata creation and serialization."""
        metadata = VideoMetadata(
            duration=120.0,
            fps=30.0,
            width=1920,
            height=1080,
            codec="h264",
            bitrate=5000000,
            format="mp4",
            total_frames=3600
        )
        
        assert metadata.duration == 120.0
        assert metadata.fps == 30.0
        assert metadata.codec == "h264"
        
        # Test serialization
        metadata_dict = metadata.to_dict()
        assert metadata_dict['duration'] == 120.0
        assert metadata_dict['codec'] == "h264"


class TestSceneInfo:
    """Test cases for SceneInfo dataclass."""
    
    def test_scene_info_creation(self):
        """Test SceneInfo creation and serialization."""
        scene = SceneInfo(
            start_time=0.0,
            end_time=30.0,
            duration=30.0,
            scene_id=0,
            confidence=0.8
        )
        
        assert scene.start_time == 0.0
        assert scene.end_time == 30.0
        assert scene.duration == 30.0
        assert scene.scene_id == 0
        assert scene.confidence == 0.8
        
        # Test serialization
        scene_dict = scene.to_dict()
        assert scene_dict['start_time'] == 0.0
        assert scene_dict['scene_id'] == 0


class TestFrameAnalysis:
    """Test cases for FrameAnalysis dataclass."""
    
    def test_frame_analysis_creation(self, mock_face_detection):
        """Test FrameAnalysis creation and serialization."""
        analysis = FrameAnalysis(
            timestamp=15.5,
            frame_number=465,
            faces=[mock_face_detection],
            visual_elements=["text_overlay", "red_dominant"],
            motion_score=0.6,
            quality_score=0.8,
            thumbnail_potential=0.75
        )
        
        assert analysis.timestamp == 15.5
        assert analysis.frame_number == 465
        assert len(analysis.faces) == 1
        assert "text_overlay" in analysis.visual_elements
        assert analysis.motion_score == 0.6
        
        # Test serialization
        analysis_dict = analysis.to_dict()
        assert analysis_dict['timestamp'] == 15.5
        assert len(analysis_dict['faces']) == 1


class TestErrorHandling:
    """Test error handling in VideoAnalyzer."""
    
    def test_analyze_video_with_invalid_path(self, video_analyzer, sample_context):
        """Test video analysis with invalid file path."""
        with patch.object(video_analyzer, '_extract_video_metadata') as mock_metadata:
            mock_metadata.side_effect = Exception("File not found")
            
            with pytest.raises(Exception):
                video_analyzer.analyze_video("invalid_path.mp4", sample_context)
    
    def test_face_detection_error_handling(self, video_analyzer):
        """Test face detection error handling."""
        video_analyzer.face_detector = None
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        faces = video_analyzer._detect_faces_in_frame(mock_frame)
        assert faces == []
    
    def test_visual_elements_error_handling(self, video_analyzer):
        """Test visual elements analysis error handling."""
        # Test with invalid frame
        invalid_frame = None
        
        with patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.cvtColor') as mock_cvt:
            mock_cvt.side_effect = Exception("Invalid frame")
            
            elements = video_analyzer._analyze_visual_elements(np.zeros((100, 100, 3)))
            assert elements == []


if __name__ == "__main__":
    pytest.main([__file__])


class TestEnhancedVisualDetection:
    """Test cases for enhanced visual element detection."""
    
    def test_detect_text_regions_with_confidence(self, video_analyzer):
        """Test enhanced text region detection with confidence scoring."""
        # Create mock frame with text-like contours
        mock_frame = np.zeros((480, 640), dtype=np.uint8)
        
        with patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.Canny') as mock_canny, \
             patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.findContours') as mock_contours:
            
            mock_canny.return_value = mock_frame
            
            # Mock text-like contours with proper format for cv2.boundingRect
            mock_text_contours = []
            # Create contours that will produce text-like bounding boxes
            for i in range(5):  # Create 5 contours to exceed threshold
                x, y = 50 + i * 100, 100 + i * 30
                w, h = 100, 25  # Text-like dimensions
                contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)
                mock_text_contours.append(contour)
            
            mock_contours.return_value = (mock_text_contours, None)
            
            detection = video_analyzer._detect_text_regions(mock_frame)
            
            assert detection is not None
            assert detection.element_type == "text_overlay"
            assert detection.confidence > 0.3
            assert detection.bbox is not None
            assert 'text_regions_count' in detection.properties
            assert detection.properties['text_regions_count'] == 5
    
    def test_detect_chart_elements_with_confidence(self, video_analyzer):
        """Test enhanced chart element detection with confidence scoring."""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.cvtColor') as mock_cvt, \
             patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.Canny') as mock_canny, \
             patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.HoughLinesP') as mock_lines, \
             patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.HoughCircles') as mock_circles, \
             patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.findContours') as mock_contours:
            
            mock_cvt.return_value = np.zeros((480, 640), dtype=np.uint8)
            mock_canny.return_value = np.zeros((480, 640), dtype=np.uint8)
            
            # Mock many lines (indicating chart)
            mock_lines.return_value = [[[0, 0, 100, 100]]] * 12  # 12 lines
            mock_circles.return_value = None
            mock_contours.return_value = ([], None)
            
            detection = video_analyzer._detect_chart_elements(mock_frame)
            
            assert detection is not None
            assert detection.element_type == "data_visualization"
            assert detection.confidence > 0.3
            assert 'line_count' in detection.properties
            assert detection.properties['line_count'] == 12
    
    def test_detect_motion_elements_with_confidence(self, video_analyzer):
        """Test enhanced motion element detection with confidence scoring."""
        # Create frame with high edge density
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch.object(video_analyzer, '_calculate_texture_complexity') as mock_texture:
            mock_texture.return_value = 0.7  # High texture complexity
            
            detection = video_analyzer._detect_motion_elements(mock_frame)
            
            # Should detect motion due to high texture complexity
            assert detection is not None
            assert detection.element_type == "gesture_content"
            assert detection.confidence > 0.4
            assert 'texture_complexity' in detection.properties
    
    def test_detect_dominant_colors_with_confidence(self, video_analyzer):
        """Test enhanced dominant color detection with confidence scoring."""
        # Create frame with dominant red color
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_frame[:, :, 0] = 10  # Red hue
        mock_frame[:, :, 1] = 200  # High saturation
        mock_frame[:, :, 2] = 200  # High value
        
        detections = video_analyzer._detect_dominant_colors(mock_frame)
        
        assert len(detections) > 0
        
        # Should detect red dominance and high saturation
        element_types = [d.element_type for d in detections]
        assert any('red_dominant' in et for et in element_types)
        assert any('high_saturation' in et for et in element_types)
    
    def test_detect_geometric_patterns(self, video_analyzer):
        """Test geometric pattern detection."""
        mock_frame = np.zeros((480, 640), dtype=np.uint8)
        
        with patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.cornerHarris') as mock_corners, \
             patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.Canny') as mock_canny, \
             patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.findContours') as mock_contours:
            
            # Mock high corner count
            mock_corner_response = np.zeros((480, 640))
            mock_corner_response[100:200, 100:200] = 1.0  # High corner response in region
            mock_corners.return_value = mock_corner_response
            
            mock_canny.return_value = mock_frame
            
            # Mock regular shape contours
            square_contour = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
            triangle_contour = np.array([[300, 100], [400, 100], [350, 200]])
            mock_contours.return_value = ([square_contour, triangle_contour] * 3, None)  # 6 shapes
            
            detection = video_analyzer._detect_geometric_patterns(mock_frame)
            
            assert detection is not None
            assert detection.element_type == "geometric_patterns"
            assert detection.confidence > 0.3
            assert 'corner_count' in detection.properties
            assert 'regular_shapes' in detection.properties


class TestVideoQualityAssessment:
    """Test cases for video quality assessment functionality."""
    
    @pytest.fixture
    def mock_quality_metrics(self):
        """Mock video quality metrics for testing."""
        from ai_video_editor.core.content_context import VideoQualityMetrics
        return VideoQualityMetrics(
            resolution_score=0.9,
            actual_resolution=(1920, 1080),
            resolution_category="high",
            lighting_score=0.8,
            brightness_mean=128.0,
            brightness_std=32.0,
            exposure_quality="optimal",
            stability_score=0.7,
            motion_blur_level=0.2,
            camera_shake_detected=False,
            stability_category="good",
            color_balance_score=0.75,
            saturation_level=150.0,
            contrast_score=0.8,
            color_temperature="neutral",
            overall_quality_score=0.8,
            quality_category="good",
            enhancement_recommendations=["Increase color saturation"],
            color_correction_needed=False,
            lighting_adjustment_needed=False,
            stabilization_needed=False,
            assessment_time=2.5,
            frames_analyzed=50
        )
    
    @patch.object(VideoAnalyzer, '_perform_quality_assessment_with_profiling')
    @patch('ai_video_editor.modules.content_analysis.video_analyzer.create_quality_assessment_profiler')
    @patch('ai_video_editor.modules.content_analysis.video_analyzer.create_benchmark_manager')
    def test_assess_video_quality(self, mock_benchmark_manager, mock_profiler_factory, 
                                mock_quality_assessment, video_analyzer, sample_context, 
                                mock_video_metadata, mock_quality_metrics):
        """Test comprehensive video quality assessment."""
        # Setup mocks
        mock_profiler = Mock()
        mock_profiler_factory.return_value = mock_profiler
        mock_benchmark = Mock()
        mock_benchmark.processing_time = 2.5
        mock_benchmark.memory_peak_usage = 1000000000  # 1GB
        mock_benchmark.frames_per_second = 20.0
        mock_benchmark.cpu_usage_percent = 60.0
        mock_profiler.end_profiling.return_value = mock_benchmark
        mock_profiler.check_performance_targets.return_value = {
            'processing_time_target': True,
            'memory_usage_target': True,
            'frames_per_second_target': True,
            'quality_accuracy_target': True,
            'cpu_usage_target': True
        }
        
        mock_benchmark_manager_instance = Mock()
        mock_benchmark_manager.return_value = mock_benchmark_manager_instance
        
        mock_quality_assessment.return_value = mock_quality_metrics
        
        # Set up context with video metadata
        sample_context.video_metadata = mock_video_metadata.to_dict()
        
        # Run quality assessment
        result_context = video_analyzer.assess_video_quality("test_video.mp4", sample_context)
        
        # Verify results
        assert result_context.video_quality_metrics is not None
        assert result_context.video_quality_metrics.overall_quality_score == 0.8
        assert result_context.video_quality_metrics.quality_category == "good"
        
        # Verify profiler was used
        mock_profiler.start_profiling.assert_called_once()
        mock_profiler.end_profiling.assert_called_once()
        
        # Verify benchmark was stored
        mock_benchmark_manager_instance.add_benchmark.assert_called_once()
    
    def test_assess_single_frame(self, video_analyzer):
        """Test single frame quality assessment."""
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        prev_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        metrics = video_analyzer._assess_single_frame(mock_frame, prev_frame)
        
        assert 'brightness' in metrics
        assert 'contrast' in metrics
        assert 'sharpness' in metrics
        assert 'motion_blur' in metrics
        assert 'saturation' in metrics
        assert 'color_balance' in metrics
        
        # All metrics should be numeric
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert value >= 0.0
    
    def test_assess_motion_blur(self, video_analyzer):
        """Test motion blur assessment."""
        # Create frames with different motion characteristics
        static_frame = np.ones((100, 100), dtype=np.uint8) * 128
        moving_frame = np.ones((100, 100), dtype=np.uint8) * 128
        moving_frame[40:60, 40:60] = 200  # Bright moving object
        
        # Test with motion but sharp edges (low blur)
        blur_level = video_analyzer._assess_motion_blur(moving_frame, static_frame)
        assert isinstance(blur_level, float)
        assert 0.0 <= blur_level <= 1.0
    
    def test_assess_color_balance(self, video_analyzer):
        """Test color balance assessment."""
        # Create frame with balanced colors
        balanced_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        balance_score = video_analyzer._assess_color_balance(balanced_frame)
        assert balance_score > 0.8  # Should be well balanced
        
        # Create frame with color cast (too much red)
        red_cast_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        red_cast_frame[:, :, 2] = 200  # Increase red channel
        balance_score = video_analyzer._assess_color_balance(red_cast_frame)
        assert balance_score < 0.8  # Should be poorly balanced
    
    def test_assess_resolution_quality(self, video_analyzer):
        """Test resolution quality assessment."""
        # Test different resolutions
        assert video_analyzer._assess_resolution_quality(3840, 2160) == 1.0  # 4K
        assert video_analyzer._assess_resolution_quality(1920, 1080) == 0.9  # 1080p
        assert video_analyzer._assess_resolution_quality(1280, 720) == 0.7   # 720p
        assert video_analyzer._assess_resolution_quality(854, 480) == 0.5    # 480p
        assert video_analyzer._assess_resolution_quality(640, 480) == 0.3    # Below 480p
        assert video_analyzer._assess_resolution_quality(320, 240) == 0.3    # Below 480p
    
    def test_categorize_resolution(self, video_analyzer):
        """Test resolution categorization."""
        assert video_analyzer._categorize_resolution(3840, 2160) == "ultra"
        assert video_analyzer._categorize_resolution(1920, 1080) == "high"
        assert video_analyzer._categorize_resolution(1280, 720) == "medium"
        assert video_analyzer._categorize_resolution(640, 480) == "low"
    
    def test_calculate_lighting_score(self, video_analyzer):
        """Test lighting quality score calculation."""
        # Optimal lighting (brightness around 128, moderate std)
        score = video_analyzer._calculate_lighting_score(128.0, 40.0)
        assert score > 0.8
        
        # Poor lighting (too dark)
        score = video_analyzer._calculate_lighting_score(50.0, 20.0)
        assert score < 0.6
        
        # Poor lighting (too bright)
        score = video_analyzer._calculate_lighting_score(200.0, 60.0)
        assert score < 0.6
    
    def test_categorize_exposure(self, video_analyzer):
        """Test exposure categorization."""
        assert video_analyzer._categorize_exposure(50.0) == "underexposed"
        assert video_analyzer._categorize_exposure(128.0) == "optimal"
        assert video_analyzer._categorize_exposure(200.0) == "overexposed"
    
    def test_categorize_stability(self, video_analyzer):
        """Test stability categorization."""
        assert video_analyzer._categorize_stability(0.9) == "excellent"
        assert video_analyzer._categorize_stability(0.7) == "good"
        assert video_analyzer._categorize_stability(0.5) == "fair"
        assert video_analyzer._categorize_stability(0.3) == "poor"
    
    def test_calculate_contrast_score(self, video_analyzer):
        """Test contrast score calculation."""
        # Optimal contrast
        score = video_analyzer._calculate_contrast_score([50.0, 45.0, 55.0])
        assert score == 1.0
        
        # Moderate contrast
        score = video_analyzer._calculate_contrast_score([30.0, 25.0, 35.0])
        assert score == 0.8
        
        # Poor contrast
        score = video_analyzer._calculate_contrast_score([5.0, 8.0, 3.0])
        assert score == 0.4
    
    def test_assess_color_temperature(self, video_analyzer):
        """Test color temperature assessment."""
        assert video_analyzer._assess_color_temperature(128.0, 0.9) == "neutral"
        assert video_analyzer._assess_color_temperature(160.0, 0.5) == "warm"
        assert video_analyzer._assess_color_temperature(100.0, 0.5) == "cool"
    
    def test_calculate_overall_quality_score(self, video_analyzer, mock_quality_metrics):
        """Test overall quality score calculation."""
        score = video_analyzer._calculate_overall_quality_score(mock_quality_metrics)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
        
        # Should be weighted combination of individual scores
        expected_range = (0.7, 0.9)  # Based on mock metrics
        assert expected_range[0] <= score <= expected_range[1]
    
    def test_categorize_overall_quality(self, video_analyzer):
        """Test overall quality categorization."""
        assert video_analyzer._categorize_overall_quality(0.9) == "excellent"
        assert video_analyzer._categorize_overall_quality(0.7) == "good"
        assert video_analyzer._categorize_overall_quality(0.5) == "fair"
        assert video_analyzer._categorize_overall_quality(0.3) == "poor"
    
    def test_generate_enhancement_recommendations(self, video_analyzer):
        """Test enhancement recommendations generation."""
        from ai_video_editor.core.content_context import VideoQualityMetrics
        
        # Create metrics that need various enhancements
        poor_metrics = VideoQualityMetrics(
            resolution_score=0.5,
            exposure_quality="underexposed",
            lighting_score=0.4,
            camera_shake_detected=True,
            motion_blur_level=0.5,
            color_balance_score=0.4,
            saturation_level=80.0,
            contrast_score=0.4,
            color_temperature="cool"
        )
        
        recommendations = video_analyzer._generate_enhancement_recommendations(poor_metrics)
        
        assert len(recommendations) > 0
        assert any("resolution" in rec.lower() for rec in recommendations)
        assert any("brightness" in rec.lower() for rec in recommendations)
        assert any("stabilization" in rec.lower() for rec in recommendations)
        assert any("color balance" in rec.lower() for rec in recommendations)
        assert any("saturation" in rec.lower() for rec in recommendations)
    
    def test_quality_assessment_error_handling(self, video_analyzer, sample_context):
        """Test quality assessment error handling."""
        with patch('ai_video_editor.modules.content_analysis.video_analyzer.create_quality_assessment_profiler') as mock_profiler_factory:
            mock_profiler = Mock()
            mock_profiler_factory.return_value = mock_profiler
            
            with patch('ai_video_editor.modules.content_analysis.video_analyzer.cv2.VideoCapture') as mock_cap:
                mock_cap.return_value.isOpened.return_value = False
                
                # Should handle error gracefully
                with pytest.raises(Exception):
                    video_analyzer.assess_video_quality("invalid_video.mp4", sample_context)


class TestPerformanceBenchmarking:
    """Test cases for performance benchmarking functionality."""
    
    def test_quality_assessment_profiler(self):
        """Test quality assessment profiler."""
        from ai_video_editor.utils.performance_benchmarks import create_quality_assessment_profiler
        
        profiler = create_quality_assessment_profiler()
        
        # Test profiling lifecycle
        profiler.start_profiling()
        assert profiler.start_time is not None
        assert profiler.start_memory is not None
        
        # Sample resources
        profiler.sample_resources()
        assert len(profiler.memory_samples) > 1
        
        # Add small delay to ensure processing time > 0
        import time
        time.sleep(0.01)
        
        # End profiling
        benchmark = profiler.end_profiling(
            video_duration=60.0,
            video_resolution=(1920, 1080),
            frames_analyzed=30,
            quality_score_accuracy=0.85
        )
        
        assert benchmark.video_duration == 60.0
        assert benchmark.video_resolution == (1920, 1080)
        assert benchmark.frames_analyzed == 30
        assert benchmark.processing_time > 0.0
        assert benchmark.frames_per_second > 0.0
    
    def test_performance_targets_checking(self):
        """Test performance targets checking."""
        from ai_video_editor.utils.performance_benchmarks import (
            create_quality_assessment_profiler, QualityAssessmentBenchmark
        )
        
        profiler = create_quality_assessment_profiler()
        
        # Create a benchmark that meets targets
        good_benchmark = QualityAssessmentBenchmark(
            video_duration=180.0,  # 3 minutes
            video_resolution=(1920, 1080),
            frames_analyzed=90,
            processing_time=120.0,  # 2 minutes (under 3 minute target)
            memory_peak_usage=8 * 1024**3,  # 8GB (under 16GB target)
            memory_average_usage=6 * 1024**3,
            cpu_usage_percent=70.0,  # Under 80% target
            frames_per_second=10.0,  # Above 5 fps target
            quality_score_accuracy=0.9  # Above 0.85 target
        )
        
        results = profiler.check_performance_targets(good_benchmark, "general")
        
        assert results['processing_time_target'] is True
        assert results['memory_usage_target'] is True
        assert results['frames_per_second_target'] is True
        assert results['quality_accuracy_target'] is True
        assert results['cpu_usage_target'] is True
    
    def test_benchmark_manager(self):
        """Test benchmark manager functionality."""
        from ai_video_editor.utils.performance_benchmarks import (
            create_benchmark_manager, QualityAssessmentBenchmark
        )
        import tempfile
        import os
        
        # Use temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            manager = create_benchmark_manager(temp_file)
            
            # Add benchmark
            benchmark = QualityAssessmentBenchmark(
                video_duration=60.0,
                video_resolution=(1920, 1080),
                frames_analyzed=30,
                processing_time=45.0,
                memory_peak_usage=4 * 1024**3,
                memory_average_usage=3 * 1024**3,
                cpu_usage_percent=60.0,
                frames_per_second=8.0,
                quality_score_accuracy=0.8
            )
            
            manager.add_benchmark(benchmark)
            assert len(manager.benchmarks) == 1
            
            # Test statistics
            stats = manager.get_performance_statistics()
            assert stats['total_benchmarks'] == 1
            assert 'processing_time' in stats
            assert 'memory_usage_gb' in stats
            assert 'frames_per_second' in stats
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestMemoryIntegration:
    """Test cases for Memory integration functionality."""
    
    def test_memory_pattern_loading(self, video_analyzer_with_memory):
        """Test loading visual patterns from Memory."""
        # Memory client is already mocked in fixture
        assert video_analyzer_with_memory.memory_client is not None
        assert hasattr(video_analyzer_with_memory, 'visual_patterns')
        assert hasattr(video_analyzer_with_memory, 'element_detection_weights')
    
    def test_store_visual_patterns(self, video_analyzer_with_memory, sample_context, mock_frame_analysis):
        """Test storing visual patterns in Memory."""
        frame_analyses = [mock_frame_analysis]
        
        # Should not raise exception
        video_analyzer_with_memory._store_visual_patterns(sample_context, frame_analyses)
        
        # Verify Memory client was called
        video_analyzer_with_memory.memory_client.create_entities.assert_called_once()
        
        # Check the call arguments
        call_args = video_analyzer_with_memory.memory_client.create_entities.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]['name'] == 'Visual Analysis Patterns'
        assert call_args[0]['entityType'] == 'analysis_insights'
        assert len(call_args[0]['observations']) > 0
    
    def test_calculate_pattern_insights(self, video_analyzer_with_memory, sample_context, mock_frame_analysis):
        """Test calculation of pattern insights for Memory storage."""
        frame_analyses = [mock_frame_analysis]
        
        insights = video_analyzer_with_memory._calculate_pattern_insights(sample_context, frame_analyses)
        
        assert 'high_potential_frames' in insights
        assert 'element_counts' in insights
        assert 'faces_detected' in insights
        assert 'avg_face_confidence' in insights
        assert 'avg_quality_score' in insights
        assert 'content_type' in insights
        
        # Check specific values
        assert insights['high_potential_frames'] == 1  # mock_frame_analysis has 0.75 potential
        assert insights['content_type'] == 'educational'
        assert insights['avg_quality_score'] == 0.8


class TestEnhancedThumbnailScoring:
    """Test cases for enhanced thumbnail potential scoring."""
    
    def test_enhanced_thumbnail_potential_calculation(self, video_analyzer, mock_face_detection, 
                                                    mock_visual_element, sample_context):
        """Test enhanced thumbnail potential calculation with Memory insights."""
        faces = [mock_face_detection]
        visual_elements = [mock_visual_element]
        quality_score = 0.8
        
        potential = video_analyzer._calculate_thumbnail_potential(
            faces, visual_elements, quality_score, sample_context
        )
        
        assert 0.0 <= potential <= 1.0
        assert potential > 0.4  # Should be reasonably high with face + text + good quality
    
    def test_content_type_specific_adjustments(self, video_analyzer, mock_face_detection, sample_context):
        """Test content type specific adjustments in thumbnail scoring."""
        # Create data visualization element for educational content
        data_viz_element = VisualElementDetection(
            element_type="data_visualization",
            confidence=0.8,
            properties={'chart_type': 'bar_chart'}
        )
        
        faces = [mock_face_detection]
        visual_elements = [data_viz_element]
        quality_score = 0.7
        
        # Test with educational content
        sample_context.content_type = ContentType.EDUCATIONAL
        potential_edu = video_analyzer._calculate_thumbnail_potential(
            faces, visual_elements, quality_score, sample_context
        )
        
        # Test with general content
        sample_context.content_type = ContentType.GENERAL
        potential_general = video_analyzer._calculate_thumbnail_potential(
            faces, visual_elements, quality_score, sample_context
        )
        
        # Educational content should score higher for data visualization
        assert potential_edu >= potential_general


class TestVisualElementDetection:
    """Test cases for VisualElementDetection dataclass."""
    
    def test_visual_element_creation(self):
        """Test VisualElementDetection creation and serialization."""
        element = VisualElementDetection(
            element_type="text_overlay",
            confidence=0.85,
            bbox=[100.0, 200.0, 300.0, 50.0],
            properties={'text_regions_count': 3, 'average_size': 150.0}
        )
        
        assert element.element_type == "text_overlay"
        assert element.confidence == 0.85
        assert element.bbox == [100.0, 200.0, 300.0, 50.0]
        assert element.properties['text_regions_count'] == 3
        
        # Test serialization
        element_dict = element.to_dict()
        assert element_dict['element_type'] == "text_overlay"
        assert element_dict['confidence'] == 0.85
        assert element_dict['bbox'] == [100.0, 200.0, 300.0, 50.0]
        assert element_dict['properties']['text_regions_count'] == 3


class TestEnhancedFrameAnalysis:
    """Test cases for enhanced FrameAnalysis functionality."""
    
    def test_enhanced_frame_analysis_creation(self, mock_face_detection, mock_visual_element):
        """Test enhanced FrameAnalysis creation with new fields."""
        analysis = FrameAnalysis(
            timestamp=15.5,
            frame_number=465,
            faces=[mock_face_detection],
            visual_elements=[mock_visual_element],
            motion_score=0.6,
            quality_score=0.8,
            thumbnail_potential=0.75,
            scene_context="scene_1",
            emotional_context="excitement_peak"
        )
        
        assert analysis.scene_context == "scene_1"
        assert analysis.emotional_context == "excitement_peak"
        
        # Test backward compatibility method
        element_types = analysis.get_element_types()
        assert element_types == ["text_overlay"]
        
        # Test serialization
        analysis_dict = analysis.to_dict()
        assert analysis_dict['scene_context'] == "scene_1"
        assert analysis_dict['emotional_context'] == "excitement_peak"
        assert len(analysis_dict['visual_elements']) == 1
        assert analysis_dict['visual_elements'][0]['element_type'] == "text_overlay"