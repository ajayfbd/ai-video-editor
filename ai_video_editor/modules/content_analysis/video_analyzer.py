"""
Video Analysis Module - OpenCV-based video content analysis with Memory integration.

This module implements comprehensive video analysis including scene detection,
face detection with expression analysis, frame-by-frame description generation
for B-roll detection opportunities, and intelligent pattern learning through Memory.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import os
from pathlib import Path
import ffmpeg
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector, ThresholdDetector
from scenedetect.video_splitter import split_video_ffmpeg

from ...core.content_context import ContentContext, VisualHighlight, FaceDetection
from ...utils.cache_manager import CacheManager
from ...utils.error_handling import ContentContextError, ResourceConstraintError
from ...utils.performance_benchmarks import create_quality_assessment_profiler, create_benchmark_manager

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Video metadata extracted using ffmpeg."""
    duration: float
    fps: float
    width: int
    height: int
    codec: str
    bitrate: Optional[int] = None
    format: Optional[str] = None
    total_frames: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'duration': self.duration,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'codec': self.codec,
            'bitrate': self.bitrate,
            'format': self.format,
            'total_frames': self.total_frames
        }


@dataclass
class SceneInfo:
    """Scene detection information."""
    start_time: float
    end_time: float
    duration: float
    scene_id: int
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'scene_id': self.scene_id,
            'confidence': self.confidence
        }


@dataclass
class VisualElementDetection:
    """Enhanced visual element detection with confidence scoring."""
    element_type: str
    confidence: float
    bbox: Optional[List[float]] = None  # [x, y, width, height] if applicable
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'element_type': self.element_type,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'properties': self.properties
        }


@dataclass
class FrameAnalysis:
    """Enhanced analysis results for a single frame with confidence scoring."""
    timestamp: float
    frame_number: int
    faces: List[FaceDetection]
    visual_elements: List[VisualElementDetection]
    motion_score: float
    quality_score: float
    thumbnail_potential: float
    scene_context: Optional[str] = None
    emotional_context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'frame_number': self.frame_number,
            'faces': [face.to_dict() for face in self.faces],
            'visual_elements': [elem.to_dict() for elem in self.visual_elements],
            'motion_score': self.motion_score,
            'quality_score': self.quality_score,
            'thumbnail_potential': self.thumbnail_potential,
            'scene_context': self.scene_context,
            'emotional_context': self.emotional_context
        }
    
    def get_element_types(self) -> List[str]:
        """Get list of detected element types for backward compatibility."""
        return [elem.element_type for elem in self.visual_elements]


class VideoAnalyzer:
    """
    Enhanced video analysis using OpenCV and PySceneDetect with Memory integration.
    
    Provides scene detection, face detection with expression analysis,
    frame-by-frame analysis for B-roll detection opportunities, and intelligent
    pattern learning through Memory storage.
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None, memory_client=None):
        """Initialize VideoAnalyzer with optional caching and Memory integration."""
        self.cache_manager = cache_manager
        self.memory_client = memory_client
        self.face_detector = None
        self.expression_classifier = None
        self._initialize_models()
        
        # Analysis parameters
        self.scene_threshold = 30.0  # Scene detection sensitivity
        self.face_confidence_threshold = 0.7
        self.thumbnail_sample_rate = 1.0  # Sample every N seconds for thumbnails
        self.max_faces_per_frame = 10
        
        # Memory-based learning parameters
        self.visual_patterns = {}
        self.thumbnail_success_patterns = {}
        self.element_detection_weights = {
            'text_overlay': 1.0,
            'data_visualization': 1.2,
            'gesture_content': 0.8,
            'face_presence': 1.1,
            'color_contrast': 0.9
        }
        
        # Load existing patterns from Memory
        self._load_visual_patterns()
        
        logger.info("Enhanced VideoAnalyzer initialized successfully")
    
    def _load_visual_patterns(self):
        """Load visual analysis patterns from Memory."""
        if not self.memory_client:
            return
        
        try:
            # Search for existing visual analysis patterns
            search_results = self.memory_client.search_nodes("visual analysis patterns")
            
            for result in search_results.get('nodes', []):
                if result['name'] == 'Visual Analysis Patterns':
                    for observation in result.get('observations', []):
                        if 'thumbnail_success_rate' in observation:
                            # Parse and store pattern data
                            self._parse_pattern_observation(observation)
                        elif 'element_detection_accuracy' in observation:
                            self._parse_accuracy_observation(observation)
            
            logger.info("Loaded visual patterns from Memory")
            
        except Exception as e:
            logger.warning(f"Failed to load visual patterns from Memory: {e}")
    
    def _parse_pattern_observation(self, observation: str):
        """Parse pattern observation from Memory."""
        try:
            # Simple parsing - in production this would be more sophisticated
            if 'text_overlay' in observation and 'success_rate' in observation:
                # Extract success rate and update weights
                parts = observation.split()
                for i, part in enumerate(parts):
                    if part == 'success_rate' and i + 1 < len(parts):
                        rate = float(parts[i + 1].rstrip('%')) / 100
                        self.element_detection_weights['text_overlay'] = max(0.5, min(2.0, rate * 2))
                        break
        except Exception as e:
            logger.debug(f"Failed to parse pattern observation: {e}")
    
    def _parse_accuracy_observation(self, observation: str):
        """Parse accuracy observation from Memory."""
        try:
            # Update detection confidence thresholds based on historical accuracy
            if 'face_detection_accuracy' in observation:
                parts = observation.split()
                for i, part in enumerate(parts):
                    if part == 'accuracy' and i + 1 < len(parts):
                        accuracy = float(parts[i + 1].rstrip('%')) / 100
                        if accuracy > 0.9:
                            self.face_confidence_threshold = max(0.6, self.face_confidence_threshold - 0.05)
                        elif accuracy < 0.7:
                            self.face_confidence_threshold = min(0.8, self.face_confidence_threshold + 0.05)
                        break
        except Exception as e:
            logger.debug(f"Failed to parse accuracy observation: {e}")
    
    def _store_visual_patterns(self, context: ContentContext, analysis_results: List[FrameAnalysis]):
        """Store visual analysis patterns and insights in Memory."""
        if not self.memory_client:
            return
        
        try:
            # Calculate pattern insights
            insights = self._calculate_pattern_insights(context, analysis_results)
            
            # Store or update visual analysis patterns entity
            observations = []
            
            # Thumbnail success patterns
            if insights['high_potential_frames'] > 0:
                success_rate = (insights['high_potential_frames'] / len(analysis_results)) * 100
                observations.append(f"Thumbnail potential success rate: {success_rate:.1f}% for {context.content_type.value} content")
            
            # Visual element detection patterns
            for element_type, count in insights['element_counts'].items():
                if count > 0:
                    frequency = (count / len(analysis_results)) * 100
                    observations.append(f"Visual element {element_type} detected in {frequency:.1f}% of frames")
            
            # Face detection insights
            if insights['faces_detected'] > 0:
                avg_confidence = insights['avg_face_confidence']
                observations.append(f"Face detection average confidence: {avg_confidence:.2f} across {insights['faces_detected']} detections")
            
            # Quality insights
            avg_quality = insights['avg_quality_score']
            observations.append(f"Average frame quality score: {avg_quality:.2f} for video analysis session")
            
            # Store in Memory
            self.memory_client.create_entities([{
                'name': 'Visual Analysis Patterns',
                'entityType': 'analysis_insights',
                'observations': observations
            }])
            
            logger.info("Stored visual analysis patterns in Memory")
            
        except Exception as e:
            logger.warning(f"Failed to store visual patterns in Memory: {e}")
    
    def _calculate_pattern_insights(self, context: ContentContext, 
                                  analysis_results: List[FrameAnalysis]) -> Dict[str, Any]:
        """Calculate insights from analysis results for Memory storage."""
        insights = {
            'high_potential_frames': 0,
            'element_counts': {},
            'faces_detected': 0,
            'avg_face_confidence': 0.0,
            'avg_quality_score': 0.0,
            'content_type': context.content_type.value
        }
        
        total_face_confidence = 0.0
        total_quality = 0.0
        
        for analysis in analysis_results:
            # Count high potential frames
            if analysis.thumbnail_potential > 0.7:
                insights['high_potential_frames'] += 1
            
            # Count visual elements
            for element in analysis.visual_elements:
                element_type = element.element_type
                insights['element_counts'][element_type] = insights['element_counts'].get(element_type, 0) + 1
            
            # Face detection metrics
            if analysis.faces:
                insights['faces_detected'] += len(analysis.faces)
                total_face_confidence += sum(face.confidence for face in analysis.faces)
            
            # Quality metrics
            total_quality += analysis.quality_score
        
        # Calculate averages
        if insights['faces_detected'] > 0:
            insights['avg_face_confidence'] = total_face_confidence / insights['faces_detected']
        
        if analysis_results:
            insights['avg_quality_score'] = total_quality / len(analysis_results)
        
        return insights
    
    def _initialize_models(self):
        """Initialize OpenCV models for face detection and analysis."""
        try:
            # Initialize modern face detector (YuNet)
            model_path = self._get_face_model_path()
            if model_path and os.path.exists(model_path):
                self.face_detector = cv2.FaceDetectorYN.create(
                    model_path, "", (320, 240)
                )
                self.face_detector.setScoreThreshold(self.face_confidence_threshold)
                self.face_detector.setNmsThreshold(0.3)
                logger.info("YuNet face detector initialized")
            else:
                # Fallback to Haar cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_detector = cv2.CascadeClassifier(cascade_path)
                logger.info("Haar cascade face detector initialized as fallback")
                
        except Exception as e:
            logger.warning(f"Face detector initialization failed: {e}")
            self.face_detector = None
    
    def _get_face_model_path(self) -> Optional[str]:
        """Get path to YuNet face detection model."""
        # In a real implementation, this would download or locate the model
        # For now, return None to use Haar cascade fallback
        return None
    
    def analyze_video(self, video_path: str, context: ContentContext) -> ContentContext:
        """
        Perform comprehensive video analysis.
        
        Args:
            video_path: Path to video file
            context: ContentContext to store results
            
        Returns:
            Updated ContentContext with video analysis results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting video analysis for: {video_path}")
            
            # Extract video metadata
            metadata = self._extract_video_metadata(video_path)
            context.video_metadata = metadata.to_dict()
            
            # Perform scene detection
            scenes = self._detect_scenes(video_path)
            
            # Analyze frames for faces and visual elements
            frame_analyses = self._analyze_frames(video_path, metadata, context)
            
            # Convert frame analyses to visual highlights
            visual_highlights = self._create_visual_highlights(frame_analyses, scenes)
            
            # Add visual highlights to context
            for highlight in visual_highlights:
                # Convert enhanced visual elements back to simple list for compatibility
                element_types = [elem.element_type for elem in highlight.visual_elements] if hasattr(highlight, 'visual_elements') else highlight.visual_elements
                context.add_visual_highlight(
                    highlight.timestamp,
                    highlight.description,
                    highlight.faces,
                    element_types,
                    highlight.thumbnail_potential
                )
            
            # Perform comprehensive video quality assessment
            context = self.assess_video_quality(video_path, context)
            
            # Store visual analysis patterns in Memory
            self._store_visual_patterns(context, frame_analyses)
            
            # Update processing metrics
            processing_time = time.time() - start_time
            context.processing_metrics.add_module_metrics(
                'video_analyzer', processing_time, 0
            )
            
            logger.info(f"Video analysis completed in {processing_time:.2f}s")
            logger.info(f"Found {len(visual_highlights)} visual highlights")
            logger.info(f"Analyzed {len(frame_analyses)} frames with enhanced detection")
            
            return context
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            raise ContentContextError(f"Video analysis failed: {e}", context)
    
    def analyze_batch(self, video_paths: List[str], context: ContentContext) -> ContentContext:
        """
        Analyze multiple video files in batch.
        
        Args:
            video_paths: List of video file paths
            context: ContentContext to store results
            
        Returns:
            Updated ContentContext with batch analysis results
        """
        logger.info(f"Starting batch video analysis for {len(video_paths)} files")
        
        for i, video_path in enumerate(video_paths):
            try:
                logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
                context = self.analyze_video(video_path, context)
                
            except Exception as e:
                logger.error(f"Failed to analyze video {video_path}: {e}")
                # Continue with other videos
                continue
        
        logger.info("Batch video analysis completed")
        return context
    
    def _extract_video_metadata(self, video_path: str) -> VideoMetadata:
        """Extract video metadata using ffmpeg-python."""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Extract metadata
            duration = float(probe['format']['duration'])
            fps = eval(video_stream['r_frame_rate'])  # Convert fraction to float
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            codec = video_stream['codec_name']
            
            # Optional metadata
            bitrate = int(probe['format'].get('bit_rate', 0)) if 'bit_rate' in probe['format'] else None
            format_name = probe['format']['format_name']
            total_frames = int(video_stream.get('nb_frames', 0)) if 'nb_frames' in video_stream else None
            
            return VideoMetadata(
                duration=duration,
                fps=fps,
                width=width,
                height=height,
                codec=codec,
                bitrate=bitrate,
                format=format_name,
                total_frames=total_frames
            )
            
        except Exception as e:
            logger.error(f"Failed to extract video metadata: {e}")
            # Return default metadata
            return VideoMetadata(
                duration=0.0,
                fps=30.0,
                width=1920,
                height=1080,
                codec="unknown"
            )
    
    def _detect_scenes(self, video_path: str) -> List[SceneInfo]:
        """Detect scene changes using PySceneDetect."""
        try:
            # Create video manager and scene manager
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            
            # Add content detector for scene changes
            scene_manager.add_detector(ContentDetector(threshold=self.scene_threshold))
            
            # Detect scenes
            video_manager.set_duration()
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            
            # Get scene list
            scene_list = scene_manager.get_scene_list()
            
            scenes = []
            for i, (start_time, end_time) in enumerate(scene_list):
                scene = SceneInfo(
                    start_time=start_time.get_seconds(),
                    end_time=end_time.get_seconds(),
                    duration=(end_time - start_time).get_seconds(),
                    scene_id=i,
                    confidence=0.8  # PySceneDetect doesn't provide confidence scores
                )
                scenes.append(scene)
            
            video_manager.release()
            logger.info(f"Detected {len(scenes)} scenes")
            return scenes
            
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            return []
    
    def _analyze_frames(self, video_path: str, metadata: VideoMetadata, context: ContentContext) -> List[FrameAnalysis]:
        """Analyze video frames for faces and visual elements."""
        frame_analyses = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Calculate sampling interval
            sample_interval = int(metadata.fps * self.thumbnail_sample_rate)
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at specified interval
                if frame_count % sample_interval == 0:
                    timestamp = frame_count / metadata.fps
                    analysis = self._analyze_single_frame(frame, timestamp, frame_count, context)
                    if analysis:
                        frame_analyses.append(analysis)
                
                frame_count += 1
                
                # Memory management - limit analysis count
                if len(frame_analyses) > 1000:  # Prevent memory overflow
                    logger.warning("Frame analysis limit reached, stopping early")
                    break
            
            cap.release()
            logger.info(f"Analyzed {len(frame_analyses)} frames")
            return frame_analyses
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}")
            return []
    
    def _analyze_single_frame(self, frame: np.ndarray, timestamp: float, 
                            frame_number: int, context: Optional[ContentContext] = None) -> Optional[FrameAnalysis]:
        """Analyze a single frame for faces and visual elements with enhanced detection."""
        try:
            # Detect faces
            faces = self._detect_faces_in_frame(frame)
            
            # Analyze visual elements with confidence scoring
            visual_elements = self._analyze_visual_elements(frame)
            
            # Calculate quality scores
            motion_score = self._calculate_motion_score(frame)
            quality_score = self._calculate_quality_score(frame)
            
            # Calculate enhanced thumbnail potential
            thumbnail_potential = self._calculate_thumbnail_potential(
                faces, visual_elements, quality_score, context
            )
            
            # Determine scene and emotional context
            scene_context = self._determine_scene_context(timestamp, context)
            emotional_context = self._determine_emotional_context(timestamp, context)
            
            return FrameAnalysis(
                timestamp=timestamp,
                frame_number=frame_number,
                faces=faces,
                visual_elements=visual_elements,
                motion_score=motion_score,
                quality_score=quality_score,
                thumbnail_potential=thumbnail_potential,
                scene_context=scene_context,
                emotional_context=emotional_context
            )
            
        except Exception as e:
            logger.error(f"Single frame analysis failed: {e}")
            return None
    
    def _determine_scene_context(self, timestamp: float, context: Optional[ContentContext]) -> Optional[str]:
        """Determine scene context for the frame timestamp."""
        if not context or not hasattr(context, 'video_metadata'):
            return None
        
        # This would use scene detection results to determine which scene the frame belongs to
        # For now, return a simple context based on timestamp
        if timestamp < 30:
            return "opening_scene"
        elif timestamp < 60:
            return "middle_scene"
        else:
            return "closing_scene"
    
    def _determine_emotional_context(self, timestamp: float, context: Optional[ContentContext]) -> Optional[str]:
        """Determine emotional context for the frame timestamp."""
        if not context or not context.emotional_markers:
            return None
        
        # Find the closest emotional marker
        closest_marker = None
        min_distance = float('inf')
        
        for marker in context.emotional_markers:
            distance = abs(marker.timestamp - timestamp)
            if distance < min_distance and distance < 5.0:  # Within 5 seconds
                min_distance = distance
                closest_marker = marker
        
        if closest_marker and closest_marker.intensity > 0.6:
            return f"{closest_marker.emotion}_peak"
        
        return None
    
    def _detect_faces_in_frame(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a single frame."""
        faces = []
        
        if self.face_detector is None:
            return faces
        
        try:
            if hasattr(self.face_detector, 'detect'):
                # YuNet detector
                self.face_detector.setInputSize(frame.shape[:2])
                _, face_data = self.face_detector.detect(frame)
                
                if face_data is not None:
                    for face in face_data:
                        x, y, w, h = face[:4].astype(int)
                        confidence = float(face[14])
                        
                        # Extract landmarks if available
                        landmarks = []
                        if len(face) > 4:
                            landmarks = [[float(face[i]), float(face[i+1])] 
                                       for i in range(4, 14, 2)]
                        
                        # Simple expression analysis (placeholder)
                        expression = self._analyze_expression(frame[y:y+h, x:x+w])
                        
                        face_detection = FaceDetection(
                            bbox=[float(x), float(y), float(w), float(h)],
                            confidence=confidence,
                            expression=expression,
                            landmarks=landmarks if landmarks else None
                        )
                        faces.append(face_detection)
            
            else:
                # Haar cascade detector
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_rects = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                for (x, y, w, h) in face_rects:
                    # Simple expression analysis
                    face_roi = frame[y:y+h, x:x+w]
                    expression = self._analyze_expression(face_roi)
                    
                    face_detection = FaceDetection(
                        bbox=[float(x), float(y), float(w), float(h)],
                        confidence=0.8,  # Haar cascade doesn't provide confidence
                        expression=expression
                    )
                    faces.append(face_detection)
            
            # Limit number of faces to prevent memory issues
            return faces[:self.max_faces_per_frame]
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _analyze_expression(self, face_roi: np.ndarray) -> Optional[str]:
        """
        Analyze facial expression (simplified implementation).
        
        In a full implementation, this would use a trained expression
        classification model. For now, we return a placeholder.
        """
        if face_roi.size == 0:
            return None
        
        # Placeholder expression analysis
        # In reality, this would use a CNN model for expression classification
        expressions = ["neutral", "happy", "surprised", "focused", "speaking"]
        
        # Simple heuristic based on face region properties
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray_face)
        
        if mean_intensity > 120:
            return "happy"
        elif mean_intensity < 80:
            return "focused"
        else:
            return "neutral"
    
    def _analyze_visual_elements(self, frame: np.ndarray) -> List[VisualElementDetection]:
        """Analyze visual elements in frame with confidence scoring."""
        elements = []
        
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Detect text regions with confidence
            text_detection = self._detect_text_regions(gray)
            if text_detection:
                elements.append(text_detection)
            
            # Detect charts/graphs with confidence
            chart_detection = self._detect_chart_elements(frame)
            if chart_detection:
                elements.append(chart_detection)
            
            # Detect motion/gestures with confidence
            motion_detection = self._detect_motion_elements(frame)
            if motion_detection:
                elements.append(motion_detection)
            
            # Analyze color distribution with confidence
            color_detections = self._detect_dominant_colors(hsv)
            elements.extend(color_detections)
            
            # Detect geometric patterns
            geometry_detection = self._detect_geometric_patterns(gray)
            if geometry_detection:
                elements.append(geometry_detection)
            
            return elements
            
        except Exception as e:
            logger.error(f"Visual element analysis failed: {e}")
            return []
    
    def _detect_text_regions(self, gray_frame: np.ndarray) -> Optional[VisualElementDetection]:
        """Detect text regions with confidence scoring."""
        try:
            # Use edge detection to find text-like regions
            edges = cv2.Canny(gray_frame, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours for text-like properties
            text_contours = []
            text_score = 0.0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                bbox_area = w * h
                
                # Text typically has certain aspect ratios and minimum size
                if 0.2 < aspect_ratio < 5.0 and w > 20 and h > 10 and bbox_area > 200:
                    text_contours.append((x, y, w, h))
                    
                    # Score based on text-like properties
                    aspect_score = 1.0 - abs(aspect_ratio - 2.0) / 2.0  # Prefer ~2:1 ratio
                    size_score = min(1.0, bbox_area / 1000.0)  # Larger text gets higher score
                    text_score += aspect_score * size_score * 0.1
            
            if len(text_contours) > 3:  # Minimum threshold for text presence
                confidence = min(0.95, 0.3 + (len(text_contours) * 0.05) + text_score)
                
                # Find bounding box of all text regions
                if text_contours:
                    min_x = min(x for x, y, w, h in text_contours)
                    min_y = min(y for x, y, w, h in text_contours)
                    max_x = max(x + w for x, y, w, h in text_contours)
                    max_y = max(y + h for x, y, w, h in text_contours)
                    bbox = [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]
                else:
                    bbox = None
                
                return VisualElementDetection(
                    element_type="text_overlay",
                    confidence=confidence,
                    bbox=bbox,
                    properties={
                        'text_regions_count': len(text_contours),
                        'average_aspect_ratio': np.mean([w/h for x, y, w, h in text_contours if h > 0]),
                        'total_text_area': sum(w*h for x, y, w, h in text_contours)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Text detection failed: {e}")
            return None
    
    def _detect_chart_elements(self, frame: np.ndarray) -> Optional[VisualElementDetection]:
        """Detect chart/graph elements with confidence scoring."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            chart_score = 0.0
            chart_properties = {}
            
            # Look for straight lines (common in charts)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                  minLineLength=50, maxLineGap=10)
            
            line_score = 0.0
            if lines is not None and len(lines) > 5:
                line_score = min(0.6, len(lines) * 0.03)
                chart_properties['line_count'] = len(lines)
                
                # Analyze line orientations for grid patterns
                horizontal_lines = 0
                vertical_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    if abs(angle) < 15 or abs(angle) > 165:  # Horizontal
                        horizontal_lines += 1
                    elif 75 < abs(angle) < 105:  # Vertical
                        vertical_lines += 1
                
                if horizontal_lines > 2 and vertical_lines > 2:
                    line_score += 0.2  # Grid pattern bonus
                    chart_properties['grid_pattern'] = True
            
            # Look for circular elements (pie charts)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=20, maxRadius=200)
            
            circle_score = 0.0
            if circles is not None and len(circles[0]) > 0:
                circle_score = min(0.4, len(circles[0]) * 0.2)
                chart_properties['circle_count'] = len(circles[0])
            
            # Look for rectangular regions (bar charts)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangular_regions = 0
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Rectangular
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 20 and h > 20:  # Minimum size
                        rectangular_regions += 1
            
            rect_score = 0.0
            if rectangular_regions > 3:
                rect_score = min(0.3, rectangular_regions * 0.05)
                chart_properties['rectangular_regions'] = rectangular_regions
            
            chart_score = line_score + circle_score + rect_score
            
            if chart_score > 0.3:  # Minimum threshold for chart detection
                confidence = min(0.95, chart_score)
                
                return VisualElementDetection(
                    element_type="data_visualization",
                    confidence=confidence,
                    properties=chart_properties
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Chart detection failed: {e}")
            return None
    
    def _detect_motion_elements(self, frame: np.ndarray) -> Optional[VisualElementDetection]:
        """Detect motion/gesture elements with confidence scoring."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Edge density analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Gradient magnitude analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(magnitude)
            
            # Texture analysis using Local Binary Patterns concept
            texture_score = self._calculate_texture_complexity(gray)
            
            motion_score = 0.0
            properties = {}
            
            # High edge density suggests motion or complex content
            if edge_density > 0.08:
                motion_score += min(0.4, edge_density * 4)
                properties['edge_density'] = edge_density
            
            # High gradient magnitude suggests motion blur or dynamic content
            if avg_gradient > 30:
                motion_score += min(0.3, avg_gradient / 100)
                properties['gradient_magnitude'] = avg_gradient
            
            # Complex texture suggests gesture or movement
            if texture_score > 0.5:
                motion_score += min(0.3, texture_score * 0.6)
                properties['texture_complexity'] = texture_score
            
            if motion_score > 0.4:  # Threshold for motion detection
                confidence = min(0.9, motion_score)
                
                return VisualElementDetection(
                    element_type="gesture_content",
                    confidence=confidence,
                    properties=properties
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Motion detection failed: {e}")
            return None
    
    def _calculate_texture_complexity(self, gray_frame: np.ndarray) -> float:
        """Calculate texture complexity score."""
        try:
            # Use standard deviation of pixel intensities as texture measure
            std_dev = np.std(gray_frame)
            
            # Use Laplacian variance as sharpness measure
            laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
            laplacian_var = laplacian.var()
            
            # Combine measures and normalize
            texture_score = (std_dev / 64.0) * 0.6 + (laplacian_var / 1000.0) * 0.4
            return min(1.0, texture_score)
            
        except Exception:
            return 0.0
    
    def _detect_dominant_colors(self, hsv_frame: np.ndarray) -> List[VisualElementDetection]:
        """Detect dominant colors with confidence scoring."""
        color_detections = []
        
        try:
            # Analyze hue, saturation, and value channels
            hue = hsv_frame[:, :, 0]
            saturation = hsv_frame[:, :, 1]
            value = hsv_frame[:, :, 2]
            
            # Calculate histograms
            hue_hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
            sat_hist = cv2.calcHist([saturation], [0], None, [256], [0, 256])
            val_hist = cv2.calcHist([value], [0], None, [256], [0, 256])
            
            # Analyze hue distribution
            hue_mean = np.mean(hue_hist)
            for i, count in enumerate(hue_hist.flatten()):
                if count > hue_mean * 2.5:  # Significantly above average
                    confidence = min(0.9, (count / hue_mean) * 0.2)
                    
                    color_name = self._hue_to_color_name(i)
                    if color_name:
                        color_detections.append(VisualElementDetection(
                            element_type=f"{color_name}_dominant",
                            confidence=confidence,
                            properties={
                                'hue_value': i,
                                'pixel_count': int(count),
                                'dominance_ratio': count / np.sum(hue_hist)
                            }
                        ))
            
            # Analyze saturation (colorfulness)
            avg_saturation = np.mean(saturation)
            if avg_saturation > 100:  # High saturation
                confidence = min(0.8, avg_saturation / 255.0)
                color_detections.append(VisualElementDetection(
                    element_type="high_saturation",
                    confidence=confidence,
                    properties={'average_saturation': avg_saturation}
                ))
            
            # Analyze contrast (value distribution)
            value_std = np.std(value)
            if value_std > 50:  # High contrast
                confidence = min(0.8, value_std / 128.0)
                color_detections.append(VisualElementDetection(
                    element_type="high_contrast",
                    confidence=confidence,
                    properties={'value_std': value_std}
                ))
            
            return color_detections[:5]  # Limit to top 5
            
        except Exception as e:
            logger.debug(f"Color detection failed: {e}")
            return []
    
    def _hue_to_color_name(self, hue_value: int) -> Optional[str]:
        """Convert hue value to color name."""
        if 0 <= hue_value < 15 or 165 <= hue_value < 180:
            return "red"
        elif 15 <= hue_value < 45:
            return "orange"
        elif 45 <= hue_value < 75:
            return "yellow"
        elif 75 <= hue_value < 105:
            return "green"
        elif 105 <= hue_value < 135:
            return "blue"
        elif 135 <= hue_value < 165:
            return "purple"
        return None
    
    def _detect_geometric_patterns(self, gray_frame: np.ndarray) -> Optional[VisualElementDetection]:
        """Detect geometric patterns that might indicate structured content."""
        try:
            # Detect corners using Harris corner detection
            corners = cv2.cornerHarris(gray_frame, 2, 3, 0.04)
            corner_count = np.sum(corners > 0.01 * corners.max())
            
            # Detect contours for shape analysis
            edges = cv2.Canny(gray_frame, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            geometric_score = 0.0
            properties = {}
            
            # Analyze corner density
            if corner_count > 20:
                corner_score = min(0.4, corner_count / 100.0)
                geometric_score += corner_score
                properties['corner_count'] = int(corner_count)
            
            # Analyze shape regularity
            regular_shapes = 0
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Minimum area
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Count regular shapes (3-8 sides)
                    if 3 <= len(approx) <= 8:
                        regular_shapes += 1
            
            if regular_shapes > 3:
                shape_score = min(0.3, regular_shapes * 0.05)
                geometric_score += shape_score
                properties['regular_shapes'] = regular_shapes
            
            if geometric_score > 0.3:
                confidence = min(0.85, geometric_score)
                
                return VisualElementDetection(
                    element_type="geometric_patterns",
                    confidence=confidence,
                    properties=properties
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Geometric pattern detection failed: {e}")
            return None
    
    def _calculate_motion_score(self, frame: np.ndarray) -> float:
        """Calculate motion score for frame (simplified)."""
        try:
            # Use gradient magnitude as motion proxy
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            return float(np.mean(magnitude) / 255.0)  # Normalize to 0-1
            
        except Exception:
            return 0.0
    
    def _calculate_quality_score(self, frame: np.ndarray) -> float:
        """Calculate frame quality score."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast
            contrast = gray.std()
            
            # Combine metrics (normalized)
            quality = (
                min(sharpness / 1000.0, 1.0) * 0.4 +  # Sharpness weight
                min(abs(brightness - 128) / 128.0, 1.0) * 0.3 +  # Brightness weight
                min(contrast / 64.0, 1.0) * 0.3  # Contrast weight
            )
            
            return float(quality)
            
        except Exception:
            return 0.5  # Default quality
    
    def assess_video_quality(self, video_path: str, context: ContentContext) -> ContentContext:
        """
        Perform comprehensive video quality assessment with performance benchmarking.
        
        Args:
            video_path: Path to video file
            context: ContentContext to store results
            
        Returns:
            Updated ContentContext with video quality metrics
        """
        # Initialize performance profiler
        profiler = create_quality_assessment_profiler()
        profiler.start_profiling()
        
        try:
            logger.info(f"Starting video quality assessment for: {video_path}")
            
            # Extract video metadata if not already available
            if not context.video_metadata:
                metadata = self._extract_video_metadata(video_path)
                context.video_metadata = metadata.to_dict()
            else:
                # Create metadata object from existing data
                metadata = VideoMetadata(
                    duration=context.video_metadata.get('duration', 0.0),
                    fps=context.video_metadata.get('fps', 30.0),
                    width=context.video_metadata.get('width', 1920),
                    height=context.video_metadata.get('height', 1080),
                    codec=context.video_metadata.get('codec', 'unknown')
                )
            
            # Perform comprehensive quality assessment with profiling
            quality_metrics = self._perform_quality_assessment_with_profiling(video_path, metadata, profiler)
            
            # Store quality metrics in context
            context.set_video_quality_metrics(quality_metrics)
            
            # Create and store performance benchmark
            benchmark = profiler.end_profiling(
                video_duration=metadata.duration,
                video_resolution=(metadata.width, metadata.height),
                frames_analyzed=quality_metrics.frames_analyzed,
                quality_score_accuracy=quality_metrics.overall_quality_score
            )
            
            # Check performance targets
            target_results = profiler.check_performance_targets(benchmark, context.content_type.value)
            
            # Store benchmark
            benchmark_manager = create_benchmark_manager()
            benchmark_manager.add_benchmark(benchmark)
            
            # Log performance results
            self._log_performance_results(benchmark, target_results)
            
            # Update processing metrics
            context.processing_metrics.add_module_metrics(
                'video_quality_assessment', benchmark.processing_time, benchmark.memory_peak_usage
            )
            
            logger.info(f"Video quality assessment completed in {benchmark.processing_time:.2f}s")
            logger.info(f"Overall quality score: {quality_metrics.overall_quality_score:.2f} ({quality_metrics.quality_category})")
            
            return context
            
        except Exception as e:
            logger.error(f"Video quality assessment failed: {e}")
            raise ContentContextError(f"Video quality assessment failed: {e}", context)
    
    def _perform_quality_assessment_with_profiling(self, video_path: str, metadata: VideoMetadata, 
                                                 profiler) -> 'VideoQualityMetrics':
        """Perform quality assessment with resource monitoring."""
        from ...core.content_context import VideoQualityMetrics
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Initialize metrics
            quality_metrics = VideoQualityMetrics()
            quality_metrics.actual_resolution = (metadata.width, metadata.height)
            
            # Sample frames for analysis (every 2 seconds)
            sample_interval = int(metadata.fps * 2.0)
            frame_count = 0
            analyzed_frames = 0
            
            # Accumulate metrics
            brightness_values = []
            contrast_values = []
            sharpness_values = []
            motion_blur_values = []
            saturation_values = []
            color_balance_values = []
            
            prev_frame = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames at specified interval
                if frame_count % sample_interval == 0:
                    # Sample resources for profiling
                    profiler.sample_resources()
                    
                    # Assess individual frame
                    frame_metrics = self._assess_single_frame(frame, prev_frame)
                    
                    # Accumulate values
                    brightness_values.append(frame_metrics['brightness'])
                    contrast_values.append(frame_metrics['contrast'])
                    sharpness_values.append(frame_metrics['sharpness'])
                    motion_blur_values.append(frame_metrics['motion_blur'])
                    saturation_values.append(frame_metrics['saturation'])
                    color_balance_values.append(frame_metrics['color_balance'])
                    
                    analyzed_frames += 1
                    prev_frame = frame.copy()
                
                frame_count += 1
                
                # Limit analysis to prevent memory issues
                if analyzed_frames > 100:
                    break
            
            cap.release()
            
            # Calculate aggregate metrics
            if analyzed_frames > 0:
                quality_metrics = self._calculate_aggregate_quality_metrics(
                    quality_metrics, metadata, analyzed_frames,
                    brightness_values, contrast_values, sharpness_values,
                    motion_blur_values, saturation_values, color_balance_values
                )
            
            quality_metrics.frames_analyzed = analyzed_frames
            
            logger.info(f"Analyzed {analyzed_frames} frames for quality assessment")
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            # Return default metrics
            return VideoQualityMetrics()
    
    def _log_performance_results(self, benchmark, target_results: Dict[str, bool]):
        """Log performance benchmark results."""
        logger.info("=== Video Quality Assessment Performance Results ===")
        logger.info(f"Processing time: {benchmark.processing_time:.2f}s")
        logger.info(f"Memory peak usage: {benchmark.memory_peak_usage / (1024**3):.2f}GB")
        logger.info(f"Processing rate: {benchmark.frames_per_second:.1f} fps")
        logger.info(f"CPU usage: {benchmark.cpu_usage_percent:.1f}%")
        
        logger.info("=== Performance Target Results ===")
        for target, met in target_results.items():
            status = "✓ PASS" if met else "✗ FAIL"
            logger.info(f"{target}: {status}")
        
        # Overall performance assessment
        passed_targets = sum(target_results.values())
        total_targets = len(target_results)
        performance_score = passed_targets / total_targets if total_targets > 0 else 0.0
        
        if performance_score >= 0.8:
            logger.info("🎉 Overall performance: EXCELLENT")
        elif performance_score >= 0.6:
            logger.info("👍 Overall performance: GOOD")
        elif performance_score >= 0.4:
            logger.info("⚠️  Overall performance: NEEDS IMPROVEMENT")
        else:
            logger.warning("❌ Overall performance: POOR - Optimization required")
    

    
    def _assess_single_frame(self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Assess quality metrics for a single frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Brightness assessment
            brightness = np.mean(gray)
            
            # Contrast assessment
            contrast = np.std(gray)
            
            # Sharpness assessment (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Motion blur assessment
            motion_blur = self._assess_motion_blur(gray, prev_frame)
            
            # Saturation assessment
            saturation = np.mean(hsv[:, :, 1])
            
            # Color balance assessment
            color_balance = self._assess_color_balance(frame)
            
            return {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': sharpness,
                'motion_blur': motion_blur,
                'saturation': saturation,
                'color_balance': color_balance
            }
            
        except Exception as e:
            logger.debug(f"Single frame assessment failed: {e}")
            return {
                'brightness': 128.0,
                'contrast': 32.0,
                'sharpness': 100.0,
                'motion_blur': 0.0,
                'saturation': 128.0,
                'color_balance': 0.5
            }
    
    def _assess_motion_blur(self, gray_frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> float:
        """Assess motion blur level in frame."""
        try:
            if prev_frame is None:
                return 0.0
            
            # Convert previous frame to grayscale if needed
            if len(prev_frame.shape) == 3:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = prev_frame
            
            # Calculate frame difference
            diff = cv2.absdiff(gray_frame, prev_gray)
            
            # Calculate motion magnitude
            motion_magnitude = np.mean(diff)
            
            # Assess blur using gradient magnitude
            grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            
            # Motion blur is high when there's motion but low gradient
            if motion_magnitude > 10 and avg_gradient < 20:
                return min(1.0, motion_magnitude / 50.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _assess_color_balance(self, frame: np.ndarray) -> float:
        """Assess color balance of frame."""
        try:
            # Calculate mean values for each color channel
            b_mean = np.mean(frame[:, :, 0])
            g_mean = np.mean(frame[:, :, 1])
            r_mean = np.mean(frame[:, :, 2])
            
            # Calculate color balance score (closer to equal means better balance)
            total_mean = (b_mean + g_mean + r_mean) / 3
            
            if total_mean == 0:
                return 0.5
            
            # Calculate deviation from balanced
            b_dev = abs(b_mean - total_mean) / total_mean
            g_dev = abs(g_mean - total_mean) / total_mean
            r_dev = abs(r_mean - total_mean) / total_mean
            
            avg_deviation = (b_dev + g_dev + r_dev) / 3
            
            # Convert to balance score (1.0 = perfect balance, 0.0 = poor balance)
            balance_score = max(0.0, 1.0 - avg_deviation)
            
            return balance_score
            
        except Exception:
            return 0.5
    
    def _calculate_aggregate_quality_metrics(self, quality_metrics: 'VideoQualityMetrics', 
                                           metadata: VideoMetadata, analyzed_frames: int,
                                           brightness_values: List[float], contrast_values: List[float],
                                           sharpness_values: List[float], motion_blur_values: List[float],
                                           saturation_values: List[float], color_balance_values: List[float]) -> 'VideoQualityMetrics':
        """Calculate aggregate quality metrics from frame samples."""
        try:
            # Resolution assessment
            quality_metrics.resolution_score = self._assess_resolution_quality(metadata.width, metadata.height)
            quality_metrics.resolution_category = self._categorize_resolution(metadata.width, metadata.height)
            
            # Lighting assessment
            quality_metrics.brightness_mean = np.mean(brightness_values)
            quality_metrics.brightness_std = np.std(brightness_values)
            quality_metrics.lighting_score = self._calculate_lighting_score(quality_metrics.brightness_mean, quality_metrics.brightness_std)
            quality_metrics.exposure_quality = self._categorize_exposure(quality_metrics.brightness_mean)
            
            # Stability assessment
            avg_motion_blur = np.mean(motion_blur_values)
            quality_metrics.motion_blur_level = avg_motion_blur
            quality_metrics.camera_shake_detected = avg_motion_blur > 0.3
            quality_metrics.stability_score = max(0.0, 1.0 - avg_motion_blur)
            quality_metrics.stability_category = self._categorize_stability(quality_metrics.stability_score)
            
            # Color assessment
            quality_metrics.color_balance_score = np.mean(color_balance_values)
            quality_metrics.saturation_level = np.mean(saturation_values)
            quality_metrics.contrast_score = self._calculate_contrast_score(contrast_values)
            quality_metrics.color_temperature = self._assess_color_temperature(quality_metrics.brightness_mean, quality_metrics.color_balance_score)
            
            # Overall quality score
            quality_metrics.overall_quality_score = self._calculate_overall_quality_score(quality_metrics)
            quality_metrics.quality_category = self._categorize_overall_quality(quality_metrics.overall_quality_score)
            
            # Generate enhancement recommendations
            quality_metrics.enhancement_recommendations = self._generate_enhancement_recommendations(quality_metrics)
            quality_metrics.color_correction_needed = quality_metrics.color_balance_score < 0.6
            quality_metrics.lighting_adjustment_needed = quality_metrics.lighting_score < 0.6
            quality_metrics.stabilization_needed = quality_metrics.stability_score < 0.7
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate aggregate quality metrics: {e}")
            return quality_metrics
    
    def _assess_resolution_quality(self, width: int, height: int) -> float:
        """Assess resolution quality score."""
        total_pixels = width * height
        
        if total_pixels >= 3840 * 2160:  # 4K+
            return 1.0
        elif total_pixels >= 1920 * 1080:  # 1080p
            return 0.9
        elif total_pixels >= 1280 * 720:  # 720p
            return 0.7
        elif total_pixels >= 854 * 480:  # 480p
            return 0.5
        else:  # Below 480p
            return 0.3
    
    def _categorize_resolution(self, width: int, height: int) -> str:
        """Categorize resolution quality."""
        total_pixels = width * height
        
        if total_pixels >= 3840 * 2160:
            return "ultra"
        elif total_pixels >= 1920 * 1080:
            return "high"
        elif total_pixels >= 1280 * 720:
            return "medium"
        else:
            return "low"
    
    def _calculate_lighting_score(self, brightness_mean: float, brightness_std: float) -> float:
        """Calculate lighting quality score."""
        # Optimal brightness is around 128 (middle gray)
        brightness_score = 1.0 - abs(brightness_mean - 128) / 128
        
        # Good lighting has moderate variation (not too flat, not too contrasty)
        std_score = 1.0 - abs(brightness_std - 40) / 40 if brightness_std <= 80 else 0.5
        
        return (brightness_score * 0.7 + std_score * 0.3)
    
    def _categorize_exposure(self, brightness_mean: float) -> str:
        """Categorize exposure quality."""
        if brightness_mean < 80:
            return "underexposed"
        elif brightness_mean > 180:
            return "overexposed"
        else:
            return "optimal"
    
    def _categorize_stability(self, stability_score: float) -> str:
        """Categorize stability quality."""
        if stability_score >= 0.8:
            return "excellent"
        elif stability_score >= 0.6:
            return "good"
        elif stability_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _calculate_contrast_score(self, contrast_values: List[float]) -> float:
        """Calculate contrast quality score."""
        avg_contrast = np.mean(contrast_values)
        
        # Optimal contrast is around 40-60
        if 40 <= avg_contrast <= 60:
            return 1.0
        elif 20 <= avg_contrast <= 80:
            return 0.8
        elif 10 <= avg_contrast <= 100:
            return 0.6
        else:
            return 0.4
    
    def _assess_color_temperature(self, brightness_mean: float, color_balance_score: float) -> str:
        """Assess color temperature."""
        if color_balance_score > 0.8:
            return "neutral"
        elif brightness_mean > 140:
            return "warm"
        else:
            return "cool"
    
    def _calculate_overall_quality_score(self, metrics: 'VideoQualityMetrics') -> float:
        """Calculate overall quality score from individual metrics."""
        # Weighted combination of all quality factors
        score = (
            metrics.resolution_score * 0.25 +
            metrics.lighting_score * 0.25 +
            metrics.stability_score * 0.20 +
            metrics.color_balance_score * 0.15 +
            metrics.contrast_score * 0.15
        )
        
        return min(1.0, max(0.0, score))
    
    def _categorize_overall_quality(self, overall_score: float) -> str:
        """Categorize overall quality."""
        if overall_score >= 0.8:
            return "excellent"
        elif overall_score >= 0.6:
            return "good"
        elif overall_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _generate_enhancement_recommendations(self, metrics: 'VideoQualityMetrics') -> List[str]:
        """Generate specific enhancement recommendations."""
        recommendations = []
        
        # Resolution recommendations
        if metrics.resolution_score < 0.7:
            recommendations.append("Consider upscaling to higher resolution")
        
        # Lighting recommendations
        if metrics.exposure_quality == "underexposed":
            recommendations.append("Increase brightness and lift shadows")
        elif metrics.exposure_quality == "overexposed":
            recommendations.append("Reduce highlights and lower exposure")
        
        if metrics.lighting_score < 0.6:
            recommendations.append("Adjust lighting balance and contrast")
        
        # Stability recommendations
        if metrics.camera_shake_detected:
            recommendations.append("Apply video stabilization")
        
        if metrics.motion_blur_level > 0.3:
            recommendations.append("Reduce motion blur with deblurring filters")
        
        # Color recommendations
        if metrics.color_balance_score < 0.6:
            recommendations.append("Correct color balance and white balance")
        
        if metrics.saturation_level < 100:
            recommendations.append("Increase color saturation")
        elif metrics.saturation_level > 200:
            recommendations.append("Reduce color saturation")
        
        if metrics.contrast_score < 0.6:
            recommendations.append("Adjust contrast and dynamic range")
        
        # Color temperature recommendations
        if metrics.color_temperature != "neutral":
            recommendations.append(f"Adjust color temperature (currently {metrics.color_temperature})")
        
        return recommendations
    
    def _calculate_thumbnail_potential(self, faces: List[FaceDetection], 
                                     visual_elements: List[VisualElementDetection], 
                                     quality_score: float,
                                     context: Optional[ContentContext] = None) -> float:
        """Calculate enhanced thumbnail potential score using Memory insights."""
        try:
            score = 0.0
            
            # Face presence and quality (enhanced with expression analysis)
            if faces:
                face_score = 0.0
                for face in faces:
                    # Base face score
                    individual_score = 0.15 * face.confidence
                    
                    # Expression bonus
                    if face.expression in ['happy', 'surprised', 'excited']:
                        individual_score *= 1.3
                    elif face.expression in ['focused', 'speaking']:
                        individual_score *= 1.1
                    
                    face_score += individual_score
                
                # Cap face score but allow multiple faces to contribute
                face_score = min(face_score, 0.45)
                score += face_score * self.element_detection_weights.get('face_presence', 1.0)
            
            # Enhanced visual elements scoring
            element_score = 0.0
            element_types = set()
            
            for element in visual_elements:
                element_weight = self.element_detection_weights.get(element.element_type, 1.0)
                element_contribution = element.confidence * 0.08 * element_weight
                element_score += element_contribution
                element_types.add(element.element_type)
            
            # Diversity bonus for multiple element types
            if len(element_types) > 2:
                element_score *= 1.2
            
            score += min(element_score, 0.35)  # Cap element score
            
            # Quality score with adaptive weighting
            quality_weight = 0.25
            if context and context.content_type.value == 'educational':
                quality_weight = 0.3  # Higher quality importance for educational content
            
            score += quality_score * quality_weight
            
            # Specific element bonuses with Memory-based weighting
            for element in visual_elements:
                if element.element_type == "text_overlay" and element.confidence > 0.7:
                    score += 0.08 * self.element_detection_weights.get('text_overlay', 1.0)
                elif element.element_type == "data_visualization" and element.confidence > 0.6:
                    score += 0.12 * self.element_detection_weights.get('data_visualization', 1.0)
                elif element.element_type == "high_contrast" and element.confidence > 0.5:
                    score += 0.05 * self.element_detection_weights.get('color_contrast', 1.0)
            
            # Content type specific adjustments
            if context:
                if context.content_type.value == 'educational':
                    # Educational content benefits from data visualizations and text
                    for element in visual_elements:
                        if element.element_type in ['data_visualization', 'text_overlay']:
                            score += 0.03
                elif context.content_type.value == 'music':
                    # Music content benefits from high saturation and motion
                    for element in visual_elements:
                        if element.element_type in ['high_saturation', 'gesture_content']:
                            score += 0.03
            
            # Temporal context bonus (if frame is near emotional peaks)
            if context and hasattr(context, 'emotional_markers'):
                # This would be implemented with frame timestamp context
                pass
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.debug(f"Thumbnail potential calculation failed: {e}")
            return 0.0
    
    def _create_visual_highlights(self, frame_analyses: List[FrameAnalysis], 
                                scenes: List[SceneInfo]) -> List[VisualHighlight]:
        """Create visual highlights from enhanced frame analyses and scene information."""
        highlights = []
        
        try:
            # Sort frame analyses by thumbnail potential
            sorted_frames = sorted(frame_analyses, 
                                 key=lambda x: x.thumbnail_potential, reverse=True)
            
            # Take top frames as highlights, but ensure diversity
            top_frames = self._select_diverse_highlights(sorted_frames, 20)
            
            for frame_analysis in top_frames:
                # Find corresponding scene
                scene_context = frame_analysis.scene_context or "unknown_scene"
                for scene in scenes:
                    if scene.start_time <= frame_analysis.timestamp <= scene.end_time:
                        scene_context = f"scene_{scene.scene_id}"
                        break
                
                # Create enhanced description
                description = self._create_enhanced_frame_description(frame_analysis, scene_context)
                
                # Convert enhanced visual elements to simple list for VisualHighlight compatibility
                element_types = [elem.element_type for elem in frame_analysis.visual_elements]
                
                highlight = VisualHighlight(
                    timestamp=frame_analysis.timestamp,
                    description=description,
                    faces=frame_analysis.faces,
                    visual_elements=element_types,
                    thumbnail_potential=frame_analysis.thumbnail_potential
                )
                highlights.append(highlight)
            
            return highlights
            
        except Exception as e:
            logger.error(f"Failed to create visual highlights: {e}")
            return []
    
    def _select_diverse_highlights(self, sorted_frames: List[FrameAnalysis], max_count: int) -> List[FrameAnalysis]:
        """Select diverse highlights to avoid clustering similar frames."""
        if not sorted_frames:
            return []
        
        selected = [sorted_frames[0]]  # Always include the best frame
        min_time_gap = 5.0  # Minimum 5 seconds between highlights
        
        for frame in sorted_frames[1:]:
            if len(selected) >= max_count:
                break
            
            # Check if this frame is far enough from existing selections
            too_close = False
            for selected_frame in selected:
                if abs(frame.timestamp - selected_frame.timestamp) < min_time_gap:
                    too_close = True
                    break
            
            if not too_close:
                selected.append(frame)
        
        return selected
    
    def _create_enhanced_frame_description(self, frame_analysis: FrameAnalysis, 
                                         scene_context: str) -> str:
        """Create enhanced descriptive text for frame analysis."""
        try:
            parts = []
            
            # Add scene and emotional context
            context_parts = [scene_context]
            if frame_analysis.emotional_context:
                context_parts.append(frame_analysis.emotional_context)
            parts.append(f"Frame from {', '.join(context_parts)}")
            
            # Add face information with expressions
            if frame_analysis.faces:
                face_count = len(frame_analysis.faces)
                expressions = [face.expression for face in frame_analysis.faces 
                             if face.expression]
                avg_confidence = np.mean([face.confidence for face in frame_analysis.faces])
                
                if expressions:
                    parts.append(f"{face_count} face(s) with {', '.join(set(expressions))} expression(s) (conf: {avg_confidence:.2f})")
                else:
                    parts.append(f"{face_count} face(s) detected (conf: {avg_confidence:.2f})")
            
            # Add enhanced visual elements with confidence
            if frame_analysis.visual_elements:
                high_conf_elements = [elem for elem in frame_analysis.visual_elements if elem.confidence > 0.7]
                if high_conf_elements:
                    elements_str = ', '.join([f"{elem.element_type}({elem.confidence:.2f})" 
                                            for elem in high_conf_elements])
                    parts.append(f"High-confidence elements: {elements_str}")
                
                all_elements = [elem.element_type for elem in frame_analysis.visual_elements]
                parts.append(f"All elements: {', '.join(set(all_elements))}")
            
            # Add quality and motion information
            quality_desc = "high quality" if frame_analysis.quality_score > 0.7 else \
                          "low quality" if frame_analysis.quality_score < 0.3 else "medium quality"
            motion_desc = "high motion" if frame_analysis.motion_score > 0.7 else \
                         "low motion" if frame_analysis.motion_score < 0.3 else "medium motion"
            
            parts.append(f"{quality_desc}, {motion_desc}")
            
            # Add thumbnail potential
            potential_desc = "excellent" if frame_analysis.thumbnail_potential > 0.8 else \
                           "good" if frame_analysis.thumbnail_potential > 0.6 else \
                           "fair" if frame_analysis.thumbnail_potential > 0.4 else "poor"
            parts.append(f"thumbnail potential: {potential_desc} ({frame_analysis.thumbnail_potential:.2f})")
            
            return "; ".join(parts)
            
        except Exception as e:
            logger.debug(f"Failed to create enhanced description: {e}")
            return f"Enhanced frame at {frame_analysis.timestamp:.2f}s"


def create_video_analyzer(cache_manager: Optional[CacheManager] = None, 
                         memory_client=None) -> VideoAnalyzer:
    """Factory function to create enhanced VideoAnalyzer instance with Memory integration."""
    return VideoAnalyzer(cache_manager, memory_client)