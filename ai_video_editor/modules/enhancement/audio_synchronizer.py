"""
Audio Synchronizer - Handles precise audio-video synchronization with movis.

This module ensures frame-accurate synchronization between enhanced audio and video
content, managing sync points, timing adjustments, and integration with the
movis-based video composition system.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    import movis as mv
    MOVIS_AVAILABLE = True
except ImportError:
    MOVIS_AVAILABLE = False
    mv = None

from ...core.content_context import ContentContext
from ...core.exceptions import ProcessingError, ContentContextError, InputValidationError
from ...utils.logging_config import get_logger
from .audio_enhancement import AudioEnhancementResult


@dataclass
class SyncPoint:
    """Represents a critical audio-video synchronization point."""
    timestamp: float
    sync_type: str  # "cut", "transition", "emotional_peak", "level_adjustment"
    audio_adjustment: float = 0.0  # Audio timing adjustment in seconds
    video_adjustment: float = 0.0  # Video timing adjustment in seconds
    tolerance: float = 0.033  # Sync tolerance (1 frame at 30fps)
    priority: str = "medium"  # "low", "medium", "high", "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'sync_type': self.sync_type,
            'audio_adjustment': self.audio_adjustment,
            'video_adjustment': self.video_adjustment,
            'tolerance': self.tolerance,
            'priority': self.priority,
            'metadata': self.metadata
        }


@dataclass
class AudioTrackInfo:
    """Information about an audio track in the composition."""
    track_id: str
    source_path: str
    start_time: float
    duration: float
    volume: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    sync_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'track_id': self.track_id,
            'source_path': self.source_path,
            'start_time': self.start_time,
            'duration': self.duration,
            'volume': self.volume,
            'fade_in': self.fade_in,
            'fade_out': self.fade_out,
            'sync_adjustments': self.sync_adjustments
        }


@dataclass
class SynchronizationResult:
    """Results of audio-video synchronization processing."""
    processing_time: float
    sync_points_processed: int
    adjustments_applied: int
    max_sync_error: float  # Maximum sync error in seconds
    average_sync_error: float  # Average sync error in seconds
    frame_accurate_points: int  # Number of frame-accurate sync points
    
    # Audio track information
    audio_tracks: List[AudioTrackInfo] = field(default_factory=list)
    
    # Movis integration data
    movis_audio_layers: List[Dict[str, Any]] = field(default_factory=list)
    composition_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'processing_time': self.processing_time,
            'sync_points_processed': self.sync_points_processed,
            'adjustments_applied': self.adjustments_applied,
            'max_sync_error': self.max_sync_error,
            'average_sync_error': self.average_sync_error,
            'frame_accurate_points': self.frame_accurate_points,
            'audio_tracks': [track.to_dict() for track in self.audio_tracks],
            'movis_audio_layers': self.movis_audio_layers,
            'composition_settings': self.composition_settings
        }


class TimingAnalyzer:
    """Analyzes timing relationships between audio and video content."""
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.logger = get_logger(__name__)
    
    def analyze_sync_requirements(self, context: ContentContext, 
                                enhancement_result: AudioEnhancementResult) -> List[SyncPoint]:
        """
        Analyze synchronization requirements based on content and enhancement data.
        
        Args:
            context: ContentContext with video and audio analysis
            enhancement_result: Results from audio enhancement
            
        Returns:
            List of SyncPoint objects defining synchronization requirements
        """
        sync_points = []
        
        # Add sync points from enhancement result
        for sync_data in enhancement_result.sync_points:
            sync_point = SyncPoint(
                timestamp=sync_data['timestamp'],
                sync_type=sync_data['type'],
                priority=sync_data.get('priority', 'medium'),
                metadata=sync_data.get('metadata', {})
            )
            sync_points.append(sync_point)
        
        # Add sync points from video editing decisions
        if context.processed_video and 'editing_decisions' in context.processed_video:
            for decision in context.processed_video['editing_decisions']:
                if decision.get('decision_type') in ['cut', 'trim', 'transition']:
                    sync_point = SyncPoint(
                        timestamp=decision.get('timestamp', 0.0),
                        sync_type=decision['decision_type'],
                        priority='high',  # Video edits require precise sync
                        metadata={
                            'decision_id': decision.get('decision_id', ''),
                            'parameters': decision.get('parameters', {}),
                            'rationale': decision.get('rationale', '')
                        }
                    )
                    sync_points.append(sync_point)
        
        # Add sync points for filler word removals
        if context.audio_analysis and hasattr(context.audio_analysis, 'filler_words_detected'):
            for filler_segment in context.audio_analysis.filler_words_detected:
                if filler_segment.should_remove:
                    sync_point = SyncPoint(
                        timestamp=filler_segment.timestamp,
                        sync_type='filler_removal',
                        priority='medium',
                        metadata={
                            'filler_words': filler_segment.filler_words,
                            'original_text': filler_segment.original_text,
                            'cleaned_text': filler_segment.cleaned_text
                        }
                    )
                    sync_points.append(sync_point)
        
        # Sort by timestamp and remove duplicates
        sync_points.sort(key=lambda x: x.timestamp)
        unique_sync_points = self._remove_duplicate_sync_points(sync_points)
        
        self.logger.info(f"Analyzed {len(unique_sync_points)} synchronization requirements")
        return unique_sync_points
    
    def _remove_duplicate_sync_points(self, sync_points: List[SyncPoint]) -> List[SyncPoint]:
        """Remove duplicate sync points that are too close together."""
        if not sync_points:
            return []
        
        unique_points = [sync_points[0]]
        min_distance = self.frame_duration * 2  # Minimum 2 frames apart
        
        for point in sync_points[1:]:
            # Check if this point is too close to the last unique point
            if point.timestamp - unique_points[-1].timestamp >= min_distance:
                unique_points.append(point)
            else:
                # Merge with previous point if same type and higher priority
                last_point = unique_points[-1]
                if (point.sync_type == last_point.sync_type and 
                    self._get_priority_value(point.priority) > self._get_priority_value(last_point.priority)):
                    unique_points[-1] = point
        
        return unique_points
    
    def _get_priority_value(self, priority: str) -> int:
        """Convert priority string to numeric value for comparison."""
        priority_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        return priority_map.get(priority, 2)
    
    def calculate_timing_adjustments(self, sync_points: List[SyncPoint], 
                                   original_duration: float, 
                                   enhanced_duration: float) -> List[SyncPoint]:
        """
        Calculate precise timing adjustments for each sync point.
        
        Args:
            sync_points: List of sync points to process
            original_duration: Original audio duration
            enhanced_duration: Enhanced audio duration
            
        Returns:
            List of sync points with calculated adjustments
        """
        if not sync_points:
            return []
        
        # Calculate global time scaling factor
        time_scale_factor = enhanced_duration / original_duration if original_duration > 0 else 1.0
        
        adjusted_points = []
        
        for point in sync_points:
            adjusted_point = SyncPoint(
                timestamp=point.timestamp,
                sync_type=point.sync_type,
                priority=point.priority,
                tolerance=point.tolerance,
                metadata=point.metadata.copy()
            )
            
            # Calculate adjustments based on sync type
            if point.sync_type == 'cut':
                # Cuts require frame-accurate alignment
                adjusted_point.audio_adjustment = self._snap_to_frame(point.timestamp) - point.timestamp
                adjusted_point.video_adjustment = adjusted_point.audio_adjustment
                adjusted_point.tolerance = self.frame_duration / 2
                
            elif point.sync_type == 'transition':
                # Transitions need smooth alignment
                adjusted_point.audio_adjustment = 0.0  # Keep audio timing
                adjusted_point.video_adjustment = 0.0  # Keep video timing
                adjusted_point.tolerance = self.frame_duration
                
            elif point.sync_type == 'filler_removal':
                # Filler removals create timing gaps that need compensation
                filler_duration = self._estimate_filler_duration(point.metadata)
                adjusted_point.audio_adjustment = -filler_duration  # Audio is shorter
                adjusted_point.video_adjustment = 0.0  # Video timing unchanged
                
            elif point.sync_type == 'level_adjustment':
                # Level adjustments don't affect timing
                adjusted_point.audio_adjustment = 0.0
                adjusted_point.video_adjustment = 0.0
                
            elif point.sync_type == 'emotional_peak':
                # Emotional peaks should be precisely aligned
                adjusted_point.audio_adjustment = 0.0
                adjusted_point.video_adjustment = 0.0
                adjusted_point.tolerance = self.frame_duration / 4  # Very tight tolerance
            
            # Apply global time scaling
            adjusted_point.timestamp *= time_scale_factor
            
            adjusted_points.append(adjusted_point)
        
        self.logger.info(f"Calculated timing adjustments for {len(adjusted_points)} sync points")
        return adjusted_points
    
    def _snap_to_frame(self, timestamp: float) -> float:
        """Snap timestamp to nearest frame boundary."""
        frame_number = round(timestamp * self.fps)
        return frame_number / self.fps
    
    def _estimate_filler_duration(self, metadata: Dict[str, Any]) -> float:
        """Estimate duration of removed filler words."""
        filler_words = metadata.get('filler_words', [])
        # Rough estimate: 0.3 seconds per filler word
        return len(filler_words) * 0.3


class AudioSynchronizer:
    """
    Handles precise audio-video synchronization with movis integration.
    
    Ensures frame-accurate synchronization between enhanced audio and video content,
    managing timing adjustments and creating synchronized movis audio layers.
    """
    
    def __init__(self, fps: float = 30.0, sample_rate: int = 48000):
        """
        Initialize AudioSynchronizer.
        
        Args:
            fps: Video frame rate for synchronization calculations
            sample_rate: Audio sample rate for precise timing
        """
        if not MOVIS_AVAILABLE:
            raise ImportError(
                "movis library is required for AudioSynchronizer. "
                "Install with: pip install movis"
            )
        
        self.logger = get_logger(__name__)
        self.fps = fps
        self.sample_rate = sample_rate
        self.frame_duration = 1.0 / fps
        
        # Initialize timing analyzer
        self.timing_analyzer = TimingAnalyzer(fps)
        
        # Synchronization state
        self.current_context: Optional[ContentContext] = None
        self.sync_result: Optional[SynchronizationResult] = None
        self.audio_tracks: List[AudioTrackInfo] = []
        
        self.logger.info(f"AudioSynchronizer initialized - {fps}fps, {sample_rate}Hz")
    
    def synchronize_audio_video(self, context: ContentContext, 
                              enhancement_result: AudioEnhancementResult) -> SynchronizationResult:
        """
        Perform comprehensive audio-video synchronization.
        
        Args:
            context: ContentContext with video and audio data
            enhancement_result: Results from audio enhancement
            
        Returns:
            SynchronizationResult with synchronization data and movis layers
            
        Raises:
            ContentContextError: If required data is missing
            ProcessingError: If synchronization fails
        """
        start_time = time.time()
        self.current_context = context
        
        try:
            # Validate inputs
            self._validate_synchronization_inputs(context, enhancement_result)
            
            self.logger.info("Starting audio-video synchronization")
            
            # Step 1: Analyze synchronization requirements
            sync_points = self.timing_analyzer.analyze_sync_requirements(context, enhancement_result)
            
            # Step 2: Calculate timing adjustments
            original_duration = enhancement_result.original_duration
            enhanced_duration = enhancement_result.enhanced_duration
            
            adjusted_sync_points = self.timing_analyzer.calculate_timing_adjustments(
                sync_points, original_duration, enhanced_duration
            )
            
            # Step 3: Create audio tracks
            audio_tracks = self._create_audio_tracks(context, enhancement_result, adjusted_sync_points)
            
            # Step 4: Apply synchronization adjustments
            synchronized_tracks = self._apply_synchronization_adjustments(audio_tracks, adjusted_sync_points)
            
            # Step 5: Create movis audio layers
            movis_layers = self._create_movis_audio_layers(synchronized_tracks, context)
            
            # Step 6: Calculate synchronization metrics
            sync_metrics = self._calculate_synchronization_metrics(adjusted_sync_points)
            
            # Create result
            processing_time = time.time() - start_time
            
            result = SynchronizationResult(
                processing_time=processing_time,
                sync_points_processed=len(adjusted_sync_points),
                adjustments_applied=len([p for p in adjusted_sync_points 
                                       if p.audio_adjustment != 0 or p.video_adjustment != 0]),
                max_sync_error=sync_metrics['max_error'],
                average_sync_error=sync_metrics['avg_error'],
                frame_accurate_points=sync_metrics['frame_accurate'],
                audio_tracks=synchronized_tracks,
                movis_audio_layers=movis_layers,
                composition_settings=self._get_composition_settings(context)
            )
            
            self.sync_result = result
            self.audio_tracks = synchronized_tracks
            
            # Update ContentContext with synchronization results
            self._update_context_with_sync_results(context, result)
            
            self.logger.info(
                f"Audio-video synchronization completed in {processing_time:.2f}s - "
                f"{len(adjusted_sync_points)} sync points, "
                f"{result.frame_accurate_points} frame-accurate"
            )
            
            return result
            
        except Exception as e:
            raise ContentContextError(
                f"Audio-video synchronization failed: {str(e)}",
                context_state=context
            )
    
    def _validate_synchronization_inputs(self, context: ContentContext, 
                                       enhancement_result: AudioEnhancementResult):
        """Validate inputs for synchronization processing."""
        if not context.video_files:
            raise ContentContextError(
                "No video files found in ContentContext",
                context_state=context
            )
        
        if not enhancement_result.enhanced_audio_path:
            raise ContentContextError(
                "No enhanced audio path in enhancement result",
                context_state=context
            )
        
        if not Path(enhancement_result.enhanced_audio_path).exists():
            raise ContentContextError(
                f"Enhanced audio file not found: {enhancement_result.enhanced_audio_path}",
                context_state=context
            )
    
    def _create_audio_tracks(self, context: ContentContext, 
                           enhancement_result: AudioEnhancementResult,
                           sync_points: List[SyncPoint]) -> List[AudioTrackInfo]:
        """Create audio track information from context and enhancement data."""
        tracks = []
        
        # Main enhanced audio track
        main_track = AudioTrackInfo(
            track_id="main_audio",
            source_path=enhancement_result.enhanced_audio_path,
            start_time=0.0,
            duration=enhancement_result.enhanced_duration,
            volume=1.0,
            fade_in=0.1,  # 100ms fade in
            fade_out=0.1   # 100ms fade out
        )
        
        # Add sync adjustments from sync points
        for point in sync_points:
            if point.audio_adjustment != 0:
                adjustment = {
                    'timestamp': point.timestamp,
                    'adjustment': point.audio_adjustment,
                    'type': point.sync_type,
                    'priority': point.priority
                }
                main_track.sync_adjustments.append(adjustment)
        
        tracks.append(main_track)
        
        # Add additional audio tracks if needed (e.g., background music, sound effects)
        # This would be expanded based on AI Director decisions
        
        return tracks
    
    def _apply_synchronization_adjustments(self, tracks: List[AudioTrackInfo], 
                                         sync_points: List[SyncPoint]) -> List[AudioTrackInfo]:
        """Apply calculated synchronization adjustments to audio tracks."""
        adjusted_tracks = []
        
        for track in tracks:
            adjusted_track = AudioTrackInfo(
                track_id=track.track_id,
                source_path=track.source_path,
                start_time=track.start_time,
                duration=track.duration,
                volume=track.volume,
                fade_in=track.fade_in,
                fade_out=track.fade_out,
                sync_adjustments=track.sync_adjustments.copy()
            )
            
            # Apply timing adjustments
            cumulative_adjustment = 0.0
            
            for point in sync_points:
                if point.audio_adjustment != 0:
                    # Adjust track timing based on sync point
                    if point.timestamp <= adjusted_track.start_time:
                        # Adjustment before track start - shift entire track
                        adjusted_track.start_time += point.audio_adjustment
                    else:
                        # Adjustment during track - this would require more complex processing
                        # For now, we record it for later processing
                        cumulative_adjustment += point.audio_adjustment
            
            # Apply cumulative adjustment to duration
            adjusted_track.duration = max(0.1, adjusted_track.duration + cumulative_adjustment)
            
            adjusted_tracks.append(adjusted_track)
        
        return adjusted_tracks
    
    def _create_movis_audio_layers(self, tracks: List[AudioTrackInfo], 
                                 context: ContentContext) -> List[Dict[str, Any]]:
        """Create movis audio layer configurations from synchronized tracks."""
        movis_layers = []
        
        for track in tracks:
            try:
                # Create basic movis audio layer configuration
                layer_config = {
                    'layer_id': track.track_id,
                    'layer_type': 'audio',
                    'source_path': track.source_path,
                    'start_time': track.start_time,
                    'duration': track.duration,
                    'volume': track.volume,
                    'fade_in': track.fade_in,
                    'fade_out': track.fade_out,
                    'sync_adjustments': track.sync_adjustments,
                    'movis_params': {
                        'sample_rate': self.sample_rate,
                        'channels': 2,  # Stereo output
                        'format': 'float32'
                    }
                }
                
                movis_layers.append(layer_config)
                
            except Exception as e:
                self.logger.error(f"Failed to create movis layer for track {track.track_id}: {str(e)}")
                continue
        
        self.logger.info(f"Created {len(movis_layers)} movis audio layer configurations")
        return movis_layers
    
    def _calculate_synchronization_metrics(self, sync_points: List[SyncPoint]) -> Dict[str, Any]:
        """Calculate synchronization quality metrics."""
        if not sync_points:
            return {
                'max_error': 0.0,
                'avg_error': 0.0,
                'frame_accurate': 0
            }
        
        # Calculate sync errors
        sync_errors = []
        frame_accurate_count = 0
        
        for point in sync_points:
            # Calculate total adjustment magnitude
            total_adjustment = abs(point.audio_adjustment) + abs(point.video_adjustment)
            sync_errors.append(total_adjustment)
            
            # Check if frame-accurate
            if total_adjustment <= self.frame_duration / 2:
                frame_accurate_count += 1
        
        return {
            'max_error': max(sync_errors) if sync_errors else 0.0,
            'avg_error': np.mean(sync_errors) if sync_errors else 0.0,
            'frame_accurate': frame_accurate_count
        }
    
    def _get_composition_settings(self, context: ContentContext) -> Dict[str, Any]:
        """Get composition settings for movis integration."""
        return {
            'fps': self.fps,
            'sample_rate': self.sample_rate,
            'audio_channels': 2,
            'audio_format': 'float32',
            'sync_tolerance': self.frame_duration / 2,
            'quality_mode': context.user_preferences.quality_mode
        }
    
    def _update_context_with_sync_results(self, context: ContentContext, 
                                        result: SynchronizationResult):
        """Update ContentContext with synchronization results."""
        try:
            # Add processing metrics
            context.processing_metrics.add_module_metrics(
                "audio_synchronization",
                result.processing_time,
                0  # Memory usage tracking would require additional monitoring
            )
            
            # Store synchronization result in processed_video
            if not context.processed_video:
                context.processed_video = {}
            
            context.processed_video['audio_synchronization'] = result.to_dict()
            
            self.logger.info("ContentContext updated with synchronization results")
            
        except Exception as e:
            self.logger.error(f"Failed to update ContentContext with sync results: {str(e)}")
    
    def create_synchronized_movis_composition(self, context: ContentContext) -> Optional[Any]:
        """
        Create a movis composition with synchronized audio layers.
        
        Args:
            context: ContentContext with synchronization data
            
        Returns:
            Configured movis Composition with synchronized audio
        """
        if not self.sync_result:
            self.logger.error("No synchronization result available")
            return None
        
        try:
            # Get composition settings
            settings = self.sync_result.composition_settings
            
            # Create composition
            composition = mv.Composition(
                size=(1920, 1080),  # Default HD resolution
                duration=max(track.duration for track in self.sync_result.audio_tracks),
                fps=settings['fps']
            )
            
            # Add synchronized audio layers
            for layer_config in self.sync_result.movis_audio_layers:
                try:
                    # Create movis audio layer
                    audio_layer = mv.layer.AudioFile(
                        path=layer_config['source_path'],
                        start_time=layer_config['start_time'],
                        duration=layer_config['duration']
                    )
                    
                    # Apply volume adjustment
                    if layer_config['volume'] != 1.0:
                        audio_layer = mv.layer.Volume(layer_config['volume'])(audio_layer)
                    
                    # Apply fade effects
                    if layer_config['fade_in'] > 0:
                        audio_layer = mv.layer.FadeIn(layer_config['fade_in'])(audio_layer)
                    
                    if layer_config['fade_out'] > 0:
                        audio_layer = mv.layer.FadeOut(layer_config['fade_out'])(audio_layer)
                    
                    composition.add_layer(audio_layer)
                    
                except Exception as e:
                    self.logger.error(f"Failed to add audio layer {layer_config['layer_id']}: {str(e)}")
                    continue
            
            self.logger.info("Created synchronized movis composition")
            return composition
            
        except Exception as e:
            self.logger.error(f"Failed to create synchronized movis composition: {str(e)}")
            return None
    
    def get_synchronization_report(self) -> Dict[str, Any]:
        """Get detailed synchronization report."""
        if not self.sync_result:
            return {}
        
        return {
            'summary': {
                'processing_time': self.sync_result.processing_time,
                'sync_points_processed': self.sync_result.sync_points_processed,
                'adjustments_applied': self.sync_result.adjustments_applied,
                'frame_accurate_percentage': (
                    self.sync_result.frame_accurate_points / 
                    max(1, self.sync_result.sync_points_processed) * 100
                )
            },
            'quality_metrics': {
                'max_sync_error': self.sync_result.max_sync_error,
                'average_sync_error': self.sync_result.average_sync_error,
                'frame_accurate_points': self.sync_result.frame_accurate_points
            },
            'audio_tracks': [track.to_dict() for track in self.sync_result.audio_tracks],
            'composition_settings': self.sync_result.composition_settings
        }