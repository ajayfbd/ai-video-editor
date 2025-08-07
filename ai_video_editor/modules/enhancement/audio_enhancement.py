"""
Audio Enhancement Engine - Advanced audio cleanup and enhancement pipeline.

This module implements comprehensive audio enhancement including noise reduction,
dynamic level adjustment, and integration with existing filler word removal.
Designed to work seamlessly with the movis-based video composition system.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    import librosa
    import librosa.display
    import soundfile as sf
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
    from scipy import signal
    from scipy.signal import butter, filtfilt
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    librosa = None
    sf = None
    AudioSegment = None

from ...core.content_context import ContentContext, AudioAnalysisResult, EmotionalPeak
from ...core.exceptions import ProcessingError, ResourceConstraintError, ContentContextError
from ...utils.logging_config import get_logger


@dataclass
class AudioEnhancementSettings:
    """Configuration settings for audio enhancement."""
    
    # Noise reduction settings
    noise_reduction_strength: float = 0.5  # 0.0 to 1.0
    spectral_gate_threshold: float = -20.0  # dB
    
    # Dynamic level adjustment
    enable_dynamic_levels: bool = True
    emotional_boost_factor: float = 1.2  # Boost factor for emotional peaks
    explanation_boost_factor: float = 1.1  # Boost for explanation segments
    filler_reduction_factor: float = 0.7  # Reduction for filler segments
    
    # Audio normalization
    target_lufs: float = -16.0  # Target loudness (LUFS)
    peak_limit: float = -1.0  # Peak limiter threshold (dB)
    
    # Compression settings
    compression_ratio: float = 3.0
    compression_threshold: float = -12.0  # dB
    attack_time: float = 0.003  # seconds
    release_time: float = 0.1  # seconds
    
    # EQ settings
    enable_eq: bool = True
    high_pass_freq: float = 80.0  # Hz - remove low frequency noise
    presence_boost_freq: float = 3000.0  # Hz - boost speech clarity
    presence_boost_gain: float = 2.0  # dB
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'noise_reduction_strength': self.noise_reduction_strength,
            'spectral_gate_threshold': self.spectral_gate_threshold,
            'enable_dynamic_levels': self.enable_dynamic_levels,
            'emotional_boost_factor': self.emotional_boost_factor,
            'explanation_boost_factor': self.explanation_boost_factor,
            'filler_reduction_factor': self.filler_reduction_factor,
            'target_lufs': self.target_lufs,
            'peak_limit': self.peak_limit,
            'compression_ratio': self.compression_ratio,
            'compression_threshold': self.compression_threshold,
            'attack_time': self.attack_time,
            'release_time': self.release_time,
            'enable_eq': self.enable_eq,
            'high_pass_freq': self.high_pass_freq,
            'presence_boost_freq': self.presence_boost_freq,
            'presence_boost_gain': self.presence_boost_gain
        }


@dataclass
class AudioEnhancementResult:
    """Results of audio enhancement processing."""
    
    # Processing metrics
    processing_time: float
    original_duration: float
    enhanced_duration: float
    
    # Enhancement statistics
    noise_reduction_applied: bool
    dynamic_adjustments_made: int
    peak_reductions: int
    level_boosts: int
    
    # Quality metrics
    snr_improvement: float  # Signal-to-noise ratio improvement in dB
    dynamic_range_improvement: float  # Dynamic range improvement in dB
    loudness_consistency_score: float  # 0.0 to 1.0
    
    # File paths
    enhanced_audio_path: Optional[str] = None
    analysis_data_path: Optional[str] = None
    
    # Integration data
    sync_points: List[Dict[str, Any]] = field(default_factory=list)
    level_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'processing_time': self.processing_time,
            'original_duration': self.original_duration,
            'enhanced_duration': self.enhanced_duration,
            'noise_reduction_applied': self.noise_reduction_applied,
            'dynamic_adjustments_made': self.dynamic_adjustments_made,
            'peak_reductions': self.peak_reductions,
            'level_boosts': self.level_boosts,
            'snr_improvement': self.snr_improvement,
            'dynamic_range_improvement': self.dynamic_range_improvement,
            'loudness_consistency_score': self.loudness_consistency_score,
            'enhanced_audio_path': self.enhanced_audio_path,
            'analysis_data_path': self.analysis_data_path,
            'sync_points': self.sync_points,
            'level_adjustments': self.level_adjustments
        }


class AudioCleanupPipeline:
    """Advanced audio cleanup pipeline with noise reduction and normalization."""
    
    def __init__(self, settings: AudioEnhancementSettings):
        self.settings = settings
        self.logger = get_logger(__name__)
    
    def apply_noise_reduction(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply spectral noise reduction to audio data.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Noise-reduced audio data
        """
        if not AUDIO_LIBS_AVAILABLE:
            self.logger.warning("Audio libraries not available, skipping noise reduction")
            return audio_data
        
        try:
            # Estimate noise profile from first 0.5 seconds (assumed to be quiet)
            noise_sample_length = min(int(0.5 * sample_rate), len(audio_data) // 4)
            noise_sample = audio_data[:noise_sample_length]
            
            # Compute spectral statistics
            noise_fft = np.fft.rfft(noise_sample)
            noise_magnitude = np.abs(noise_fft)
            noise_phase = np.angle(noise_fft)
            
            # Apply spectral gating
            audio_fft = np.fft.rfft(audio_data)
            audio_magnitude = np.abs(audio_fft)
            audio_phase = np.angle(audio_fft)
            
            # Calculate noise threshold
            noise_threshold = np.mean(noise_magnitude) * (10 ** (self.settings.spectral_gate_threshold / 20))
            
            # Apply noise reduction
            reduction_factor = self.settings.noise_reduction_strength
            mask = audio_magnitude > noise_threshold
            
            # Ensure noise_magnitude matches audio_magnitude length
            if len(noise_magnitude) != len(audio_magnitude):
                # Repeat or truncate noise_magnitude to match audio_magnitude
                if len(noise_magnitude) < len(audio_magnitude):
                    # Repeat the noise pattern
                    repeat_factor = len(audio_magnitude) // len(noise_magnitude) + 1
                    noise_magnitude_extended = np.tile(noise_magnitude, repeat_factor)[:len(audio_magnitude)]
                else:
                    # Truncate to match
                    noise_magnitude_extended = noise_magnitude[:len(audio_magnitude)]
            else:
                noise_magnitude_extended = noise_magnitude
            
            enhanced_magnitude = np.where(
                mask,
                audio_magnitude,
                audio_magnitude * (1 - reduction_factor) + noise_magnitude_extended * reduction_factor
            )
            
            # Reconstruct audio
            enhanced_fft = enhanced_magnitude * np.exp(1j * audio_phase)
            enhanced_audio = np.fft.irfft(enhanced_fft, n=len(audio_data))
            
            self.logger.info(f"Applied noise reduction with strength {reduction_factor}")
            return enhanced_audio.astype(audio_data.dtype)
            
        except Exception as e:
            self.logger.error(f"Noise reduction failed: {str(e)}")
            return audio_data
    
    def apply_eq_filtering(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply EQ filtering for speech clarity.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            EQ-filtered audio data
        """
        if not self.settings.enable_eq:
            return audio_data
        
        try:
            # High-pass filter to remove low frequency noise
            nyquist = sample_rate / 2
            high_pass_normalized = self.settings.high_pass_freq / nyquist
            
            if high_pass_normalized < 1.0:
                b, a = butter(4, high_pass_normalized, btype='high')
                audio_data = filtfilt(b, a, audio_data)
            
            # Presence boost for speech clarity
            presence_normalized = self.settings.presence_boost_freq / nyquist
            if presence_normalized < 1.0:
                # Simple peaking EQ implementation
                gain_linear = 10 ** (self.settings.presence_boost_gain / 20)
                q_factor = 2.0
                
                # Calculate filter coefficients for peaking EQ
                w0 = 2 * np.pi * self.settings.presence_boost_freq / sample_rate
                cos_w0 = np.cos(w0)
                sin_w0 = np.sin(w0)
                alpha = sin_w0 / (2 * q_factor)
                A = gain_linear
                
                # Peaking EQ coefficients
                b0 = 1 + alpha * A
                b1 = -2 * cos_w0
                b2 = 1 - alpha * A
                a0 = 1 + alpha / A
                a1 = -2 * cos_w0
                a2 = 1 - alpha / A
                
                # Normalize coefficients
                b = np.array([b0, b1, b2]) / a0
                a = np.array([1, a1 / a0, a2 / a0])
                
                audio_data = filtfilt(b, a, audio_data)
            
            self.logger.info("Applied EQ filtering for speech clarity")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"EQ filtering failed: {str(e)}")
            return audio_data
    
    def apply_dynamic_range_compression(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply dynamic range compression for consistent levels.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Compressed audio data
        """
        try:
            # Convert to dB scale for processing
            audio_db = 20 * np.log10(np.abs(audio_data) + 1e-10)
            
            # Apply compression
            threshold_db = self.settings.compression_threshold
            ratio = self.settings.compression_ratio
            
            # Simple compression algorithm
            compressed_db = np.where(
                audio_db > threshold_db,
                threshold_db + (audio_db - threshold_db) / ratio,
                audio_db
            )
            
            # Convert back to linear scale
            gain_adjustment = compressed_db - audio_db
            gain_linear = 10 ** (gain_adjustment / 20)
            
            compressed_audio = audio_data * gain_linear
            
            self.logger.info(f"Applied compression with ratio {ratio}:1")
            return compressed_audio
            
        except Exception as e:
            self.logger.error(f"Dynamic range compression failed: {str(e)}")
            return audio_data


class DynamicLevelAdjuster:
    """Handles dynamic audio level adjustment based on content analysis."""
    
    def __init__(self, settings: AudioEnhancementSettings):
        self.settings = settings
        self.logger = get_logger(__name__)
    
    def calculate_level_adjustments(self, context: ContentContext) -> List[Dict[str, Any]]:
        """
        Calculate level adjustments based on content analysis.
        
        Args:
            context: ContentContext with audio analysis and emotional data
            
        Returns:
            List of level adjustment instructions
        """
        adjustments = []
        
        if not context.audio_analysis or not self.settings.enable_dynamic_levels:
            return adjustments
        
        # Process emotional peaks
        for emotion in context.emotional_markers:
            if emotion.intensity > 0.7:  # High intensity emotions
                adjustment = {
                    'timestamp': emotion.timestamp,
                    'duration': 2.0,  # 2-second adjustment window
                    'type': 'emotional_boost',
                    'factor': self.settings.emotional_boost_factor,
                    'reason': f"Boost for {emotion.emotion} (intensity: {emotion.intensity:.2f})"
                }
                adjustments.append(adjustment)
        
        # Process explanation segments
        for segment in context.audio_analysis.explanation_segments:
            adjustment = {
                'timestamp': segment['timestamp'],
                'duration': 3.0,  # 3-second adjustment window
                'type': 'explanation_boost',
                'factor': self.settings.explanation_boost_factor,
                'reason': "Boost for explanation segment"
            }
            adjustments.append(adjustment)
        
        # Process filler word segments
        if hasattr(context.audio_analysis, 'filler_words_detected'):
            for filler_segment in context.audio_analysis.filler_words_detected:
                if filler_segment.should_remove:
                    adjustment = {
                        'timestamp': filler_segment.timestamp,
                        'duration': 1.0,  # 1-second adjustment window
                        'type': 'filler_reduction',
                        'factor': self.settings.filler_reduction_factor,
                        'reason': f"Reduce for filler words: {', '.join(filler_segment.filler_words)}"
                    }
                    adjustments.append(adjustment)
        
        # Sort by timestamp
        adjustments.sort(key=lambda x: x['timestamp'])
        
        self.logger.info(f"Calculated {len(adjustments)} dynamic level adjustments")
        return adjustments
    
    def apply_level_adjustments(self, audio_data: np.ndarray, sample_rate: int, 
                              adjustments: List[Dict[str, Any]]) -> np.ndarray:
        """
        Apply calculated level adjustments to audio data.
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            adjustments: List of adjustment instructions
            
        Returns:
            Audio data with level adjustments applied
        """
        if not adjustments:
            return audio_data
        
        try:
            enhanced_audio = audio_data.copy()
            adjustments_applied = 0
            
            for adjustment in adjustments:
                start_sample = int(adjustment['timestamp'] * sample_rate)
                duration_samples = int(adjustment['duration'] * sample_rate)
                end_sample = min(start_sample + duration_samples, len(enhanced_audio))
                
                if start_sample < len(enhanced_audio):
                    # Apply adjustment with smooth fade in/out
                    factor = adjustment['factor']
                    segment = enhanced_audio[start_sample:end_sample]
                    
                    # Create smooth envelope
                    envelope_length = min(int(0.1 * sample_rate), len(segment) // 4)  # 100ms fade
                    envelope = np.ones(len(segment))
                    
                    if len(segment) > 2 * envelope_length:
                        # Fade in
                        envelope[:envelope_length] = np.linspace(1.0, factor, envelope_length)
                        # Sustain
                        envelope[envelope_length:-envelope_length] = factor
                        # Fade out
                        envelope[-envelope_length:] = np.linspace(factor, 1.0, envelope_length)
                    else:
                        # Short segment - linear interpolation
                        envelope = np.linspace(1.0, factor, len(segment))
                    
                    enhanced_audio[start_sample:end_sample] = segment * envelope
                    adjustments_applied += 1
            
            self.logger.info(f"Applied {adjustments_applied} level adjustments")
            return enhanced_audio
            
        except Exception as e:
            self.logger.error(f"Level adjustment application failed: {str(e)}")
            return audio_data


class AudioEnhancementEngine:
    """
    Main audio enhancement engine that orchestrates all enhancement operations.
    
    Integrates with existing filler word removal and provides comprehensive
    audio enhancement for the movis-based video composition system.
    """
    
    def __init__(self, output_dir: str = "temp/audio_enhancement", 
                 settings: Optional[AudioEnhancementSettings] = None):
        """
        Initialize AudioEnhancementEngine.
        
        Args:
            output_dir: Directory for enhanced audio files
            settings: Enhancement settings (uses defaults if None)
        """
        if not AUDIO_LIBS_AVAILABLE:
            raise ImportError(
                "Audio processing libraries are required. "
                "Install with: pip install librosa pydub scipy"
            )
        
        self.logger = get_logger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.settings = settings or AudioEnhancementSettings()
        
        # Initialize processing components
        self.cleanup_pipeline = AudioCleanupPipeline(self.settings)
        self.level_adjuster = DynamicLevelAdjuster(self.settings)
        
        # Processing state
        self.current_context: Optional[ContentContext] = None
        self.enhancement_result: Optional[AudioEnhancementResult] = None
        
        self.logger.info("AudioEnhancementEngine initialized")
    
    def enhance_audio(self, context: ContentContext, 
                     audio_file_path: Optional[str] = None) -> AudioEnhancementResult:
        """
        Perform comprehensive audio enhancement on ContentContext audio data.
        
        Args:
            context: ContentContext with audio analysis and emotional data
            audio_file_path: Optional path to audio file (extracted from video if None)
            
        Returns:
            AudioEnhancementResult with enhanced audio and processing metrics
            
        Raises:
            ContentContextError: If audio data is missing or invalid
            ProcessingError: If enhancement processing fails
        """
        start_time = time.time()
        self.current_context = context
        
        try:
            # Validate input
            if not context.audio_analysis:
                raise ContentContextError(
                    "No audio analysis found in ContentContext",
                    context_state=context
                )
            
            # Determine audio source
            if not audio_file_path and context.video_files:
                audio_file_path = context.video_files[0]  # Extract from first video file
            
            if not audio_file_path:
                raise ContentContextError(
                    "No audio source available for enhancement",
                    context_state=context
                )
            
            self.logger.info(f"Starting audio enhancement for: {audio_file_path}")
            
            # Load audio data
            audio_data, sample_rate = self._load_audio_data(audio_file_path)
            original_duration = len(audio_data) / sample_rate
            
            # Initialize metrics
            noise_reduction_applied = False
            dynamic_adjustments_made = 0
            peak_reductions = 0
            level_boosts = 0
            
            # Step 1: Apply noise reduction
            if self.settings.noise_reduction_strength > 0:
                self.logger.info("Applying noise reduction")
                enhanced_audio = self.cleanup_pipeline.apply_noise_reduction(audio_data, sample_rate)
                noise_reduction_applied = True
            else:
                enhanced_audio = audio_data.copy()
            
            # Step 2: Apply EQ filtering
            enhanced_audio = self.cleanup_pipeline.apply_eq_filtering(enhanced_audio, sample_rate)
            
            # Step 3: Calculate and apply dynamic level adjustments
            level_adjustments = self.level_adjuster.calculate_level_adjustments(context)
            if level_adjustments:
                enhanced_audio = self.level_adjuster.apply_level_adjustments(
                    enhanced_audio, sample_rate, level_adjustments
                )
                dynamic_adjustments_made = len(level_adjustments)
                
                # Count boost vs reduction adjustments
                for adj in level_adjustments:
                    if adj['factor'] > 1.0:
                        level_boosts += 1
                    elif adj['factor'] < 1.0:
                        peak_reductions += 1
            
            # Step 4: Apply dynamic range compression
            enhanced_audio = self.cleanup_pipeline.apply_dynamic_range_compression(
                enhanced_audio, sample_rate
            )
            
            # Step 5: Final normalization
            enhanced_audio = self._apply_final_normalization(enhanced_audio)
            
            # Calculate quality metrics
            snr_improvement = self._calculate_snr_improvement(audio_data, enhanced_audio)
            dynamic_range_improvement = self._calculate_dynamic_range_improvement(
                audio_data, enhanced_audio
            )
            loudness_consistency_score = self._calculate_loudness_consistency(enhanced_audio)
            
            # Save enhanced audio
            enhanced_audio_path = self._save_enhanced_audio(
                enhanced_audio, sample_rate, context.project_id
            )
            
            # Create sync points for movis integration
            sync_points = self._create_sync_points(context, level_adjustments)
            
            # Create result
            processing_time = time.time() - start_time
            enhanced_duration = len(enhanced_audio) / sample_rate
            
            result = AudioEnhancementResult(
                processing_time=processing_time,
                original_duration=original_duration,
                enhanced_duration=enhanced_duration,
                noise_reduction_applied=noise_reduction_applied,
                dynamic_adjustments_made=dynamic_adjustments_made,
                peak_reductions=peak_reductions,
                level_boosts=level_boosts,
                snr_improvement=snr_improvement,
                dynamic_range_improvement=dynamic_range_improvement,
                loudness_consistency_score=loudness_consistency_score,
                enhanced_audio_path=enhanced_audio_path,
                sync_points=sync_points,
                level_adjustments=level_adjustments
            )
            
            self.enhancement_result = result
            
            # Update ContentContext with enhancement results
            self._update_context_with_results(context, result)
            
            self.logger.info(
                f"Audio enhancement completed in {processing_time:.2f}s - "
                f"SNR improved by {snr_improvement:.1f}dB, "
                f"{dynamic_adjustments_made} level adjustments applied"
            )
            
            return result
            
        except Exception as e:
            raise ProcessingError(
                f"Audio enhancement failed: {str(e)}"
            )
    
    def _load_audio_data(self, audio_file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio data from file."""
        try:
            if Path(audio_file_path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                # Extract audio from video file
                audio_data, sample_rate = librosa.load(audio_file_path, sr=None, mono=True)
            else:
                # Load audio file directly
                audio_data, sample_rate = librosa.load(audio_file_path, sr=None, mono=True)
            
            self.logger.info(f"Loaded audio: {len(audio_data)} samples at {sample_rate}Hz")
            return audio_data, sample_rate
            
        except Exception as e:
            raise ProcessingError(f"Failed to load audio from {audio_file_path}: {str(e)}")
    
    def _apply_final_normalization(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply final normalization to prevent clipping."""
        try:
            # Peak normalization to prevent clipping
            peak_level = np.max(np.abs(audio_data))
            if peak_level > 0:
                target_peak = 10 ** (self.settings.peak_limit / 20)
                if peak_level > target_peak:
                    audio_data = audio_data * (target_peak / peak_level)
            
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Final normalization failed: {str(e)}")
            return audio_data
    
    def _calculate_snr_improvement(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate signal-to-noise ratio improvement."""
        try:
            # Simple SNR estimation based on dynamic range
            original_dynamic_range = np.max(original) - np.min(original)
            enhanced_dynamic_range = np.max(enhanced) - np.min(enhanced)
            
            if original_dynamic_range > 0:
                improvement = 20 * np.log10(enhanced_dynamic_range / original_dynamic_range)
                return max(0.0, improvement)  # Only positive improvements
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_dynamic_range_improvement(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate dynamic range improvement."""
        try:
            original_std = np.std(original)
            enhanced_std = np.std(enhanced)
            
            if original_std > 0:
                improvement = 20 * np.log10(enhanced_std / original_std)
                return improvement
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_loudness_consistency(self, audio_data: np.ndarray) -> float:
        """Calculate loudness consistency score (0.0 to 1.0)."""
        try:
            # Calculate RMS levels in overlapping windows
            window_size = len(audio_data) // 20  # 20 windows
            if window_size < 1024:
                return 1.0  # Too short to analyze
            
            rms_levels = []
            for i in range(0, len(audio_data) - window_size, window_size // 2):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                if rms > 0:
                    rms_levels.append(20 * np.log10(rms))
            
            if len(rms_levels) < 2:
                return 1.0
            
            # Calculate consistency as inverse of standard deviation
            std_dev = np.std(rms_levels)
            consistency = 1.0 / (1.0 + std_dev / 10.0)  # Normalize to 0-1 range
            
            return min(1.0, max(0.0, consistency))
            
        except Exception:
            return 0.5  # Default moderate consistency
    
    def _save_enhanced_audio(self, audio_data: np.ndarray, sample_rate: int, 
                           project_id: str) -> str:
        """Save enhanced audio to file."""
        try:
            output_path = self.output_dir / f"{project_id}_enhanced.wav"
            
            # Convert to 16-bit integer for saving
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Use soundfile to save
            sf.write(str(output_path), audio_data, sample_rate)
            
            self.logger.info(f"Enhanced audio saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save enhanced audio: {str(e)}")
            return ""
    
    def _create_sync_points(self, context: ContentContext, 
                          level_adjustments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create synchronization points for movis integration."""
        sync_points = []
        
        # Add sync points for emotional peaks
        for emotion in context.emotional_markers:
            sync_point = {
                'timestamp': emotion.timestamp,
                'type': 'emotional_peak',
                'priority': 'high' if emotion.intensity > 0.8 else 'medium',
                'metadata': {
                    'emotion': emotion.emotion,
                    'intensity': emotion.intensity,
                    'context': emotion.context
                }
            }
            sync_points.append(sync_point)
        
        # Add sync points for level adjustments
        for adjustment in level_adjustments:
            sync_point = {
                'timestamp': adjustment['timestamp'],
                'type': 'level_adjustment',
                'priority': 'medium',
                'metadata': {
                    'adjustment_type': adjustment['type'],
                    'factor': adjustment['factor'],
                    'reason': adjustment['reason']
                }
            }
            sync_points.append(sync_point)
        
        # Sort by timestamp
        sync_points.sort(key=lambda x: x['timestamp'])
        
        return sync_points
    
    def _update_context_with_results(self, context: ContentContext, 
                                   result: AudioEnhancementResult):
        """Update ContentContext with enhancement results."""
        try:
            # Update audio analysis with enhancement data
            if context.audio_analysis:
                context.audio_analysis.enhanced_duration = result.enhanced_duration
                context.audio_analysis.quality_improvement_score = (
                    result.snr_improvement / 10.0 +  # Normalize SNR improvement
                    result.loudness_consistency_score +
                    (1.0 if result.noise_reduction_applied else 0.0)
                ) / 3.0
            
            # Add processing metrics
            context.processing_metrics.add_module_metrics(
                "audio_enhancement",
                result.processing_time,
                0  # Memory usage tracking would require additional monitoring
            )
            
            # Store enhancement result in processed_video for downstream use
            if not context.processed_video:
                context.processed_video = {}
            
            context.processed_video['audio_enhancement'] = result.to_dict()
            
            self.logger.info("ContentContext updated with enhancement results")
            
        except Exception as e:
            self.logger.error(f"Failed to update ContentContext: {str(e)}")
    
    def get_enhancement_settings(self) -> AudioEnhancementSettings:
        """Get current enhancement settings."""
        return self.settings
    
    def update_enhancement_settings(self, new_settings: AudioEnhancementSettings):
        """Update enhancement settings."""
        self.settings = new_settings
        self.cleanup_pipeline.settings = new_settings
        self.level_adjuster.settings = new_settings
        self.logger.info("Enhancement settings updated")