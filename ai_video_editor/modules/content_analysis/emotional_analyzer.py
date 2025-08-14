"""
Emotional and Engagement Analysis Module.

This module implements emotional peak detection combining audio and visual cues,
engagement prediction based on content patterns, and integration with the
ContentContext system for unified emotional insights.
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ai_video_editor.core.content_context import (
    ContentContext, EmotionalPeak, AudioAnalysisResult, VisualHighlight
)
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.exceptions import ContentContextError
from ai_video_editor.utils.logging_config import get_logger


class EmotionType(Enum):
    """Enumeration of detectable emotion types."""
    EXCITEMENT = "excitement"
    CURIOSITY = "curiosity"
    CONCERN = "concern"
    CONFIDENCE = "confidence"
    SURPRISE = "surprise"
    SATISFACTION = "satisfaction"
    ANTICIPATION = "anticipation"
    FOCUS = "focus"


@dataclass
class EmotionalPattern:
    """Pattern for detecting specific emotions in content."""
    emotion: EmotionType
    audio_keywords: List[str]
    intensity_multipliers: Dict[str, float]
    visual_indicators: List[str]
    context_patterns: List[str]
    confidence_threshold: float = 0.6
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'emotion': self.emotion.value,
            'audio_keywords': self.audio_keywords,
            'intensity_multipliers': self.intensity_multipliers,
            'visual_indicators': self.visual_indicators,
            'context_patterns': self.context_patterns,
            'confidence_threshold': self.confidence_threshold
        }


@dataclass
class EngagementMetrics:
    """Metrics for predicting content engagement."""
    emotional_variety_score: float  # 0.0 to 1.0
    peak_intensity_score: float     # 0.0 to 1.0
    pacing_score: float            # 0.0 to 1.0
    visual_engagement_score: float  # 0.0 to 1.0
    audio_clarity_score: float     # 0.0 to 1.0
    overall_engagement_score: float # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'emotional_variety_score': self.emotional_variety_score,
            'peak_intensity_score': self.peak_intensity_score,
            'pacing_score': self.pacing_score,
            'visual_engagement_score': self.visual_engagement_score,
            'audio_clarity_score': self.audio_clarity_score,
            'overall_engagement_score': self.overall_engagement_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EngagementMetrics':
        return cls(**data)


@dataclass
class EmotionalAnalysisResult:
    """Complete emotional analysis results."""
    detected_peaks: List[EmotionalPeak]
    engagement_metrics: EngagementMetrics
    emotional_timeline: List[Dict[str, Any]]
    cross_modal_correlations: Dict[str, float]
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'detected_peaks': [peak.to_dict() for peak in self.detected_peaks],
            'engagement_metrics': self.engagement_metrics.to_dict(),
            'emotional_timeline': self.emotional_timeline,
            'cross_modal_correlations': self.cross_modal_correlations,
            'processing_time': self.processing_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalAnalysisResult':
        detected_peaks = [EmotionalPeak.from_dict(peak) for peak in data['detected_peaks']]
        engagement_metrics = EngagementMetrics.from_dict(data['engagement_metrics'])
        return cls(
            detected_peaks=detected_peaks,
            engagement_metrics=engagement_metrics,
            emotional_timeline=data['emotional_timeline'],
            cross_modal_correlations=data['cross_modal_correlations'],
            processing_time=data['processing_time']
        )


class EmotionalAnalyzer:
    """
    Emotional and engagement analysis combining audio and visual cues.
    
    Detects emotional peaks from multiple modalities and predicts engagement
    based on content patterns, integrating with the ContentContext system.
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None, memory_client=None):
        """Initialize EmotionalAnalyzer with optional caching and Memory integration."""
        self.logger = get_logger(__name__)
        self.cache_manager = cache_manager
        self.memory_client = memory_client
        
        # Initialize emotional patterns
        self.emotional_patterns = self._initialize_emotional_patterns()
        
        # Analysis parameters
        self.min_peak_intensity = 0.5
        self.peak_merge_threshold = 3.0  # seconds
        self.engagement_window_size = 10.0  # seconds for engagement analysis
        
        # Memory-based learning parameters
        self.pattern_weights = {}
        self.engagement_success_patterns = {}
        
        # Load existing patterns from Memory
        self._load_emotional_patterns()
        
        self.logger.info("EmotionalAnalyzer initialized successfully")
    
    def _initialize_emotional_patterns(self) -> Dict[EmotionType, EmotionalPattern]:
        """Initialize emotional detection patterns."""
        patterns = {
            EmotionType.EXCITEMENT: EmotionalPattern(
                emotion=EmotionType.EXCITEMENT,
                audio_keywords=[
                    'amazing', 'incredible', 'fantastic', 'wow', 'great', 'excellent',
                    'outstanding', 'awesome', 'brilliant', 'spectacular', 'wonderful',
                    'thrilled', 'excited', 'love', 'perfect'
                ],
                intensity_multipliers={
                    'amazing': 1.0, 'incredible': 0.95, 'fantastic': 0.9,
                    'wow': 0.8, 'awesome': 0.85, 'perfect': 0.9
                },
                visual_indicators=[
                    'animated_gesture', 'bright_lighting', 'dynamic_movement',
                    'expressive_face', 'energetic_posture'
                ],
                context_patterns=[
                    'achievement', 'breakthrough', 'success', 'discovery',
                    'positive_outcome', 'milestone'
                ]
            ),
            
            EmotionType.CURIOSITY: EmotionalPattern(
                emotion=EmotionType.CURIOSITY,
                audio_keywords=[
                    'interesting', 'wonder', 'question', 'why', 'how', 'what if',
                    'curious', 'explore', 'discover', 'investigate', 'examine',
                    'fascinating', 'intriguing', 'puzzling'
                ],
                intensity_multipliers={
                    'fascinating': 0.9, 'intriguing': 0.8, 'wonder': 0.7,
                    'curious': 0.75, 'interesting': 0.6
                },
                visual_indicators=[
                    'thoughtful_expression', 'head_tilt', 'focused_gaze',
                    'questioning_gesture', 'exploratory_movement'
                ],
                context_patterns=[
                    'mystery', 'unknown', 'investigation', 'research',
                    'hypothesis', 'theory'
                ]
            ),
            
            EmotionType.CONCERN: EmotionalPattern(
                emotion=EmotionType.CONCERN,
                audio_keywords=[
                    'careful', 'warning', 'risk', 'danger', 'problem', 'issue',
                    'trouble', 'worried', 'concerned', 'caution', 'beware',
                    'alert', 'serious', 'critical'
                ],
                intensity_multipliers={
                    'danger': 1.0, 'critical': 0.95, 'warning': 0.8,
                    'serious': 0.85, 'careful': 0.6
                },
                visual_indicators=[
                    'serious_expression', 'furrowed_brow', 'cautionary_gesture',
                    'tense_posture', 'alert_stance'
                ],
                context_patterns=[
                    'risk_assessment', 'precaution', 'safety', 'protection',
                    'prevention', 'mitigation'
                ]
            ),
            
            EmotionType.CONFIDENCE: EmotionalPattern(
                emotion=EmotionType.CONFIDENCE,
                audio_keywords=[
                    'definitely', 'certainly', 'absolutely', 'sure', 'confident',
                    'guaranteed', 'proven', 'established', 'confirmed',
                    'undoubtedly', 'clearly', 'obviously'
                ],
                intensity_multipliers={
                    'absolutely': 1.0, 'definitely': 0.9, 'certainly': 0.85,
                    'guaranteed': 0.95, 'confident': 0.8
                },
                visual_indicators=[
                    'upright_posture', 'direct_gaze', 'assertive_gesture',
                    'stable_stance', 'authoritative_presence'
                ],
                context_patterns=[
                    'expertise', 'authority', 'experience', 'knowledge',
                    'mastery', 'competence'
                ]
            ),
            
            EmotionType.SURPRISE: EmotionalPattern(
                emotion=EmotionType.SURPRISE,
                audio_keywords=[
                    'surprising', 'unexpected', 'shocking', 'unbelievable',
                    'astonishing', 'remarkable', 'startling', 'stunning',
                    'mind-blowing', 'jaw-dropping'
                ],
                intensity_multipliers={
                    'shocking': 1.0, 'astonishing': 0.95, 'stunning': 0.9,
                    'surprising': 0.7, 'unexpected': 0.75
                },
                visual_indicators=[
                    'wide_eyes', 'raised_eyebrows', 'open_mouth',
                    'sudden_movement', 'startled_reaction'
                ],
                context_patterns=[
                    'revelation', 'plot_twist', 'discovery', 'breakthrough',
                    'contradiction', 'anomaly'
                ]
            ),
            
            EmotionType.SATISFACTION: EmotionalPattern(
                emotion=EmotionType.SATISFACTION,
                audio_keywords=[
                    'satisfied', 'pleased', 'content', 'fulfilled', 'accomplished',
                    'achieved', 'completed', 'successful', 'rewarding',
                    'gratifying', 'worthwhile'
                ],
                intensity_multipliers={
                    'accomplished': 0.9, 'fulfilled': 0.85, 'successful': 0.8,
                    'satisfied': 0.75, 'pleased': 0.7
                },
                visual_indicators=[
                    'relaxed_expression', 'gentle_smile', 'calm_posture',
                    'peaceful_demeanor', 'content_look'
                ],
                context_patterns=[
                    'completion', 'achievement', 'goal_reached', 'success',
                    'resolution', 'closure'
                ]
            )
        }
        
        return patterns
    
    def _load_emotional_patterns(self):
        """Load emotional analysis patterns from Memory."""
        if not self.memory_client:
            return
        
        try:
            search_results = self.memory_client.search_nodes("emotional analysis patterns")
            
            for result in search_results.get('nodes', []):
                if result['name'] == 'Emotional Analysis Patterns':
                    for observation in result.get('observations', []):
                        self._parse_emotional_pattern(observation)
            
            self.logger.info("Loaded emotional patterns from Memory")
            
        except Exception as e:
            self.logger.warning(f"Failed to load emotional patterns from Memory: {e}")
    
    def _parse_emotional_pattern(self, observation: str):
        """Parse emotional pattern observation from Memory."""
        try:
            # Extract pattern weights based on historical success
            if 'emotion_detection_accuracy' in observation:
                parts = observation.split()
                for i, part in enumerate(parts):
                    if part in ['excitement', 'curiosity', 'concern', 'confidence']:
                        emotion = part
                        if i + 2 < len(parts) and parts[i + 1] == 'accuracy':
                            accuracy = float(parts[i + 2].rstrip('%')) / 100
                            self.pattern_weights[emotion] = max(0.5, min(2.0, accuracy * 2))
                            break
            
            elif 'engagement_prediction_success' in observation:
                # Update engagement prediction parameters
                if 'high_engagement' in observation:
                    self.min_peak_intensity = max(0.3, self.min_peak_intensity - 0.05)
                elif 'low_engagement' in observation:
                    self.min_peak_intensity = min(0.7, self.min_peak_intensity + 0.05)
                    
        except Exception as e:
            self.logger.debug(f"Failed to parse emotional pattern: {e}")
    
    def analyze_emotional_content(self, context: ContentContext) -> ContentContext:
        """
        Perform comprehensive emotional and engagement analysis.
        
        Args:
            context: ContentContext to analyze and update
            
        Returns:
            Updated ContentContext with emotional analysis results
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting emotional and engagement analysis")
            
            # Detect emotional peaks from multiple modalities
            detected_peaks = self._detect_multi_modal_emotional_peaks(context)
            
            # Calculate engagement metrics
            engagement_metrics = self._calculate_engagement_metrics(context, detected_peaks)
            
            # Create emotional timeline
            emotional_timeline = self._create_emotional_timeline(detected_peaks, context)
            
            # Analyze cross-modal correlations
            cross_modal_correlations = self._analyze_cross_modal_correlations(context, detected_peaks)
            
            # Create comprehensive analysis result
            analysis_result = EmotionalAnalysisResult(
                detected_peaks=detected_peaks,
                engagement_metrics=engagement_metrics,
                emotional_timeline=emotional_timeline,
                cross_modal_correlations=cross_modal_correlations,
                processing_time=time.time() - start_time
            )
            
            # Update context with emotional markers
            for peak in detected_peaks:
                context.add_emotional_marker(
                    peak.timestamp, peak.emotion, peak.intensity, 
                    peak.confidence, peak.context
                )
            
            # Store engagement predictions in context
            context.engagement_predictions = engagement_metrics.to_dict()
            
            # Store analysis patterns in Memory
            self._store_emotional_patterns(context, analysis_result)
            
            # Update processing metrics
            context.processing_metrics.add_module_metrics(
                'emotional_analyzer',
                analysis_result.processing_time,
                0
            )
            
            self.logger.info(
                f"Emotional analysis completed in {analysis_result.processing_time:.2f}s"
            )
            self.logger.info(f"Detected {len(detected_peaks)} emotional peaks")
            self.logger.info(f"Overall engagement score: {engagement_metrics.overall_engagement_score:.2f}")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Emotional analysis failed: {e}")
            raise ContentContextError(f"Emotional analysis failed: {e}", context)
    
    def _detect_multi_modal_emotional_peaks(self, context: ContentContext) -> List[EmotionalPeak]:
        """Detect emotional peaks combining audio and visual cues."""
        self.logger.info("Detecting multi-modal emotional peaks")
        
        all_peaks = []
        
        # Detect peaks from audio analysis
        if context.audio_analysis:
            audio_peaks = self._detect_audio_emotional_peaks(context.audio_analysis)
            all_peaks.extend(audio_peaks)
        
        # Detect peaks from visual highlights
        if context.visual_highlights:
            visual_peaks = self._detect_visual_emotional_peaks(context.visual_highlights)
            all_peaks.extend(visual_peaks)
        
        # Merge and enhance peaks with cross-modal information
        merged_peaks = self._merge_emotional_peaks(all_peaks)
        
        # Filter peaks by minimum intensity and confidence
        filtered_peaks = [
            peak for peak in merged_peaks 
            if peak.intensity >= self.min_peak_intensity and peak.confidence >= 0.6
        ]
        
        self.logger.info(f"Detected {len(filtered_peaks)} emotional peaks from {len(all_peaks)} candidates")
        
        return filtered_peaks
    
    def _detect_audio_emotional_peaks(self, audio_analysis: AudioAnalysisResult) -> List[EmotionalPeak]:
        """Detect emotional peaks from audio transcript analysis."""
        peaks = []
        
        for segment in audio_analysis.segments:
            segment_text = segment.text.lower()
            segment_peaks = []
            
            # Check each emotional pattern
            for emotion_type, pattern in self.emotional_patterns.items():
                emotion_score = 0.0
                matched_keywords = []
                
                # Check for keyword matches
                for keyword in pattern.audio_keywords:
                    if keyword in segment_text:
                        multiplier = pattern.intensity_multipliers.get(keyword, 0.5)
                        emotion_score += multiplier
                        matched_keywords.append(keyword)
                
                # Check for context patterns
                for context_pattern in pattern.context_patterns:
                    if context_pattern in segment_text:
                        emotion_score += 0.3
                
                # Apply Memory-based pattern weights
                pattern_weight = self.pattern_weights.get(emotion_type.value, 1.0)
                emotion_score *= pattern_weight
                
                # Create peak if score is significant
                if emotion_score > 0.3 and matched_keywords:
                    intensity = min(emotion_score / len(pattern.audio_keywords), 1.0)
                    confidence = min(segment.confidence * (emotion_score / 2.0), 1.0)
                    
                    context_description = f"Audio keywords: {', '.join(matched_keywords[:3])}"
                    
                    peak = EmotionalPeak(
                        timestamp=segment.start,
                        emotion=emotion_type.value,
                        intensity=intensity,
                        confidence=confidence,
                        context=context_description
                    )
                    segment_peaks.append(peak)
            
            # Keep only the strongest peak per segment to avoid over-detection
            if segment_peaks:
                strongest_peak = max(segment_peaks, key=lambda p: p.intensity * p.confidence)
                peaks.append(strongest_peak)
        
        return peaks
    
    def _detect_visual_emotional_peaks(self, visual_highlights: List[VisualHighlight]) -> List[EmotionalPeak]:
        """Detect emotional peaks from visual analysis."""
        peaks = []
        
        for highlight in visual_highlights:
            # Analyze facial expressions if faces are present
            if highlight.faces:
                for face in highlight.faces:
                    if face.expression:
                        emotion_mapping = {
                            'happy': (EmotionType.EXCITEMENT, 0.7),
                            'surprised': (EmotionType.SURPRISE, 0.8),
                            'focused': (EmotionType.FOCUS, 0.6),
                            'neutral': (EmotionType.CONFIDENCE, 0.4)
                        }
                        
                        if face.expression in emotion_mapping:
                            emotion_type, base_intensity = emotion_mapping[face.expression]
                            
                            # Adjust intensity based on face confidence and visual elements
                            intensity = base_intensity * face.confidence
                            if 'animated_gesture' in highlight.visual_elements:
                                intensity = min(intensity * 1.2, 1.0)
                            
                            confidence = face.confidence * 0.8  # Visual confidence is generally lower
                            
                            context_description = f"Visual expression: {face.expression}, elements: {', '.join(highlight.visual_elements[:2])}"
                            
                            peak = EmotionalPeak(
                                timestamp=highlight.timestamp,
                                emotion=emotion_type.value,
                                intensity=intensity,
                                confidence=confidence,
                                context=context_description
                            )
                            peaks.append(peak)
            
            # Analyze visual elements for emotional cues
            emotional_visual_elements = {
                'bright_lighting': (EmotionType.EXCITEMENT, 0.5),
                'dynamic_movement': (EmotionType.EXCITEMENT, 0.6),
                'data_visualization': (EmotionType.CURIOSITY, 0.4),
                'text_overlay': (EmotionType.FOCUS, 0.3)
            }
            
            for element in highlight.visual_elements:
                if element in emotional_visual_elements:
                    emotion_type, intensity = emotional_visual_elements[element]
                    
                    # Adjust intensity based on thumbnail potential
                    adjusted_intensity = intensity * highlight.thumbnail_potential
                    confidence = 0.6  # Moderate confidence for visual element emotions
                    
                    context_description = f"Visual element: {element}, thumbnail potential: {highlight.thumbnail_potential:.2f}"
                    
                    peak = EmotionalPeak(
                        timestamp=highlight.timestamp,
                        emotion=emotion_type.value,
                        intensity=adjusted_intensity,
                        confidence=confidence,
                        context=context_description
                    )
                    peaks.append(peak)
        
        return peaks
    
    def _merge_emotional_peaks(self, peaks: List[EmotionalPeak]) -> List[EmotionalPeak]:
        """Merge nearby emotional peaks to avoid over-detection."""
        if not peaks:
            return []
        
        # Sort peaks by timestamp
        sorted_peaks = sorted(peaks, key=lambda p: p.timestamp)
        merged_peaks = []
        
        current_peak = sorted_peaks[0]
        
        for next_peak in sorted_peaks[1:]:
            time_diff = abs(next_peak.timestamp - current_peak.timestamp)
            
            # Merge if peaks are close in time and same emotion
            if (time_diff <= self.peak_merge_threshold and 
                next_peak.emotion == current_peak.emotion):
                
                # Create merged peak with enhanced properties
                merged_intensity = max(current_peak.intensity, next_peak.intensity)
                merged_confidence = (current_peak.confidence + next_peak.confidence) / 2
                merged_context = f"{current_peak.context}; {next_peak.context}"
                
                current_peak = EmotionalPeak(
                    timestamp=(current_peak.timestamp + next_peak.timestamp) / 2,
                    emotion=current_peak.emotion,
                    intensity=merged_intensity,
                    confidence=merged_confidence,
                    context=merged_context
                )
            else:
                # Add current peak and move to next
                merged_peaks.append(current_peak)
                current_peak = next_peak
        
        # Add the last peak
        merged_peaks.append(current_peak)
        
        return merged_peaks
    
    def _calculate_engagement_metrics(self, context: ContentContext, 
                                    peaks: List[EmotionalPeak]) -> EngagementMetrics:
        """Calculate comprehensive engagement metrics."""
        self.logger.info("Calculating engagement metrics")
        
        # Emotional variety score
        unique_emotions = set(peak.emotion for peak in peaks)
        emotional_variety_score = min(len(unique_emotions) / 4.0, 1.0)  # Normalize to 4 emotions
        
        # Peak intensity score
        if peaks:
            avg_intensity = sum(peak.intensity for peak in peaks) / len(peaks)
            max_intensity = max(peak.intensity for peak in peaks)
            peak_intensity_score = (avg_intensity + max_intensity) / 2
        else:
            peak_intensity_score = 0.0
        
        # Pacing score (based on peak distribution)
        pacing_score = self._calculate_pacing_score(peaks, context)
        
        # Visual engagement score
        visual_engagement_score = self._calculate_visual_engagement_score(context)
        
        # Audio clarity score
        audio_clarity_score = self._calculate_audio_clarity_score(context)
        
        # Overall engagement score (weighted average)
        weights = {
            'emotional_variety': 0.25,
            'peak_intensity': 0.25,
            'pacing': 0.20,
            'visual_engagement': 0.15,
            'audio_clarity': 0.15
        }
        
        overall_engagement_score = (
            emotional_variety_score * weights['emotional_variety'] +
            peak_intensity_score * weights['peak_intensity'] +
            pacing_score * weights['pacing'] +
            visual_engagement_score * weights['visual_engagement'] +
            audio_clarity_score * weights['audio_clarity']
        )
        
        return EngagementMetrics(
            emotional_variety_score=emotional_variety_score,
            peak_intensity_score=peak_intensity_score,
            pacing_score=pacing_score,
            visual_engagement_score=visual_engagement_score,
            audio_clarity_score=audio_clarity_score,
            overall_engagement_score=overall_engagement_score
        )
    
    def _calculate_pacing_score(self, peaks: List[EmotionalPeak], context: ContentContext) -> float:
        """Calculate pacing score based on emotional peak distribution."""
        if not peaks or not context.video_metadata:
            return 0.5  # Neutral score
        
        video_duration = context.video_metadata.get('duration', 120.0)
        
        # Calculate ideal peak distribution (one peak every 15-30 seconds)
        ideal_peak_count = max(1, int(video_duration / 20))
        actual_peak_count = len(peaks)
        
        # Score based on peak count relative to ideal
        count_score = 1.0 - abs(actual_peak_count - ideal_peak_count) / max(ideal_peak_count, 1)
        count_score = max(0.0, count_score)
        
        # Score based on peak distribution evenness
        if len(peaks) > 1:
            peak_times = [peak.timestamp for peak in peaks]
            time_intervals = [peak_times[i+1] - peak_times[i] for i in range(len(peak_times)-1)]
            
            if time_intervals:
                avg_interval = sum(time_intervals) / len(time_intervals)
                interval_variance = sum((interval - avg_interval) ** 2 for interval in time_intervals) / len(time_intervals)
                distribution_score = 1.0 / (1.0 + interval_variance / 100)  # Normalize variance
            else:
                distribution_score = 0.5
        else:
            distribution_score = 0.3  # Low score for single peak
        
        # Combine scores
        pacing_score = (count_score * 0.6 + distribution_score * 0.4)
        return min(max(pacing_score, 0.0), 1.0)
    
    def _calculate_visual_engagement_score(self, context: ContentContext) -> float:
        """Calculate visual engagement score from visual highlights."""
        if not context.visual_highlights:
            return 0.3  # Low score for no visual analysis
        
        # Average thumbnail potential as base score
        avg_thumbnail_potential = sum(
            highlight.thumbnail_potential for highlight in context.visual_highlights
        ) / len(context.visual_highlights)
        
        # Bonus for face presence
        faces_present = sum(1 for highlight in context.visual_highlights if highlight.faces)
        face_bonus = min(faces_present / len(context.visual_highlights), 0.3)
        
        # Bonus for visual variety
        all_elements = []
        for highlight in context.visual_highlights:
            all_elements.extend(highlight.visual_elements)
        
        unique_elements = set(all_elements)
        variety_bonus = min(len(unique_elements) / 10, 0.2)  # Normalize to 10 element types
        
        visual_score = avg_thumbnail_potential + face_bonus + variety_bonus
        return min(visual_score, 1.0)
    
    def _calculate_audio_clarity_score(self, context: ContentContext) -> float:
        """Calculate audio clarity score from audio analysis."""
        if not context.audio_analysis:
            return 0.5  # Neutral score for no audio analysis
        
        # Base score from transcription confidence
        base_score = context.audio_analysis.overall_confidence
        
        # Bonus for audio enhancement
        if context.audio_analysis.filler_words_removed > 0:
            enhancement_bonus = min(context.audio_analysis.quality_improvement_score, 0.2)
        else:
            enhancement_bonus = 0.0
        
        # Penalty for low confidence segments
        low_confidence_segments = sum(
            1 for segment in context.audio_analysis.segments 
            if segment.confidence < 0.7
        )
        
        if context.audio_analysis.segments:
            confidence_penalty = (low_confidence_segments / len(context.audio_analysis.segments)) * 0.3
        else:
            confidence_penalty = 0.0
        
        audio_score = base_score + enhancement_bonus - confidence_penalty
        return min(max(audio_score, 0.0), 1.0)
    
    def _create_emotional_timeline(self, peaks: List[EmotionalPeak], 
                                 context: ContentContext) -> List[Dict[str, Any]]:
        """Create emotional timeline for visualization and analysis."""
        timeline = []
        
        # Sort peaks by timestamp
        sorted_peaks = sorted(peaks, key=lambda p: p.timestamp)
        
        for i, peak in enumerate(sorted_peaks):
            timeline_entry = {
                'timestamp': peak.timestamp,
                'emotion': peak.emotion,
                'intensity': peak.intensity,
                'confidence': peak.confidence,
                'context': peak.context,
                'sequence_position': i + 1,
                'total_peaks': len(sorted_peaks)
            }
            
            # Add relative timing information
            if i > 0:
                timeline_entry['time_since_previous'] = peak.timestamp - sorted_peaks[i-1].timestamp
            
            if i < len(sorted_peaks) - 1:
                timeline_entry['time_to_next'] = sorted_peaks[i+1].timestamp - peak.timestamp
            
            timeline.append(timeline_entry)
        
        return timeline
    
    def _analyze_cross_modal_correlations(self, context: ContentContext, 
                                        peaks: List[EmotionalPeak]) -> Dict[str, float]:
        """Analyze correlations between audio and visual emotional cues."""
        correlations = {
            'audio_visual_sync': 0.0,
            'emotion_consistency': 0.0,
            'intensity_correlation': 0.0,
            'temporal_alignment': 0.0
        }
        
        if not peaks or not context.visual_highlights:
            return correlations
        
        # Separate audio and visual peaks
        audio_peaks = [p for p in peaks if 'Audio keywords' in p.context]
        visual_peaks = [p for p in peaks if 'Visual' in p.context]
        
        if not audio_peaks or not visual_peaks:
            return correlations
        
        # Calculate temporal alignment
        aligned_pairs = 0
        total_comparisons = 0
        
        for audio_peak in audio_peaks:
            for visual_peak in visual_peaks:
                time_diff = abs(audio_peak.timestamp - visual_peak.timestamp)
                total_comparisons += 1
                
                if time_diff <= 5.0:  # Within 5 seconds
                    aligned_pairs += 1
                    
                    # Check emotion consistency
                    if audio_peak.emotion == visual_peak.emotion:
                        correlations['emotion_consistency'] += 1
        
        if total_comparisons > 0:
            correlations['temporal_alignment'] = aligned_pairs / total_comparisons
            correlations['emotion_consistency'] /= total_comparisons
        
        # Calculate intensity correlation
        if aligned_pairs > 0 and len(audio_peaks) > 0 and len(visual_peaks) > 0:
            # Get matching pairs for correlation
            audio_intensities = []
            visual_intensities = []
            
            for audio_peak in audio_peaks:
                for visual_peak in visual_peaks:
                    time_diff = abs(audio_peak.timestamp - visual_peak.timestamp)
                    if time_diff <= 5.0:  # Within 5 seconds
                        audio_intensities.append(audio_peak.intensity)
                        visual_intensities.append(visual_peak.intensity)
                        break  # Only match each audio peak once
            
            if len(audio_intensities) > 1 and len(visual_intensities) > 1:
                try:
                    # Check for constant arrays which cause correlation issues
                    if np.std(audio_intensities) > 1e-10 and np.std(visual_intensities) > 1e-10:
                        correlation = np.corrcoef(audio_intensities, visual_intensities)[0, 1]
                        correlations['intensity_correlation'] = max(0.0, correlation) if not np.isnan(correlation) else 0.0
                    else:
                        correlations['intensity_correlation'] = 0.0
                except (ValueError, np.linalg.LinAlgError, IndexError):
                    correlations['intensity_correlation'] = 0.0
        
        # Overall audio-visual sync score
        correlations['audio_visual_sync'] = (
            correlations['temporal_alignment'] * 0.4 +
            correlations['emotion_consistency'] * 0.4 +
            correlations['intensity_correlation'] * 0.2
        )
        
        return correlations
    
    def _store_emotional_patterns(self, context: ContentContext, 
                                analysis_result: EmotionalAnalysisResult):
        """Store emotional analysis patterns and insights in Memory."""
        if not self.memory_client:
            return
        
        try:
            observations = []
            
            # Emotional detection success
            emotion_counts = {}
            for peak in analysis_result.detected_peaks:
                emotion_counts[peak.emotion] = emotion_counts.get(peak.emotion, 0) + 1
            
            for emotion, count in emotion_counts.items():
                if count > 0:
                    avg_confidence = sum(
                        p.confidence for p in analysis_result.detected_peaks 
                        if p.emotion == emotion
                    ) / count
                    observations.append(
                        f"Emotion detection {emotion} accuracy {avg_confidence:.1%} "
                        f"with {count} peaks detected"
                    )
            
            # Engagement prediction insights
            engagement_score = analysis_result.engagement_metrics.overall_engagement_score
            if engagement_score > 0.7:
                observations.append(f"High engagement prediction success: {engagement_score:.1%}")
            elif engagement_score < 0.4:
                observations.append(f"Low engagement prediction: {engagement_score:.1%}")
            
            # Cross-modal correlation insights
            correlations = analysis_result.cross_modal_correlations
            if correlations['audio_visual_sync'] > 0.6:
                observations.append(
                    f"Strong audio-visual emotional sync: {correlations['audio_visual_sync']:.1%}"
                )
            
            # Processing performance
            observations.append(
                f"Emotional analysis completed in {analysis_result.processing_time:.2f}s "
                f"for {len(analysis_result.detected_peaks)} peaks"
            )
            
            # Store in Memory
            self.memory_client.create_entities([{
                'name': 'Emotional Analysis Patterns',
                'entityType': 'analysis_insights',
                'observations': observations
            }])
            
            self.logger.info("Stored emotional analysis patterns in Memory")
            
        except Exception as e:
            self.logger.warning(f"Failed to store emotional patterns in Memory: {e}")


def create_emotional_analyzer(cache_manager: Optional[CacheManager] = None, 
                            memory_client=None) -> EmotionalAnalyzer:
    """
    Factory function to create EmotionalAnalyzer instance.
    
    Args:
        cache_manager: Optional CacheManager for caching
        memory_client: Optional Memory client for pattern learning
        
    Returns:
        EmotionalAnalyzer instance
    """
    return EmotionalAnalyzer(cache_manager=cache_manager, memory_client=memory_client)