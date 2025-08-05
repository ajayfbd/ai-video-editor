"""
Content Analysis Module - Multi-Modal Content Understanding.

This module implements the ContentAnalyzer base class that provides a unified
interface for analyzing content across audio and visual modalities. It integrates
with existing audio and video analyzers to provide comprehensive content understanding
for optimal editing decisions.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from ai_video_editor.core.content_context import ContentContext, ContentType, EmotionalPeak
from ai_video_editor.modules.content_analysis.audio_analyzer import FinancialContentAnalyzer
from ai_video_editor.modules.content_analysis.video_analyzer import VideoAnalyzer
from ai_video_editor.modules.content_analysis.emotional_analyzer import EmotionalAnalyzer
from ai_video_editor.utils.cache_manager import CacheManager
from ai_video_editor.utils.error_handling import ContentContextError
from ai_video_editor.utils.logging_config import get_logger


@dataclass
class ConceptExtraction:
    """Represents an extracted concept from multi-modal analysis."""
    concept: str
    confidence: float
    sources: List[str]  # ['audio', 'visual', 'metadata']
    context: str
    timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'concept': self.concept,
            'confidence': self.confidence,
            'sources': self.sources,
            'context': self.context,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptExtraction':
        return cls(**data)


@dataclass
class ContentTypeDetection:
    """Results of content type detection analysis."""
    detected_type: ContentType
    confidence: float
    reasoning: str
    alternative_types: List[Tuple[ContentType, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'detected_type': self.detected_type.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'alternative_types': [(t.value, c) for t, c in self.alternative_types]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentTypeDetection':
        detected_type = ContentType(data['detected_type'])
        alternative_types = [(ContentType(t), c) for t, c in data.get('alternative_types', [])]
        return cls(
            detected_type=detected_type,
            confidence=data['confidence'],
            reasoning=data['reasoning'],
            alternative_types=alternative_types
        )


@dataclass
class MultiModalAnalysisResult:
    """Comprehensive multi-modal analysis results."""
    content_type_detection: ContentTypeDetection
    extracted_concepts: List[ConceptExtraction]
    cross_modal_insights: Dict[str, Any]
    engagement_predictions: Dict[str, float]
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content_type_detection': self.content_type_detection.to_dict(),
            'extracted_concepts': [concept.to_dict() for concept in self.extracted_concepts],
            'cross_modal_insights': self.cross_modal_insights,
            'engagement_predictions': self.engagement_predictions,
            'processing_time': self.processing_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiModalAnalysisResult':
        content_type_detection = ContentTypeDetection.from_dict(data['content_type_detection'])
        extracted_concepts = [ConceptExtraction.from_dict(c) for c in data['extracted_concepts']]
        return cls(
            content_type_detection=content_type_detection,
            extracted_concepts=extracted_concepts,
            cross_modal_insights=data['cross_modal_insights'],
            engagement_predictions=data['engagement_predictions'],
            processing_time=data['processing_time']
        )


class ContentAnalyzer(ABC):
    """
    Abstract base class for unified content analysis interface.
    
    Provides a common interface for analyzing content across different modalities
    and integrating with the ContentContext system for optimal editing decisions.
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None, memory_client=None):
        """Initialize ContentAnalyzer with optional caching and Memory integration."""
        self.logger = get_logger(__name__)
        self.cache_manager = cache_manager
        self.memory_client = memory_client
        
        # Analysis patterns learned from Memory
        self.analysis_patterns = {}
        self._load_analysis_patterns()
    
    def _load_analysis_patterns(self):
        """Load analysis patterns from Memory for improved accuracy."""
        if not self.memory_client:
            return
        
        try:
            search_results = self.memory_client.search_nodes("content analysis patterns")
            
            for result in search_results.get('nodes', []):
                if result['name'] == 'Content Analysis Patterns':
                    for observation in result.get('observations', []):
                        self._parse_analysis_pattern(observation)
            
            self.logger.info("Loaded content analysis patterns from Memory")
            
        except Exception as e:
            self.logger.warning(f"Failed to load analysis patterns from Memory: {e}")
    
    def _parse_analysis_pattern(self, observation: str):
        """Parse analysis pattern observation from Memory."""
        try:
            # Extract patterns for content type detection accuracy
            if 'educational_content_accuracy' in observation:
                # Update detection weights based on historical accuracy
                parts = observation.split()
                for i, part in enumerate(parts):
                    if part == 'accuracy' and i + 1 < len(parts):
                        accuracy = float(parts[i + 1].rstrip('%')) / 100
                        self.analysis_patterns['educational_weight'] = max(0.5, min(2.0, accuracy * 2))
                        break
            
            elif 'concept_extraction_success' in observation:
                # Update concept extraction confidence thresholds
                if 'financial_concepts' in observation:
                    self.analysis_patterns['financial_concept_threshold'] = 0.7
                elif 'general_concepts' in observation:
                    self.analysis_patterns['general_concept_threshold'] = 0.6
                    
        except Exception as e:
            self.logger.debug(f"Failed to parse analysis pattern: {e}")
    
    @abstractmethod
    def analyze_content(self, context: ContentContext) -> ContentContext:
        """
        Analyze content and update ContentContext with results.
        
        Args:
            context: ContentContext to analyze and update
            
        Returns:
            Updated ContentContext with analysis results
        """
        pass
    
    @abstractmethod
    def detect_content_type(self, context: ContentContext) -> ContentTypeDetection:
        """
        Detect content type based on multi-modal analysis.
        
        Args:
            context: ContentContext with audio/video data
            
        Returns:
            ContentTypeDetection with detected type and confidence
        """
        pass
    
    @abstractmethod
    def extract_concepts(self, context: ContentContext) -> List[ConceptExtraction]:
        """
        Extract key concepts from audio transcripts and visual elements.
        
        Args:
            context: ContentContext with analysis data
            
        Returns:
            List of extracted concepts with confidence scores
        """
        pass
    
    def _store_analysis_patterns(self, context: ContentContext, results: MultiModalAnalysisResult):
        """Store analysis patterns and insights in Memory."""
        if not self.memory_client:
            return
        
        try:
            observations = []
            
            # Content type detection accuracy
            if results.content_type_detection.confidence > 0.8:
                observations.append(
                    f"Content type detection: {results.content_type_detection.detected_type.value} "
                    f"with {results.content_type_detection.confidence:.1%} confidence"
                )
            
            # Concept extraction success
            high_confidence_concepts = [c for c in results.extracted_concepts if c.confidence > 0.8]
            if high_confidence_concepts:
                observations.append(
                    f"Extracted {len(high_confidence_concepts)} high-confidence concepts "
                    f"from {context.content_type.value} content"
                )
            
            # Cross-modal insights
            if results.cross_modal_insights:
                for insight_type, insight_data in results.cross_modal_insights.items():
                    if isinstance(insight_data, dict) and 'confidence' in insight_data:
                        observations.append(
                            f"Cross-modal insight: {insight_type} with "
                            f"{insight_data['confidence']:.1%} confidence"
                        )
            
            # Processing performance
            observations.append(
                f"Multi-modal analysis completed in {results.processing_time:.2f}s "
                f"for {len(context.video_files)} video files"
            )
            
            # Store in Memory
            self.memory_client.create_entities([{
                'name': 'Content Analysis Patterns',
                'entityType': 'analysis_insights',
                'observations': observations
            }])
            
            self.logger.info("Stored content analysis patterns in Memory")
            
        except Exception as e:
            self.logger.warning(f"Failed to store analysis patterns in Memory: {e}")


class MultiModalContentAnalyzer(ContentAnalyzer):
    """
    Concrete implementation of ContentAnalyzer for multi-modal content understanding.
    
    Integrates audio and video analysis to provide comprehensive content understanding
    with content type detection, concept extraction, and cross-modal insights.
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None, memory_client=None):
        """Initialize MultiModalContentAnalyzer with audio and video analyzers."""
        super().__init__(cache_manager, memory_client)
        
        # Initialize specialized analyzers
        self.audio_analyzer = FinancialContentAnalyzer(
            cache_manager=cache_manager
        )
        self.video_analyzer = VideoAnalyzer(
            cache_manager=cache_manager,
            memory_client=memory_client
        )
        self.emotional_analyzer = EmotionalAnalyzer(
            cache_manager=cache_manager,
            memory_client=memory_client
        )
        
        # Content type detection patterns
        self.content_type_patterns = {
            ContentType.EDUCATIONAL: {
                'audio_keywords': [
                    'explain', 'learn', 'understand', 'concept', 'definition',
                    'example', 'tutorial', 'lesson', 'course', 'education',
                    'teach', 'instruction', 'guide', 'how to', 'step by step'
                ],
                'financial_keywords': [
                    'investment', 'portfolio', 'financial', 'money', 'budget',
                    'savings', 'retirement', 'compound interest', 'stocks', 'bonds'
                ],
                'visual_indicators': [
                    'text_overlay', 'data_visualization', 'presentation_slide',
                    'diagram', 'chart', 'graph'
                ],
                'engagement_patterns': ['explanation_segments', 'data_references']
            },
            ContentType.MUSIC: {
                'audio_keywords': [
                    'music', 'song', 'beat', 'rhythm', 'melody', 'lyrics',
                    'artist', 'album', 'track', 'performance', 'concert'
                ],
                'visual_indicators': [
                    'performance_stage', 'musical_instrument', 'dance_movement',
                    'concert_lighting', 'music_video_effects'
                ],
                'engagement_patterns': ['rhythm_sync', 'visual_effects']
            },
            ContentType.GENERAL: {
                'audio_keywords': [
                    'general', 'discussion', 'conversation', 'talk', 'chat',
                    'opinion', 'review', 'commentary', 'vlog', 'update'
                ],
                'visual_indicators': [
                    'talking_head', 'casual_setting', 'personal_space',
                    'informal_presentation'
                ],
                'engagement_patterns': ['personal_stories', 'casual_conversation']
            }
        }
        
        # Concept extraction weights based on user preferences and Memory patterns
        self.concept_weights = {
            'audio_transcript': 1.0,
            'visual_elements': 0.8,
            'emotional_markers': 0.6,
            'metadata': 0.4
        }
        
        self.logger.info("MultiModalContentAnalyzer initialized successfully")
    
    def analyze_content(self, context: ContentContext) -> ContentContext:
        """
        Perform comprehensive multi-modal content analysis.
        
        Args:
            context: ContentContext to analyze and update
            
        Returns:
            Updated ContentContext with multi-modal analysis results
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting multi-modal content analysis")
            
            # Ensure audio and video analysis are complete
            context = self._ensure_prerequisite_analysis(context)
            
            # Detect content type
            content_type_detection = self.detect_content_type(context)
            
            # Update context content type if detection confidence is high
            if content_type_detection.confidence > 0.8:
                context.content_type = content_type_detection.detected_type
                self.logger.info(
                    f"Updated content type to {content_type_detection.detected_type.value} "
                    f"with {content_type_detection.confidence:.1%} confidence"
                )
            
            # Extract concepts from multiple modalities
            extracted_concepts = self.extract_concepts(context)
            
            # Perform emotional and engagement analysis
            context = self.emotional_analyzer.analyze_emotional_content(context)
            
            # Generate cross-modal insights
            cross_modal_insights = self._generate_cross_modal_insights(context, extracted_concepts)
            
            # Predict engagement based on multi-modal analysis
            engagement_predictions = self._predict_engagement(context, extracted_concepts)
            
            # Create comprehensive analysis result
            analysis_result = MultiModalAnalysisResult(
                content_type_detection=content_type_detection,
                extracted_concepts=extracted_concepts,
                cross_modal_insights=cross_modal_insights,
                engagement_predictions=engagement_predictions,
                processing_time=time.time() - start_time
            )
            
            # Update context with extracted concepts
            for concept in extracted_concepts:
                if concept.concept not in context.key_concepts:
                    context.key_concepts.append(concept.concept)
            
            # Store analysis patterns in Memory
            self._store_analysis_patterns(context, analysis_result)
            
            # Update processing metrics
            context.processing_metrics.add_module_metrics(
                'multi_modal_content_analyzer',
                analysis_result.processing_time,
                0
            )
            
            self.logger.info(
                f"Multi-modal content analysis completed in {analysis_result.processing_time:.2f}s"
            )
            self.logger.info(f"Extracted {len(extracted_concepts)} concepts")
            self.logger.info(f"Content type: {content_type_detection.detected_type.value}")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Multi-modal content analysis failed: {e}")
            raise ContentContextError(f"Multi-modal content analysis failed: {e}", context)
    
    def _ensure_prerequisite_analysis(self, context: ContentContext) -> ContentContext:
        """Ensure audio and video analysis are completed before multi-modal analysis."""
        # Check if audio analysis is needed
        if not context.audio_analysis and context.video_files:
            self.logger.info("Audio analysis not found, performing audio analysis")
            # For now, we'll assume audio analysis is done separately
            # In a full implementation, we might trigger it here
        
        # Check if video analysis is needed
        if not context.visual_highlights and context.video_files:
            self.logger.info("Video analysis not found, performing video analysis")
            # For now, we'll assume video analysis is done separately
            # In a full implementation, we might trigger it here
        
        return context
    
    def detect_content_type(self, context: ContentContext) -> ContentTypeDetection:
        """
        Detect content type based on multi-modal analysis.
        
        Args:
            context: ContentContext with audio/video data
            
        Returns:
            ContentTypeDetection with detected type and confidence
        """
        self.logger.info("Detecting content type from multi-modal analysis")
        
        # Initialize scores for each content type
        type_scores = {content_type: 0.0 for content_type in ContentType}
        evidence = {content_type: [] for content_type in ContentType}
        
        # Analyze audio transcript if available
        if context.audio_analysis and context.audio_analysis.transcript_text:
            transcript = context.audio_analysis.transcript_text.lower()
            
            for content_type, patterns in self.content_type_patterns.items():
                audio_score = 0.0
                matched_keywords = []
                
                # Check audio keywords
                for keyword in patterns['audio_keywords']:
                    if keyword in transcript:
                        audio_score += 1.0
                        matched_keywords.append(keyword)
                
                # Check financial keywords for educational content
                if content_type == ContentType.EDUCATIONAL:
                    for keyword in patterns['financial_keywords']:
                        if keyword in transcript:
                            audio_score += 1.5  # Higher weight for financial education
                            matched_keywords.append(keyword)
                
                # Normalize score
                total_keywords = len(patterns['audio_keywords'])
                if content_type == ContentType.EDUCATIONAL:
                    total_keywords += len(patterns['financial_keywords'])
                
                if total_keywords > 0:
                    audio_score = min(audio_score / total_keywords, 1.0)
                
                type_scores[content_type] += audio_score * self.concept_weights['audio_transcript']
                
                if matched_keywords:
                    evidence[content_type].append(f"Audio keywords: {', '.join(matched_keywords[:3])}")
        
        # Analyze visual elements if available
        if context.visual_highlights:
            for content_type, patterns in self.content_type_patterns.items():
                visual_score = 0.0
                matched_elements = []
                
                for highlight in context.visual_highlights:
                    for element in highlight.visual_elements:
                        if element in patterns['visual_indicators']:
                            visual_score += 1.0
                            matched_elements.append(element)
                
                # Normalize score
                total_highlights = len(context.visual_highlights)
                if total_highlights > 0:
                    visual_score = min(visual_score / total_highlights, 1.0)
                
                type_scores[content_type] += visual_score * self.concept_weights['visual_elements']
                
                if matched_elements:
                    evidence[content_type].append(f"Visual elements: {', '.join(set(matched_elements[:3]))}")
        
        # Analyze engagement patterns if available
        if context.audio_analysis:
            for content_type, patterns in self.content_type_patterns.items():
                engagement_score = 0.0
                matched_patterns = []
                
                # Check for explanation segments (educational indicator)
                if 'explanation_segments' in patterns['engagement_patterns']:
                    if context.audio_analysis.explanation_segments:
                        engagement_score += len(context.audio_analysis.explanation_segments) * 0.2
                        matched_patterns.append('explanation_segments')
                
                # Check for data references (educational indicator)
                if 'data_references' in patterns['engagement_patterns']:
                    if context.audio_analysis.data_references:
                        engagement_score += len(context.audio_analysis.data_references) * 0.3
                        matched_patterns.append('data_references')
                
                engagement_score = min(engagement_score, 1.0)
                type_scores[content_type] += engagement_score * self.concept_weights['emotional_markers']
                
                if matched_patterns:
                    evidence[content_type].append(f"Engagement patterns: {', '.join(matched_patterns)}")
        
        # Apply Memory-based pattern weights
        for content_type in type_scores:
            pattern_key = f"{content_type.value}_weight"
            if pattern_key in self.analysis_patterns:
                type_scores[content_type] *= self.analysis_patterns[pattern_key]
        
        # Determine the best match
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        
        # Create alternative types list
        alternative_types = []
        for content_type, score in type_scores.items():
            if content_type != best_type and score > 0.1:
                alternative_types.append((content_type, score))
        
        alternative_types.sort(key=lambda x: x[1], reverse=True)
        
        # Generate reasoning
        reasoning_parts = []
        if evidence[best_type]:
            reasoning_parts.extend(evidence[best_type])
        else:
            reasoning_parts.append(f"Default classification based on user preference")
        
        reasoning = "; ".join(reasoning_parts)
        
        # Ensure minimum confidence for meaningful detection
        confidence = max(best_score, 0.3)  # Minimum 30% confidence
        
        detection = ContentTypeDetection(
            detected_type=best_type,
            confidence=confidence,
            reasoning=reasoning,
            alternative_types=alternative_types[:2]  # Top 2 alternatives
        )
        
        self.logger.info(
            f"Content type detected: {best_type.value} "
            f"(confidence: {confidence:.1%})"
        )
        
        return detection
    
    def extract_concepts(self, context: ContentContext) -> List[ConceptExtraction]:
        """
        Extract key concepts from audio transcripts and visual elements.
        
        Args:
            context: ContentContext with analysis data
            
        Returns:
            List of extracted concepts with confidence scores
        """
        self.logger.info("Extracting concepts from multi-modal analysis")
        
        extracted_concepts = []
        concept_sources = {}  # Track sources for each concept
        
        # Extract concepts from audio transcript
        if context.audio_analysis and context.audio_analysis.transcript_text:
            audio_concepts = self._extract_audio_concepts(context.audio_analysis)
            
            for concept, confidence, timestamp in audio_concepts:
                if concept not in concept_sources:
                    concept_sources[concept] = {
                        'sources': [],
                        'confidences': [],
                        'contexts': [],
                        'timestamps': []
                    }
                
                concept_sources[concept]['sources'].append('audio')
                concept_sources[concept]['confidences'].append(confidence)
                concept_sources[concept]['contexts'].append('transcript_analysis')
                concept_sources[concept]['timestamps'].append(timestamp)
        
        # Extract concepts from visual elements
        if context.visual_highlights:
            visual_concepts = self._extract_visual_concepts(context.visual_highlights)
            
            for concept, confidence, timestamp in visual_concepts:
                if concept not in concept_sources:
                    concept_sources[concept] = {
                        'sources': [],
                        'confidences': [],
                        'contexts': [],
                        'timestamps': []
                    }
                
                concept_sources[concept]['sources'].append('visual')
                concept_sources[concept]['confidences'].append(confidence)
                concept_sources[concept]['contexts'].append('visual_analysis')
                concept_sources[concept]['timestamps'].append(timestamp)
        
        # Extract concepts from emotional markers
        if context.emotional_markers:
            emotional_concepts = self._extract_emotional_concepts(context.emotional_markers)
            
            for concept, confidence, timestamp in emotional_concepts:
                if concept not in concept_sources:
                    concept_sources[concept] = {
                        'sources': [],
                        'confidences': [],
                        'contexts': [],
                        'timestamps': []
                    }
                
                concept_sources[concept]['sources'].append('emotional')
                concept_sources[concept]['confidences'].append(confidence)
                concept_sources[concept]['contexts'].append('emotional_analysis')
                concept_sources[concept]['timestamps'].append(timestamp)
        
        # Create ConceptExtraction objects
        for concept, data in concept_sources.items():
            # Calculate weighted confidence based on multiple sources
            total_weight = 0.0
            weighted_confidence = 0.0
            
            for source, confidence in zip(data['sources'], data['confidences']):
                weight = self.concept_weights.get(f"{source}_transcript", 
                                                self.concept_weights.get(f"{source}_elements", 0.5))
                weighted_confidence += confidence * weight
                total_weight += weight
            
            if total_weight > 0:
                final_confidence = weighted_confidence / total_weight
            else:
                final_confidence = max(data['confidences']) if data['confidences'] else 0.5
            
            # Apply Memory-based threshold adjustments
            threshold_key = f"{context.content_type.value}_concept_threshold"
            min_threshold = self.analysis_patterns.get(threshold_key, 0.6)
            
            if final_confidence >= min_threshold:
                # Get the most relevant timestamp
                best_timestamp = None
                if data['timestamps']:
                    # Use timestamp from highest confidence source
                    max_conf_idx = data['confidences'].index(max(data['confidences']))
                    best_timestamp = data['timestamps'][max_conf_idx]
                
                concept_extraction = ConceptExtraction(
                    concept=concept,
                    confidence=final_confidence,
                    sources=list(set(data['sources'])),  # Remove duplicates
                    context='; '.join(set(data['contexts'])),  # Remove duplicates
                    timestamp=best_timestamp
                )
                
                extracted_concepts.append(concept_extraction)
        
        # Sort by confidence
        extracted_concepts.sort(key=lambda x: x.confidence, reverse=True)
        
        self.logger.info(f"Extracted {len(extracted_concepts)} concepts")
        
        return extracted_concepts
    
    def _extract_audio_concepts(self, audio_analysis) -> List[Tuple[str, float, float]]:
        """
        Extract concepts from audio analysis results.
        
        Args:
            audio_analysis: AudioAnalysisResult with transcript and analysis data
            
        Returns:
            List of tuples (concept, confidence, timestamp)
        """
        concepts = []
        
        # Extract financial concepts with high confidence
        for concept in audio_analysis.financial_concepts:
            concepts.append((concept, 0.9, 0.0))  # High confidence for detected financial concepts
        
        # Extract concepts from explanation segments
        for explanation in audio_analysis.explanation_segments:
            if 'concept' in explanation:
                timestamp = explanation.get('timestamp', 0.0)
                confidence = explanation.get('confidence', 0.7)
                concepts.append((explanation['concept'], confidence, timestamp))
        
        # Extract concepts from transcript using keyword matching
        transcript = audio_analysis.transcript_text.lower()
        
        # Educational concepts
        educational_keywords = {
            'investment': 0.8, 'portfolio': 0.8, 'compound interest': 0.9,
            'diversification': 0.8, 'risk management': 0.8, 'asset allocation': 0.8,
            'financial planning': 0.8, 'retirement': 0.7, 'savings': 0.7,
            'budgeting': 0.7, 'debt management': 0.7, 'credit score': 0.7,
            'inflation': 0.8, 'market volatility': 0.8, 'dollar cost averaging': 0.8
        }
        
        for keyword, confidence in educational_keywords.items():
            if keyword in transcript:
                concepts.append((keyword, confidence, 0.0))
        
        # Remove duplicates and sort by confidence
        unique_concepts = {}
        for concept, confidence, timestamp in concepts:
            if concept not in unique_concepts or confidence > unique_concepts[concept][1]:
                unique_concepts[concept] = (concept, confidence, timestamp)
        
        return list(unique_concepts.values())
    
    def _extract_visual_concepts(self, visual_highlights) -> List[Tuple[str, float, float]]:
        """
        Extract concepts from visual highlights.
        
        Args:
            visual_highlights: List of VisualHighlight objects
            
        Returns:
            List of tuples (concept, confidence, timestamp)
        """
        concepts = []
        
        for highlight in visual_highlights:
            timestamp = highlight.timestamp
            
            # Extract concepts from visual elements
            for element in highlight.visual_elements:
                confidence = 0.7  # Base confidence for visual elements
                
                # Higher confidence for specific educational elements
                if element in ['chart', 'graph', 'data_visualization', 'presentation_slide']:
                    confidence = 0.8
                elif element in ['text_overlay', 'diagram']:
                    confidence = 0.75
                
                concepts.append((element, confidence, timestamp))
            
            # Extract concepts from description
            description = highlight.description.lower()
            
            # Look for financial/educational indicators in description
            if 'chart' in description or 'graph' in description:
                concepts.append(('data_visualization', 0.8, timestamp))
            if 'explanation' in description or 'teaching' in description:
                concepts.append(('educational_content', 0.7, timestamp))
            if 'presentation' in description or 'slide' in description:
                concepts.append(('presentation_format', 0.7, timestamp))
        
        return concepts
    
    def _extract_emotional_concepts(self, emotional_markers) -> List[Tuple[str, float, float]]:
        """
        Extract concepts from emotional markers.
        
        Args:
            emotional_markers: List of EmotionalPeak objects
            
        Returns:
            List of tuples (concept, confidence, timestamp)
        """
        concepts = []
        
        for marker in emotional_markers:
            timestamp = marker.timestamp
            confidence = marker.confidence
            
            # Convert emotions to content concepts
            emotion_concepts = {
                'excitement': 'engaging_content',
                'curiosity': 'educational_content',
                'surprise': 'revelation_moment',
                'satisfaction': 'achievement_content',
                'confusion': 'complex_concept',
                'understanding': 'clarity_moment'
            }
            
            if marker.emotion in emotion_concepts:
                concept = emotion_concepts[marker.emotion]
                concepts.append((concept, confidence * marker.intensity, timestamp))
            
            # High intensity emotions indicate important moments
            if marker.intensity > 0.8:
                concepts.append(('high_engagement_moment', confidence, timestamp))
        
        return concepts
    
    def _extract_audio_concepts_legacy(self, audio_analysis) -> List[Tuple[str, float, Optional[float]]]:
        """Extract concepts from audio analysis results (legacy method)."""
        concepts = []
        
        # Extract financial concepts if available
        if hasattr(audio_analysis, 'financial_concepts') and audio_analysis.financial_concepts:
            for concept in audio_analysis.financial_concepts:
                concepts.append((concept, 0.8, None))  # High confidence for detected financial concepts
        
        # Extract concepts from transcript text analysis
        if audio_analysis.transcript_text:
            text_lower = audio_analysis.transcript_text.lower()
            
            # Educational concepts
            educational_terms = [
                'learning', 'education', 'tutorial', 'lesson', 'course',
                'explanation', 'concept', 'understanding', 'knowledge'
            ]
            
            for term in educational_terms:
                if term in text_lower:
                    concepts.append((term, 0.7, None))
            
            # General topic concepts (simplified NLP)
            important_words = self._extract_important_words(text_lower)
            for word in important_words:
                if len(word) > 4:  # Filter short words
                    concepts.append((word, 0.6, None))
        
        return concepts
    
    def _extract_visual_concepts(self, visual_highlights) -> List[Tuple[str, float, float]]:
        """Extract concepts from visual analysis results."""
        concepts = []
        
        for highlight in visual_highlights:
            timestamp = highlight.timestamp
            
            # Extract concepts from visual elements
            for element in highlight.visual_elements:
                if element in ['text_overlay', 'data_visualization']:
                    concepts.append((f"visual_{element}", 0.8, timestamp))
                elif element in ['presentation_slide', 'diagram']:
                    concepts.append(('educational_visual', 0.7, timestamp))
                else:
                    concepts.append((element, 0.6, timestamp))
            
            # Extract concepts from description
            if highlight.description:
                desc_lower = highlight.description.lower()
                if 'chart' in desc_lower or 'graph' in desc_lower:
                    concepts.append(('data_visualization', 0.8, timestamp))
                if 'text' in desc_lower:
                    concepts.append(('text_content', 0.7, timestamp))
        
        return concepts
    
    def _extract_emotional_concepts(self, emotional_markers) -> List[Tuple[str, float, float]]:
        """Extract concepts from emotional analysis results."""
        concepts = []
        
        for marker in emotional_markers:
            # High-intensity emotions become concepts
            if marker.intensity > 0.7:
                concept_name = f"{marker.emotion}_moment"
                concepts.append((concept_name, marker.confidence, marker.timestamp))
            
            # Context-based concepts
            if marker.context:
                context_lower = marker.context.lower()
                if any(word in context_lower for word in ['explain', 'teach', 'learn']):
                    concepts.append(('educational_moment', 0.7, marker.timestamp))
        
        return concepts
    
    def _extract_important_words(self, text: str) -> List[str]:
        """Extract important words from text (simplified NLP)."""
        # Simple word extraction - in production, use proper NLP
        words = text.split()
        
        # Filter common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        important_words = []
        for word in words:
            clean_word = word.strip('.,!?;:').lower()
            if clean_word not in stop_words and len(clean_word) > 3:
                important_words.append(clean_word)
        
        # Return most frequent words (simplified)
        word_freq = {}
        for word in important_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10] if freq > 1]
    
    def _generate_cross_modal_insights(self, context: ContentContext, 
                                     concepts: List[ConceptExtraction]) -> Dict[str, Any]:
        """Generate insights from cross-modal analysis."""
        insights = {}
        
        # Audio-visual synchronization insights
        if context.audio_analysis and context.visual_highlights:
            insights['audio_visual_sync'] = self._analyze_audio_visual_sync(
                context.audio_analysis, context.visual_highlights
            )
        
        # Concept reinforcement across modalities
        multi_modal_concepts = [c for c in concepts if len(c.sources) > 1]
        if multi_modal_concepts:
            insights['cross_modal_reinforcement'] = {
                'count': len(multi_modal_concepts),
                'concepts': [c.concept for c in multi_modal_concepts[:5]],
                'confidence': sum(c.confidence for c in multi_modal_concepts) / len(multi_modal_concepts)
            }
        
        # Emotional-visual correlation
        if context.emotional_markers and context.visual_highlights:
            insights['emotional_visual_correlation'] = self._analyze_emotional_visual_correlation(
                context.emotional_markers, context.visual_highlights
            )
        
        return insights
    
    def _analyze_audio_visual_sync(self, audio_analysis, visual_highlights) -> Dict[str, Any]:
        """Analyze synchronization between audio and visual elements."""
        sync_insights = {
            'synchronized_moments': 0,
            'explanation_visual_matches': 0,
            'confidence': 0.0
        }
        
        # Check for explanation segments that align with visual highlights
        if hasattr(audio_analysis, 'explanation_segments'):
            for explanation in audio_analysis.explanation_segments:
                explanation_time = explanation.get('timestamp', 0)
                
                # Find visual highlights within 5 seconds
                for highlight in visual_highlights:
                    time_diff = abs(highlight.timestamp - explanation_time)
                    if time_diff <= 5.0:  # Within 5 seconds
                        sync_insights['explanation_visual_matches'] += 1
                        sync_insights['synchronized_moments'] += 1
        
        # Calculate confidence based on synchronization rate
        total_explanations = len(audio_analysis.explanation_segments) if hasattr(audio_analysis, 'explanation_segments') else 0
        if total_explanations > 0:
            sync_insights['confidence'] = sync_insights['explanation_visual_matches'] / total_explanations
        
        return sync_insights
    
    def _analyze_emotional_visual_correlation(self, emotional_markers, visual_highlights) -> Dict[str, Any]:
        """Analyze correlation between emotional peaks and visual highlights."""
        correlation = {
            'correlated_moments': 0,
            'high_engagement_sync': 0,
            'confidence': 0.0
        }
        
        for emotion in emotional_markers:
            if emotion.intensity > 0.6:  # High intensity emotions
                # Find visual highlights within 3 seconds
                for highlight in visual_highlights:
                    time_diff = abs(highlight.timestamp - emotion.timestamp)
                    if time_diff <= 3.0:  # Within 3 seconds
                        correlation['correlated_moments'] += 1
                        if highlight.thumbnail_potential > 0.7:
                            correlation['high_engagement_sync'] += 1
        
        # Calculate confidence
        high_intensity_emotions = len([e for e in emotional_markers if e.intensity > 0.6])
        if high_intensity_emotions > 0:
            correlation['confidence'] = correlation['correlated_moments'] / high_intensity_emotions
        
        return correlation
    
    def _predict_engagement(self, context: ContentContext, 
                          concepts: List[ConceptExtraction]) -> Dict[str, float]:
        """Predict engagement based on multi-modal analysis."""
        predictions = {
            'overall_engagement': 0.0,
            'educational_value': 0.0,
            'visual_appeal': 0.0,
            'content_clarity': 0.0
        }
        
        # Overall engagement based on concept diversity and confidence
        if concepts:
            avg_confidence = sum(c.confidence for c in concepts) / len(concepts)
            concept_diversity = len(set(c.concept for c in concepts)) / max(len(concepts), 1)
            predictions['overall_engagement'] = (avg_confidence + concept_diversity) / 2
        
        # Educational value for educational content
        if context.content_type == ContentType.EDUCATIONAL:
            educational_concepts = [c for c in concepts if 'educational' in c.concept.lower()]
            if educational_concepts:
                predictions['educational_value'] = sum(c.confidence for c in educational_concepts) / len(educational_concepts)
            
            # Boost for financial education
            financial_concepts = [c for c in concepts if any(fin in c.concept.lower() 
                                for fin in ['financial', 'investment', 'money', 'budget'])]
            if financial_concepts:
                predictions['educational_value'] = min(predictions['educational_value'] + 0.2, 1.0)
        
        # Visual appeal based on visual highlights
        if context.visual_highlights:
            high_potential_highlights = [h for h in context.visual_highlights if h.thumbnail_potential > 0.7]
            if context.visual_highlights:
                predictions['visual_appeal'] = len(high_potential_highlights) / len(context.visual_highlights)
        
        # Content clarity based on audio quality and concept extraction
        if context.audio_analysis:
            audio_quality = context.audio_analysis.overall_confidence
            concept_clarity = len([c for c in concepts if c.confidence > 0.8]) / max(len(concepts), 1)
            predictions['content_clarity'] = (audio_quality + concept_clarity) / 2
        
        return predictions
    
    def _generate_cross_modal_insights(self, context: ContentContext, 
                                     extracted_concepts: List[ConceptExtraction]) -> Dict[str, Any]:
        """
        Generate insights from cross-modal analysis.
        
        Args:
            context: ContentContext with multi-modal data
            extracted_concepts: List of extracted concepts
            
        Returns:
            Dictionary of cross-modal insights
        """
        insights = {}
        
        # Audio-visual synchronization analysis
        if context.audio_analysis and context.visual_highlights:
            insights['audio_visual_sync'] = self._analyze_audio_visual_sync(
                context.audio_analysis, context.visual_highlights
            )
        
        # Concept consistency across modalities
        audio_concepts = [c.concept for c in extracted_concepts if 'audio' in c.sources]
        visual_concepts = [c.concept for c in extracted_concepts if 'visual' in c.sources]
        
        common_concepts = set(audio_concepts) & set(visual_concepts)
        insights['cross_modal_consistency'] = {
            'common_concepts': list(common_concepts),
            'consistency_score': len(common_concepts) / max(len(set(audio_concepts + visual_concepts)), 1)
        }
        
        # Emotional-visual correlation
        if context.emotional_markers and context.visual_highlights:
            insights['emotional_visual_correlation'] = self._analyze_emotional_visual_correlation(
                context.emotional_markers, context.visual_highlights
            )
        
        return insights
    
    def _predict_engagement(self, context: ContentContext, 
                          extracted_concepts: List[ConceptExtraction]) -> Dict[str, float]:
        """
        Predict engagement metrics based on multi-modal analysis.
        
        Args:
            context: ContentContext with analysis data
            extracted_concepts: List of extracted concepts
            
        Returns:
            Dictionary of engagement predictions
        """
        predictions = {}
        
        # Content quality prediction
        audio_quality = 0.5
        if context.audio_analysis:
            audio_quality = context.audio_analysis.overall_confidence
        
        concept_clarity = len([c for c in extracted_concepts if c.confidence > 0.8]) / max(len(extracted_concepts), 1)
        predictions['content_clarity'] = (audio_quality + concept_clarity) / 2
        
        # Engagement potential based on emotional peaks
        high_intensity_emotions = len([e for e in context.emotional_markers if e.intensity > 0.7])
        predictions['emotional_engagement'] = min(high_intensity_emotions / 10.0, 1.0)
        
        # Visual appeal based on highlights
        high_potential_visuals = len([h for h in context.visual_highlights if h.thumbnail_potential > 0.7])
        predictions['visual_appeal'] = min(high_potential_visuals / 5.0, 1.0)
        
        # Educational value for educational content
        if context.content_type == ContentType.EDUCATIONAL:
            educational_concepts = len([c for c in extracted_concepts 
                                      if any(edu_word in c.concept.lower() 
                                           for edu_word in ['financial', 'investment', 'education', 'explain'])])
            predictions['educational_value'] = min(educational_concepts / 5.0, 1.0)
        
        # Overall engagement prediction
        engagement_factors = [
            predictions.get('content_clarity', 0.5),
            predictions.get('emotional_engagement', 0.5),
            predictions.get('visual_appeal', 0.5)
        ]
        
        if 'educational_value' in predictions:
            engagement_factors.append(predictions['educational_value'])
        
        predictions['overall_engagement'] = sum(engagement_factors) / len(engagement_factors)
        
        return predictions


# Factory function for easy instantiation
def create_content_analyzer(cache_manager: Optional[CacheManager] = None, 
                          memory_client=None) -> ContentAnalyzer:
    """
    Factory function to create a ContentAnalyzer instance.
    
    Args:
        cache_manager: Optional CacheManager for caching expensive operations
        memory_client: Optional Memory client for pattern learning
        
    Returns:
        ContentAnalyzer instance
    """
    return MultiModalContentAnalyzer(cache_manager=cache_manager, memory_client=memory_client)