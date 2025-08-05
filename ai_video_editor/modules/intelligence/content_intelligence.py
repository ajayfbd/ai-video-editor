"""
Content Intelligence Engine - Intelligent decision coordination for video editing.

This module implements the ContentIntelligenceEngine that analyzes content patterns
to make data-driven recommendations for editing decisions, B-roll placement, 
transitions, and pacing optimization, coordinating with the AI Director.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import re

from .ai_director import AIDirectorPlan, EditingDecision, BRollPlan, MetadataStrategy
from ...core.content_context import ContentContext, ContentType, EmotionalPeak, VisualHighlight
from ...core.exceptions import (
    ContentContextError, 
    ModuleIntegrationError,
    handle_errors
)


logger = logging.getLogger(__name__)


@dataclass
class EditingOpportunity:
    """Represents an editing opportunity identified by content analysis."""
    timestamp: float
    opportunity_type: str  # "cut", "emphasis", "pace_change", "hook_placement"
    parameters: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    rationale: str
    priority: int  # 1-10, higher is more important
    content_trigger: str  # What in the content triggered this opportunity
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EditingOpportunity':
        return cls(**data)


@dataclass
class BRollPlacement:
    """Represents a B-roll placement recommendation."""
    timestamp: float
    duration: float
    content_type: str  # "chart", "animation", "concept_visual", "process_diagram"
    description: str
    visual_elements: List[str]
    priority: int
    trigger_keywords: List[str]  # Keywords that triggered this placement
    educational_value: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BRollPlacement':
        return cls(**data)


@dataclass
class TransitionSuggestion:
    """Represents a transition recommendation between content segments."""
    from_timestamp: float
    to_timestamp: float
    transition_type: str  # "cut", "fade", "slide", "zoom"
    parameters: Dict[str, Any]
    reason: str  # Why this transition is recommended
    content_context: str  # What content surrounds this transition
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransitionSuggestion':
        return cls(**data)


@dataclass
class PacingSegment:
    """Represents a pacing recommendation for a content segment."""
    start_timestamp: float
    end_timestamp: float
    recommended_speed: float  # 0.5 to 2.0, where 1.0 is normal speed
    reason: str
    content_complexity: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PacingSegment':
        return cls(**data)


@dataclass
class PacingPlan:
    """Represents a comprehensive pacing optimization plan."""
    segments: List[PacingSegment]
    overall_strategy: str  # "educational_slow", "engagement_varied", "retention_focused"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'segments': [segment.to_dict() for segment in self.segments],
            'overall_strategy': self.overall_strategy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PacingPlan':
        segments = [PacingSegment.from_dict(seg) for seg in data['segments']]
        return cls(segments=segments, overall_strategy=data['overall_strategy'])


@dataclass
class EnhancedEditingPlan:
    """Represents an enhanced editing plan combining AI Director and Intelligence Engine recommendations."""
    ai_director_plan: AIDirectorPlan  # Original plan from AI Director
    intelligence_recommendations: List[EditingOpportunity]
    broll_enhancements: List[BRollPlacement]
    transition_improvements: List[TransitionSuggestion]
    pacing_optimizations: PacingPlan
    coordination_notes: List[str]  # How recommendations integrate with AI Director plan
    confidence_score: float  # Overall confidence in enhanced plan
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ai_director_plan': self.ai_director_plan.to_dict(),
            'intelligence_recommendations': [rec.to_dict() for rec in self.intelligence_recommendations],
            'broll_enhancements': [broll.to_dict() for broll in self.broll_enhancements],
            'transition_improvements': [trans.to_dict() for trans in self.transition_improvements],
            'pacing_optimizations': self.pacing_optimizations.to_dict(),
            'coordination_notes': self.coordination_notes,
            'confidence_score': self.confidence_score
        }


class ContentIntelligenceEngine:
    """
    Content Intelligence Engine for intelligent video editing decisions.
    
    This engine analyzes content patterns to make data-driven recommendations
    for editing decisions, B-roll placement, transitions, and pacing optimization.
    It coordinates with the AI Director to create enhanced editing plans.
    """
    
    def __init__(self, enable_advanced_analysis: bool = True):
        """
        Initialize the Content Intelligence Engine.
        
        Args:
            enable_advanced_analysis: Enable advanced content analysis features
        """
        self.enable_advanced_analysis = enable_advanced_analysis
        
        # Keyword dictionaries for B-roll detection
        self.broll_triggers = {
            'chart': ['percent', 'percentage', 'growth', 'decline', 'data', 'statistics', 
                     'comparison', 'chart', 'graph', 'numbers', 'rate', 'ratio'],
            'animation': ['process', 'how it works', 'step by step', 'concept', 'mechanism',
                         'workflow', 'procedure', 'method', 'technique'],
            'concept_visual': ['compound interest', 'diversification', 'portfolio', 'risk',
                              'investment', 'asset allocation', 'financial planning'],
            'process_diagram': ['structure', 'relationship', 'flow', 'system', 'framework',
                               'model', 'strategy', 'approach', 'steps']
        }
        
        # Transition mapping based on content relationships
        self.transition_mapping = {
            'topic_change': 'cut',
            'emotional_shift': 'fade',
            'sequential_content': 'slide',
            'emphasis_point': 'zoom',
            'concept_continuation': 'cut',
            'example_introduction': 'slide'
        }
        
        # Complexity indicators for pacing analysis
        self.complexity_indicators = {
            'financial_terms': ['compound', 'diversification', 'allocation', 'volatility', 
                               'correlation', 'derivative', 'amortization'],
            'abstract_concepts': ['concept', 'theory', 'principle', 'framework', 'model'],
            'mathematical_content': ['calculate', 'formula', 'equation', 'percentage', 'ratio'],
            'process_descriptions': ['first', 'second', 'then', 'next', 'finally', 'step']
        }
        
        logger.info(f"ContentIntelligenceEngine initialized with advanced_analysis={enable_advanced_analysis}")
    
    @handle_errors()
    def analyze_editing_opportunities(self, context: ContentContext) -> List[EditingOpportunity]:
        """
        Analyze content for editing opportunities based on transcript, emotional peaks, and visual highlights.
        
        Args:
            context: ContentContext with analyzed content data
            
        Returns:
            List of EditingOpportunity objects with recommendations
            
        Raises:
            ContentContextError: If context is invalid or missing required data
        """
        if not context:
            raise ContentContextError("Invalid ContentContext provided")
        
        opportunities = []
        
        try:
            # Analyze transcript segments for natural cut points
            transcript_opportunities = self._analyze_transcript_cuts(context)
            opportunities.extend(transcript_opportunities)
            
            # Identify emotional peaks as emphasis opportunities
            emotional_opportunities = self._analyze_emotional_peaks(context)
            opportunities.extend(emotional_opportunities)
            
            # Detect concept transitions for pacing changes
            concept_opportunities = self._analyze_concept_transitions(context)
            opportunities.extend(concept_opportunities)
            
            # Find engagement drop points for hook placement
            engagement_opportunities = self._analyze_engagement_points(context)
            opportunities.extend(engagement_opportunities)
            
            # Sort by priority and timestamp
            opportunities.sort(key=lambda x: (-x.priority, x.timestamp))
            
            logger.info(f"Identified {len(opportunities)} editing opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Failed to analyze editing opportunities: {str(e)}")
            raise ContentContextError(
                f"Failed to analyze editing opportunities: {str(e)}",
                context_state=context
            )
    
    def _analyze_transcript_cuts(self, context: ContentContext) -> List[EditingOpportunity]:
        """Analyze transcript for natural cut points."""
        opportunities = []
        
        if not hasattr(context, 'audio_transcript') or not context.audio_transcript:
            return opportunities
        
        # Get transcript segments
        transcript_segments = getattr(context.audio_transcript, 'segments', [])
        
        for i, segment in enumerate(transcript_segments):
            text = getattr(segment, 'text', str(segment)).strip()
            timestamp = getattr(segment, 'start', i * 5.0)
            
            # Look for natural pause indicators
            if any(indicator in text.lower() for indicator in ['. ', '? ', '! ', ', and ', ', but ', ', so ']):
                opportunities.append(EditingOpportunity(
                    timestamp=timestamp,
                    opportunity_type='cut',
                    parameters={'cut_type': 'natural_pause', 'fade_duration': 0.2},
                    confidence=0.8,
                    rationale='Natural pause detected in speech pattern',
                    priority=6,
                    content_trigger=text[:50] + '...' if len(text) > 50 else text
                ))
        
        return opportunities
    
    def _analyze_emotional_peaks(self, context: ContentContext) -> List[EditingOpportunity]:
        """Analyze emotional peaks for emphasis opportunities."""
        opportunities = []
        
        if not hasattr(context, 'emotional_markers') or not context.emotional_markers:
            return opportunities
        
        for peak in context.emotional_markers:
            if hasattr(peak, 'timestamp') and hasattr(peak, 'intensity'):
                # High intensity peaks are good emphasis points
                if peak.intensity > 0.7:
                    opportunities.append(EditingOpportunity(
                        timestamp=peak.timestamp,
                        opportunity_type='emphasis',
                        parameters={'emphasis_type': 'zoom_in', 'duration': 2.0},
                        confidence=peak.confidence if hasattr(peak, 'confidence') else 0.8,
                        rationale=f'High emotional intensity ({peak.emotion}) detected',
                        priority=8,
                        content_trigger=getattr(peak, 'context', f'{peak.emotion} peak')
                    ))
        
        return opportunities
    
    def _analyze_concept_transitions(self, context: ContentContext) -> List[EditingOpportunity]:
        """Analyze concept transitions for pacing changes."""
        opportunities = []
        
        if not hasattr(context, 'key_concepts') or not context.key_concepts:
            return opportunities
        
        # Look for complex financial concepts that need slower pacing
        complex_concepts = ['compound interest', 'diversification', 'asset allocation', 'risk management']
        
        if hasattr(context, 'audio_transcript') and context.audio_transcript:
            transcript_segments = getattr(context.audio_transcript, 'segments', [])
            
            for segment in transcript_segments:
                text = getattr(segment, 'text', str(segment)).lower()
                timestamp = getattr(segment, 'start', 0.0)
                
                for concept in complex_concepts:
                    if concept.lower() in text:
                        opportunities.append(EditingOpportunity(
                            timestamp=timestamp,
                            opportunity_type='pace_change',
                            parameters={'speed_multiplier': 0.8, 'duration': 10.0},
                            confidence=0.9,
                            rationale=f'Complex concept "{concept}" requires slower pacing',
                            priority=7,
                            content_trigger=f'Complex concept: {concept}'
                        ))
        
        return opportunities
    
    def _analyze_engagement_points(self, context: ContentContext) -> List[EditingOpportunity]:
        """Find engagement drop points for hook placement."""
        opportunities = []
        
        # Add hook opportunities every 30-45 seconds for retention
        total_duration = getattr(context, 'total_duration', 180)  # Default 3 minutes
        hook_interval = 35  # seconds
        
        for timestamp in range(hook_interval, int(total_duration), hook_interval):
            opportunities.append(EditingOpportunity(
                timestamp=float(timestamp),
                opportunity_type='hook_placement',
                parameters={'hook_type': 'text_overlay', 'duration': 3.0},
                confidence=0.7,
                rationale='Regular engagement hook for viewer retention',
                priority=5,
                content_trigger='Retention optimization point'
            ))
        
        return opportunities    

    @handle_errors()
    def detect_broll_placements(self, context: ContentContext) -> List[BRollPlacement]:
        """
        Detect optimal B-roll placement opportunities based on content analysis.
        
        Args:
            context: ContentContext with transcript and concept analysis
            
        Returns:
            List of BRollPlacement objects with timing and content specifications
            
        Raises:
            ContentContextError: If context is invalid or missing required data
        """
        if not context:
            raise ContentContextError("Invalid ContentContext provided")
        
        placements = []
        
        try:
            if hasattr(context, 'audio_transcript') and context.audio_transcript:
                transcript_segments = getattr(context.audio_transcript, 'segments', [])
                
                for segment in transcript_segments:
                    text = getattr(segment, 'text', str(segment)).lower()
                    timestamp = getattr(segment, 'start', 0.0)
                    duration = getattr(segment, 'end', timestamp + 5.0) - timestamp
                    
                    # Check each B-roll type
                    for broll_type, keywords in self.broll_triggers.items():
                        matching_keywords = [kw for kw in keywords if kw in text]
                        
                        if matching_keywords:
                            # Calculate educational value based on concept complexity
                            educational_value = self._calculate_educational_value(text, matching_keywords)
                            
                            # Determine priority based on B-roll type and educational value
                            priority = self._calculate_broll_priority(broll_type, educational_value)
                            
                            placement = BRollPlacement(
                                timestamp=timestamp,
                                duration=min(duration, 8.0),  # Cap at 8 seconds
                                content_type=broll_type,
                                description=self._generate_broll_description(broll_type, matching_keywords, text),
                                visual_elements=self._suggest_visual_elements(broll_type, matching_keywords),
                                priority=priority,
                                trigger_keywords=matching_keywords,
                                educational_value=educational_value
                            )
                            
                            placements.append(placement)
            
            # Remove overlapping placements and sort by priority
            placements = self._resolve_broll_conflicts(placements)
            placements.sort(key=lambda x: (-x.priority, x.timestamp))
            
            logger.info(f"Detected {len(placements)} B-roll placement opportunities")
            return placements
            
        except Exception as e:
            logger.error(f"Failed to detect B-roll placements: {str(e)}")
            raise ContentContextError(
                f"Failed to detect B-roll placements: {str(e)}",
                context_state=context
            )
    
    def _calculate_educational_value(self, text: str, keywords: List[str]) -> float:
        """Calculate educational value of content segment."""
        # Base value from keyword match
        base_value = min(len(keywords) * 0.2, 0.8)
        
        # Bonus for financial complexity
        complexity_bonus = 0.0
        for term_list in self.complexity_indicators.values():
            if any(term in text for term in term_list):
                complexity_bonus += 0.1
        
        return min(base_value + complexity_bonus, 1.0)
    
    def _calculate_broll_priority(self, broll_type: str, educational_value: float) -> int:
        """Calculate priority for B-roll placement."""
        base_priorities = {
            'chart': 8,
            'animation': 7,
            'concept_visual': 9,
            'process_diagram': 6
        }
        
        base_priority = base_priorities.get(broll_type, 5)
        educational_bonus = int(educational_value * 2)  # 0-2 bonus points
        
        return min(base_priority + educational_bonus, 10)
    
    def _generate_broll_description(self, broll_type: str, keywords: List[str], text: str) -> str:
        """Generate description for B-roll content."""
        descriptions = {
            'chart': f"Data visualization for: {', '.join(keywords[:3])}",
            'animation': f"Animated explanation of: {', '.join(keywords[:2])}",
            'concept_visual': f"Visual representation of: {', '.join(keywords[:2])}",
            'process_diagram': f"Step-by-step diagram for: {', '.join(keywords[:2])}"
        }
        
        return descriptions.get(broll_type, f"Visual aid for: {', '.join(keywords[:2])}")
    
    def _suggest_visual_elements(self, broll_type: str, keywords: List[str]) -> List[str]:
        """Suggest specific visual elements for B-roll."""
        element_suggestions = {
            'chart': ['bar_chart', 'line_graph', 'pie_chart', 'comparison_table'],
            'animation': ['motion_graphics', 'icon_animation', 'text_animation'],
            'concept_visual': ['infographic', 'diagram', 'illustration'],
            'process_diagram': ['flowchart', 'step_diagram', 'timeline']
        }
        
        return element_suggestions.get(broll_type, ['generic_visual'])
    
    def _resolve_broll_conflicts(self, placements: List[BRollPlacement]) -> List[BRollPlacement]:
        """Remove overlapping B-roll placements."""
        if not placements:
            return placements
        
        # Sort by timestamp
        sorted_placements = sorted(placements, key=lambda x: x.timestamp)
        resolved = [sorted_placements[0]]
        
        for placement in sorted_placements[1:]:
            last_placement = resolved[-1]
            last_end = last_placement.timestamp + last_placement.duration
            
            # If no overlap, add placement
            if placement.timestamp >= last_end:
                resolved.append(placement)
            # If overlap, keep higher priority placement
            elif placement.priority > last_placement.priority:
                resolved[-1] = placement
        
        return resolved 
   
    @handle_errors()
    def suggest_transitions(self, context: ContentContext) -> List[TransitionSuggestion]:
        """
        Suggest optimal transitions between content segments.
        
        Args:
            context: ContentContext with segment analysis
            
        Returns:
            List of TransitionSuggestion objects with timing and type
            
        Raises:
            ContentContextError: If context is invalid or missing required data
        """
        if not context:
            raise ContentContextError("Invalid ContentContext provided")
        
        suggestions = []
        
        try:
            if hasattr(context, 'audio_transcript') and context.audio_transcript:
                transcript_segments = getattr(context.audio_transcript, 'segments', [])
                
                for i in range(len(transcript_segments) - 1):
                    current_segment = transcript_segments[i]
                    next_segment = transcript_segments[i + 1]
                    
                    current_text = getattr(current_segment, 'text', '').lower()
                    next_text = getattr(next_segment, 'text', '').lower()
                    
                    from_timestamp = getattr(current_segment, 'end', i * 5.0)
                    to_timestamp = getattr(next_segment, 'start', (i + 1) * 5.0)
                    
                    # Analyze content relationship
                    relationship = self._analyze_content_relationship(current_text, next_text)
                    transition_type = self.transition_mapping.get(relationship, 'cut')
                    
                    suggestion = TransitionSuggestion(
                        from_timestamp=from_timestamp,
                        to_timestamp=to_timestamp,
                        transition_type=transition_type,
                        parameters=self._get_transition_parameters(transition_type),
                        reason=f"Content relationship: {relationship}",
                        content_context=f"From: {current_text[:30]}... To: {next_text[:30]}..."
                    )
                    
                    suggestions.append(suggestion)
            
            logger.info(f"Generated {len(suggestions)} transition suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest transitions: {str(e)}")
            raise ContentContextError(
                f"Failed to suggest transitions: {str(e)}",
                context_state=context
            )
    
    def _analyze_content_relationship(self, current_text: str, next_text: str) -> str:
        """Analyze relationship between adjacent content segments."""
        next_text_lower = next_text.lower()
        
        # Check for topic change indicators
        topic_change_indicators = ['now', 'next', 'however', 'but', 'on the other hand', 'alternatively']
        if any(indicator in next_text_lower for indicator in topic_change_indicators):
            return 'topic_change'
        
        # Check for emotional shift indicators
        emotional_indicators = ['exciting', 'unfortunately', 'surprisingly', 'importantly']
        if any(indicator in next_text_lower for indicator in emotional_indicators):
            return 'emotional_shift'
        
        # Check for sequential content
        sequence_indicators = ['first', 'second', 'then', 'next', 'finally', 'step']
        if any(indicator in next_text_lower for indicator in sequence_indicators):
            return 'sequential_content'
        
        # Check for emphasis points
        emphasis_indicators = ['remember', 'important', 'key point', 'crucial', 'essential']
        if any(indicator in next_text_lower for indicator in emphasis_indicators):
            return 'emphasis_point'
        
        # Check for examples
        example_indicators = ['for example', 'for instance', 'let\'s say', 'imagine']
        if any(indicator in next_text_lower for indicator in example_indicators):
            return 'example_introduction'
        
        # Default to concept continuation
        return 'concept_continuation'
    
    def _get_transition_parameters(self, transition_type: str) -> Dict[str, Any]:
        """Get parameters for specific transition types."""
        parameters = {
            'cut': {'duration': 0.0},
            'fade': {'duration': 0.5, 'fade_type': 'cross_fade'},
            'slide': {'duration': 0.3, 'direction': 'left_to_right'},
            'zoom': {'duration': 0.4, 'zoom_factor': 1.2}
        }
        
        return parameters.get(transition_type, {'duration': 0.0})    

    @handle_errors()
    def optimize_pacing(self, context: ContentContext) -> PacingPlan:
        """
        Create pacing optimization plan based on content complexity and engagement.
        
        Args:
            context: ContentContext with content analysis
            
        Returns:
            PacingPlan with segment-by-segment recommendations
            
        Raises:
            ContentContextError: If context is invalid or missing required data
        """
        if not context:
            raise ContentContextError("Invalid ContentContext provided")
        
        try:
            segments = []
            overall_strategy = self._determine_pacing_strategy(context)
            
            if hasattr(context, 'audio_transcript') and context.audio_transcript:
                transcript_segments = getattr(context.audio_transcript, 'segments', [])
                
                for segment in transcript_segments:
                    text = getattr(segment, 'text', '').lower()
                    start_time = getattr(segment, 'start', 0.0)
                    end_time = getattr(segment, 'end', start_time + 5.0)
                    
                    # Analyze content complexity
                    complexity = self._analyze_content_complexity(text)
                    
                    # Determine recommended speed based on complexity
                    recommended_speed = self._calculate_recommended_speed(complexity, overall_strategy)
                    
                    # Generate reason for pacing decision
                    reason = self._generate_pacing_reason(complexity, recommended_speed)
                    
                    pacing_segment = PacingSegment(
                        start_timestamp=start_time,
                        end_timestamp=end_time,
                        recommended_speed=recommended_speed,
                        reason=reason,
                        content_complexity=complexity
                    )
                    
                    segments.append(pacing_segment)
            
            pacing_plan = PacingPlan(
                segments=segments,
                overall_strategy=overall_strategy
            )
            
            logger.info(f"Created pacing plan with {len(segments)} segments, strategy: {overall_strategy}")
            return pacing_plan
            
        except Exception as e:
            logger.error(f"Failed to optimize pacing: {str(e)}")
            raise ContentContextError(
                f"Failed to optimize pacing: {str(e)}",
                context_state=context
            )
    
    def _determine_pacing_strategy(self, context: ContentContext) -> str:
        """Determine overall pacing strategy based on content type."""
        if hasattr(context, 'content_type'):
            if context.content_type == ContentType.EDUCATIONAL:
                return 'educational_slow'
            elif context.content_type == ContentType.MUSIC:
                return 'engagement_varied'
        
        return 'retention_focused'
    
    def _analyze_content_complexity(self, text: str) -> float:
        """Analyze complexity of content segment."""
        complexity_score = 0.0
        
        # Check for financial terms
        financial_matches = sum(1 for term in self.complexity_indicators['financial_terms'] if term in text)
        complexity_score += financial_matches * 0.2
        
        # Check for abstract concepts
        abstract_matches = sum(1 for term in self.complexity_indicators['abstract_concepts'] if term in text)
        complexity_score += abstract_matches * 0.15
        
        # Check for mathematical content
        math_matches = sum(1 for term in self.complexity_indicators['mathematical_content'] if term in text)
        complexity_score += math_matches * 0.25
        
        # Check sentence complexity (words per sentence)
        sentences = text.split('.')
        if sentences:
            avg_words_per_sentence = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
            if avg_words_per_sentence > 15:
                complexity_score += 0.1
        
        return min(complexity_score, 1.0)
    
    def _calculate_recommended_speed(self, complexity: float, strategy: str) -> float:
        """Calculate recommended playback speed based on complexity and strategy."""
        base_speeds = {
            'educational_slow': 0.9,
            'engagement_varied': 1.0,
            'retention_focused': 1.1
        }
        
        base_speed = base_speeds.get(strategy, 1.0)
        
        # Adjust based on complexity
        if complexity > 0.7:
            speed_adjustment = -0.3  # Slow down for complex content
        elif complexity > 0.4:
            speed_adjustment = -0.1  # Slightly slow down
        elif complexity < 0.2:
            speed_adjustment = 0.2   # Speed up for simple content
        else:
            speed_adjustment = 0.0   # Normal speed
        
        recommended_speed = base_speed + speed_adjustment
        return max(0.5, min(2.0, recommended_speed))  # Clamp between 0.5x and 2.0x
    
    def _generate_pacing_reason(self, complexity: float, recommended_speed: float) -> str:
        """Generate human-readable reason for pacing decision."""
        if recommended_speed < 0.9:
            return f"Slow pacing for complex content (complexity: {complexity:.2f})"
        elif recommended_speed > 1.1:
            return f"Faster pacing for simple content (complexity: {complexity:.2f})"
        else:
            return f"Normal pacing for moderate complexity (complexity: {complexity:.2f})" 
   
    @handle_errors()
    def coordinate_with_ai_director(
        self, 
        context: ContentContext, 
        director_plan: AIDirectorPlan
    ) -> EnhancedEditingPlan:
        """
        Coordinate intelligence recommendations with AI Director's creative plan.
        
        Args:
            context: ContentContext with analysis data
            director_plan: AIDirectorPlan from FinancialVideoEditor
            
        Returns:
            EnhancedEditingPlan combining both recommendation sets
            
        Raises:
            ContentContextError: If context is invalid
            ModuleIntegrationError: If coordination fails
        """
        if not context:
            raise ContentContextError("Invalid ContentContext provided")
        
        if not director_plan:
            raise ModuleIntegrationError(
                "content_intelligence",
                reason="AI Director plan is required for coordination"
            )
        
        try:
            start_time = time.time()
            
            # Generate intelligence recommendations
            intelligence_recommendations = self.analyze_editing_opportunities(context)
            broll_enhancements = self.detect_broll_placements(context)
            transition_improvements = self.suggest_transitions(context)
            pacing_optimizations = self.optimize_pacing(context)
            
            # Coordinate with AI Director plan
            coordination_notes = []
            
            # Merge B-roll recommendations
            merged_broll = self._merge_broll_recommendations(
                director_plan.broll_plans, 
                broll_enhancements,
                coordination_notes
            )
            
            # Resolve editing decision conflicts
            resolved_editing = self._resolve_editing_conflicts(
                director_plan.editing_decisions,
                intelligence_recommendations,
                coordination_notes
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(
                intelligence_recommendations,
                broll_enhancements,
                director_plan
            )
            
            # Create enhanced plan
            enhanced_plan = EnhancedEditingPlan(
                ai_director_plan=director_plan,
                intelligence_recommendations=intelligence_recommendations,
                broll_enhancements=merged_broll,
                transition_improvements=transition_improvements,
                pacing_optimizations=pacing_optimizations,
                coordination_notes=coordination_notes,
                confidence_score=confidence_score
            )
            
            # Store enhanced plan in context
            context.enhanced_editing_plan = enhanced_plan
            
            # Update processing metrics
            processing_time = time.time() - start_time
            context.processing_metrics.add_module_processing_time('content_intelligence', processing_time)
            
            logger.info(f"Enhanced editing plan created with confidence: {confidence_score:.2f}")
            return enhanced_plan
            
        except Exception as e:
            logger.error(f"Failed to coordinate with AI Director: {str(e)}")
            raise ModuleIntegrationError(
                "content_intelligence",
                reason=f"Coordination failed: {str(e)}"
            )
    
    def _merge_broll_recommendations(
        self, 
        director_broll: List[BRollPlan], 
        intelligence_broll: List[BRollPlacement],
        coordination_notes: List[str]
    ) -> List[BRollPlacement]:
        """Merge B-roll recommendations from both sources."""
        merged = []
        
        # Convert AI Director B-roll plans to BRollPlacement format
        for director_plan in director_broll:
            placement = BRollPlacement(
                timestamp=director_plan.timestamp,
                duration=director_plan.duration,
                content_type=director_plan.content_type,
                description=director_plan.description,
                visual_elements=director_plan.visual_elements,
                priority=director_plan.priority,
                trigger_keywords=[],  # AI Director doesn't provide keywords
                educational_value=0.8  # Assume high educational value from AI Director
            )
            merged.append(placement)
        
        # Add intelligence recommendations that don't conflict
        for intel_broll in intelligence_broll:
            conflicts = [m for m in merged if abs(m.timestamp - intel_broll.timestamp) < 5.0]
            
            if not conflicts:
                merged.append(intel_broll)
                coordination_notes.append(f"Added intelligence B-roll at {intel_broll.timestamp:.1f}s")
            elif intel_broll.priority > max(c.priority for c in conflicts):
                # Replace lower priority conflicts
                merged = [m for m in merged if m not in conflicts]
                merged.append(intel_broll)
                coordination_notes.append(f"Replaced lower priority B-roll at {intel_broll.timestamp:.1f}s")
        
        return sorted(merged, key=lambda x: x.timestamp)
    
    def _resolve_editing_conflicts(
        self,
        director_decisions: List[EditingDecision],
        intelligence_opportunities: List[EditingOpportunity],
        coordination_notes: List[str]
    ) -> List[EditingOpportunity]:
        """Resolve conflicts between AI Director and Intelligence recommendations."""
        resolved = []
        
        # Convert AI Director decisions to EditingOpportunity format for comparison
        director_opportunities = []
        for decision in director_decisions:
            opportunity = EditingOpportunity(
                timestamp=decision.timestamp,
                opportunity_type=decision.decision_type,
                parameters=decision.parameters,
                confidence=decision.confidence,
                rationale=decision.rationale,
                priority=decision.priority,
                content_trigger="AI Director decision"
            )
            director_opportunities.append(opportunity)
        
        # Merge without conflicts
        all_opportunities = director_opportunities + intelligence_opportunities
        all_opportunities.sort(key=lambda x: x.timestamp)
        
        for opportunity in all_opportunities:
            conflicts = [r for r in resolved if abs(r.timestamp - opportunity.timestamp) < 2.0]
            
            if not conflicts:
                resolved.append(opportunity)
            elif opportunity.confidence > max(c.confidence for c in conflicts):
                # Keep higher confidence recommendation
                resolved = [r for r in resolved if r not in conflicts]
                resolved.append(opportunity)
                coordination_notes.append(f"Resolved conflict at {opportunity.timestamp:.1f}s using higher confidence")
        
        return resolved
    
    def _calculate_overall_confidence(
        self,
        intelligence_recommendations: List[EditingOpportunity],
        broll_enhancements: List[BRollPlacement],
        director_plan: AIDirectorPlan
    ) -> float:
        """Calculate overall confidence score for enhanced plan."""
        confidence_scores = []
        
        # Intelligence recommendations confidence
        if intelligence_recommendations:
            intel_confidence = sum(r.confidence for r in intelligence_recommendations) / len(intelligence_recommendations)
            confidence_scores.append(intel_confidence)
        
        # B-roll enhancements confidence (based on educational value)
        if broll_enhancements:
            broll_confidence = sum(b.educational_value for b in broll_enhancements) / len(broll_enhancements)
            confidence_scores.append(broll_confidence)
        
        # AI Director plan confidence
        if hasattr(director_plan, 'confidence_score'):
            confidence_scores.append(director_plan.confidence_score)
        
        # Return weighted average
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        else:
            return 0.7  # Default confidence