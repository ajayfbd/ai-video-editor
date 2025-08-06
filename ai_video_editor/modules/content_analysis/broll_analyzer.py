"""
B-Roll Opportunity Analysis - Automated detection of B-roll insertion opportunities.

This module implements the FinancialBRollAnalyzer class that identifies optimal
timing and content for B-roll graphics, charts, and animations in financial
educational content.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .content_analyzer import FinancialContentAnalyzer
from ...core.content_context import ContentContext, AudioSegment
from ...core.exceptions import ProcessingError
from ...core.cache_manager import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class BRollOpportunity:
    """Represents a detected B-roll insertion opportunity."""
    
    timestamp: float
    duration: float
    opportunity_type: str  # 'data_visualization', 'concept_explanation', 'process_diagram'
    content: str
    graphics_type: str  # 'chart_or_graph', 'animated_explanation', 'step_by_step_visual'
    confidence: float
    priority: int  # 1-10, higher is more important
    keywords: List[str] = field(default_factory=list)
    suggested_elements: List[str] = field(default_factory=list)
    animation_style: str = "fade_in"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'duration': self.duration,
            'opportunity_type': self.opportunity_type,
            'content': self.content,
            'graphics_type': self.graphics_type,
            'confidence': self.confidence,
            'priority': self.priority,
            'keywords': self.keywords,
            'suggested_elements': self.suggested_elements,
            'animation_style': self.animation_style
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BRollOpportunity':
        return cls(**data)


class FinancialBRollAnalyzer(FinancialContentAnalyzer):
    """
    Specialized analyzer for detecting B-roll opportunities in financial educational content.
    
    This class extends FinancialContentAnalyzer to identify specific moments where
    visual graphics, charts, or animations would enhance understanding and engagement.
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        super().__init__(cache_manager)
        
        # Visual trigger keywords organized by category
        self.visual_triggers = {
            'chart_keywords': [
                'percent', 'percentage', 'growth', 'decline', 'increase', 'decrease',
                'chart', 'graph', 'data', 'statistics', 'numbers', 'figure',
                'rate', 'ratio', 'comparison', 'versus', 'trend', 'pattern'
            ],
            'concept_keywords': [
                'compound interest', 'compounding', 'diversification', 'portfolio',
                'risk', 'return', 'investment', 'dividend', 'yield', 'allocation',
                'asset', 'equity', 'bond', 'stock', 'market', 'inflation',
                'volatility', 'correlation', 'beta', 'alpha'
            ],
            'comparison_keywords': [
                'versus', 'vs', 'compared to', 'better than', 'worse than',
                'difference between', 'contrast', 'alternative', 'option',
                'choice', 'versus', 'against', 'competition'
            ],
            'process_keywords': [
                'steps', 'process', 'how to', 'method', 'strategy', 'approach',
                'technique', 'procedure', 'workflow', 'sequence', 'order',
                'first', 'second', 'third', 'next', 'then', 'finally'
            ],
            'mathematical_keywords': [
                'formula', 'equation', 'calculate', 'computation', 'math',
                'multiply', 'divide', 'add', 'subtract', 'percentage',
                'interest rate', 'compound', 'exponential', 'logarithmic'
            ],
            'time_keywords': [
                'years', 'months', 'decades', 'long term', 'short term',
                'time period', 'duration', 'timeline', 'over time',
                'historical', 'future', 'projection', 'forecast'
            ]
        }
        
        # Duration recommendations based on content type
        self.duration_guidelines = {
            'data_visualization': {'min': 3.0, 'max': 8.0, 'optimal': 5.0},
            'concept_explanation': {'min': 4.0, 'max': 12.0, 'optimal': 8.0},
            'process_diagram': {'min': 5.0, 'max': 15.0, 'optimal': 10.0},
            'mathematical_formula': {'min': 3.0, 'max': 10.0, 'optimal': 6.0},
            'comparison_chart': {'min': 4.0, 'max': 8.0, 'optimal': 6.0},
            'timeline_visualization': {'min': 6.0, 'max': 12.0, 'optimal': 9.0}
        }
        
        # Graphics specifications for different content types
        self.graphics_specs = {
            'chart_or_graph': {
                'elements': ['x_axis', 'y_axis', 'data_points', 'trend_line', 'labels'],
                'animation_options': ['fade_in', 'draw_on', 'slide_up', 'zoom_in'],
                'priority_base': 8
            },
            'animated_explanation': {
                'elements': ['icons', 'arrows', 'text_overlays', 'highlights'],
                'animation_options': ['fade_in', 'slide_in', 'bounce', 'morph'],
                'priority_base': 7
            },
            'step_by_step_visual': {
                'elements': ['numbered_steps', 'flowchart', 'progress_indicator'],
                'animation_options': ['sequential_reveal', 'step_by_step', 'cascade'],
                'priority_base': 9
            },
            'formula_visualization': {
                'elements': ['mathematical_notation', 'variables', 'calculations'],
                'animation_options': ['typewriter', 'build_up', 'solve_animation'],
                'priority_base': 6
            },
            'comparison_table': {
                'elements': ['side_by_side', 'highlighting', 'differences'],
                'animation_options': ['side_slide', 'highlight_differences', 'flip'],
                'priority_base': 7
            }
        }
        
        # Processing metrics
        self.analysis_time = 0.0
        self.opportunities_detected = 0
        
        logger.info("FinancialBRollAnalyzer initialized with visual trigger detection")
    
    def detect_broll_opportunities(self, context: ContentContext) -> List[BRollOpportunity]:
        """
        Detect B-roll insertion opportunities in financial educational content.
        
        Args:
            context: ContentContext with audio analysis and transcript data
            
        Returns:
            List of BRollOpportunity objects with timing and content specifications
        """
        start_time = time.time()
        
        if not context.audio_analysis or not context.audio_analysis.segments:
            logger.warning("No audio analysis data available for B-roll detection")
            return []
        
        logger.info("Analyzing content for B-roll opportunities")
        
        opportunities = []
        segments = context.audio_analysis.segments
        
        for segment in segments:
            segment_opportunities = self._analyze_segment_for_broll(segment)
            opportunities.extend(segment_opportunities)
        
        # Post-process opportunities
        opportunities = self._optimize_opportunity_timing(opportunities)
        opportunities = self._assign_priorities(opportunities)
        opportunities = self._filter_and_merge_opportunities(opportunities)
        
        self.analysis_time = time.time() - start_time
        self.opportunities_detected = len(opportunities)
        
        logger.info(f"Detected {len(opportunities)} B-roll opportunities in {self.analysis_time:.2f}s")
        
        return opportunities
    
    def _analyze_segment_for_broll(self, segment: AudioSegment) -> List[BRollOpportunity]:
        """Analyze individual audio segment for B-roll opportunities."""
        opportunities = []
        text = segment.text.lower()
        timestamp = segment.start
        segment_duration = segment.end - segment.start
        
        # Check for data visualization opportunities
        if self._contains_keywords(text, self.visual_triggers['chart_keywords']):
            opportunity = self._create_data_visualization_opportunity(
                segment, text, timestamp, segment_duration
            )
            if opportunity:
                opportunities.append(opportunity)
        
        # Check for concept explanation opportunities
        if self._contains_keywords(text, self.visual_triggers['concept_keywords']):
            opportunity = self._create_concept_explanation_opportunity(
                segment, text, timestamp, segment_duration
            )
            if opportunity:
                opportunities.append(opportunity)
        
        # Check for process diagram opportunities
        if self._contains_keywords(text, self.visual_triggers['process_keywords']):
            opportunity = self._create_process_diagram_opportunity(
                segment, text, timestamp, segment_duration
            )
            if opportunity:
                opportunities.append(opportunity)
        
        # Check for mathematical formula opportunities
        if self._contains_keywords(text, self.visual_triggers['mathematical_keywords']):
            opportunity = self._create_formula_opportunity(
                segment, text, timestamp, segment_duration
            )
            if opportunity:
                opportunities.append(opportunity)
        
        # Check for comparison opportunities
        if self._contains_keywords(text, self.visual_triggers['comparison_keywords']):
            opportunity = self._create_comparison_opportunity(
                segment, text, timestamp, segment_duration
            )
            if opportunity:
                opportunities.append(opportunity)
        
        return opportunities
    
    def _create_data_visualization_opportunity(self, segment: AudioSegment, text: str, 
                                             timestamp: float, segment_duration: float) -> Optional[BRollOpportunity]:
        """Create opportunity for data visualization (charts, graphs)."""
        keywords = self._extract_matching_keywords(text, self.visual_triggers['chart_keywords'])
        
        # Determine optimal duration
        duration = min(
            max(segment_duration, self.duration_guidelines['data_visualization']['min']),
            self.duration_guidelines['data_visualization']['max']
        )
        
        # Calculate confidence based on keyword density and financial concepts
        confidence = self._calculate_confidence(text, keywords, segment.financial_concepts)
        
        # Suggest specific visual elements
        suggested_elements = ['chart', 'data_points', 'trend_lines']
        if 'percentage' in text or 'percent' in text:
            suggested_elements.append('percentage_labels')
        if 'growth' in text or 'increase' in text:
            suggested_elements.append('upward_trend')
        if 'decline' in text or 'decrease' in text:
            suggested_elements.append('downward_trend')
        
        return BRollOpportunity(
            timestamp=timestamp,
            duration=duration,
            opportunity_type='data_visualization',
            content=segment.text,
            graphics_type='chart_or_graph',
            confidence=confidence,
            priority=0,  # Will be assigned later
            keywords=keywords,
            suggested_elements=suggested_elements,
            animation_style='draw_on' if 'growth' in text else 'fade_in'
        )
    
    def _create_concept_explanation_opportunity(self, segment: AudioSegment, text: str,
                                              timestamp: float, segment_duration: float) -> Optional[BRollOpportunity]:
        """Create opportunity for concept explanation graphics."""
        keywords = self._extract_matching_keywords(text, self.visual_triggers['concept_keywords'])
        
        duration = min(
            max(segment_duration, self.duration_guidelines['concept_explanation']['min']),
            self.duration_guidelines['concept_explanation']['max']
        )
        
        confidence = self._calculate_confidence(text, keywords, segment.financial_concepts)
        
        # Suggest elements based on specific concepts
        suggested_elements = ['concept_icon', 'explanation_text', 'visual_metaphor']
        if 'compound interest' in text:
            suggested_elements.extend(['snowball_effect', 'exponential_curve'])
        if 'diversification' in text:
            suggested_elements.extend(['portfolio_pie_chart', 'risk_distribution'])
        if 'risk' in text:
            suggested_elements.extend(['risk_meter', 'volatility_indicator'])
        
        return BRollOpportunity(
            timestamp=timestamp,
            duration=duration,
            opportunity_type='concept_explanation',
            content=segment.text,
            graphics_type='animated_explanation',
            confidence=confidence,
            priority=0,
            keywords=keywords,
            suggested_elements=suggested_elements,
            animation_style='slide_in'
        )
    
    def _create_process_diagram_opportunity(self, segment: AudioSegment, text: str,
                                          timestamp: float, segment_duration: float) -> Optional[BRollOpportunity]:
        """Create opportunity for process/step diagrams."""
        keywords = self._extract_matching_keywords(text, self.visual_triggers['process_keywords'])
        
        duration = min(
            max(segment_duration, self.duration_guidelines['process_diagram']['min']),
            self.duration_guidelines['process_diagram']['max']
        )
        
        confidence = self._calculate_confidence(text, keywords, segment.financial_concepts)
        
        suggested_elements = ['numbered_steps', 'arrows', 'flowchart']
        if 'step' in text:
            suggested_elements.append('step_indicators')
        if 'process' in text:
            suggested_elements.append('process_flow')
        
        return BRollOpportunity(
            timestamp=timestamp,
            duration=duration,
            opportunity_type='process_diagram',
            content=segment.text,
            graphics_type='step_by_step_visual',
            confidence=confidence,
            priority=0,
            keywords=keywords,
            suggested_elements=suggested_elements,
            animation_style='sequential_reveal'
        )
    
    def _create_formula_opportunity(self, segment: AudioSegment, text: str,
                                  timestamp: float, segment_duration: float) -> Optional[BRollOpportunity]:
        """Create opportunity for mathematical formula visualization."""
        keywords = self._extract_matching_keywords(text, self.visual_triggers['mathematical_keywords'])
        
        duration = min(
            max(segment_duration, self.duration_guidelines['mathematical_formula']['min']),
            self.duration_guidelines['mathematical_formula']['max']
        )
        
        confidence = self._calculate_confidence(text, keywords, segment.financial_concepts)
        
        suggested_elements = ['formula', 'variables', 'calculation_steps']
        if 'compound' in text:
            suggested_elements.extend(['compound_formula', 'time_variable'])
        if 'interest' in text:
            suggested_elements.append('interest_calculation')
        
        return BRollOpportunity(
            timestamp=timestamp,
            duration=duration,
            opportunity_type='formula_visualization',
            content=segment.text,
            graphics_type='formula_visualization',
            confidence=confidence,
            priority=0,
            keywords=keywords,
            suggested_elements=suggested_elements,
            animation_style='typewriter'
        )
    
    def _create_comparison_opportunity(self, segment: AudioSegment, text: str,
                                     timestamp: float, segment_duration: float) -> Optional[BRollOpportunity]:
        """Create opportunity for comparison visualizations."""
        keywords = self._extract_matching_keywords(text, self.visual_triggers['comparison_keywords'])
        
        duration = min(
            max(segment_duration, self.duration_guidelines['comparison_chart']['min']),
            self.duration_guidelines['comparison_chart']['max']
        )
        
        confidence = self._calculate_confidence(text, keywords, segment.financial_concepts)
        
        suggested_elements = ['side_by_side_comparison', 'highlight_differences', 'versus_indicator']
        
        return BRollOpportunity(
            timestamp=timestamp,
            duration=duration,
            opportunity_type='comparison_visualization',
            content=segment.text,
            graphics_type='comparison_table',
            confidence=confidence,
            priority=0,
            keywords=keywords,
            suggested_elements=suggested_elements,
            animation_style='side_slide'
        )
    
    def _contains_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the specified keywords."""
        return any(keyword in text for keyword in keywords)
    
    def _extract_matching_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Extract keywords that are found in the text."""
        return [keyword for keyword in keywords if keyword in text]
    
    def _calculate_confidence(self, text: str, keywords: List[str], financial_concepts: List[str]) -> float:
        """Calculate confidence score for B-roll opportunity."""
        base_confidence = 0.5
        
        # Increase confidence based on keyword density
        keyword_density = len(keywords) / max(len(text.split()), 1)
        confidence_boost = min(keyword_density * 2, 0.3)
        
        # Increase confidence based on financial concepts
        if financial_concepts:
            confidence_boost += min(len(financial_concepts) * 0.1, 0.2)
        
        # Increase confidence for specific high-value keywords
        high_value_keywords = ['compound interest', 'diversification', 'growth', 'chart', 'data']
        for hvk in high_value_keywords:
            if hvk in text:
                confidence_boost += 0.1
        
        return min(base_confidence + confidence_boost, 0.95)
    
    def _optimize_opportunity_timing(self, opportunities: List[BRollOpportunity]) -> List[BRollOpportunity]:
        """Optimize timing and prevent overlapping opportunities."""
        if not opportunities:
            return opportunities
        
        # Sort by timestamp
        opportunities.sort(key=lambda x: x.timestamp)
        
        optimized = [opportunities[0]]
        
        for current in opportunities[1:]:
            last = optimized[-1]
            
            # Check for overlap
            if current.timestamp < last.timestamp + last.duration:
                # Merge or adjust timing
                if current.confidence > last.confidence:
                    # Replace with higher confidence opportunity
                    optimized[-1] = current
                else:
                    # Adjust timing to avoid overlap
                    current.timestamp = last.timestamp + last.duration + 0.5
                    optimized.append(current)
            else:
                optimized.append(current)
        
        return optimized
    
    def _assign_priorities(self, opportunities: List[BRollOpportunity]) -> List[BRollOpportunity]:
        """Assign priority scores to opportunities."""
        for opportunity in opportunities:
            base_priority = self.graphics_specs.get(
                opportunity.graphics_type, {'priority_base': 5}
            )['priority_base']
            
            # Adjust priority based on confidence
            confidence_adjustment = int(opportunity.confidence * 2)
            
            # Adjust priority based on content value
            content_value = self._assess_content_value(opportunity.content)
            
            opportunity.priority = min(base_priority + confidence_adjustment + content_value, 10)
        
        return opportunities
    
    def _assess_content_value(self, content: str) -> int:
        """Assess the educational value of content for priority adjustment."""
        value = 0
        content_lower = content.lower()
        
        # High-value financial concepts
        high_value_concepts = [
            'compound interest', 'diversification', 'risk management',
            'asset allocation', 'portfolio', 'investment strategy'
        ]
        
        for concept in high_value_concepts:
            if concept in content_lower:
                value += 2
        
        # Mathematical or data-driven content
        if any(term in content_lower for term in ['calculate', 'formula', 'percentage', 'data']):
            value += 1
        
        return min(value, 3)
    
    def _filter_and_merge_opportunities(self, opportunities: List[BRollOpportunity]) -> List[BRollOpportunity]:
        """Filter low-quality opportunities and merge similar ones."""
        # Filter by minimum confidence threshold
        filtered = [opp for opp in opportunities if opp.confidence >= 0.6]
        
        # Limit to top opportunities to avoid overwhelming the viewer
        max_opportunities = 8
        if len(filtered) > max_opportunities:
            filtered.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
            filtered = filtered[:max_opportunities]
        
        return filtered
    
    def integrate_with_ai_director(self, context: ContentContext, opportunities: List[BRollOpportunity]) -> List[Dict[str, Any]]:
        """
        Convert B-roll opportunities to AI Director-compatible format.
        
        Args:
            context: ContentContext with existing AI Director plan
            opportunities: List of detected B-roll opportunities
            
        Returns:
            List of B-roll plans compatible with AI Director structure
        """
        ai_director_plans = []
        
        for opportunity in opportunities:
            plan = {
                'timestamp': opportunity.timestamp,
                'duration': opportunity.duration,
                'content_type': opportunity.graphics_type,
                'description': f"{opportunity.opportunity_type}: {opportunity.content[:100]}...",
                'visual_elements': opportunity.suggested_elements,
                'animation_style': opportunity.animation_style,
                'priority': opportunity.priority,
                'confidence': opportunity.confidence,
                'keywords': opportunity.keywords
            }
            ai_director_plans.append(plan)
        
        logger.info(f"Converted {len(opportunities)} opportunities to AI Director format")
        return ai_director_plans
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get statistics about the B-roll analysis process."""
        return {
            'analysis_time': self.analysis_time,
            'opportunities_detected': self.opportunities_detected,
            'trigger_categories': len(self.visual_triggers),
            'graphics_types_supported': len(self.graphics_specs)
        }
