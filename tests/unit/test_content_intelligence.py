"""
Unit tests for ContentIntelligenceEngine.

This module tests the ContentIntelligenceEngine class with comprehensive mocking
of ContentContext and AI Director integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from ai_video_editor.modules.intelligence.content_intelligence import (
    ContentIntelligenceEngine,
    EditingOpportunity,
    BRollPlacement,
    TransitionSuggestion,
    PacingSegment,
    PacingPlan,
    EnhancedEditingPlan
)
from ai_video_editor.modules.intelligence.ai_director import (
    AIDirectorPlan,
    EditingDecision,
    BRollPlan,
    MetadataStrategy
)
from ai_video_editor.core.content_context import (
    ContentContext,
    ContentType,
    EmotionalPeak,
    VisualHighlight
)
from ai_video_editor.core.exceptions import (
    ContentContextError,
    ModuleIntegrationError
)


class TestContentIntelligenceEngine:
    """Test suite for ContentIntelligenceEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create ContentIntelligenceEngine instance for testing."""
        return ContentIntelligenceEngine(enable_advanced_analysis=True)
    
    @pytest.fixture
    def mock_transcript_segment(self):
        """Create mock transcript segment."""
        segment = Mock()
        segment.text = "Let's talk about compound interest and how it works."
        segment.start = 10.0
        segment.end = 15.0
        return segment
    
    @pytest.fixture
    def mock_transcript(self):
        """Create mock transcript with segments."""
        transcript = Mock()
        segments = []
        
        # Create multiple segments
        segment1 = Mock()
        segment1.text = "Welcome to financial education. Today we'll discuss compound interest."
        segment1.start = 0.0
        segment1.end = 5.0
        
        segment2 = Mock()
        segment2.text = "Compound interest is the process where your money grows exponentially."
        segment2.start = 5.0
        segment2.end = 10.0
        
        segment3 = Mock()
        segment3.text = "For example, if you invest $1000 at 5% annual interest rate."
        segment3.start = 10.0
        segment3.end = 15.0
        
        segments = [segment1, segment2, segment3]
        transcript.segments = segments
        return transcript
    
    @pytest.fixture
    def mock_emotional_peaks(self):
        """Create mock emotional peaks."""
        return [
            EmotionalPeak(
                timestamp=12.5,
                emotion="excitement",
                intensity=0.8,
                confidence=0.9,
                context="Explaining compound interest benefits"
            ),
            EmotionalPeak(
                timestamp=25.0,
                emotion="curiosity",
                intensity=0.6,
                confidence=0.7,
                context="Introducing investment example"
            )
        ]
    
    @pytest.fixture
    def mock_content_context(self, mock_transcript, mock_emotional_peaks):
        """Create mock ContentContext for testing."""
        context = Mock(spec=ContentContext)
        context.content_type = ContentType.EDUCATIONAL
        context.audio_transcript = mock_transcript
        context.emotional_markers = mock_emotional_peaks
        context.key_concepts = ["compound interest", "investment", "growth"]
        context.total_duration = 180.0
        context.processing_metrics = Mock()
        context.processing_metrics.add_module_processing_time = Mock()
        return context 
   
    @pytest.fixture
    def mock_ai_director_plan(self):
        """Create mock AI Director plan."""
        editing_decisions = [
            EditingDecision(
                timestamp=5.0,
                decision_type="cut",
                parameters={"fade_duration": 0.5},
                rationale="Natural break in content",
                confidence=0.8,
                priority=7
            )
        ]
        
        broll_plans = [
            BRollPlan(
                timestamp=12.0,
                duration=6.0,
                content_type="chart",
                description="Compound interest growth chart",
                visual_elements=["line_graph", "data_points"],
                animation_style="fade_in",
                priority=8
            )
        ]
        
        metadata_strategy = MetadataStrategy(
            primary_title="Understanding Compound Interest",
            title_variations=["Compound Interest Explained", "How Compound Interest Works"],
            description="Learn about compound interest and its benefits",
            tags=["finance", "investment", "compound interest"],
            thumbnail_concepts=["growth chart", "money tree"],
            hook_text="Discover the power of compound interest",
            target_keywords=["compound interest", "investment growth"]
        )
        
        return AIDirectorPlan(
            editing_decisions=editing_decisions,
            broll_plans=broll_plans,
            metadata_strategy=metadata_strategy,
            quality_enhancements=["audio_cleanup"],
            pacing_adjustments=[],
            engagement_hooks=[],
            created_at=datetime.now(),
            confidence_score=0.85,
            processing_time=2.5,
            model_used="gemini-2.5-pro-latest"
        )
    
    def test_initialization(self, engine):
        """Test ContentIntelligenceEngine initialization."""
        assert engine.enable_advanced_analysis is True
        assert 'chart' in engine.broll_triggers
        assert 'cut' in engine.transition_mapping.values()
        assert 'financial_terms' in engine.complexity_indicators
    
    def test_analyze_editing_opportunities_success(self, engine, mock_content_context):
        """Test successful editing opportunities analysis."""
        opportunities = engine.analyze_editing_opportunities(mock_content_context)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        # Check that opportunities are properly structured
        for opportunity in opportunities:
            assert isinstance(opportunity, EditingOpportunity)
            assert hasattr(opportunity, 'timestamp')
            assert hasattr(opportunity, 'opportunity_type')
            assert hasattr(opportunity, 'confidence')
            assert 0.0 <= opportunity.confidence <= 1.0
            assert 1 <= opportunity.priority <= 10
    
    def test_analyze_editing_opportunities_invalid_context(self, engine):
        """Test editing opportunities analysis with invalid context."""
        with pytest.raises(ContentContextError):
            engine.analyze_editing_opportunities(None)
    
    def test_detect_broll_placements_success(self, engine, mock_content_context):
        """Test successful B-roll placement detection."""
        placements = engine.detect_broll_placements(mock_content_context)
        
        assert isinstance(placements, list)
        assert len(placements) > 0
        
        # Check that placements are properly structured
        for placement in placements:
            assert isinstance(placement, BRollPlacement)
            assert hasattr(placement, 'timestamp')
            assert hasattr(placement, 'content_type')
            assert hasattr(placement, 'educational_value')
            assert 0.0 <= placement.educational_value <= 1.0
            assert 1 <= placement.priority <= 10
    
    def test_detect_broll_placements_keyword_matching(self, engine, mock_content_context):
        """Test B-roll placement keyword matching."""
        placements = engine.detect_broll_placements(mock_content_context)
        
        # Should detect compound interest as concept_visual
        concept_placements = [p for p in placements if p.content_type == 'concept_visual']
        assert len(concept_placements) > 0
        
        # Check trigger keywords
        compound_placement = next((p for p in concept_placements 
                                 if 'compound interest' in p.trigger_keywords), None)
        assert compound_placement is not None
    
    def test_suggest_transitions_success(self, engine, mock_content_context):
        """Test successful transition suggestions."""
        suggestions = engine.suggest_transitions(mock_content_context)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Check that suggestions are properly structured
        for suggestion in suggestions:
            assert isinstance(suggestion, TransitionSuggestion)
            assert hasattr(suggestion, 'from_timestamp')
            assert hasattr(suggestion, 'to_timestamp')
            assert hasattr(suggestion, 'transition_type')
            assert suggestion.transition_type in ['cut', 'fade', 'slide', 'zoom']    

    def test_optimize_pacing_success(self, engine, mock_content_context):
        """Test successful pacing optimization."""
        pacing_plan = engine.optimize_pacing(mock_content_context)
        
        assert isinstance(pacing_plan, PacingPlan)
        assert hasattr(pacing_plan, 'segments')
        assert hasattr(pacing_plan, 'overall_strategy')
        assert pacing_plan.overall_strategy in ['educational_slow', 'engagement_varied', 'retention_focused']
        
        # Check segments
        assert len(pacing_plan.segments) > 0
        for segment in pacing_plan.segments:
            assert isinstance(segment, PacingSegment)
            assert 0.5 <= segment.recommended_speed <= 2.0
            assert 0.0 <= segment.content_complexity <= 1.0
    
    def test_optimize_pacing_educational_content(self, engine, mock_content_context):
        """Test pacing optimization for educational content."""
        mock_content_context.content_type = ContentType.EDUCATIONAL
        pacing_plan = engine.optimize_pacing(mock_content_context)
        
        assert pacing_plan.overall_strategy == 'educational_slow'
    
    def test_coordinate_with_ai_director_success(self, engine, mock_content_context, mock_ai_director_plan):
        """Test successful coordination with AI Director."""
        enhanced_plan = engine.coordinate_with_ai_director(mock_content_context, mock_ai_director_plan)
        
        assert isinstance(enhanced_plan, EnhancedEditingPlan)
        assert enhanced_plan.ai_director_plan == mock_ai_director_plan
        assert hasattr(enhanced_plan, 'intelligence_recommendations')
        assert hasattr(enhanced_plan, 'broll_enhancements')
        assert hasattr(enhanced_plan, 'transition_improvements')
        assert hasattr(enhanced_plan, 'pacing_optimizations')
        assert hasattr(enhanced_plan, 'coordination_notes')
        assert 0.0 <= enhanced_plan.confidence_score <= 1.0
    
    def test_coordinate_with_ai_director_invalid_context(self, engine, mock_ai_director_plan):
        """Test coordination with invalid context."""
        with pytest.raises(ContentContextError):
            engine.coordinate_with_ai_director(None, mock_ai_director_plan)
    
    def test_coordinate_with_ai_director_missing_plan(self, engine, mock_content_context):
        """Test coordination with missing AI Director plan."""
        with pytest.raises(ModuleIntegrationError):
            engine.coordinate_with_ai_director(mock_content_context, None)
    
    def test_calculate_educational_value(self, engine):
        """Test educational value calculation."""
        # High complexity text
        complex_text = "compound interest diversification portfolio allocation"
        keywords = ["compound", "diversification", "portfolio"]
        value = engine._calculate_educational_value(complex_text, keywords)
        assert 0.0 <= value <= 1.0
        assert value > 0.5  # Should be high for complex financial content
        
        # Simple text
        simple_text = "hello world"
        simple_keywords = ["hello"]
        simple_value = engine._calculate_educational_value(simple_text, simple_keywords)
        assert simple_value < value  # Should be lower than complex content
    
    def test_calculate_broll_priority(self, engine):
        """Test B-roll priority calculation."""
        # High educational value concept visual
        priority = engine._calculate_broll_priority('concept_visual', 0.9)
        assert priority >= 9  # Should be high priority
        
        # Low educational value process diagram
        low_priority = engine._calculate_broll_priority('process_diagram', 0.1)
        assert low_priority <= 7  # Should be lower priority
    
    def test_analyze_content_relationship(self, engine):
        """Test content relationship analysis."""
        # Topic change
        current = "We discussed stocks."
        next_text = "However, let's talk about bonds."
        relationship = engine._analyze_content_relationship(current, next_text)
        assert relationship == 'topic_change'
        
        # Also test with "now" which should work
        next_text_now = "Now let's talk about bonds."
        relationship_now = engine._analyze_content_relationship(current, next_text_now)
        assert relationship_now == 'topic_change'
        
        # Sequential content
        current = "First step is to save money."
        next_text = "Second, you need to choose investments."
        relationship = engine._analyze_content_relationship(current, next_text)
        assert relationship == 'sequential_content'
        
        # Example introduction
        current = "This concept is important."
        next_text = "For example, if you invest $1000..."
        relationship = engine._analyze_content_relationship(current, next_text)
        assert relationship == 'example_introduction'
    
    def test_analyze_content_complexity(self, engine):
        """Test content complexity analysis."""
        # Complex financial text
        complex_text = "compound interest diversification portfolio allocation calculate formula"
        complexity = engine._analyze_content_complexity(complex_text)
        assert complexity > 0.5
        
        # Simple text
        simple_text = "hello world this is simple"
        simple_complexity = engine._analyze_content_complexity(simple_text)
        assert simple_complexity < complexity
    
    def test_resolve_broll_conflicts(self, engine):
        """Test B-roll conflict resolution."""
        # Create overlapping placements
        placement1 = BRollPlacement(
            timestamp=10.0, duration=5.0, content_type="chart",
            description="Chart 1", visual_elements=[], priority=7,
            trigger_keywords=[], educational_value=0.8
        )
        
        placement2 = BRollPlacement(
            timestamp=12.0, duration=4.0, content_type="animation",
            description="Animation 1", visual_elements=[], priority=9,
            trigger_keywords=[], educational_value=0.9
        )
        
        placements = [placement1, placement2]
        resolved = engine._resolve_broll_conflicts(placements)
        
        # Should keep higher priority placement
        assert len(resolved) == 1
        assert resolved[0].priority == 9
    
    def test_error_handling_with_missing_transcript(self, engine):
        """Test error handling when transcript is missing."""
        context = Mock(spec=ContentContext)
        context.audio_transcript = None
        context.emotional_markers = []
        context.key_concepts = []
        context.total_duration = 180.0
        
        # Should not raise error, but return empty results
        opportunities = engine.analyze_editing_opportunities(context)
        assert isinstance(opportunities, list)
        
        placements = engine.detect_broll_placements(context)
        assert isinstance(placements, list)
    
    def test_performance_benchmarks(self, engine, mock_content_context):
        """Test that methods meet performance benchmarks."""
        import time
        
        # Test analyze_editing_opportunities performance
        start_time = time.time()
        engine.analyze_editing_opportunities(mock_content_context)
        elapsed = time.time() - start_time
        assert elapsed < 2.0  # Should complete in under 2 seconds
        
        # Test detect_broll_placements performance
        start_time = time.time()
        engine.detect_broll_placements(mock_content_context)
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Should complete in under 1 second
        
        # Test suggest_transitions performance
        start_time = time.time()
        engine.suggest_transitions(mock_content_context)
        elapsed = time.time() - start_time
        assert elapsed < 0.5  # Should complete in under 0.5 seconds
        
        # Test optimize_pacing performance
        start_time = time.time()
        engine.optimize_pacing(mock_content_context)
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Should complete in under 1 second


class TestDataStructures:
    """Test data structure serialization and deserialization."""
    
    def test_editing_opportunity_serialization(self):
        """Test EditingOpportunity to_dict and from_dict."""
        opportunity = EditingOpportunity(
            timestamp=10.0,
            opportunity_type="cut",
            parameters={"fade_duration": 0.5},
            confidence=0.8,
            rationale="Natural break",
            priority=7,
            content_trigger="Sentence end"
        )
        
        # Test serialization
        data = opportunity.to_dict()
        assert isinstance(data, dict)
        assert data['timestamp'] == 10.0
        assert data['opportunity_type'] == "cut"
        
        # Test deserialization
        restored = EditingOpportunity.from_dict(data)
        assert restored.timestamp == opportunity.timestamp
        assert restored.opportunity_type == opportunity.opportunity_type
        assert restored.confidence == opportunity.confidence
    
    def test_broll_placement_serialization(self):
        """Test BRollPlacement to_dict and from_dict."""
        placement = BRollPlacement(
            timestamp=15.0,
            duration=6.0,
            content_type="chart",
            description="Growth chart",
            visual_elements=["line_graph"],
            priority=8,
            trigger_keywords=["growth", "data"],
            educational_value=0.9
        )
        
        # Test serialization
        data = placement.to_dict()
        assert isinstance(data, dict)
        assert data['timestamp'] == 15.0
        assert data['content_type'] == "chart"
        
        # Test deserialization
        restored = BRollPlacement.from_dict(data)
        assert restored.timestamp == placement.timestamp
        assert restored.content_type == placement.content_type
        assert restored.educational_value == placement.educational_value
    
    def test_pacing_plan_serialization(self):
        """Test PacingPlan to_dict and from_dict."""
        segment = PacingSegment(
            start_timestamp=0.0,
            end_timestamp=10.0,
            recommended_speed=0.8,
            reason="Complex content",
            content_complexity=0.7
        )
        
        plan = PacingPlan(
            segments=[segment],
            overall_strategy="educational_slow"
        )
        
        # Test serialization
        data = plan.to_dict()
        assert isinstance(data, dict)
        assert data['overall_strategy'] == "educational_slow"
        assert len(data['segments']) == 1
        
        # Test deserialization
        restored = PacingPlan.from_dict(data)
        assert restored.overall_strategy == plan.overall_strategy
        assert len(restored.segments) == 1
        assert restored.segments[0].recommended_speed == 0.8