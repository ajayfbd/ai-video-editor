"""
Unit tests for AI Director module.

This module tests the FinancialVideoEditor and related components with comprehensive
mocking to avoid actual API calls and video processing.
"""

import pytest
import json
import time
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from ai_video_editor.modules.intelligence.ai_director import (
    FinancialVideoEditor,
    EditingDecision,
    BRollPlan,
    MetadataStrategy,
    AIDirectorPlan,
    create_financial_video_editor,
    validate_editing_plan,
    merge_editing_plans,
    get_content_level_prompt
)
from ai_video_editor.modules.intelligence.gemini_client import GeminiClient, GeminiConfig, GeminiResponse
from ai_video_editor.core.content_context import ContentContext, ContentType, EmotionalPeak, VisualHighlight
from ai_video_editor.core.exceptions import GeminiAPIError, ContentContextError


class TestEditingDecision:
    """Test EditingDecision data class."""
    
    def test_editing_decision_creation(self):
        """Test creating EditingDecision with valid data."""
        decision = EditingDecision(
            timestamp=30.5,
            decision_type="cut",
            parameters={"fade_duration": 1.0},
            rationale="Remove filler word",
            confidence=0.95,
            priority=8
        )
        
        assert decision.timestamp == 30.5
        assert decision.decision_type == "cut"
        assert decision.parameters == {"fade_duration": 1.0}
        assert decision.rationale == "Remove filler word"
        assert decision.confidence == 0.95
        assert decision.priority == 8
    
    def test_editing_decision_to_dict(self):
        """Test converting EditingDecision to dictionary."""
        decision = EditingDecision(
            timestamp=30.5,
            decision_type="cut",
            parameters={"fade_duration": 1.0},
            rationale="Remove filler word",
            confidence=0.95,
            priority=8
        )
        
        decision_dict = decision.to_dict()
        
        assert decision_dict["timestamp"] == 30.5
        assert decision_dict["decision_type"] == "cut"
        assert decision_dict["parameters"] == {"fade_duration": 1.0}
        assert decision_dict["rationale"] == "Remove filler word"
        assert decision_dict["confidence"] == 0.95
        assert decision_dict["priority"] == 8
    
    def test_editing_decision_from_dict(self):
        """Test creating EditingDecision from dictionary."""
        decision_data = {
            "timestamp": 30.5,
            "decision_type": "cut",
            "parameters": {"fade_duration": 1.0},
            "rationale": "Remove filler word",
            "confidence": 0.95,
            "priority": 8
        }
        
        decision = EditingDecision.from_dict(decision_data)
        
        assert decision.timestamp == 30.5
        assert decision.decision_type == "cut"
        assert decision.parameters == {"fade_duration": 1.0}
        assert decision.rationale == "Remove filler word"
        assert decision.confidence == 0.95
        assert decision.priority == 8


class TestBRollPlan:
    """Test BRollPlan data class."""
    
    def test_broll_plan_creation(self):
        """Test creating BRollPlan with valid data."""
        plan = BRollPlan(
            timestamp=45.0,
            duration=8.0,
            content_type="chart",
            description="Compound interest visualization",
            visual_elements=["growth_curve", "time_axis"],
            animation_style="fade_in",
            priority=7
        )
        
        assert plan.timestamp == 45.0
        assert plan.duration == 8.0
        assert plan.content_type == "chart"
        assert plan.description == "Compound interest visualization"
        assert plan.visual_elements == ["growth_curve", "time_axis"]
        assert plan.animation_style == "fade_in"
        assert plan.priority == 7
    
    def test_broll_plan_serialization(self):
        """Test BRollPlan serialization and deserialization."""
        original_plan = BRollPlan(
            timestamp=45.0,
            duration=8.0,
            content_type="chart",
            description="Compound interest visualization",
            visual_elements=["growth_curve", "time_axis"],
            animation_style="fade_in",
            priority=7
        )
        
        # Convert to dict and back
        plan_dict = original_plan.to_dict()
        restored_plan = BRollPlan.from_dict(plan_dict)
        
        assert restored_plan.timestamp == original_plan.timestamp
        assert restored_plan.duration == original_plan.duration
        assert restored_plan.content_type == original_plan.content_type
        assert restored_plan.description == original_plan.description
        assert restored_plan.visual_elements == original_plan.visual_elements
        assert restored_plan.animation_style == original_plan.animation_style
        assert restored_plan.priority == original_plan.priority


class TestMetadataStrategy:
    """Test MetadataStrategy data class."""
    
    def test_metadata_strategy_creation(self):
        """Test creating MetadataStrategy with valid data."""
        strategy = MetadataStrategy(
            primary_title="Master Compound Interest in 10 Minutes",
            title_variations=["Compound Interest Explained", "The Power of Compound Interest"],
            description="Learn how compound interest can grow your wealth exponentially...",
            tags=["compound interest", "investing", "financial education"],
            thumbnail_concepts=["growth chart", "money tree"],
            hook_text="This ONE concept changed my financial life",
            target_keywords=["compound interest", "investment growth"]
        )
        
        assert strategy.primary_title == "Master Compound Interest in 10 Minutes"
        assert len(strategy.title_variations) == 2
        assert "compound interest" in strategy.tags
        assert strategy.hook_text == "This ONE concept changed my financial life"
    
    def test_metadata_strategy_serialization(self):
        """Test MetadataStrategy serialization."""
        strategy = MetadataStrategy(
            primary_title="Master Compound Interest in 10 Minutes",
            title_variations=["Compound Interest Explained"],
            description="Learn compound interest...",
            tags=["compound interest", "investing"],
            thumbnail_concepts=["growth chart"],
            hook_text="This ONE concept changed my financial life",
            target_keywords=["compound interest"]
        )
        
        strategy_dict = strategy.to_dict()
        restored_strategy = MetadataStrategy.from_dict(strategy_dict)
        
        assert restored_strategy.primary_title == strategy.primary_title
        assert restored_strategy.title_variations == strategy.title_variations
        assert restored_strategy.description == strategy.description
        assert restored_strategy.tags == strategy.tags
        assert restored_strategy.thumbnail_concepts == strategy.thumbnail_concepts
        assert restored_strategy.hook_text == strategy.hook_text
        assert restored_strategy.target_keywords == strategy.target_keywords


class TestAIDirectorPlan:
    """Test AIDirectorPlan data class."""
    
    @pytest.fixture
    def sample_plan(self):
        """Create a sample AIDirectorPlan for testing."""
        editing_decisions = [
            EditingDecision(
                timestamp=30.0,
                decision_type="cut",
                parameters={},
                rationale="Remove filler",
                confidence=0.9,
                priority=8
            )
        ]
        
        broll_plans = [
            BRollPlan(
                timestamp=45.0,
                duration=8.0,
                content_type="chart",
                description="Growth visualization",
                visual_elements=["curve"],
                animation_style="fade_in",
                priority=7
            )
        ]
        
        metadata_strategy = MetadataStrategy(
            primary_title="Test Video",
            title_variations=["Alt Title"],
            description="Test description",
            tags=["test"],
            thumbnail_concepts=["concept"],
            hook_text="Hook",
            target_keywords=["keyword"]
        )
        
        return AIDirectorPlan(
            editing_decisions=editing_decisions,
            broll_plans=broll_plans,
            metadata_strategy=metadata_strategy,
            quality_enhancements=["Audio cleanup"],
            pacing_adjustments=[{"timestamp": 60.0, "adjustment": "slow_down"}],
            engagement_hooks=[{"timestamp": 30.0, "type": "question"}],
            created_at=datetime.now(),
            confidence_score=0.85,
            processing_time=5.2,
            model_used="gemini-2.5-pro-latest"
        )
    
    def test_ai_director_plan_creation(self, sample_plan):
        """Test creating AIDirectorPlan with valid data."""
        assert len(sample_plan.editing_decisions) == 1
        assert len(sample_plan.broll_plans) == 1
        assert sample_plan.metadata_strategy.primary_title == "Test Video"
        assert sample_plan.confidence_score == 0.85
        assert sample_plan.processing_time == 5.2
        assert sample_plan.model_used == "gemini-2.5-pro-latest"
    
    def test_ai_director_plan_serialization(self, sample_plan):
        """Test AIDirectorPlan serialization and deserialization."""
        plan_dict = sample_plan.to_dict()
        restored_plan = AIDirectorPlan.from_dict(plan_dict)
        
        assert len(restored_plan.editing_decisions) == len(sample_plan.editing_decisions)
        assert len(restored_plan.broll_plans) == len(sample_plan.broll_plans)
        assert restored_plan.metadata_strategy.primary_title == sample_plan.metadata_strategy.primary_title
        assert restored_plan.confidence_score == sample_plan.confidence_score
        assert restored_plan.processing_time == sample_plan.processing_time
        assert restored_plan.model_used == sample_plan.model_used


class TestFinancialVideoEditor:
    """Test FinancialVideoEditor class."""
    
    @pytest.fixture
    def mock_gemini_client(self):
        """Create a mock GeminiClient for testing."""
        client = Mock(spec=GeminiClient)
        client.get_usage_stats.return_value = {
            'total_requests': 10,
            'successful_requests': 9,
            'cache_hit_rate': 0.3
        }
        return client
    
    @pytest.fixture
    def mock_content_context(self):
        """Create a mock ContentContext for testing."""
        context = Mock(spec=ContentContext)
        context.content_type = ContentType.EDUCATIONAL
        context.total_duration = 180.0
        context.key_concepts = ["compound interest", "investment", "growth"]
        context.emotional_markers = []
        context.visual_highlights = []
        
        # Mock audio transcript
        mock_transcript = Mock()
        mock_transcript.text = "Welcome to this lesson on compound interest. Compound interest is the eighth wonder of the world."
        mock_transcript.segments = [
            Mock(text="Welcome to this lesson on compound interest", start=0.0, end=3.0),
            Mock(text="Compound interest is the eighth wonder of the world", start=3.0, end=7.0)
        ]
        context.audio_transcript = mock_transcript
        
        # Mock processing metrics
        context.processing_metrics = Mock()
        context.processing_metrics.add_module_processing_time = Mock()
        context.processing_metrics.add_api_call = Mock()
        
        # Mock cost tracking
        context.cost_tracking = Mock()
        context.cost_tracking.add_cost = Mock()
        
        return context
    
    @pytest.fixture
    def financial_editor(self, mock_gemini_client):
        """Create a FinancialVideoEditor for testing."""
        return FinancialVideoEditor(
            gemini_client=mock_gemini_client,
            quality_focused=True,
            streaming_enabled=False,
            max_processing_time=300.0
        )
    
    def test_financial_video_editor_initialization(self, financial_editor, mock_gemini_client):
        """Test FinancialVideoEditor initialization."""
        assert financial_editor.gemini_client == mock_gemini_client
        assert financial_editor.quality_focused is True
        assert financial_editor.streaming_enabled is False
        assert financial_editor.max_processing_time == 300.0
        assert len(financial_editor.financial_keywords) > 10
        assert 'compound interest' in financial_editor.financial_keywords
    
    def test_create_financial_editing_prompt(self, financial_editor, mock_content_context):
        """Test creating financial editing prompt."""
        prompt = financial_editor.create_financial_editing_prompt(mock_content_context)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 500  # Should be a comprehensive prompt
        assert "financial educational content" in prompt.lower()
        assert "compound interest" in prompt.lower()
        assert "JSON format" in prompt
        assert "editing_decisions" in prompt
        assert "broll_plans" in prompt
        assert "metadata_strategy" in prompt
    
    def test_extract_key_concepts(self, financial_editor, mock_content_context):
        """Test extracting key concepts from context."""
        concepts = financial_editor._extract_key_concepts(mock_content_context)
        
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        assert "compound interest" in concepts
        assert "investment" in concepts
    
    def test_extract_key_concepts_fallback(self, financial_editor):
        """Test key concepts extraction with minimal context."""
        minimal_context = Mock(spec=ContentContext)
        minimal_context.key_concepts = None
        minimal_context.audio_transcript = None
        
        concepts = financial_editor._extract_key_concepts(minimal_context)
        
        assert isinstance(concepts, list)
        assert len(concepts) >= 2  # Should have fallback concepts
        assert "financial education" in concepts
    
    def test_extract_transcript_summary(self, financial_editor, mock_content_context):
        """Test extracting transcript summary."""
        summary = financial_editor._extract_transcript_summary(mock_content_context)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "compound interest" in summary.lower()
    
    def test_extract_transcript_summary_no_transcript(self, financial_editor):
        """Test transcript summary extraction with no transcript."""
        context_no_transcript = Mock(spec=ContentContext)
        context_no_transcript.audio_transcript = None
        
        summary = financial_editor._extract_transcript_summary(context_no_transcript)
        
        assert summary == "No transcript available"
    
    @patch('ai_video_editor.modules.intelligence.ai_director.time.time')
    def test_generate_editing_plan_success(self, mock_time, financial_editor, mock_content_context):
        """Test successful editing plan generation."""
        # Mock time for processing time calculation
        mock_time.return_value = 1000.0  # Fixed time for consistent testing
        
        # Mock successful API response
        mock_response_data = {
            "editing_decisions": [
                {
                    "timestamp": 30.0,
                    "decision_type": "cut",
                    "parameters": {"fade_duration": 1.0},
                    "rationale": "Remove filler word",
                    "confidence": 0.9,
                    "priority": 8
                }
            ],
            "broll_plans": [
                {
                    "timestamp": 45.0,
                    "duration": 8.0,
                    "content_type": "chart",
                    "description": "Compound interest visualization",
                    "visual_elements": ["growth_curve"],
                    "animation_style": "fade_in",
                    "priority": 7
                }
            ],
            "metadata_strategy": {
                "primary_title": "Master Compound Interest in 10 Minutes",
                "title_variations": ["Compound Interest Explained"],
                "description": "Learn how compound interest works...",
                "tags": ["compound interest", "investing", "finance"],
                "thumbnail_concepts": ["growth chart", "money tree"],
                "hook_text": "This ONE concept changed my life",
                "target_keywords": ["compound interest", "investment growth"]
            },
            "quality_enhancements": ["Audio cleanup at 1:23-1:45"],
            "pacing_adjustments": [
                {
                    "timestamp": 60.0,
                    "adjustment": "slow_down",
                    "duration": 3.0,
                    "reason": "Complex concept explanation"
                }
            ],
            "engagement_hooks": [
                {
                    "timestamp": 30.0,
                    "type": "question",
                    "content": "What if I told you...",
                    "visual_treatment": "text_overlay"
                }
            ]
        }
        
        financial_editor.gemini_client.generate_structured_response.return_value = mock_response_data
        
        # Test the method
        plan = asyncio.run(financial_editor.generate_editing_plan(mock_content_context))
        
        # Verify the plan
        assert isinstance(plan, AIDirectorPlan)
        assert len(plan.editing_decisions) == 1
        assert len(plan.broll_plans) == 1
        assert plan.metadata_strategy.primary_title == "Master Compound Interest in 10 Minutes"
        assert plan.confidence_score == 0.9  # From editing decision
        assert plan.processing_time == 0.0  # Since we're using fixed time
        assert plan.model_used == "gemini-2.5-pro-latest"
        
        # Verify context was updated
        assert hasattr(mock_content_context, 'ai_director_plan')
        mock_content_context.processing_metrics.add_module_processing_time.assert_called_once()
    
    def test_generate_editing_plan_api_error(self, financial_editor, mock_content_context):
        """Test editing plan generation with API error."""
        # Mock API error
        financial_editor.gemini_client.generate_structured_response.side_effect = GeminiAPIError(
            "api_request", "API request failed"
        )
        
        # Test that the error is propagated
        with pytest.raises(GeminiAPIError):
            asyncio.run(financial_editor.generate_editing_plan(mock_content_context))
    
    def test_generate_editing_plan_invalid_context(self, financial_editor):
        """Test editing plan generation with invalid context."""
        with pytest.raises(ContentContextError):
            asyncio.run(financial_editor.generate_editing_plan(None))
    
    def test_analyze_content_for_broll(self, financial_editor, mock_content_context):
        """Test B-roll opportunity analysis."""
        opportunities = financial_editor.analyze_content_for_broll(mock_content_context)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        # Check that compound interest triggered a concept explanation
        concept_opportunities = [op for op in opportunities if op['type'] == 'concept_explanation']
        assert len(concept_opportunities) > 0
        
        # Verify opportunity structure
        opportunity = opportunities[0]
        assert 'type' in opportunity
        assert 'timestamp' in opportunity
        assert 'duration' in opportunity
        assert 'content' in opportunity
        assert 'priority' in opportunity
        assert 'visual_type' in opportunity
    
    def test_analyze_content_for_broll_no_transcript(self, financial_editor):
        """Test B-roll analysis with no transcript."""
        context_no_transcript = Mock(spec=ContentContext)
        context_no_transcript.audio_transcript = None
        
        opportunities = financial_editor.analyze_content_for_broll(context_no_transcript)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) == 0
    
    def test_optimize_for_engagement(self, financial_editor, mock_content_context):
        """Test engagement optimization."""
        optimization = financial_editor.optimize_for_engagement(mock_content_context)
        
        assert isinstance(optimization, dict)
        assert 'hook_placements' in optimization
        assert 'retention_techniques' in optimization
        assert 'pacing_adjustments' in optimization
        assert 'visual_enhancements' in optimization
        
        # Should have regular engagement hooks
        assert len(optimization['hook_placements']) > 0
    
    def test_get_processing_stats(self, financial_editor):
        """Test getting processing statistics."""
        stats = financial_editor.get_processing_stats()
        
        assert isinstance(stats, dict)
        assert 'quality_focused' in stats
        assert 'streaming_enabled' in stats
        assert 'max_processing_time' in stats
        assert 'financial_keywords_count' in stats
        assert 'retention_techniques' in stats
        assert 'gemini_client_stats' in stats
        
        assert stats['quality_focused'] is True
        assert stats['financial_keywords_count'] > 10


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('ai_video_editor.modules.intelligence.ai_director.GeminiClient')
    def test_create_financial_video_editor(self, mock_gemini_client_class):
        """Test factory function for creating FinancialVideoEditor."""
        mock_client_instance = Mock()
        mock_gemini_client_class.return_value = mock_client_instance
        
        editor = create_financial_video_editor(
            api_key="test_key",
            quality_focused=True,
            streaming_enabled=False
        )
        
        assert isinstance(editor, FinancialVideoEditor)
        assert editor.quality_focused is True
        assert editor.streaming_enabled is False
        mock_gemini_client_class.assert_called_once()
    
    def test_validate_editing_plan_valid(self):
        """Test validating a valid editing plan."""
        plan = AIDirectorPlan(
            editing_decisions=[
                EditingDecision(
                    timestamp=30.0,
                    decision_type="cut",
                    parameters={},
                    rationale="Test",
                    confidence=0.9,
                    priority=8
                )
            ],
            broll_plans=[
                BRollPlan(
                    timestamp=45.0,
                    duration=8.0,
                    content_type="chart",
                    description="Test",
                    visual_elements=[],
                    animation_style="fade_in",
                    priority=7
                )
            ],
            metadata_strategy=MetadataStrategy(
                primary_title="Test",
                title_variations=[],
                description="Test",
                tags=[],
                thumbnail_concepts=[],
                hook_text="Test",
                target_keywords=[]
            ),
            quality_enhancements=[],
            pacing_adjustments=[],
            engagement_hooks=[],
            created_at=datetime.now(),
            confidence_score=0.85,
            processing_time=5.0,
            model_used="test"
        )
        
        assert validate_editing_plan(plan) is True
    
    def test_validate_editing_plan_invalid(self):
        """Test validating an invalid editing plan."""
        # Plan with no editing decisions
        plan = AIDirectorPlan(
            editing_decisions=[],  # Empty - should be invalid
            broll_plans=[],
            metadata_strategy=None,  # None - should be invalid
            quality_enhancements=[],
            pacing_adjustments=[],
            engagement_hooks=[],
            created_at=datetime.now(),
            confidence_score=0.85,
            processing_time=5.0,
            model_used="test"
        )
        
        assert validate_editing_plan(plan) is False
    
    def test_merge_editing_plans(self):
        """Test merging multiple editing plans."""
        plan1 = AIDirectorPlan(
            editing_decisions=[
                EditingDecision(
                    timestamp=30.0,
                    decision_type="cut",
                    parameters={},
                    rationale="Test 1",
                    confidence=0.9,
                    priority=8
                )
            ],
            broll_plans=[],
            metadata_strategy=MetadataStrategy(
                primary_title="Plan 1",
                title_variations=[],
                description="Test 1",
                tags=[],
                thumbnail_concepts=[],
                hook_text="Test 1",
                target_keywords=[]
            ),
            quality_enhancements=["Enhancement 1"],
            pacing_adjustments=[],
            engagement_hooks=[],
            created_at=datetime.now(),
            confidence_score=0.85,
            processing_time=3.0,
            model_used="test1"
        )
        
        plan2 = AIDirectorPlan(
            editing_decisions=[
                EditingDecision(
                    timestamp=60.0,
                    decision_type="trim",
                    parameters={},
                    rationale="Test 2",
                    confidence=0.8,
                    priority=7
                )
            ],
            broll_plans=[],
            metadata_strategy=MetadataStrategy(
                primary_title="Plan 2",
                title_variations=[],
                description="Test 2",
                tags=[],
                thumbnail_concepts=[],
                hook_text="Test 2",
                target_keywords=[]
            ),
            quality_enhancements=["Enhancement 2"],
            pacing_adjustments=[],
            engagement_hooks=[],
            created_at=datetime.now(),
            confidence_score=0.90,  # Higher confidence
            processing_time=2.0,
            model_used="test2"
        )
        
        merged_plan = merge_editing_plans([plan1, plan2])
        
        assert len(merged_plan.editing_decisions) == 2
        assert merged_plan.metadata_strategy.primary_title == "Plan 2"  # Higher confidence plan
        assert len(merged_plan.quality_enhancements) == 2
        assert merged_plan.confidence_score == 0.875  # Average of 0.85 and 0.90
        assert merged_plan.processing_time == 5.0  # Sum of processing times
        assert "merged_from_2_plans" in merged_plan.model_used
    
    def test_merge_editing_plans_empty(self):
        """Test merging empty list of plans."""
        with pytest.raises(ContentContextError):
            merge_editing_plans([])
    
    def test_merge_editing_plans_single(self):
        """Test merging single plan."""
        plan = AIDirectorPlan(
            editing_decisions=[],
            broll_plans=[],
            metadata_strategy=MetadataStrategy(
                primary_title="Single Plan",
                title_variations=[],
                description="Test",
                tags=[],
                thumbnail_concepts=[],
                hook_text="Test",
                target_keywords=[]
            ),
            quality_enhancements=[],
            pacing_adjustments=[],
            engagement_hooks=[],
            created_at=datetime.now(),
            confidence_score=0.85,
            processing_time=5.0,
            model_used="test"
        )
        
        result = merge_editing_plans([plan])
        assert result is plan  # Should return the same plan
    
    def test_get_content_level_prompt(self):
        """Test getting content level prompts."""
        beginner_prompt = get_content_level_prompt('beginner')
        intermediate_prompt = get_content_level_prompt('intermediate')
        advanced_prompt = get_content_level_prompt('advanced')
        default_prompt = get_content_level_prompt('unknown')
        
        assert isinstance(beginner_prompt, str)
        assert isinstance(intermediate_prompt, str)
        assert isinstance(advanced_prompt, str)
        assert isinstance(default_prompt, str)
        
        assert "simple explanations" in beginner_prompt.lower()
        assert "practical application" in intermediate_prompt.lower()
        assert "nuanced analysis" in advanced_prompt.lower()
        assert default_prompt == intermediate_prompt  # Should default to intermediate


if __name__ == "__main__":
    pytest.main([__file__])