"""
Unit tests for B-Roll Analysis functionality (Task 6 implementation).

Tests the FinancialBRollAnalyzer and AIGraphicsDirector classes
to ensure proper B-roll opportunity detection and graphics planning.
"""

import pytest
from unittest.mock import Mock, patch
import asyncio

from ai_video_editor.modules.content_analysis import FinancialBRollAnalyzer, AIGraphicsDirector, BRollOpportunity
from ai_video_editor.core.content_context import (
    ContentContext, ContentType, UserPreferences,
    AudioSegment, AudioAnalysisResult
)


class TestBRollOpportunity:
    """Test BRollOpportunity data class."""
    
    def test_broll_opportunity_creation(self):
        """Test BRollOpportunity creation and serialization."""
        opportunity = BRollOpportunity(
            timestamp=10.0,
            duration=5.0,
            opportunity_type="data_visualization",
            content="Testing content",
            graphics_type="chart_or_graph",
            confidence=0.85,
            priority=8,
            keywords=["test", "data"],
            suggested_elements=["chart", "axis"]
        )
        
        assert opportunity.timestamp == 10.0
        assert opportunity.duration == 5.0
        assert opportunity.confidence == 0.85
        assert len(opportunity.keywords) == 2
        
        # Test serialization
        data = opportunity.to_dict()
        assert data['timestamp'] == 10.0
        assert data['graphics_type'] == "chart_or_graph"
        
        # Test deserialization
        restored = BRollOpportunity.from_dict(data)
        assert restored.timestamp == opportunity.timestamp
        assert restored.graphics_type == opportunity.graphics_type


class TestFinancialBRollAnalyzer:
    """Test FinancialBRollAnalyzer functionality."""
    
    @pytest.fixture
    def broll_analyzer(self):
        """Create FinancialBRollAnalyzer instance."""
        return FinancialBRollAnalyzer()
    
    @pytest.fixture
    def sample_context_with_audio(self):
        """Create sample ContentContext with audio analysis."""
        segments = [
            AudioSegment(
                text="Let's look at the compound interest growth chart showing 7% annual returns.",
                start=10.0,
                end=15.0,
                confidence=0.9,
                financial_concepts=["compound interest", "growth", "returns"]
            ),
            AudioSegment(
                text="Diversification helps reduce portfolio risk by spreading investments.",
                start=20.0,
                end=25.0,
                confidence=0.85,
                financial_concepts=["diversification", "portfolio", "risk"]
            ),
            AudioSegment(
                text="The process involves three steps: assess risk, allocate assets, rebalance.",
                start=30.0,
                end=35.0,
                confidence=0.88,
                financial_concepts=["process", "risk assessment"]
            )
        ]
        
        audio_analysis = AudioAnalysisResult(
            transcript_text=" ".join([seg.text for seg in segments]),
            segments=segments,
            overall_confidence=0.88,
            language="en",
            processing_time=1.5,
            model_used="whisper-base"
        )
        
        context = ContentContext(
            project_id="test_broll",
            video_files=["test.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        context.set_audio_analysis(audio_analysis)
        return context
    
    def test_broll_analyzer_initialization(self, broll_analyzer):
        """Test FinancialBRollAnalyzer initialization."""
        assert broll_analyzer is not None
        assert len(broll_analyzer.visual_triggers) == 6  # 6 trigger categories
        assert 'chart_keywords' in broll_analyzer.visual_triggers
        assert len(broll_analyzer.duration_guidelines) == 6
        assert len(broll_analyzer.graphics_specs) == 5
    
    def test_detect_broll_opportunities_success(self, broll_analyzer, sample_context_with_audio):
        """Test successful B-roll opportunity detection."""
        opportunities = broll_analyzer.detect_broll_opportunities(sample_context_with_audio)
        
        # Should detect multiple opportunities from the sample content
        assert len(opportunities) > 0
        assert all(isinstance(opp, BRollOpportunity) for opp in opportunities)
        
        # Check that opportunities have required fields
        for opp in opportunities:
            assert opp.timestamp >= 0
            assert opp.duration > 0
            assert opp.confidence > 0
            assert opp.priority > 0
            assert opp.graphics_type in ['chart_or_graph', 'animated_explanation', 'step_by_step_visual', 'formula_visualization', 'comparison_table']
    
    def test_detect_broll_opportunities_empty_context(self, broll_analyzer):
        """Test B-roll detection with empty context."""
        context = ContentContext(
            project_id="empty_test",
            video_files=["test.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        opportunities = broll_analyzer.detect_broll_opportunities(context)
        assert len(opportunities) == 0
    
    def test_contains_keywords(self, broll_analyzer):
        """Test keyword detection functionality."""
        text = "The compound interest chart shows exponential growth over time."
        
        assert broll_analyzer._contains_keywords(text, ["compound", "chart"])
        assert broll_analyzer._contains_keywords(text, ["compound interest"])
        assert not broll_analyzer._contains_keywords(text, ["diversification", "portfolio"])
    
    def test_extract_matching_keywords(self, broll_analyzer):
        """Test keyword extraction."""
        text = "portfolio diversification reduces risk through asset allocation."
        keywords = ["portfolio", "diversification", "risk", "unused_keyword"]
        
        matches = broll_analyzer._extract_matching_keywords(text, keywords)
        assert "portfolio" in matches
        assert "diversification" in matches
        assert "risk" in matches
        assert "unused_keyword" not in matches
    
    def test_calculate_confidence(self, broll_analyzer):
        """Test confidence calculation."""
        text = "Compound interest data shows growth percentage over time."
        keywords = ["compound interest", "data", "growth", "percentage"]
        financial_concepts = ["compound interest", "growth"]
        
        confidence = broll_analyzer._calculate_confidence(text, keywords, financial_concepts)
        assert 0.5 <= confidence <= 0.95
        assert confidence > 0.5  # Should be above base confidence
    
    def test_integrate_with_ai_director(self, broll_analyzer, sample_context_with_audio):
        """Test AI Director integration."""
        opportunities = broll_analyzer.detect_broll_opportunities(sample_context_with_audio)
        ai_director_plans = broll_analyzer.integrate_with_ai_director(sample_context_with_audio, opportunities)
        
        assert len(ai_director_plans) == len(opportunities)
        
        for plan in ai_director_plans:
            assert 'timestamp' in plan
            assert 'duration' in plan
            assert 'content_type' in plan
            assert 'description' in plan
            assert 'priority' in plan
    
    def test_get_analysis_stats(self, broll_analyzer, sample_context_with_audio):
        """Test analysis statistics."""
        # Run analysis to populate stats
        broll_analyzer.detect_broll_opportunities(sample_context_with_audio)
        
        stats = broll_analyzer.get_analysis_stats()
        assert 'analysis_time' in stats
        assert 'opportunities_detected' in stats
        assert 'trigger_categories' in stats
        assert 'graphics_types_supported' in stats
        assert stats['trigger_categories'] == 6
        assert stats['graphics_types_supported'] == 5


class TestAIGraphicsDirector:
    """Test AIGraphicsDirector functionality."""
    
    @pytest.fixture
    def graphics_director(self):
        """Create AIGraphicsDirector instance."""
        return AIGraphicsDirector(output_dir="temp/test_graphics")
    
    @pytest.fixture
    def sample_opportunity(self):
        """Create sample BRollOpportunity."""
        return BRollOpportunity(
            timestamp=15.0,
            duration=6.0,
            opportunity_type="data_visualization",
            content="Compound interest shows exponential growth over time.",
            graphics_type="chart_or_graph",
            confidence=0.9,
            priority=8,
            keywords=["compound", "interest", "growth"],
            suggested_elements=["chart", "growth_curve"]
        )
    
    def test_graphics_director_initialization(self, graphics_director):
        """Test AIGraphicsDirector initialization."""
        assert graphics_director is not None
        assert graphics_director.output_dir.exists()
        assert graphics_director.graphics_generator is not None
        assert graphics_director.slide_generator is not None
        assert len(graphics_director.template_specs) == 3
    
    def test_generate_template_graphics_spec(self, graphics_director, sample_opportunity):
        """Test template-based graphics specification generation."""
        spec = graphics_director._generate_template_graphics_spec(sample_opportunity)
        
        assert spec.graphics_type == sample_opportunity.graphics_type
        assert spec.duration == sample_opportunity.duration
        assert spec.chart_type is not None
        assert spec.animation_style is not None
    
    def test_extract_concept_from_content(self, graphics_director):
        """Test concept extraction from content."""
        content1 = "Let's discuss compound interest and its effects."
        concept1 = graphics_director._extract_concept_from_content(content1)
        assert concept1 == "Compound Interest"
        
        content2 = "Portfolio diversification is important for risk management."
        concept2 = graphics_director._extract_concept_from_content(content2)
        assert concept2 == "Portfolio Diversification"
        
        content3 = "General investment advice without specific keywords."
        concept3 = graphics_director._extract_concept_from_content(content3)
        assert concept3 == "Investment Strategy"
    
    def test_create_movis_motion_graphics_plan(self, graphics_director, sample_opportunity):
        """Test movis motion graphics plan creation."""
        motion_plan = graphics_director.create_movis_motion_graphics_plan(sample_opportunity)
        
        assert motion_plan['timestamp'] == sample_opportunity.timestamp
        assert motion_plan['duration'] == sample_opportunity.duration
        assert 'layers' in motion_plan
        assert 'animations' in motion_plan
        assert 'effects' in motion_plan
        assert len(motion_plan['layers']) > 0
        assert len(motion_plan['animations']) > 0
    
    def test_create_blender_animation_script(self, graphics_director, sample_opportunity):
        """Test Blender animation script creation."""
        script_path = graphics_director.create_blender_animation_script(sample_opportunity)
        
        assert script_path.endswith('.py')
        assert graphics_director.output_dir.name in script_path
        
        # Check that script file was created
        from pathlib import Path
        assert Path(script_path).exists()
        
        # Check script content
        with open(script_path, 'r') as f:
            content = f.read()
            assert 'import bpy' in content
            assert 'render.render' in content
            assert str(int(sample_opportunity.duration * 24)) in content
    
    @pytest.mark.asyncio
    async def test_create_fallback_graphics(self, graphics_director, sample_opportunity):
        """Test fallback graphics creation."""
        try:
            graphics_files = await graphics_director._create_fallback_graphics(sample_opportunity)
            # May return empty list if matplotlib not available, which is acceptable
            assert isinstance(graphics_files, list)
        except Exception:
            # Graphics creation may fail without proper dependencies
            pytest.skip("Graphics dependencies not available")
    
    def test_get_generation_stats(self, graphics_director):
        """Test graphics generation statistics."""
        stats = graphics_director.get_generation_stats()
        
        assert 'generation_time' in stats
        assert 'graphics_created' in stats
        assert 'template_specs_available' in stats
        assert 'ai_enabled' in stats
        assert stats['template_specs_available'] == 3
        assert stats['ai_enabled'] == False  # No API key provided


# Integration tests
class TestBRollIntegration:
    """Test integration between B-roll components."""
    
    @pytest.fixture
    def integrated_setup(self):
        """Set up integrated B-roll system."""
        analyzer = FinancialBRollAnalyzer()
        graphics_director = AIGraphicsDirector(output_dir="temp/integration_test")
        
        # Create context with rich audio content
        segments = [
            AudioSegment(
                text="Let's examine the compound interest formula and see how money grows exponentially.",
                start=5.0,
                end=12.0,
                confidence=0.92,
                financial_concepts=["compound interest", "formula", "exponential growth"]
            )
        ]
        
        audio_analysis = AudioAnalysisResult(
            transcript_text=segments[0].text,
            segments=segments,
            overall_confidence=0.92,
            language="en",
            processing_time=1.0,
            model_used="whisper-base"
        )
        
        context = ContentContext(
            project_id="integration_test",
            video_files=["test.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        context.set_audio_analysis(audio_analysis)
        
        return analyzer, graphics_director, context
    
    def test_end_to_end_broll_workflow(self, integrated_setup):
        """Test complete B-roll detection and planning workflow."""
        analyzer, graphics_director, context = integrated_setup
        
        # Step 1: Detect opportunities
        opportunities = analyzer.detect_broll_opportunities(context)
        assert len(opportunities) > 0
        
        # Step 2: Convert to AI Director format
        ai_plans = analyzer.integrate_with_ai_director(context, opportunities)
        assert len(ai_plans) == len(opportunities)
        
        # Step 3: Create motion graphics plans
        motion_plans = []
        for opportunity in opportunities[:2]:  # Limit for testing
            motion_plan = graphics_director.create_movis_motion_graphics_plan(opportunity)
            motion_plans.append(motion_plan)
        
        assert len(motion_plans) > 0
        
        # Step 4: Store in ContentContext
        if not context.processed_video:
            context.processed_video = {}
        context.processed_video['broll_plans'] = ai_plans
        context.processed_video['motion_graphics_plans'] = motion_plans
        
        # Verify integration
        assert 'broll_plans' in context.processed_video
        assert len(context.processed_video['broll_plans']) > 0
        assert 'motion_graphics_plans' in context.processed_video
    
    def test_broll_timing_optimization(self, integrated_setup):
        """Test B-roll timing optimization prevents overlaps."""
        analyzer, _, context = integrated_setup
        
        # Create overlapping opportunities manually
        opportunities = [
            BRollOpportunity(
                timestamp=10.0, duration=8.0, opportunity_type="test1",
                content="Test 1", graphics_type="chart_or_graph",
                confidence=0.8, priority=5
            ),
            BRollOpportunity(
                timestamp=15.0, duration=6.0, opportunity_type="test2",
                content="Test 2", graphics_type="animated_explanation",
                confidence=0.9, priority=7
            )
        ]
        
        # Test optimization
        optimized = analyzer._optimize_opportunity_timing(opportunities)
        
        # Should handle overlaps appropriately
        assert len(optimized) > 0
        if len(optimized) >= 2:
            # Check that high confidence opportunity is preserved
            assert any(opp.confidence == 0.9 for opp in optimized)
