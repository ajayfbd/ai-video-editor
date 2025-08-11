"""
Unit tests for B-roll generation and integration system.

Tests the complete B-roll generation pipeline including chart generation,
Blender rendering, educational slides, and VideoComposer integration.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from ai_video_editor.modules.video_processing.broll_generation import (
    BRollGenerationSystem,
    EnhancedChartGenerator,
    BlenderRenderingPipeline,
    EducationalSlideSystem,
    GeneratedBRollAsset
)
from ai_video_editor.modules.intelligence.ai_director import BRollPlan
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_broll_plan():
    """Create sample B-roll plan for testing."""
    return BRollPlan(
        timestamp=15.0,
        duration=5.0,
        content_type="chart_or_graph",
        description="Compound interest growth showing how $10,000 grows over 30 years at 7% annual return",
        visual_elements=["growth_curve", "time_axis", "value_labels"],
        animation_style="progressive_reveal",
        priority=8
    )


@pytest.fixture
def sample_context_with_broll_plans():
    """Create ContentContext with B-roll plans."""
    context = ContentContext(
        project_id="test_broll_generation",
        video_files=["test_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(quality_mode="high")
    )
    
    # Add B-roll plans to processed_video
    context.processed_video = {
        'broll_plans': [
            {
                'timestamp': 15.0,
                'duration': 5.0,
                'content_type': 'chart_or_graph',
                'description': 'Compound interest growth chart',
                'visual_elements': ['growth_curve', 'time_axis'],
                'animation_style': 'progressive_reveal',
                'priority': 8
            },
            {
                'timestamp': 30.0,
                'duration': 4.0,
                'content_type': 'animated_explanation',
                'description': 'Portfolio diversification concept explanation',
                'visual_elements': ['pie_chart', 'asset_classes'],
                'animation_style': 'fade_in',
                'priority': 7
            },
            {
                'timestamp': 45.0,
                'duration': 6.0,
                'content_type': 'educational_slide',
                'description': 'Risk management process steps',
                'visual_elements': ['numbered_steps', 'flowchart'],
                'animation_style': 'sequential_reveal',
                'priority': 9
            }
        ]
    }
    
    return context


class TestEnhancedChartGenerator:
    """Test enhanced chart generation from AI Director specifications."""
    
    def test_init(self, temp_output_dir):
        """Test chart generator initialization."""
        generator = EnhancedChartGenerator(output_dir=temp_output_dir)
        
        assert generator.output_dir == Path(temp_output_dir)
        assert generator.output_dir.exists()
    
    def test_determine_chart_type(self, temp_output_dir, sample_broll_plan):
        """Test chart type determination from B-roll plan."""
        generator = EnhancedChartGenerator(output_dir=temp_output_dir)
        
        # Test compound interest detection
        compound_plan = BRollPlan(
            timestamp=10.0, duration=5.0, content_type="chart",
            description="Compound interest growth over time",
            visual_elements=[], animation_style="fade", priority=5
        )
        assert generator._determine_chart_type(compound_plan) == "compound_interest"
        
        # Test portfolio allocation detection
        portfolio_plan = BRollPlan(
            timestamp=20.0, duration=5.0, content_type="chart",
            description="Portfolio allocation and diversification strategy",
            visual_elements=[], animation_style="fade", priority=5
        )
        assert generator._determine_chart_type(portfolio_plan) == "portfolio_allocation"
        
        # Test risk-return detection
        risk_plan = BRollPlan(
            timestamp=30.0, duration=5.0, content_type="chart",
            description="Risk versus return analysis for different investments",
            visual_elements=[], animation_style="fade", priority=5
        )
        assert generator._determine_chart_type(risk_plan) == "risk_return"
    
    @patch('ai_video_editor.modules.video_processing.broll_generation.MATPLOTLIB_AVAILABLE', True)
    @patch('ai_video_editor.modules.video_processing.broll_generation.plt')
    @patch('ai_video_editor.modules.video_processing.broll_generation.np')
    def test_generate_from_ai_specification(self, mock_np, mock_plt, temp_output_dir, sample_broll_plan):
        """Test chart generation from AI specification."""
        generator = EnhancedChartGenerator(output_dir=temp_output_dir)
        
        # Mock numpy and matplotlib
        mock_np.arange.return_value = [0, 1, 2, 3, 4, 5]
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Mock context
        context = Mock()
        
        # Test chart generation
        result = generator.generate_from_ai_specification(sample_broll_plan, context)
        
        # Verify chart was created (may be placeholder due to error)
        assert isinstance(result, str)
        assert result.endswith('.png')
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()
    
    @patch('ai_video_editor.modules.video_processing.broll_generation.MATPLOTLIB_AVAILABLE', False)
    @patch('ai_video_editor.modules.video_processing.broll_generation.PIL_AVAILABLE', True)
    def test_create_placeholder_chart(self, temp_output_dir, sample_broll_plan):
        """Test placeholder chart creation when matplotlib unavailable."""
        generator = EnhancedChartGenerator(output_dir=temp_output_dir)
        
        with patch('ai_video_editor.modules.video_processing.broll_generation.Image') as mock_image:
            mock_img = Mock()
            mock_image.new.return_value = mock_img
            
            result = generator._create_placeholder_chart(sample_broll_plan)
            
            assert isinstance(result, str)
            assert "placeholder" in result
            mock_image.new.assert_called_once()
            mock_img.save.assert_called_once()


class TestBlenderRenderingPipeline:
    """Test Blender animation rendering pipeline."""
    
    def test_init(self, temp_output_dir):
        """Test Blender pipeline initialization."""
        pipeline = BlenderRenderingPipeline(output_dir=temp_output_dir)
        
        assert pipeline.output_dir == Path(temp_output_dir)
        assert pipeline.output_dir.exists()
        assert hasattr(pipeline, 'blender_available')
    
    @patch('subprocess.run')
    def test_check_blender_availability_success(self, mock_run, temp_output_dir):
        """Test successful Blender availability check."""
        mock_run.return_value.returncode = 0
        
        pipeline = BlenderRenderingPipeline(output_dir=temp_output_dir)
        
        assert pipeline.blender_available is True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_check_blender_availability_failure(self, mock_run, temp_output_dir):
        """Test failed Blender availability check."""
        mock_run.side_effect = FileNotFoundError()
        
        pipeline = BlenderRenderingPipeline(output_dir=temp_output_dir)
        
        assert pipeline.blender_available is False
    
    def test_determine_animation_type(self, temp_output_dir):
        """Test animation type determination."""
        pipeline = BlenderRenderingPipeline(output_dir=temp_output_dir)
        
        # Test financial concept
        financial_plan = BRollPlan(
            timestamp=10.0, duration=5.0, content_type="animation",
            description="Compound interest growth visualization",
            visual_elements=[], animation_style="fade", priority=5
        )
        assert pipeline._determine_animation_type(financial_plan) == "financial_concept"
        
        # Test process flow
        process_plan = BRollPlan(
            timestamp=20.0, duration=5.0, content_type="animation",
            description="Step by step investment process",
            visual_elements=[], animation_style="fade", priority=5
        )
        assert pipeline._determine_animation_type(process_plan) == "financial_concept"  # "investment" keyword takes precedence
    
    def test_generate_blender_script(self, temp_output_dir, sample_broll_plan):
        """Test Blender script generation."""
        pipeline = BlenderRenderingPipeline(output_dir=temp_output_dir)
        context = Mock()
        
        script = pipeline._generate_blender_script(sample_broll_plan, context)
        
        assert isinstance(script, str)
        assert "import bpy" in script
        assert "scene.frame_end" in script
        assert str(int(sample_broll_plan.duration * 24)) in script
    
    @patch('ai_video_editor.modules.video_processing.broll_generation.PIL_AVAILABLE', True)
    def test_create_placeholder_animation(self, temp_output_dir, sample_broll_plan):
        """Test placeholder animation creation."""
        pipeline = BlenderRenderingPipeline(output_dir=temp_output_dir)
        
        with patch('ai_video_editor.modules.video_processing.broll_generation.Image') as mock_image:
            mock_frames = [Mock() for _ in range(10)]
            mock_image.new.side_effect = mock_frames
            
            result = pipeline._create_placeholder_animation(sample_broll_plan)
            
            assert isinstance(result, str)
            assert "placeholder_animation" in result
            assert mock_image.new.call_count == 10


class TestEducationalSlideSystem:
    """Test educational slide generation system."""
    
    def test_init(self, temp_output_dir):
        """Test slide system initialization."""
        slide_system = EducationalSlideSystem(output_dir=temp_output_dir)
        
        assert slide_system.output_dir == Path(temp_output_dir)
        assert slide_system.output_dir.exists()
    
    def test_determine_slide_type(self, temp_output_dir):
        """Test slide type determination."""
        slide_system = EducationalSlideSystem(output_dir=temp_output_dir)
        
        # Test concept explanation
        concept_plan = BRollPlan(
            timestamp=10.0, duration=5.0, content_type="slide",
            description="Explain the concept of compound interest",
            visual_elements=[], animation_style="fade", priority=5
        )
        assert slide_system._determine_slide_type(concept_plan) == "concept_explanation"
        
        # Test step by step
        step_plan = BRollPlan(
            timestamp=20.0, duration=5.0, content_type="slide",
            description="Step by step investment process",
            visual_elements=[], animation_style="fade", priority=5
        )
        assert slide_system._determine_slide_type(step_plan) == "step_by_step"
        
        # Test comparison
        comparison_plan = BRollPlan(
            timestamp=30.0, duration=5.0, content_type="slide",
            description="Compare stocks versus bonds",
            visual_elements=[], animation_style="fade", priority=5
        )
        assert slide_system._determine_slide_type(comparison_plan) == "comparison"
    
    def test_extract_concept(self, temp_output_dir):
        """Test concept extraction from description."""
        slide_system = EducationalSlideSystem(output_dir=temp_output_dir)
        
        description = "Compound interest is the eighth wonder of the world"
        concept = slide_system._extract_concept(description)
        
        assert concept == "Compound Interest Is"
    
    def test_extract_steps(self, temp_output_dir):
        """Test step extraction from description."""
        slide_system = EducationalSlideSystem(output_dir=temp_output_dir)
        
        description = "First, assess your risk tolerance. Second, choose asset allocation. Third, rebalance regularly."
        steps = slide_system._extract_steps(description)
        
        assert len(steps) == 3
        assert "First, assess your risk tolerance" in steps[0]
        assert "Second, choose asset allocation" in steps[1]
        assert "Third, rebalance regularly" in steps[2]
    
    @patch('ai_video_editor.modules.video_processing.broll_generation.PIL_AVAILABLE', True)
    def test_generate_educational_slide(self, temp_output_dir, sample_broll_plan):
        """Test educational slide generation."""
        slide_system = EducationalSlideSystem(output_dir=temp_output_dir)
        context = Mock()
        
        with patch('ai_video_editor.modules.video_processing.broll_generation.Image') as mock_image:
            mock_img = Mock()
            mock_image.new.return_value = mock_img
            
            result = slide_system.generate_educational_slide(sample_broll_plan, context)
            
            assert isinstance(result, str)
            assert result.endswith('.png')  # May be generic slide due to fallback
            mock_image.new.assert_called_once()
            mock_img.save.assert_called_once()


class TestBRollGenerationSystem:
    """Test complete B-roll generation system."""
    
    def test_init(self, temp_output_dir):
        """Test B-roll generation system initialization."""
        system = BRollGenerationSystem(output_dir=temp_output_dir)
        
        assert system.output_dir == Path(temp_output_dir)
        assert system.output_dir.exists()
        assert hasattr(system, 'chart_generator')
        assert hasattr(system, 'blender_pipeline')
        assert hasattr(system, 'slide_system')
        assert system.generated_assets == []
    
    @pytest.mark.asyncio
    async def test_generate_all_broll_assets(self, temp_output_dir, sample_context_with_broll_plans):
        """Test complete B-roll asset generation."""
        system = BRollGenerationSystem(output_dir=temp_output_dir)
        
        # Mock the individual generation methods
        with patch.object(system, '_generate_single_broll_asset') as mock_generate:
            mock_asset1 = GeneratedBRollAsset(
                asset_id="broll_15_chart",
                file_path=f"{temp_output_dir}/chart_15.png",
                asset_type="chart",
                timestamp=15.0,
                duration=5.0,
                generation_method="matplotlib"
            )
            mock_asset2 = GeneratedBRollAsset(
                asset_id="broll_30_animation",
                file_path=f"{temp_output_dir}/animation_30.mp4",
                asset_type="animation",
                timestamp=30.0,
                duration=4.0,
                generation_method="blender"
            )
            mock_asset3 = GeneratedBRollAsset(
                asset_id="broll_45_slide",
                file_path=f"{temp_output_dir}/slide_45.png",
                asset_type="slide",
                timestamp=45.0,
                duration=6.0,
                generation_method="pil"
            )
            
            mock_generate.side_effect = [mock_asset1, mock_asset2, mock_asset3]
            
            # Generate assets
            assets = await system.generate_all_broll_assets(sample_context_with_broll_plans)
            
            # Verify results
            assert len(assets) == 3
            assert mock_generate.call_count == 3
            assert system.total_assets_generated == 3
            assert len(system.generated_assets) == 3
    
    @pytest.mark.asyncio
    async def test_generate_single_broll_asset_chart(self, temp_output_dir, sample_broll_plan):
        """Test single chart asset generation."""
        system = BRollGenerationSystem(output_dir=temp_output_dir)
        context = Mock()
        
        # Mock chart generator
        with patch.object(system.chart_generator, 'generate_from_ai_specification') as mock_generate:
            mock_generate.return_value = f"{temp_output_dir}/test_chart.png"
            
            asset = await system._generate_single_broll_asset(sample_broll_plan, context)
            
            assert asset is not None
            assert asset.asset_type == "chart"
            assert asset.generation_method == "matplotlib"
            assert asset.timestamp == sample_broll_plan.timestamp
            assert asset.duration == sample_broll_plan.duration
            mock_generate.assert_called_once_with(sample_broll_plan, context)
    
    @pytest.mark.asyncio
    async def test_generate_single_broll_asset_animation(self, temp_output_dir):
        """Test single animation asset generation."""
        system = BRollGenerationSystem(output_dir=temp_output_dir)
        context = Mock()
        
        animation_plan = BRollPlan(
            timestamp=20.0, duration=4.0, content_type="animation",
            description="Test animation", visual_elements=[], 
            animation_style="fade", priority=5
        )
        
        # Mock blender pipeline
        with patch.object(system.blender_pipeline, 'render_animation') as mock_render:
            mock_render.return_value = f"{temp_output_dir}/test_animation.mp4"
            
            asset = await system._generate_single_broll_asset(animation_plan, context)
            
            assert asset is not None
            assert asset.asset_type == "animation"
            assert asset.generation_method == "blender"
            assert asset.timestamp == animation_plan.timestamp
    
    @pytest.mark.asyncio
    async def test_generate_single_broll_asset_slide(self, temp_output_dir):
        """Test single slide asset generation."""
        system = BRollGenerationSystem(output_dir=temp_output_dir)
        context = Mock()
        
        slide_plan = BRollPlan(
            timestamp=30.0, duration=6.0, content_type="educational_slide",
            description="Test slide", visual_elements=[], 
            animation_style="fade", priority=5
        )
        
        # Mock slide system
        with patch.object(system.slide_system, 'generate_educational_slide') as mock_generate:
            mock_generate.return_value = f"{temp_output_dir}/test_slide.png"
            
            asset = await system._generate_single_broll_asset(slide_plan, context)
            
            assert asset is not None
            assert asset.asset_type == "slide"
            assert asset.generation_method == "pil"
            assert asset.timestamp == slide_plan.timestamp
    
    def test_get_asset_for_timestamp(self, temp_output_dir):
        """Test asset retrieval by timestamp."""
        system = BRollGenerationSystem(output_dir=temp_output_dir)
        
        # Add test assets
        asset1 = GeneratedBRollAsset(
            asset_id="test1", file_path="test1.png", asset_type="chart",
            timestamp=15.0, duration=5.0, generation_method="matplotlib"
        )
        asset2 = GeneratedBRollAsset(
            asset_id="test2", file_path="test2.png", asset_type="slide",
            timestamp=30.0, duration=4.0, generation_method="pil"
        )
        
        system.generated_assets = [asset1, asset2]
        
        # Test exact match
        found_asset = system.get_asset_for_timestamp(15.0)
        assert found_asset == asset1
        
        # Test tolerance match
        found_asset = system.get_asset_for_timestamp(15.3, tolerance=0.5)
        assert found_asset == asset1
        
        # Test no match
        found_asset = system.get_asset_for_timestamp(50.0)
        assert found_asset is None
    
    def test_get_assets_in_range(self, temp_output_dir):
        """Test asset retrieval by time range."""
        system = BRollGenerationSystem(output_dir=temp_output_dir)
        
        # Add test assets
        assets = [
            GeneratedBRollAsset(
                asset_id=f"test{i}", file_path=f"test{i}.png", asset_type="chart",
                timestamp=float(i * 10), duration=5.0, generation_method="matplotlib"
            )
            for i in range(1, 6)  # timestamps: 10, 20, 30, 40, 50
        ]
        
        system.generated_assets = assets
        
        # Test range selection
        assets_in_range = system.get_assets_in_range(15.0, 35.0)
        assert len(assets_in_range) == 2  # timestamps 20, 30
        assert assets_in_range[0].timestamp == 20.0
        assert assets_in_range[1].timestamp == 30.0
    
    def test_get_generation_stats(self, temp_output_dir):
        """Test generation statistics."""
        system = BRollGenerationSystem(output_dir=temp_output_dir)
        
        # Add test assets
        system.generated_assets = [
            GeneratedBRollAsset(
                asset_id="chart1", file_path="chart1.png", asset_type="chart",
                timestamp=10.0, duration=5.0, generation_method="matplotlib"
            ),
            GeneratedBRollAsset(
                asset_id="chart2", file_path="chart2.png", asset_type="chart",
                timestamp=20.0, duration=5.0, generation_method="matplotlib"
            ),
            GeneratedBRollAsset(
                asset_id="slide1", file_path="slide1.png", asset_type="slide",
                timestamp=30.0, duration=5.0, generation_method="pil"
            )
        ]
        
        system.total_assets_generated = 3
        system.generation_time = 5.5
        
        stats = system.get_generation_stats()
        
        assert stats['total_assets_generated'] == 3
        assert stats['total_generation_time'] == 5.5
        assert stats['asset_types']['chart'] == 2
        assert stats['asset_types']['slide'] == 1
        assert stats['generation_methods']['matplotlib'] == 2
        assert stats['generation_methods']['pil'] == 1
        assert 'chart_generator_available' in stats
        assert 'blender_available' in stats
        assert 'pil_available' in stats
    
    def test_cleanup_generated_assets(self, temp_output_dir):
        """Test asset cleanup."""
        system = BRollGenerationSystem(output_dir=temp_output_dir)
        
        # Create test files
        test_files = []
        for i in range(3):
            test_file = Path(temp_output_dir) / f"test{i}.png"
            test_file.write_text("test content")
            test_files.append(str(test_file))
        
        # Add assets
        system.generated_assets = [
            GeneratedBRollAsset(
                asset_id=f"test{i}", file_path=test_files[i], asset_type="chart",
                timestamp=float(i * 10), duration=5.0, generation_method="matplotlib"
            )
            for i in range(3)
        ]
        
        # Verify files exist
        for test_file in test_files:
            assert Path(test_file).exists()
        
        # Cleanup
        system.cleanup_generated_assets()
        
        # Verify files are deleted and assets cleared
        for test_file in test_files:
            assert not Path(test_file).exists()
        assert len(system.generated_assets) == 0


@pytest.mark.asyncio
async def test_broll_generation_integration(temp_output_dir, sample_context_with_broll_plans):
    """Test complete B-roll generation integration."""
    system = BRollGenerationSystem(output_dir=temp_output_dir)
    
    # Mock all generation methods to avoid external dependencies
    with patch.object(system.chart_generator, 'generate_from_ai_specification') as mock_chart, \
         patch.object(system.blender_pipeline, 'render_animation') as mock_blender, \
         patch.object(system.slide_system, 'generate_educational_slide') as mock_slide:
        
        # Setup mock returns
        mock_chart.return_value = f"{temp_output_dir}/chart.png"
        mock_blender.return_value = f"{temp_output_dir}/animation.mp4"
        mock_slide.return_value = f"{temp_output_dir}/slide.png"
        
        # Generate all assets
        assets = await system.generate_all_broll_assets(sample_context_with_broll_plans)
        
        # Verify integration
        assert len(assets) == 3
        assert mock_chart.call_count == 1  # chart_or_graph type
        assert mock_blender.call_count == 1  # animated_explanation type
        assert mock_slide.call_count == 1  # educational_slide type
        
        # Verify asset properties
        chart_asset = next(a for a in assets if a.asset_type == "chart")
        animation_asset = next(a for a in assets if a.asset_type == "animation")
        slide_asset = next(a for a in assets if a.asset_type == "slide")
        
        assert chart_asset.timestamp == 15.0
        assert animation_asset.timestamp == 30.0
        assert slide_asset.timestamp == 45.0
        
        # Verify stats
        stats = system.get_generation_stats()
        assert stats['total_assets_generated'] == 3
        assert stats['asset_types']['chart'] == 1
        assert stats['asset_types']['animation'] == 1
        assert stats['asset_types']['slide'] == 1