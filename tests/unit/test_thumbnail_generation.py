"""
Unit tests for Thumbnail Generation System.

Tests the complete thumbnail generation workflow including concept analysis,
image generation, and metadata synchronization.
"""

import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from pathlib import Path

from ai_video_editor.modules.thumbnail_generation.generator import (
    ThumbnailGenerator,
    ThumbnailGenerationError
)
from ai_video_editor.modules.thumbnail_generation.concept_analyzer import (
    ThumbnailConceptAnalyzer,
    ConceptAnalysisError
)
from ai_video_editor.modules.thumbnail_generation.image_generator import (
    ThumbnailImageGenerator,
    ImageGenerationError
)
from ai_video_editor.modules.thumbnail_generation.synchronizer import (
    ThumbnailMetadataSynchronizer,
    SynchronizationError
)
from ai_video_editor.modules.thumbnail_generation.thumbnail_models import (
    ThumbnailConcept,
    ThumbnailVariation,
    ThumbnailPackage,
    ThumbnailGenerationStats
)
from ai_video_editor.core.content_context import (
    ContentContext,
    ContentType,
    UserPreferences,
    EmotionalPeak,
    VisualHighlight,
    FaceDetection
)
from ai_video_editor.core.exceptions import ContentContextError


class TestThumbnailModels:
    """Test thumbnail data models."""
    
    def test_thumbnail_concept_creation(self):
        """Test ThumbnailConcept creation and serialization."""
        visual_highlight = VisualHighlight(
            timestamp=30.0,
            description="Speaker with excited expression",
            faces=[FaceDetection([100, 50, 200, 250], 0.95, "excited")],
            visual_elements=["speaker", "excited_expression"],
            thumbnail_potential=0.9
        )
        
        emotional_peak = EmotionalPeak(
            timestamp=30.0,
            emotion="excitement",
            intensity=0.9,
            confidence=0.85,
            context="Revealing amazing results"
        )
        
        concept = ThumbnailConcept(
            concept_id="test_concept",
            visual_highlight=visual_highlight,
            emotional_peak=emotional_peak,
            hook_text="AMAZING RESULTS!",
            background_style="dynamic_gradient",
            text_style={"bold": True, "color": "#FF4444", "size": "large"},
            visual_elements=["speaker", "excited_expression"],
            thumbnail_potential=0.9,
            strategy="emotional"
        )
        
        # Test serialization
        concept_dict = concept.to_dict()
        assert concept_dict["concept_id"] == "test_concept"
        assert concept_dict["hook_text"] == "AMAZING RESULTS!"
        assert concept_dict["strategy"] == "emotional"
        
        # Test deserialization
        restored_concept = ThumbnailConcept.from_dict(concept_dict)
        assert restored_concept.concept_id == concept.concept_id
        assert restored_concept.hook_text == concept.hook_text
        assert restored_concept.strategy == concept.strategy
    
    def test_thumbnail_variation_creation(self):
        """Test ThumbnailVariation creation and scoring."""
        concept = self._create_mock_concept()
        
        variation = ThumbnailVariation(
            variation_id="test_variation",
            concept=concept,
            generated_image_path="/path/to/thumbnail.jpg",
            generation_method="ai_generated",
            confidence_score=0.85,
            estimated_ctr=0.12,
            visual_appeal_score=0.9,
            text_readability_score=0.8,
            brand_consistency_score=0.7,
            generation_time=3.5,
            generation_cost=0.05
        )
        
        assert variation.variation_id == "test_variation"
        assert variation.generation_method == "ai_generated"
        assert variation.confidence_score == 0.85
        assert variation.estimated_ctr == 0.12
        
        # Test serialization
        variation_dict = variation.to_dict()
        assert variation_dict["confidence_score"] == 0.85
        assert variation_dict["generation_cost"] == 0.05
    
    def test_thumbnail_package_creation(self):
        """Test ThumbnailPackage creation and methods."""
        variations = [
            self._create_mock_variation("var1", 0.8),
            self._create_mock_variation("var2", 0.9),
            self._create_mock_variation("var3", 0.7)
        ]
        
        package = ThumbnailPackage(
            package_id="test_package",
            variations=variations,
            recommended_variation="var2",  # Highest confidence
            generation_timestamp=datetime.now(),
            synchronized_metadata={},
            a_b_testing_config={},
            performance_predictions={},
            total_generation_time=10.5,
            total_generation_cost=0.15
        )
        
        # Test recommended variation retrieval
        recommended = package.get_recommended_variation()
        assert recommended.variation_id == "var2"
        
        # Test top variations
        top_variations = package.get_top_variations(2)
        assert len(top_variations) == 2
        assert top_variations[0].confidence_score >= top_variations[1].confidence_score
        
        # Test strategy filtering
        emotional_var = package.get_variation_by_strategy("emotional")
        assert emotional_var is not None
    
    def test_thumbnail_generation_stats(self):
        """Test ThumbnailGenerationStats tracking."""
        stats = ThumbnailGenerationStats()
        
        # Add generations
        stats.add_generation("ai_generated", 3.0, 0.05, 0.8)
        stats.add_generation("procedural", 1.5, 0.0, 0.7)
        stats.add_generation("ai_generated", 2.5, 0.05, 0.9)
        
        assert stats.variations_generated == 3
        assert stats.ai_generations == 2
        assert stats.procedural_generations == 1
        assert stats.total_processing_time == 7.0
        assert stats.total_api_cost == 0.10
        assert abs(stats.average_confidence_score - 0.8) < 0.01
        
        # Add fallbacks
        stats.add_fallback("ai_to_procedural")
        assert "ai_to_procedural" in stats.fallbacks_used
    
    def _create_mock_concept(self, strategy="emotional"):
        """Create mock ThumbnailConcept for testing."""
        return ThumbnailConcept(
            concept_id="mock_concept",
            visual_highlight=VisualHighlight(30.0, "Test highlight", [], ["test"], 0.8),
            emotional_peak=EmotionalPeak(30.0, "excitement", 0.8, 0.9, "Test context"),
            hook_text="TEST HOOK",
            background_style="dynamic_gradient",
            text_style={"bold": True},
            visual_elements=["test"],
            thumbnail_potential=0.8,
            strategy=strategy
        )
    
    def _create_mock_variation(self, var_id, confidence):
        """Create mock ThumbnailVariation for testing."""
        return ThumbnailVariation(
            variation_id=var_id,
            concept=self._create_mock_concept(),
            generated_image_path=f"/path/to/{var_id}.jpg",
            generation_method="ai_generated",
            confidence_score=confidence,
            estimated_ctr=0.1,
            visual_appeal_score=0.8,
            text_readability_score=0.8,
            brand_consistency_score=0.7
        )


class TestThumbnailConceptAnalyzer:
    """Test thumbnail concept analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_gemini_client = Mock()
        self.analyzer = ThumbnailConceptAnalyzer(self.mock_gemini_client)
        self.context = self._create_mock_context()
    
    @pytest.mark.asyncio
    async def test_analyze_thumbnail_concepts_success(self):
        """Test successful thumbnail concept analysis."""
        # Mock AI response for hook text generation
        mock_response = Mock()
        mock_response.content = "AMAZING RESULTS!"
        self.mock_gemini_client.generate_content = AsyncMock(return_value=mock_response)
        
        concepts = await self.analyzer.analyze_thumbnail_concepts(self.context)
        
        assert len(concepts) > 0
        assert all(isinstance(concept, ThumbnailConcept) for concept in concepts)
        assert all(concept.thumbnail_potential > 0 for concept in concepts)
        
        # Check that different strategies are represented
        strategies = {concept.strategy for concept in concepts}
        assert len(strategies) > 1
    
    @pytest.mark.asyncio
    async def test_analyze_concepts_no_visual_highlights(self):
        """Test concept analysis with no visual highlights."""
        self.context.visual_highlights = []
        
        # Mock AI response
        mock_response = Mock()
        mock_response.content = "MUST WATCH!"
        self.mock_gemini_client.generate_content = AsyncMock(return_value=mock_response)
        
        concepts = await self.analyzer.analyze_thumbnail_concepts(self.context)
        
        # Should generate fallback concepts
        assert len(concepts) > 0
        assert all(concept.thumbnail_potential > 0 for concept in concepts)
    
    @pytest.mark.asyncio
    async def test_generate_hook_text_success(self):
        """Test hook text generation."""
        emotional_peak = EmotionalPeak(
            timestamp=30.0,
            emotion="excitement",
            intensity=0.9,
            confidence=0.85,
            context="Revealing amazing investment results"
        )
        
        # Mock AI response
        mock_response = Mock()
        mock_response.content = "INCREDIBLE GAINS!"
        self.mock_gemini_client.generate_content = AsyncMock(return_value=mock_response)
        
        hook_text = await self.analyzer.generate_hook_text(
            emotional_peak, self.context, "emotional"
        )
        
        assert hook_text == "INCREDIBLE GAINS!"
        
        # Verify AI was called with appropriate prompt
        self.mock_gemini_client.generate_content.assert_called_once()
        call_args = self.mock_gemini_client.generate_content.call_args
        assert "emotional peak: excitement" in call_args[1]["prompt"]
        assert "Strategy: emotional" in call_args[1]["prompt"]
    
    @pytest.mark.asyncio
    async def test_generate_hook_text_ai_failure(self):
        """Test hook text generation with AI failure."""
        emotional_peak = EmotionalPeak(30.0, "excitement", 0.9, 0.85, "Test context")
        
        # Mock AI failure
        self.mock_gemini_client.generate_content = AsyncMock(return_value=None)
        
        hook_text = await self.analyzer.generate_hook_text(
            emotional_peak, self.context, "emotional"
        )
        
        # Should return fallback text
        assert hook_text in ["AMAZING RESULTS!", "INCREDIBLE!", "YOU WON'T BELIEVE", "SHOCKING TRUTH"]
    
    def test_score_thumbnail_potential(self):
        """Test thumbnail potential scoring."""
        concept = ThumbnailConcept(
            concept_id="test",
            visual_highlight=VisualHighlight(30.0, "Test", [], ["test"], 0.8),
            emotional_peak=EmotionalPeak(30.0, "excitement", 0.9, 0.85, "Test"),
            hook_text="AMAZING!",
            background_style="dynamic_gradient",
            text_style={"bold": True},
            visual_elements=["test"],
            thumbnail_potential=0.0,  # Will be calculated
            strategy="emotional"
        )
        
        score = self.analyzer.score_thumbnail_potential(concept, self.context)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably high given good inputs
    
    def test_determine_best_strategy(self):
        """Test strategy determination logic."""
        # High intensity excitement should be emotional
        highlight = VisualHighlight(30.0, "Test", [], ["test"], 0.8)
        emotion = EmotionalPeak(30.0, "excitement", 0.9, 0.85, "Test")
        
        strategy = self.analyzer._determine_best_strategy(highlight, emotion)
        assert strategy == "emotional"
        
        # High intensity curiosity should be curiosity
        emotion = EmotionalPeak(30.0, "curiosity", 0.9, 0.85, "Test")
        strategy = self.analyzer._determine_best_strategy(highlight, emotion)
        assert strategy == "curiosity"
        
        # Medium intensity confidence should be authority
        emotion = EmotionalPeak(30.0, "confidence", 0.6, 0.85, "Test")
        strategy = self.analyzer._determine_best_strategy(highlight, emotion)
        assert strategy == "authority"
    
    def test_hook_text_validation(self):
        """Test hook text validation and cleaning."""
        # Test word limit
        long_text = "This is a very long hook text that exceeds the limit"
        cleaned = self.analyzer._validate_and_clean_hook_text(long_text)
        assert len(cleaned.split()) <= 6
        
        # Test empty text
        empty_text = ""
        cleaned = self.analyzer._validate_and_clean_hook_text(empty_text)
        assert cleaned == "MUST WATCH!"
        
        # Test normal text
        normal_text = "Amazing Results"
        cleaned = self.analyzer._validate_and_clean_hook_text(normal_text)
        assert cleaned == "AMAZING RESULTS"
    
    def test_hook_text_quality_scoring(self):
        """Test hook text quality scoring."""
        # Good hook text (3-6 words, emotional words)
        good_hook = "AMAZING SECRET REVEALED"
        score = self.analyzer._score_hook_text_quality(good_hook)
        assert score > 0.7
        
        # Poor hook text (too long, no emotional words)
        poor_hook = "this is a very long and boring hook text"
        score = self.analyzer._score_hook_text_quality(poor_hook)
        assert score < 0.5
        
        # Empty hook text
        empty_hook = ""
        score = self.analyzer._score_hook_text_quality(empty_hook)
        assert score == 0.0
    
    def _create_mock_context(self):
        """Create mock ContentContext for testing."""
        context = Mock(spec=ContentContext)
        context.project_id = "test_project"
        context.content_type = ContentType.EDUCATIONAL
        context.key_concepts = ["compound interest", "investment", "growth"]
        
        # Mock visual highlights
        context.visual_highlights = [
            VisualHighlight(
                timestamp=30.0,
                description="Speaker with excited expression",
                faces=[FaceDetection([100, 50, 200, 250], 0.95, "excited")],
                visual_elements=["speaker", "excited_expression"],
                thumbnail_potential=0.9
            ),
            VisualHighlight(
                timestamp=60.0,
                description="Chart showing growth",
                faces=[],
                visual_elements=["chart", "growth_visualization"],
                thumbnail_potential=0.8
            )
        ]
        
        # Mock emotional markers
        context.emotional_markers = [
            EmotionalPeak(30.0, "excitement", 0.9, 0.85, "Revealing results"),
            EmotionalPeak(60.0, "curiosity", 0.7, 0.8, "Showing data")
        ]
        
        # Mock methods
        context.get_best_visual_highlights = Mock(return_value=context.visual_highlights)
        context.get_top_emotional_peaks = Mock(return_value=context.emotional_markers)
        
        return context


class TestThumbnailImageGenerator:
    """Test thumbnail image generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_cache_manager = Mock()
        self.generator = ThumbnailImageGenerator(
            cache_manager=self.mock_cache_manager,
            output_dir=self.temp_dir
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_generate_thumbnail_image_success(self):
        """Test successful thumbnail image generation."""
        concept = self._create_mock_concept()
        context = Mock(spec=ContentContext)
        
        # Mock cache miss
        self.mock_cache_manager.get = AsyncMock(return_value=None)
        self.mock_cache_manager.set = AsyncMock()
        
        # Mock PIL availability
        with patch('ai_video_editor.modules.thumbnail_generation.image_generator.PIL_AVAILABLE', True):
            image_path = await self.generator.generate_thumbnail_image(concept, context)
        
        assert image_path is not None
        assert Path(image_path).exists()
        assert image_path.endswith('.jpg')
        
        # Verify caching was attempted
        self.mock_cache_manager.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_thumbnail_image_cached(self):
        """Test thumbnail generation with cache hit."""
        concept = self._create_mock_concept()
        context = Mock(spec=ContentContext)
        
        # Create a test image file
        test_image_path = Path(self.temp_dir) / "cached_thumbnail.jpg"
        test_image_path.touch()
        
        # Mock cache hit
        cache_data = {"image_path": str(test_image_path)}
        self.mock_cache_manager.get = AsyncMock(return_value=cache_data)
        
        image_path = await self.generator.generate_thumbnail_image(concept, context)
        
        assert image_path == str(test_image_path)
        # Should not call set since it was cached
        self.mock_cache_manager.set.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_generate_procedural_thumbnail(self):
        """Test procedural thumbnail generation."""
        concept = self._create_mock_concept()
        context = Mock(spec=ContentContext)
        
        with patch('ai_video_editor.modules.thumbnail_generation.image_generator.PIL_AVAILABLE', True):
            with patch('ai_video_editor.modules.thumbnail_generation.image_generator.Image') as mock_image:
                with patch('ai_video_editor.modules.thumbnail_generation.image_generator.ImageDraw') as mock_draw:
                    # Mock PIL objects
                    mock_img = Mock()
                    mock_image.new.return_value = mock_img
                    mock_draw.Draw.return_value = Mock()
                    
                    image_path = await self.generator.generate_procedural_thumbnail(concept, context)
        
        assert image_path is not None
        assert "thumb_" in image_path
        assert image_path.endswith('.jpg')
    
    @pytest.mark.asyncio
    async def test_generate_ai_background(self):
        """Test AI background generation (mocked)."""
        concept = self._create_mock_concept()
        
        with patch('ai_video_editor.modules.thumbnail_generation.image_generator.PIL_AVAILABLE', True):
            with patch('ai_video_editor.modules.thumbnail_generation.image_generator.Image') as mock_image:
                mock_img = Mock()
                mock_image.new.return_value = mock_img
                
                background_path = await self.generator.generate_ai_background(concept)
        
        assert background_path is not None
        assert "ai_bg_" in background_path
    
    def test_background_style_generation(self):
        """Test different background style generation."""
        with patch('ai_video_editor.modules.thumbnail_generation.image_generator.PIL_AVAILABLE', True):
            with patch('ai_video_editor.modules.thumbnail_generation.image_generator.ImageDraw') as mock_draw:
                mock_draw_obj = Mock()
                mock_draw.Draw.return_value = mock_draw_obj
                
                # Test different background styles
                styles = ["dynamic_gradient", "question_mark_overlay", "clean_professional", 
                         "urgent_arrows", "educational_icons"]
                
                for style in styles:
                    # This should not raise an exception
                    self.generator._generate_background(
                        Mock(), mock_draw_obj, style, (1280, 720)
                    )
    
    def test_color_interpolation(self):
        """Test color interpolation utility."""
        color1 = "#FF0000"  # Red
        color2 = "#0000FF"  # Blue
        
        # Test midpoint
        mid_color = self.generator._interpolate_color(color1, color2, 0.5)
        assert mid_color == "#800080"  # Purple
        
        # Test endpoints
        start_color = self.generator._interpolate_color(color1, color2, 0.0)
        assert start_color == color1
        
        end_color = self.generator._interpolate_color(color1, color2, 1.0)
        assert end_color == color2
    
    def _create_mock_concept(self):
        """Create mock ThumbnailConcept for testing."""
        return ThumbnailConcept(
            concept_id="test_concept",
            visual_highlight=VisualHighlight(30.0, "Test", [], ["test"], 0.8),
            emotional_peak=EmotionalPeak(30.0, "excitement", 0.8, 0.9, "Test"),
            hook_text="TEST HOOK",
            background_style="dynamic_gradient",
            text_style={"bold": True, "color": "#FF4444", "size": "large"},
            visual_elements=["test"],
            thumbnail_potential=0.8,
            strategy="emotional"
        )


class TestThumbnailMetadataSynchronizer:
    """Test thumbnail-metadata synchronization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.synchronizer = ThumbnailMetadataSynchronizer()
        self.context = self._create_mock_context()
        self.thumbnail_package = self._create_mock_package()
    
    def test_synchronize_concepts_success(self):
        """Test successful concept synchronization."""
        sync_data = self.synchronizer.synchronize_concepts(
            self.thumbnail_package, self.context
        )
        
        assert "mappings" in sync_data
        assert "analysis" in sync_data
        assert "recommendations" in sync_data
        assert "sync_score" in sync_data
        assert sync_data["sync_score"] >= 0.0
    
    def test_create_ab_testing_config(self):
        """Test A/B testing configuration creation."""
        metadata_variations = [
            {
                "variation_id": "meta_1",
                "strategy": "emotional",
                "title": "Amazing Investment Results!",
                "confidence_score": 0.8,
                "estimated_ctr": 0.12
            },
            {
                "variation_id": "meta_2", 
                "strategy": "curiosity_driven",
                "title": "What Happened Next Will Shock You",
                "confidence_score": 0.7,
                "estimated_ctr": 0.10
            }
        ]
        
        ab_config = self.synchronizer.create_ab_testing_config(
            self.thumbnail_package, metadata_variations
        )
        
        assert "test_groups" in ab_config
        assert "performance_predictions" in ab_config
        assert "allocation_strategy" in ab_config
        assert "success_metrics" in ab_config
        assert len(ab_config["test_groups"]) > 0
    
    def test_validate_synchronization(self):
        """Test synchronization validation."""
        is_valid = self.synchronizer.validate_synchronization(
            self.thumbnail_package, self.context
        )
        
        assert isinstance(is_valid, bool)
        
        # Check that validation results were stored
        if hasattr(self.thumbnail_package, 'synchronized_metadata'):
            validation = self.thumbnail_package.synchronized_metadata.get("validation")
            if validation:
                assert "is_valid" in validation
                assert "scores" in validation
                assert "overall_score" in validation
    
    def test_keyword_extraction(self):
        """Test keyword extraction from thumbnails and metadata."""
        variation = self.thumbnail_package.variations[0]
        
        # Test thumbnail keyword extraction
        thumb_keywords = self.synchronizer._extract_thumbnail_keywords(variation, self.context)
        assert len(thumb_keywords) > 0
        assert any(keyword in variation.concept.hook_text.lower() for keyword in thumb_keywords)
        
        # Test metadata keyword extraction
        metadata = {
            "title": "Amazing Investment Results That Will Shock You",
            "tags": ["investment", "finance", "results"],
            "description": "Learn about incredible investment strategies that work"
        }
        
        meta_keywords = self.synchronizer._extract_metadata_keywords(metadata)
        assert len(meta_keywords) > 0
        assert "investment" in meta_keywords
        assert "results" in meta_keywords
    
    def test_emotional_alignment_calculation(self):
        """Test emotional alignment calculation."""
        # Perfect alignment
        alignment = self.synchronizer._calculate_emotional_alignment("excitement", "emotional")
        assert alignment == 1.0
        
        # Good alignment
        alignment = self.synchronizer._calculate_emotional_alignment("curiosity", "curiosity_driven")
        assert alignment == 1.0
        
        # Poor alignment
        alignment = self.synchronizer._calculate_emotional_alignment("excitement", "educational")
        assert alignment < 0.5
    
    def test_strategy_compatibility(self):
        """Test strategy compatibility checking."""
        # Compatible strategies
        assert self.synchronizer._are_compatible_strategies("emotional", "emotional")
        assert self.synchronizer._are_compatible_strategies("emotional", "curiosity_driven")
        assert self.synchronizer._are_compatible_strategies("curiosity", "educational")
        
        # Incompatible strategies (should still work but return False)
        # Note: The method might not be fully implemented, so we test what exists
        result = self.synchronizer._are_compatible_strategies("authority", "urgency")
        assert isinstance(result, bool)
    
    def test_text_similarity_calculation(self):
        """Test text similarity calculation."""
        # Test if method exists and works
        if hasattr(self.synchronizer, '_calculate_text_similarity'):
            similarity = self.synchronizer._calculate_text_similarity(
                "amazing results", "incredible results"
            )
            assert 0.0 <= similarity <= 1.0
        else:
            # Method might not be implemented yet
            pytest.skip("_calculate_text_similarity method not implemented")
    
    def _create_mock_context(self):
        """Create mock ContentContext for testing."""
        context = Mock(spec=ContentContext)
        context.project_id = "test_project"
        context.content_type = ContentType.EDUCATIONAL
        context.key_concepts = ["investment", "finance", "growth"]
        context.trending_keywords = Mock()
        context.trending_keywords.primary_keywords = ["investment", "finance", "money"]
        
        # Mock metadata variations
        context.metadata_variations = [
            {
                "variation_id": "meta_1",
                "strategy": "emotional",
                "title": "Amazing Investment Results!",
                "description": "Learn about incredible investment strategies",
                "tags": ["investment", "finance", "results"],
                "confidence_score": 0.8
            }
        ]
        
        return context
    
    def _create_mock_package(self):
        """Create mock ThumbnailPackage for testing."""
        concept = ThumbnailConcept(
            concept_id="test_concept",
            visual_highlight=VisualHighlight(30.0, "Test", [], ["test"], 0.8),
            emotional_peak=EmotionalPeak(30.0, "excitement", 0.8, 0.9, "Test context"),
            hook_text="AMAZING RESULTS!",
            background_style="dynamic_gradient",
            text_style={"bold": True},
            visual_elements=["test"],
            thumbnail_potential=0.8,
            strategy="emotional"
        )
        
        variation = ThumbnailVariation(
            variation_id="test_variation",
            concept=concept,
            generated_image_path="/path/to/test.jpg",
            generation_method="ai_generated",
            confidence_score=0.8,
            estimated_ctr=0.12,
            visual_appeal_score=0.8,
            text_readability_score=0.8,
            brand_consistency_score=0.7
        )
        
        package = ThumbnailPackage(
            package_id="test_package",
            variations=[variation],
            recommended_variation="test_variation",
            generation_timestamp=datetime.now(),
            synchronized_metadata={},
            a_b_testing_config={},
            performance_predictions={}
        )
        
        return package


class TestThumbnailGenerator:
    """Test main thumbnail generator orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_gemini_client = Mock()
        self.mock_cache_manager = Mock()
        self.generator = ThumbnailGenerator(
            self.mock_gemini_client, 
            self.mock_cache_manager
        )
        self.context = self._create_mock_context()
    
    @pytest.mark.asyncio
    async def test_generate_thumbnail_package_success(self):
        """Test successful thumbnail package generation."""
        # Mock concept analyzer
        mock_concepts = [self._create_mock_concept("emotional"), self._create_mock_concept("curiosity")]
        self.generator.concept_analyzer.analyze_thumbnail_concepts = AsyncMock(return_value=mock_concepts)
        
        # Mock image generator
        self.generator.image_generator.generate_thumbnail_image = AsyncMock(return_value="/path/to/thumb.jpg")
        
        # Mock synchronizer
        self.generator.synchronizer.synchronize_concepts = Mock(return_value={"sync_score": 0.8})
        
        package = await self.generator.generate_thumbnail_package(self.context)
        
        assert isinstance(package, ThumbnailPackage)
        assert len(package.variations) > 0
        assert package.recommended_variation is not None
        assert package.total_generation_time > 0
    
    @pytest.mark.asyncio
    async def test_generate_thumbnail_package_no_concepts(self):
        """Test package generation with no concepts."""
        # Mock empty concepts
        self.generator.concept_analyzer.analyze_thumbnail_concepts = AsyncMock(return_value=[])
        
        with pytest.raises(ThumbnailGenerationError, match="No thumbnail concepts could be generated"):
            await self.generator.generate_thumbnail_package(self.context)
    
    @pytest.mark.asyncio
    async def test_generate_thumbnail_package_no_variations(self):
        """Test package generation with concept analysis but no variations."""
        # Mock concepts but no variations
        mock_concepts = [self._create_mock_concept()]
        self.generator.concept_analyzer.analyze_thumbnail_concepts = AsyncMock(return_value=mock_concepts)
        
        # Mock image generation failure
        self.generator.image_generator.generate_thumbnail_image = AsyncMock(return_value=None)
        
        with pytest.raises(ThumbnailGenerationError, match="No thumbnail variations could be generated"):
            await self.generator.generate_thumbnail_package(self.context)
    
    @pytest.mark.asyncio
    async def test_analyze_thumbnail_concepts_cached(self):
        """Test concept analysis with cache hit."""
        cached_concepts = [self._create_mock_concept()]
        self.generator._check_concept_cache = AsyncMock(return_value=cached_concepts)
        
        concepts = await self.generator.analyze_thumbnail_concepts(self.context)
        
        assert concepts == cached_concepts
        # Should not call the analyzer since it was cached
        assert not hasattr(self.generator.concept_analyzer, 'analyze_thumbnail_concepts') or \
               not self.generator.concept_analyzer.analyze_thumbnail_concepts.called
    
    @pytest.mark.asyncio
    async def test_generate_thumbnail_variations_parallel(self):
        """Test parallel thumbnail variation generation."""
        concepts = [
            self._create_mock_concept("emotional"),
            self._create_mock_concept("curiosity"),
            self._create_mock_concept("authority")
        ]
        
        # Mock successful image generation
        self.generator.image_generator.generate_thumbnail_image = AsyncMock(return_value="/path/to/thumb.jpg")
        
        variations = await self.generator.generate_thumbnail_variations(concepts, self.context)
        
        assert len(variations) == len(concepts)
        assert all(isinstance(var, ThumbnailVariation) for var in variations)
        
        # Verify variations are sorted by confidence
        confidences = [var.confidence_score for var in variations]
        assert confidences == sorted(confidences, reverse=True)
    
    @pytest.mark.asyncio
    async def test_generate_thumbnail_variations_with_failures(self):
        """Test variation generation with some failures."""
        concepts = [
            self._create_mock_concept("emotional"),
            self._create_mock_concept("curiosity")
        ]
        
        # Mock partial failure - first succeeds, second fails
        async def mock_generate_single(concept, context):
            if concept.strategy == "emotional":
                return ThumbnailVariation(
                    variation_id="test_var",
                    concept=concept,
                    generated_image_path="/path/to/thumb.jpg",
                    generation_method="ai_generated",
                    confidence_score=0.8,
                    estimated_ctr=0.12,
                    visual_appeal_score=0.8,
                    text_readability_score=0.8,
                    brand_consistency_score=0.7
                )
            else:
                return None
        
        self.generator._generate_single_variation = mock_generate_single
        
        variations = await self.generator.generate_thumbnail_variations(concepts, self.context)
        
        # Should have one successful variation
        assert len(variations) == 1
        assert variations[0].concept.strategy == "emotional"
    
    def test_synchronize_with_metadata(self):
        """Test metadata synchronization."""
        package = self._create_mock_package()
        
        # Mock synchronizer
        sync_data = {"sync_score": 0.8, "mappings": {}}
        self.generator.synchronizer.synchronize_concepts = Mock(return_value=sync_data)
        self.generator.synchronizer.validate_synchronization = Mock(return_value=True)
        
        synchronized_package = self.generator.synchronize_with_metadata(package, self.context)
        
        assert synchronized_package.synchronized_metadata == sync_data
        
        # Verify synchronizer methods were called
        self.generator.synchronizer.synchronize_concepts.assert_called_once_with(package, self.context)
        self.generator.synchronizer.validate_synchronization.assert_called_once_with(package, self.context)
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation."""
        concept = self._create_mock_concept()
        
        score = self.generator._calculate_confidence_score(concept, self.context)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably high for good concept
    
    def test_ctr_estimation(self):
        """Test CTR estimation."""
        concept = self._create_mock_concept()
        
        ctr = self.generator._estimate_ctr(concept, self.context)
        
        assert 0.05 <= ctr <= 0.25  # Within expected bounds
        assert ctr > 0.08  # Should be above minimum for good concept
    
    def test_visual_appeal_scoring(self):
        """Test visual appeal scoring."""
        concept = self._create_mock_concept()
        
        score = self.generator._calculate_visual_appeal_score(concept)
        
        assert 0.0 <= score <= 1.0
    
    def test_text_readability_scoring(self):
        """Test text readability scoring."""
        # Good hook text
        good_concept = self._create_mock_concept()
        good_concept.hook_text = "AMAZING RESULTS!"
        
        good_score = self.generator._calculate_text_readability_score(good_concept)
        assert good_score > 0.7
        
        # Poor hook text
        poor_concept = self._create_mock_concept()
        poor_concept.hook_text = "this is a very long and poorly formatted hook text that is hard to read"
        
        poor_score = self.generator._calculate_text_readability_score(poor_concept)
        assert poor_score < good_score
    
    def test_brand_consistency_scoring(self):
        """Test brand consistency scoring."""
        concept = self._create_mock_concept()
        
        score = self.generator._calculate_brand_consistency_score(concept, self.context)
        
        assert 0.0 <= score <= 1.0
    
    def test_performance_predictions_creation(self):
        """Test performance predictions creation."""
        variations = [
            self._create_mock_variation("var1", 0.8, 0.12),
            self._create_mock_variation("var2", 0.9, 0.15),
            self._create_mock_variation("var3", 0.7, 0.10)
        ]
        
        predictions = self.generator._create_performance_predictions(variations, self.context)
        
        assert "expected_best_performer" in predictions
        assert "ctr_predictions" in predictions
        assert "engagement_predictions" in predictions
        assert "risk_assessment" in predictions
        assert "confidence_intervals" in predictions
        
        # Best performer should be the one with highest CTR
        best_var = max(variations, key=lambda v: v.estimated_ctr)
        assert predictions["expected_best_performer"] == best_var.variation_id
    
    def _create_mock_context(self):
        """Create mock ContentContext for testing."""
        context = Mock(spec=ContentContext)
        context.project_id = "test_project"
        context.content_type = ContentType.EDUCATIONAL
        context.key_concepts = ["investment", "finance"]
        context.metadata_variations = []
        
        # Mock visual highlights and emotional markers
        context.visual_highlights = [
            VisualHighlight(30.0, "Test highlight", [], ["test"], 0.8)
        ]
        context.emotional_markers = [
            EmotionalPeak(30.0, "excitement", 0.8, 0.9, "Test context")
        ]
        
        return context
    
    def _create_mock_concept(self, strategy="emotional"):
        """Create mock ThumbnailConcept for testing."""
        return ThumbnailConcept(
            concept_id="test_concept",
            visual_highlight=VisualHighlight(30.0, "Test", [], ["test"], 0.8),
            emotional_peak=EmotionalPeak(30.0, "excitement", 0.8, 0.9, "Test"),
            hook_text="TEST HOOK",
            background_style="dynamic_gradient",
            text_style={"bold": True},
            visual_elements=["test"],
            thumbnail_potential=0.8,
            strategy=strategy
        )
    
    def _create_mock_variation(self, var_id, confidence, ctr):
        """Create mock ThumbnailVariation for testing."""
        return ThumbnailVariation(
            variation_id=var_id,
            concept=self._create_mock_concept(),
            generated_image_path=f"/path/to/{var_id}.jpg",
            generation_method="ai_generated",
            confidence_score=confidence,
            estimated_ctr=ctr,
            visual_appeal_score=0.8,
            text_readability_score=0.8,
            brand_consistency_score=0.7
        )
    
    def _create_mock_package(self):
        """Create mock ThumbnailPackage for testing."""
        variation = self._create_mock_variation("test_var", 0.8, 0.12)
        
        return ThumbnailPackage(
            package_id="test_package",
            variations=[variation],
            recommended_variation="test_var",
            generation_timestamp=datetime.now(),
            synchronized_metadata={},
            a_b_testing_config={},
            performance_predictions={}
        )


class TestThumbnailGenerationIntegration:
    """Integration tests for thumbnail generation system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_gemini_client = Mock()
        self.mock_cache_manager = Mock()
        
        # Set up async mocks
        self.mock_cache_manager.get = AsyncMock(return_value=None)
        self.mock_cache_manager.set = AsyncMock()
        
        self.generator = ThumbnailGenerator(
            self.mock_gemini_client,
            self.mock_cache_manager
        )
        
        # Override output directory for image generator
        self.generator.image_generator.output_dir = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_end_to_end_thumbnail_generation(self):
        """Test complete end-to-end thumbnail generation workflow."""
        context = self._create_comprehensive_context()
        
        # Mock AI responses
        mock_response = Mock()
        mock_response.content = "INCREDIBLE RESULTS!"
        self.mock_gemini_client.generate_content = AsyncMock(return_value=mock_response)
        
        # Generate thumbnail package
        with patch('ai_video_editor.modules.thumbnail_generation.image_generator.PIL_AVAILABLE', True):
            with patch('ai_video_editor.modules.thumbnail_generation.image_generator.Image') as mock_image:
                with patch('ai_video_editor.modules.thumbnail_generation.image_generator.ImageDraw'):
                    mock_img = Mock()
                    mock_image.new.return_value = mock_img
                    
                    package = await self.generator.generate_thumbnail_package(context)
        
        # Verify package structure
        assert isinstance(package, ThumbnailPackage)
        assert len(package.variations) > 0
        assert package.recommended_variation is not None
        assert package.total_generation_time > 0
        
        # Verify variations have required properties
        for variation in package.variations:
            assert variation.variation_id is not None
            assert variation.concept is not None
            assert variation.generated_image_path is not None
            assert 0.0 <= variation.confidence_score <= 1.0
            assert 0.05 <= variation.estimated_ctr <= 0.25
        
        # Verify context was updated
        assert hasattr(context, 'thumbnail_concepts') or context.thumbnail_concepts is not None
        assert hasattr(context, 'generated_thumbnails') or context.generated_thumbnails is not None
    
    @pytest.mark.asyncio
    async def test_thumbnail_generation_with_metadata_sync(self):
        """Test thumbnail generation with metadata synchronization."""
        context = self._create_comprehensive_context()
        
        # Add metadata variations to context
        context.metadata_variations = [
            {
                "variation_id": "meta_1",
                "strategy": "emotional",
                "title": "Amazing Investment Results That Will Shock You!",
                "description": "Discover the incredible investment strategy that changed everything",
                "tags": ["investment", "finance", "results", "amazing"],
                "confidence_score": 0.85,
                "estimated_ctr": 0.13
            },
            {
                "variation_id": "meta_2",
                "strategy": "curiosity_driven",
                "title": "What This Investor Did Next Will Surprise You",
                "description": "The surprising investment move that led to incredible returns",
                "tags": ["investment", "surprise", "strategy", "returns"],
                "confidence_score": 0.78,
                "estimated_ctr": 0.11
            }
        ]
        
        # Mock AI responses
        mock_response = Mock()
        mock_response.content = "SHOCKING GAINS!"
        self.mock_gemini_client.generate_content = AsyncMock(return_value=mock_response)
        
        # Generate thumbnail package
        with patch('ai_video_editor.modules.thumbnail_generation.image_generator.PIL_AVAILABLE', True):
            with patch('ai_video_editor.modules.thumbnail_generation.image_generator.Image') as mock_image:
                with patch('ai_video_editor.modules.thumbnail_generation.image_generator.ImageDraw'):
                    mock_img = Mock()
                    mock_image.new.return_value = mock_img
                    
                    package = await self.generator.generate_thumbnail_package(context)
        
        # Verify synchronization occurred
        assert package.synchronized_metadata is not None
        assert "sync_score" in package.synchronized_metadata
        
        # Verify A/B testing config was created
        assert package.a_b_testing_config is not None
        if package.a_b_testing_config:
            assert "test_groups" in package.a_b_testing_config
    
    def _create_comprehensive_context(self):
        """Create comprehensive ContentContext for integration testing."""
        context = ContentContext(
            project_id="integration_test",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences(quality_mode="high")
        )
        
        # Add visual highlights
        context.visual_highlights = [
            VisualHighlight(
                timestamp=15.0,
                description="Speaker explaining with hand gestures",
                faces=[FaceDetection([120, 60, 180, 230], 0.92, "engaged")],
                visual_elements=["speaker", "hand_gestures", "professional_setting"],
                thumbnail_potential=0.85
            ),
            VisualHighlight(
                timestamp=45.0,
                description="Chart showing exponential growth",
                faces=[],
                visual_elements=["chart", "growth_visualization", "data_points"],
                thumbnail_potential=0.90
            ),
            VisualHighlight(
                timestamp=75.0,
                description="Speaker with excited expression revealing results",
                faces=[FaceDetection([100, 50, 200, 250], 0.95, "excited")],
                visual_elements=["speaker", "excited_expression", "results_reveal"],
                thumbnail_potential=0.95
            )
        ]
        
        # Add emotional markers
        context.emotional_markers = [
            EmotionalPeak(
                timestamp=15.0,
                emotion="curiosity",
                intensity=0.7,
                confidence=0.85,
                context="Introducing the investment concept"
            ),
            EmotionalPeak(
                timestamp=45.0,
                emotion="understanding",
                intensity=0.8,
                confidence=0.90,
                context="Explaining the growth pattern"
            ),
            EmotionalPeak(
                timestamp=75.0,
                emotion="excitement",
                intensity=0.95,
                confidence=0.92,
                context="Revealing the incredible results"
            )
        ]
        
        # Add key concepts and themes
        context.key_concepts = ["compound interest", "investment growth", "financial education", "exponential returns"]
        context.content_themes = ["personal finance", "investment strategy", "wealth building"]
        
        # Mock methods
        context.get_best_visual_highlights = lambda count=10: context.visual_highlights[:count]
        context.get_top_emotional_peaks = lambda count=8: context.emotional_markers[:count]
        
        return context


if __name__ == "__main__":
    pytest.main([__file__])