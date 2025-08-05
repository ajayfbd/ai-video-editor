"""
Unit tests for MetadataPackageIntegrator module.

Tests comprehensive metadata package integration including synchronization
with video content, thumbnails, and AI Director decisions.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from ai_video_editor.modules.intelligence.metadata_integration import (
    MetadataPackageIntegrator,
    ThumbnailMetadataAlignment,
    MetadataValidationResult,
    IntegratedMetadataPackage
)
from ai_video_editor.modules.intelligence.metadata_generator import (
    MetadataGenerator,
    MetadataVariation,
    MetadataPackage
)
from ai_video_editor.core.content_context import (
    ContentContext,
    ContentType,
    TrendingKeywords,
    EmotionalPeak,
    VisualHighlight,
    FaceDetection,
    AudioAnalysisResult,
    AudioSegment,
    UserPreferences
)
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.exceptions import ContentContextError


class TestThumbnailMetadataAlignment:
    """Test ThumbnailMetadataAlignment data class."""
    
    def test_thumbnail_alignment_creation(self):
        """Test ThumbnailMetadataAlignment creation and serialization."""
        alignment = ThumbnailMetadataAlignment(
            thumbnail_concept="growth chart",
            aligned_title="This Growth Chart Reveals Everything!",
            aligned_description="See how this growth chart shows the power of compound interest...",
            hook_text_integration="This chart reveals the truth!",
            visual_consistency_score=0.85,
            keyword_overlap=["growth", "chart", "financial"]
        )
        
        assert alignment.thumbnail_concept == "growth chart"
        assert alignment.visual_consistency_score == 0.85
        assert len(alignment.keyword_overlap) == 3
        
        # Test serialization
        data = alignment.to_dict()
        assert data['thumbnail_concept'] == "growth chart"
        assert data['visual_consistency_score'] == 0.85
        assert data['keyword_overlap'] == ["growth", "chart", "financial"]


class TestMetadataValidationResult:
    """Test MetadataValidationResult data class."""
    
    def test_validation_result_creation(self):
        """Test MetadataValidationResult creation and serialization."""
        result = MetadataValidationResult(
            is_complete=True,
            missing_components=[],
            quality_score=0.85,
            synchronization_score=0.90,
            ai_director_alignment=0.80,
            validation_errors=[],
            recommendations=["Add more tags for better discoverability"]
        )
        
        assert result.is_complete is True
        assert result.quality_score == 0.85
        assert len(result.recommendations) == 1
        
        # Test serialization
        data = result.to_dict()
        assert data['is_complete'] is True
        assert data['quality_score'] == 0.85
        assert len(data['recommendations']) == 1


class TestIntegratedMetadataPackage:
    """Test IntegratedMetadataPackage data class."""
    
    def test_integrated_package_creation(self):
        """Test IntegratedMetadataPackage creation and serialization."""
        primary_metadata = MetadataVariation(
            title="Test Title",
            description="Test description",
            tags=["test"],
            variation_id="primary",
            strategy="integrated"
        )
        
        validation_result = MetadataValidationResult(
            is_complete=True,
            missing_components=[],
            quality_score=0.85,
            synchronization_score=0.90,
            ai_director_alignment=0.80,
            validation_errors=[],
            recommendations=[]
        )
        
        package = IntegratedMetadataPackage(
            primary_metadata=primary_metadata,
            alternative_variations=[],
            thumbnail_alignments=[],
            ai_director_integration={},
            content_synchronization={},
            validation_result=validation_result,
            publish_readiness_score=0.88,
            generation_timestamp=datetime.now()
        )
        
        assert package.publish_readiness_score == 0.88
        assert package.validation_result.is_complete is True
        
        # Test serialization
        data = package.to_dict()
        assert data['publish_readiness_score'] == 0.88
        assert data['validation_result']['is_complete'] is True


class TestMetadataPackageIntegrator:
    """Test MetadataPackageIntegrator class."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        cache_manager = Mock(spec=CacheManager)
        cache_manager.get.return_value = None
        cache_manager.put.return_value = None
        return cache_manager
    
    @pytest.fixture
    def mock_metadata_generator(self):
        """Create mock metadata generator."""
        generator = Mock(spec=MetadataGenerator)
        generator.generate_metadata_package = AsyncMock()
        return generator
    
    @pytest.fixture
    def integrator(self, mock_cache_manager, mock_metadata_generator):
        """Create MetadataPackageIntegrator instance."""
        return MetadataPackageIntegrator(mock_cache_manager, mock_metadata_generator)
    
    @pytest.fixture
    def sample_trending_keywords(self):
        """Create sample trending keywords."""
        return TrendingKeywords(
            primary_keywords=["financial education", "investing", "compound interest"],
            long_tail_keywords=["how to invest money", "financial planning basics"],
            trending_hashtags=["#finance", "#investing"],
            seasonal_keywords=["2024"],
            competitor_keywords=["wealth building"],
            search_volume_data={"financial education": 5000},
            research_timestamp=datetime.now(),
            keyword_difficulty={"financial education": 0.6},
            keyword_confidence={"financial education": 0.9},
            trending_topics=["AI investing"],
            research_quality_score=0.85
        )
    
    @pytest.fixture
    def sample_content_context(self, sample_trending_keywords):
        """Create sample ContentContext with comprehensive data."""
        # Create emotional markers
        emotional_markers = [
            EmotionalPeak(30.0, "excitement", 0.8, 0.9, "Discussing compound interest"),
            EmotionalPeak(120.0, "curiosity", 0.7, 0.85, "Explaining investment strategies")
        ]
        
        # Create visual highlights
        visual_highlights = [
            VisualHighlight(
                timestamp=45.0,
                description="Growth chart showing investment returns",
                faces=[FaceDetection([100, 100, 200, 200], 0.95)],
                visual_elements=["chart", "graph", "numbers"],
                thumbnail_potential=0.9
            ),
            VisualHighlight(
                timestamp=90.0,
                description="Calculator with money symbols",
                faces=[FaceDetection([150, 120, 250, 220], 0.92)],
                visual_elements=["calculator", "money", "symbols"],
                thumbnail_potential=0.85
            )
        ]
        
        # Create audio analysis
        audio_segments = [
            AudioSegment(
                text="Welcome to financial education",
                start=0.0,
                end=3.0,
                confidence=0.95,
                financial_concepts=["financial education"]
            )
        ]
        
        audio_analysis = AudioAnalysisResult(
            transcript_text="Welcome to financial education. Today we'll learn about investing.",
            segments=audio_segments,
            overall_confidence=0.92,
            language="en",
            processing_time=2.5,
            model_used="whisper-large",
            financial_concepts=["financial education", "investing", "compound interest"],
            complexity_level="beginner"
        )
        
        context = ContentContext(
            project_id="test_integration",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences(),
            key_concepts=["financial education", "investing", "compound interest"],
            content_themes=["education", "finance", "beginner-friendly"],
            emotional_markers=emotional_markers,
            visual_highlights=visual_highlights,
            trending_keywords=sample_trending_keywords
        )
        
        context.set_audio_analysis(audio_analysis)
        
        # Add thumbnail concepts
        context.thumbnail_concepts = [
            {"concept": "growth chart", "confidence": 0.9},
            {"concept": "money calculator", "confidence": 0.85}
        ]
        
        # Add AI Director plan
        context.ai_director_plan = {
            'creative_strategy': 'educational_engaging',
            'content_strategy': {'focus': 'beginner_friendly'},
            'thumbnail_concepts': ['growth chart', 'calculator']
        }
        
        return context
    
    @pytest.fixture
    def sample_metadata_variations(self):
        """Create sample metadata variations."""
        return [
            {
                'variations': [
                    {
                        'title': 'Financial Education: Complete Guide',
                        'description': 'Learn about financial education...',
                        'tags': ['finance', 'education', 'investing'],
                        'variation_id': 'var_1_seo',
                        'strategy': 'seo_focused',
                        'confidence_score': 0.85,
                        'estimated_ctr': 0.12,
                        'seo_score': 0.78
                    },
                    {
                        'title': '5 Financial Secrets That Will Shock You!',
                        'description': 'Discover shocking financial secrets...',
                        'tags': ['finance', 'secrets', 'money'],
                        'variation_id': 'var_2_emotional',
                        'strategy': 'emotional',
                        'confidence_score': 0.80,
                        'estimated_ctr': 0.18,
                        'seo_score': 0.65
                    }
                ],
                'recommended_variation': 'var_1_seo',
                'generation_timestamp': datetime.now().isoformat()
            }
        ]
    
    def test_integrator_initialization(self, integrator):
        """Test MetadataPackageIntegrator initialization."""
        assert integrator.cache_manager is not None
        assert integrator.metadata_generator is not None
        assert 'content_alignment' in integrator.scoring_weights
        assert 'required_components' in integrator.validation_requirements
        assert integrator.scoring_weights['content_alignment'] == 0.25
    
    @pytest.mark.asyncio
    async def test_create_integrated_package_success(self, integrator, sample_content_context, sample_metadata_variations):
        """Test successful integrated package creation."""
        # Setup metadata variations
        sample_content_context.metadata_variations = sample_metadata_variations
        
        result_context = await integrator.create_integrated_package(sample_content_context)
        
        assert result_context.metadata_variations is not None
        assert len(result_context.metadata_variations) == 1
        
        # Verify integrated package structure
        package_data = result_context.metadata_variations[0]
        assert 'primary_metadata' in package_data
        assert 'alternative_variations' in package_data
        assert 'thumbnail_alignments' in package_data
        assert 'validation_result' in package_data
        assert 'publish_readiness_score' in package_data
        
        # Verify publish readiness score
        assert 0.0 <= package_data['publish_readiness_score'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_create_integrated_package_cached(self, integrator, sample_content_context):
        """Test integrated package creation with cached results."""
        # Setup cache to return existing package
        cached_package = {
            'primary_metadata': {'title': 'Cached Title'},
            'publish_readiness_score': 0.9
        }
        integrator.cache_manager.get.return_value = cached_package
        
        result_context = await integrator.create_integrated_package(sample_content_context)
        
        assert result_context.metadata_variations == [cached_package]
        integrator.cache_manager.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_integrated_package_generates_base_metadata(self, integrator, sample_content_context, sample_metadata_variations):
        """Test that base metadata is generated when not present."""
        # Remove existing metadata variations to test generation
        sample_content_context.metadata_variations = []
        
        # Setup metadata generator to return context with sample variations
        async def mock_generate_metadata(context):
            context.metadata_variations = sample_metadata_variations
            return context
        
        integrator.metadata_generator.generate_metadata_package = AsyncMock(side_effect=mock_generate_metadata)
        
        result_context = await integrator.create_integrated_package(sample_content_context)
        
        # Should call metadata generator when no metadata exists initially
        integrator.metadata_generator.generate_metadata_package.assert_called_once()
        assert result_context.metadata_variations is not None
    
    @pytest.mark.asyncio
    async def test_create_integrated_package_missing_prerequisites(self, integrator):
        """Test error handling when prerequisites are missing."""
        # Create context without required data
        invalid_context = ContentContext(
            project_id="invalid",
            video_files=[],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        with pytest.raises(ContentContextError) as exc_info:
            await integrator.create_integrated_package(invalid_context)
        
        assert "Missing required components" in str(exc_info.value)
    
    def test_validate_prerequisites_success(self, integrator, sample_content_context):
        """Test successful prerequisite validation."""
        # Should not raise exception
        integrator._validate_prerequisites(sample_content_context)
    
    def test_validate_prerequisites_missing_data(self, integrator):
        """Test prerequisite validation with missing data."""
        context = ContentContext(
            project_id="test",
            video_files=[],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        with pytest.raises(ContentContextError) as exc_info:
            integrator._validate_prerequisites(context)
        
        assert "trending_keywords" in str(exc_info.value)
    
    def test_extract_ai_director_decisions_with_plan(self, integrator, sample_content_context):
        """Test AI Director decision extraction with plan available."""
        decisions = integrator._extract_ai_director_decisions(sample_content_context)
        
        assert decisions['has_director_plan'] is True
        assert decisions['creative_strategy'] == 'educational_engaging'
        assert len(decisions['target_emotions']) > 0
        assert len(decisions['key_messages']) > 0
    
    def test_extract_ai_director_decisions_without_plan(self, integrator, sample_content_context):
        """Test AI Director decision extraction without plan."""
        # Remove AI Director plan
        delattr(sample_content_context, 'ai_director_plan')
        
        decisions = integrator._extract_ai_director_decisions(sample_content_context)
        
        assert decisions['has_director_plan'] is False
        assert decisions['creative_strategy'] == 'educational_focused'
        assert len(decisions['target_emotions']) > 0  # From emotional markers
        assert len(decisions['key_messages']) > 0  # From content themes
    
    @pytest.mark.asyncio
    async def test_create_thumbnail_alignments(self, integrator, sample_content_context):
        """Test thumbnail alignment creation."""
        alignments = await integrator._create_thumbnail_alignments(sample_content_context)
        
        assert len(alignments) > 0
        
        for alignment in alignments:
            assert isinstance(alignment, ThumbnailMetadataAlignment)
            assert alignment.thumbnail_concept is not None
            assert alignment.aligned_title is not None
            assert alignment.aligned_description is not None
            assert 0.0 <= alignment.visual_consistency_score <= 1.0
            assert isinstance(alignment.keyword_overlap, list)
    
    @pytest.mark.asyncio
    async def test_create_thumbnail_alignments_fallback(self, integrator, sample_content_context):
        """Test thumbnail alignment creation with fallback to visual highlights."""
        # Remove thumbnail concepts
        sample_content_context.thumbnail_concepts = []
        delattr(sample_content_context, 'ai_director_plan')
        
        alignments = await integrator._create_thumbnail_alignments(sample_content_context)
        
        # Should use visual highlights as fallback
        assert len(alignments) > 0
        assert any("chart" in align.thumbnail_concept.lower() for align in alignments)
    
    @pytest.mark.asyncio
    async def test_generate_aligned_title(self, integrator, sample_content_context):
        """Test aligned title generation."""
        concept = "growth chart"
        
        title = await integrator._generate_aligned_title(concept, sample_content_context)
        
        assert isinstance(title, str)
        assert len(title) > 0
        assert len(title) <= 60  # Should be optimized for length
        assert "chart" in title.lower() or "growth" in title.lower()
    
    @pytest.mark.asyncio
    async def test_generate_aligned_description(self, integrator, sample_content_context):
        """Test aligned description generation."""
        concept = "growth chart"
        
        description = await integrator._generate_aligned_description(concept, sample_content_context)
        
        assert isinstance(description, str)
        assert len(description) > 100  # Should be substantial
        assert "chart" in description.lower()
        assert ":" in description  # Should contain timestamps
        assert "#" in description  # Should contain hashtags
    
    def test_generate_hook_text(self, integrator, sample_content_context):
        """Test hook text generation."""
        concept = "growth chart"
        
        hook_text = integrator._generate_hook_text(concept, sample_content_context)
        
        assert isinstance(hook_text, str)
        assert len(hook_text) > 0
        assert hook_text.endswith("!")  # Should be engaging
        assert "chart" in hook_text.lower() or "this" in hook_text.lower()
    
    def test_calculate_visual_consistency(self, integrator, sample_content_context):
        """Test visual consistency calculation."""
        concept = "growth chart"
        
        score = integrator._calculate_visual_consistency(concept, sample_content_context)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should score well since visual highlights contain "chart"
        assert score > 0.0
    
    def test_find_keyword_overlap(self, integrator, sample_content_context):
        """Test keyword overlap finding."""
        concept = "financial education chart"
        
        overlap = integrator._find_keyword_overlap(concept, sample_content_context)
        
        assert isinstance(overlap, list)
        assert len(overlap) <= 5  # Should be limited
        # Should find overlap with "financial education"
        assert any("financial" in keyword.lower() for keyword in overlap)
    
    def test_synchronize_with_content(self, integrator, sample_content_context):
        """Test content synchronization."""
        synchronization = integrator._synchronize_with_content(sample_content_context)
        
        assert isinstance(synchronization, dict)
        assert 'content_alignment_score' in synchronization
        assert 'emotional_alignment' in synchronization
        assert 'visual_alignment' in synchronization
        assert 'audio_alignment' in synchronization
        
        # Should have good alignment scores
        assert 0.0 <= synchronization['content_alignment_score'] <= 1.0
        assert synchronization['emotional_alignment']['peak_count'] == 2
        assert synchronization['visual_alignment']['highlight_count'] == 2
    
    def test_integrate_ai_director_decisions(self, integrator, sample_content_context):
        """Test AI Director decision integration."""
        ai_decisions = integrator._extract_ai_director_decisions(sample_content_context)
        
        integration = integrator._integrate_ai_director_decisions(sample_content_context, ai_decisions)
        
        assert isinstance(integration, dict)
        assert 'director_influence_score' in integration
        assert 'creative_alignment' in integration
        assert 'strategic_alignment' in integration
        
        # Should have high influence score with AI Director plan
        assert integration['director_influence_score'] > 0.7
        assert integration['creative_alignment']['strategy'] == 'educational_engaging'
    
    @pytest.mark.asyncio
    async def test_select_primary_metadata(self, integrator, sample_content_context, sample_metadata_variations):
        """Test primary metadata selection."""
        sample_content_context.metadata_variations = sample_metadata_variations
        
        # Create sample alignments
        alignments = [
            ThumbnailMetadataAlignment(
                thumbnail_concept="test concept",
                aligned_title="Test Aligned Title",
                aligned_description="Test aligned description",
                hook_text_integration="Test hook",
                visual_consistency_score=0.8,
                keyword_overlap=["test"]
            )
        ]
        
        ai_integration = {'director_influence_score': 0.8}
        
        primary = await integrator._select_primary_metadata(
            sample_content_context, alignments, ai_integration
        )
        
        assert isinstance(primary, MetadataVariation)
        assert primary.title is not None
        assert primary.description is not None
        # Should use aligned title from thumbnail alignment
        assert primary.title == "Test Aligned Title"
    
    def test_calculate_integration_score(self, integrator, sample_content_context):
        """Test integration score calculation."""
        variation = MetadataVariation(
            title="Test Title",
            description="Test description",
            tags=["test"],
            variation_id="test_var",
            strategy="test",
            confidence_score=0.8,
            seo_score=0.75,
            estimated_ctr=0.12
        )
        
        alignments = [
            ThumbnailMetadataAlignment(
                thumbnail_concept="test",
                aligned_title="Test",
                aligned_description="Test",
                hook_text_integration="Test",
                visual_consistency_score=0.85,
                keyword_overlap=[]
            )
        ]
        
        ai_integration = {'director_influence_score': 0.8}
        
        score = integrator._calculate_integration_score(
            variation, alignments, ai_integration, sample_content_context
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should score well with good inputs
    
    @pytest.mark.asyncio
    async def test_create_alternative_variations(self, integrator, sample_content_context, sample_metadata_variations):
        """Test alternative variation creation."""
        sample_content_context.metadata_variations = sample_metadata_variations
        
        primary = MetadataVariation(
            title="Primary Title",
            description="Primary description",
            tags=["primary"],
            variation_id="var_1_seo",
            strategy="primary"
        )
        
        alignments = []
        
        alternatives = await integrator._create_alternative_variations(
            sample_content_context, primary, alignments
        )
        
        assert isinstance(alternatives, list)
        assert len(alternatives) <= 3  # Should be limited
        
        # Should not include the primary variation
        for alt in alternatives:
            assert alt.variation_id != primary.variation_id
    
    def test_validate_metadata_package(self, integrator, sample_content_context):
        """Test metadata package validation."""
        primary_metadata = MetadataVariation(
            title="Complete Financial Education Guide for Beginners",
            description="Learn everything about financial education in this comprehensive guide. " * 3,  # Long enough
            tags=["finance", "education", "investing", "money", "wealth", "tutorial"],  # Enough tags
            variation_id="primary",
            strategy="integrated",
            confidence_score=0.85
        )
        
        alternatives = []
        
        alignments = [
            ThumbnailMetadataAlignment(
                thumbnail_concept="test",
                aligned_title="Test",
                aligned_description="Test",
                hook_text_integration="Test",
                visual_consistency_score=0.8,
                keyword_overlap=[]
            )
        ]
        
        result = integrator._validate_metadata_package(
            sample_content_context, primary_metadata, alternatives, alignments
        )
        
        assert isinstance(result, MetadataValidationResult)
        assert result.is_complete is True  # Should pass validation
        assert len(result.missing_components) == 0
        assert result.quality_score > 0.7
        assert result.synchronization_score > 0.0
    
    def test_validate_metadata_package_failures(self, integrator, sample_content_context):
        """Test metadata package validation with failures."""
        # Create inadequate metadata
        primary_metadata = MetadataVariation(
            title="Short",  # Too short
            description="Short desc",  # Too short
            tags=["one"],  # Too few tags
            variation_id="primary",
            strategy="integrated",
            confidence_score=0.5  # Low quality
        )
        
        result = integrator._validate_metadata_package(
            sample_content_context, primary_metadata, [], []
        )
        
        assert result.is_complete is False
        assert len(result.missing_components) > 0
        assert len(result.validation_errors) > 0
        assert len(result.recommendations) > 0
        
        # Check specific errors
        assert any("title" in error.lower() for error in result.validation_errors)
        assert any("description" in error.lower() for error in result.validation_errors)
        assert any("tags" in error.lower() for error in result.validation_errors)
    
    def test_calculate_publish_readiness(self, integrator):
        """Test publish readiness calculation."""
        primary_metadata = MetadataVariation(
            title="Test Title",
            description="Test description",
            tags=["test"],
            variation_id="test",
            strategy="test",
            seo_score=0.8,
            estimated_ctr=0.15
        )
        
        validation_result = MetadataValidationResult(
            is_complete=True,
            missing_components=[],
            quality_score=0.85,
            synchronization_score=0.90,
            ai_director_alignment=0.80,
            validation_errors=[],
            recommendations=[]
        )
        
        ai_integration = {'director_influence_score': 0.8}
        
        score = integrator._calculate_publish_readiness(
            primary_metadata, validation_result, ai_integration
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should score well with good inputs
    
    def test_calculate_publish_readiness_with_penalties(self, integrator):
        """Test publish readiness calculation with penalties."""
        primary_metadata = MetadataVariation(
            title="Test Title",
            description="Test description",
            tags=["test"],
            variation_id="test",
            strategy="test",
            seo_score=0.6,
            estimated_ctr=0.1
        )
        
        validation_result = MetadataValidationResult(
            is_complete=False,
            missing_components=["valid_title", "adequate_description"],  # Penalties
            quality_score=0.6,
            synchronization_score=0.5,
            ai_director_alignment=0.6,
            validation_errors=["Title too short", "Description inadequate"],
            recommendations=["Improve title", "Expand description"]
        )
        
        ai_integration = {'director_influence_score': 0.6}
        
        score = integrator._calculate_publish_readiness(
            primary_metadata, validation_result, ai_integration
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should have lower score due to penalties
        assert score < 0.7
    
    @pytest.mark.asyncio
    async def test_error_handling(self, integrator):
        """Test error handling in metadata integration."""
        # Test with completely invalid context
        invalid_context = ContentContext(
            project_id="invalid",
            video_files=[],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        with pytest.raises(ContentContextError):
            await integrator.create_integrated_package(invalid_context)
    
    def test_integration_with_content_context(self, integrator, sample_content_context):
        """Test integration with ContentContext data structures."""
        # Test that all ContentContext data is properly utilized
        ai_decisions = integrator._extract_ai_director_decisions(sample_content_context)
        
        # Should extract data from all relevant fields
        assert ai_decisions['has_director_plan'] is True
        assert len(ai_decisions['target_emotions']) > 0  # From emotional markers
        assert len(ai_decisions['key_messages']) > 0  # From content themes
        
        # Test content synchronization
        sync_data = integrator._synchronize_with_content(sample_content_context)
        
        # Should use all available content data
        assert sync_data['emotional_alignment']['peak_count'] == len(sample_content_context.emotional_markers)
        assert sync_data['visual_alignment']['highlight_count'] == len(sample_content_context.visual_highlights)
        assert sync_data['audio_alignment']['transcript_quality'] == sample_content_context.audio_analysis.overall_confidence


if __name__ == "__main__":
    pytest.main([__file__])