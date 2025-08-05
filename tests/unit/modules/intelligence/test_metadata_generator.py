"""
Unit tests for MetadataGenerator module.

Tests comprehensive metadata generation including titles, descriptions, tags,
A/B testing variations, and SEO optimization.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

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


class TestMetadataVariation:
    """Test MetadataVariation data class."""
    
    def test_metadata_variation_creation(self):
        """Test MetadataVariation creation and serialization."""
        variation = MetadataVariation(
            title="Test Title",
            description="Test description",
            tags=["test", "video", "tutorial"],
            variation_id="var_1_test",
            strategy="seo_focused",
            confidence_score=0.85,
            estimated_ctr=0.12,
            seo_score=0.78
        )
        
        assert variation.title == "Test Title"
        assert variation.strategy == "seo_focused"
        assert variation.confidence_score == 0.85
        
        # Test serialization
        data = variation.to_dict()
        assert data['title'] == "Test Title"
        assert data['tags'] == ["test", "video", "tutorial"]
        
        # Test deserialization
        restored = MetadataVariation.from_dict(data)
        assert restored.title == variation.title
        assert restored.confidence_score == variation.confidence_score


class TestMetadataPackage:
    """Test MetadataPackage data class."""
    
    def test_metadata_package_creation(self):
        """Test MetadataPackage creation and serialization."""
        variation = MetadataVariation(
            title="Test Title",
            description="Test description",
            tags=["test"],
            variation_id="var_1",
            strategy="test"
        )
        
        package = MetadataPackage(
            variations=[variation],
            recommended_variation="var_1",
            generation_timestamp=datetime.now(),
            content_analysis={"test": "data"},
            seo_insights={"score": 0.8},
            performance_predictions={"ctr": 0.1}
        )
        
        assert len(package.variations) == 1
        assert package.recommended_variation == "var_1"
        
        # Test serialization
        data = package.to_dict()
        assert len(data['variations']) == 1
        assert data['recommended_variation'] == "var_1"
        
        # Test deserialization
        restored = MetadataPackage.from_dict(data)
        assert len(restored.variations) == 1
        assert restored.recommended_variation == package.recommended_variation


class TestMetadataGenerator:
    """Test MetadataGenerator class."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager."""
        cache_manager = Mock(spec=CacheManager)
        cache_manager.get.return_value = None
        cache_manager.put.return_value = None
        return cache_manager
    
    @pytest.fixture
    def metadata_generator(self, mock_cache_manager):
        """Create MetadataGenerator instance."""
        return MetadataGenerator(mock_cache_manager)
    
    @pytest.fixture
    def sample_trending_keywords(self):
        """Create sample trending keywords."""
        return TrendingKeywords(
            primary_keywords=["financial education", "investing", "money management"],
            long_tail_keywords=["how to invest money", "financial planning for beginners"],
            trending_hashtags=["#finance", "#investing"],
            seasonal_keywords=["2024", "new year"],
            competitor_keywords=["wealth building", "passive income"],
            search_volume_data={"financial education": 5000, "investing": 8000},
            research_timestamp=datetime.now(),
            keyword_difficulty={"financial education": 0.6, "investing": 0.8},
            keyword_confidence={"financial education": 0.9, "investing": 0.7},
            trending_topics=["AI investing", "crypto education"],
            research_quality_score=0.85
        )
    
    @pytest.fixture
    def sample_content_context(self, sample_trending_keywords):
        """Create sample ContentContext."""
        # Create emotional markers
        emotional_markers = [
            EmotionalPeak(30.0, "excitement", 0.8, 0.9, "Discussing compound interest"),
            EmotionalPeak(120.0, "curiosity", 0.7, 0.85, "Explaining investment strategies")
        ]
        
        # Create visual highlights
        visual_highlights = [
            VisualHighlight(
                timestamp=45.0,
                description="Chart showing investment growth",
                faces=[FaceDetection([100, 100, 200, 200], 0.95)],
                visual_elements=["chart", "graph", "numbers"],
                thumbnail_potential=0.9
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
            project_id="test_project",
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
        
        return context
    
    def test_metadata_generator_initialization(self, metadata_generator):
        """Test MetadataGenerator initialization."""
        assert metadata_generator.cache_manager is not None
        assert 'emotional' in metadata_generator.title_strategies
        assert 'educational' in metadata_generator.description_templates
        assert 'broad' in metadata_generator.tag_categories
        assert metadata_generator.seo_patterns['title_length'] == (50, 60)
    
    @pytest.mark.asyncio
    async def test_generate_metadata_package_success(self, metadata_generator, sample_content_context):
        """Test successful metadata package generation."""
        # Mock the memory tracking function by patching the import
        with patch('ai_video_editor.modules.intelligence.metadata_generator.logger') as mock_logger:
            result_context = await metadata_generator.generate_metadata_package(sample_content_context)
            
            assert result_context.metadata_variations is not None
            assert len(result_context.metadata_variations) == 1
            
            # Verify metadata package structure
            package_data = result_context.metadata_variations[0]
            assert 'variations' in package_data
            assert 'recommended_variation' in package_data
            assert 'generation_timestamp' in package_data
            
            # Verify variations were generated
            variations = package_data['variations']
            assert len(variations) >= 3  # Should generate multiple strategies
            
            # Verify each variation has required fields
            for variation in variations:
                assert 'title' in variation
                assert 'description' in variation
                assert 'tags' in variation
                assert 'strategy' in variation
                assert 'confidence_score' in variation
    
    @pytest.mark.asyncio
    async def test_generate_metadata_package_cached(self, metadata_generator, sample_content_context):
        """Test metadata generation with cached results."""
        # Setup cache to return existing package
        cached_package = {
            'variations': [{'title': 'Cached Title', 'description': 'Cached desc', 'tags': ['cached']}],
            'recommended_variation': 'cached_var'
        }
        metadata_generator.cache_manager.get.return_value = cached_package
        
        result_context = await metadata_generator.generate_metadata_package(sample_content_context)
        
        assert result_context.metadata_variations == [cached_package]
        metadata_generator.cache_manager.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_metadata_package_no_trending_keywords(self, metadata_generator):
        """Test metadata generation without trending keywords."""
        context = ContentContext(
            project_id="test_project",
            video_files=["test_video.mp4"],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        with pytest.raises(ContentContextError) as exc_info:
            await metadata_generator.generate_metadata_package(context)
        
        assert "Trending keywords required" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_title_strategies(self, metadata_generator, sample_content_context):
        """Test title generation for different strategies."""
        content_analysis = metadata_generator._analyze_content_for_metadata(sample_content_context)
        
        strategies = ['emotional', 'seo_focused', 'curiosity_driven', 'educational', 'listicle']
        
        for strategy in strategies:
            title = await metadata_generator._generate_title(
                strategy, content_analysis, sample_content_context.trending_keywords
            )
            
            assert isinstance(title, str)
            assert len(title) > 0
            assert len(title) <= 60  # Should be optimized for length
            
            # Strategy-specific checks
            if strategy == 'listicle':
                # Should contain numbers
                import re
                assert re.search(r'\d+', title)
            
            if strategy == 'seo_focused':
                # Should contain primary keyword
                primary_keyword = sample_content_context.trending_keywords.primary_keywords[0]
                assert primary_keyword.lower() in title.lower() or "financial" in title.lower()
    
    @pytest.mark.asyncio
    async def test_generate_description_templates(self, metadata_generator, sample_content_context):
        """Test description generation for different templates."""
        content_analysis = metadata_generator._analyze_content_for_metadata(sample_content_context)
        
        # Test educational template
        description = await metadata_generator._generate_description(
            'educational', content_analysis, sample_content_context
        )
        
        assert isinstance(description, str)
        assert len(description) > 125  # Should meet minimum SEO length
        assert "📚" in description  # Should contain educational emoji
        assert "Timestamps:" in description
        assert "#education" in description.lower()
        
        # Verify timestamps are included
        assert "00:" in description or "01:" in description
    
    @pytest.mark.asyncio
    async def test_generate_tags_comprehensive(self, metadata_generator, sample_content_context):
        """Test comprehensive tag generation."""
        content_analysis = metadata_generator._analyze_content_for_metadata(sample_content_context)
        
        tags = await metadata_generator._generate_tags(
            content_analysis, sample_content_context.trending_keywords
        )
        
        assert isinstance(tags, list)
        assert 10 <= len(tags) <= 15  # Should be within optimal range
        
        # Should include primary keywords (converted to hashtag format)
        primary_keywords = sample_content_context.trending_keywords.primary_keywords
        tag_text = ' '.join(tags).lower()
        # Check if any primary keyword (converted to hashtag format) is in tags
        found_keyword = False
        for keyword in primary_keywords[:3]:
            hashtag_format = keyword.lower().replace(' ', '').replace('-', '')
            if hashtag_format in tag_text:
                found_keyword = True
                break
        assert found_keyword
        
        # Should include broad category tags
        broad_tags = ['tutorial', 'guide', 'tips', 'education']
        assert any(tag in tags for tag in broad_tags)
        
        # All tags should be valid (no spaces, reasonable length)
        for tag in tags:
            assert ' ' not in tag
            assert len(tag) <= 20
            assert len(tag) >= 2
    
    def test_analyze_content_for_metadata(self, metadata_generator, sample_content_context):
        """Test content analysis for metadata generation."""
        analysis = metadata_generator._analyze_content_for_metadata(sample_content_context)
        
        assert isinstance(analysis, dict)
        assert 'primary_concept' in analysis
        assert 'target_audience' in analysis
        assert 'content_type' in analysis
        assert 'key_themes' in analysis
        
        # Should extract primary concept from key concepts
        assert analysis['primary_concept'] in sample_content_context.key_concepts
        
        # Should determine audience from complexity
        assert analysis['target_audience'] == 'Beginners'  # Based on audio analysis complexity
        
        # Should include emotional analysis
        assert 'dominant_emotion' in analysis
    
    def test_calculate_seo_score(self, metadata_generator, sample_content_context):
        """Test SEO score calculation."""
        variation = MetadataVariation(
            title="Financial Education: Complete 2024 Guide",  # Good length, contains keyword
            description="Learn about financial education in this comprehensive guide. " * 3,  # Good length
            tags=["financial", "education", "investing", "tutorial", "guide", "money", "finance", "tips", "beginner", "2024"],  # Good count
            variation_id="test_var",
            strategy="seo_focused"
        )
        
        seo_score = metadata_generator._calculate_seo_score(
            variation, sample_content_context.trending_keywords
        )
        
        assert isinstance(seo_score, float)
        assert 0.0 <= seo_score <= 1.0
        assert seo_score > 0.5  # Should score well with optimized content
    
    def test_estimate_click_through_rate(self, metadata_generator, sample_content_context):
        """Test CTR estimation."""
        content_analysis = metadata_generator._analyze_content_for_metadata(sample_content_context)
        
        # Test emotional title
        emotional_variation = MetadataVariation(
            title="5 Shocking Financial Secrets That Will Amaze You!",
            description="Test description",
            tags=["test"],
            variation_id="emotional_var",
            strategy="emotional"
        )
        
        ctr_score = metadata_generator._estimate_click_through_rate(
            emotional_variation, content_analysis
        )
        
        assert isinstance(ctr_score, float)
        assert 0.0 <= ctr_score <= 1.0
        assert ctr_score > 0.3  # Should score well with emotional words and numbers
    
    def test_optimize_title_length(self, metadata_generator):
        """Test title length optimization."""
        # Test long title
        long_title = "This is a very long title that exceeds the optimal length for YouTube SEO and should be truncated"
        optimized = metadata_generator._optimize_title_length(long_title)
        
        assert len(optimized) <= 60
        assert len(optimized) >= 50 or "..." in optimized
        
        # Test short title
        short_title = "Short Title"
        optimized_short = metadata_generator._optimize_title_length(short_title)
        assert optimized_short == short_title
    
    def test_optimize_description_seo(self, metadata_generator, sample_trending_keywords):
        """Test description SEO optimization."""
        description = "This is a test description without the primary keyword in the first part."
        
        optimized = metadata_generator._optimize_description_seo(description, sample_trending_keywords)
        
        # Should prepend keyword mention
        primary_keyword = sample_trending_keywords.primary_keywords[0]
        assert primary_keyword.lower() in optimized[:125].lower()
    
    def test_generate_key_points(self, metadata_generator, sample_content_context):
        """Test key points generation."""
        content_analysis = metadata_generator._analyze_content_for_metadata(sample_content_context)
        
        key_points = metadata_generator._generate_key_points(content_analysis, sample_content_context)
        
        assert isinstance(key_points, str)
        assert "•" in key_points  # Should use bullet points
        assert len(key_points.split('\n')) >= 3  # Should have at least 3 points
        
        # Should include content themes
        for theme in content_analysis['key_themes'][:2]:
            assert theme.lower() in key_points.lower()
    
    def test_generate_timestamps(self, metadata_generator, sample_content_context):
        """Test timestamp generation."""
        timestamps = metadata_generator._generate_timestamps(sample_content_context)
        
        assert isinstance(timestamps, str)
        assert ":" in timestamps  # Should contain time format
        assert len(timestamps.split('\n')) >= 2  # Should have multiple timestamps
        
        # Should use emotional markers for timestamps
        for marker in sample_content_context.emotional_markers:
            minutes = int(marker.timestamp // 60)
            seconds = int(marker.timestamp % 60)
            expected_time = f"{minutes:02d}:{seconds:02d}"
            assert expected_time in timestamps
    
    def test_generate_seo_insights(self, metadata_generator, sample_content_context):
        """Test SEO insights generation."""
        variations = [
            MetadataVariation(
                title="Test Title 1",
                description="Test description",
                tags=["financial", "education"],
                variation_id="var_1",
                strategy="test",
                seo_score=0.8
            ),
            MetadataVariation(
                title="Test Title 2",
                description="Test description",
                tags=["investing", "money"],
                variation_id="var_2",
                strategy="test",
                seo_score=0.6
            )
        ]
        
        insights = metadata_generator._generate_seo_insights(
            variations, sample_content_context.trending_keywords
        )
        
        assert isinstance(insights, dict)
        assert 'keyword_coverage' in insights
        assert 'optimization_suggestions' in insights
        assert 'trending_alignment' in insights
        
        # Should calculate trending alignment
        assert isinstance(insights['trending_alignment'], float)
        assert 0.0 <= insights['trending_alignment'] <= 1.0
    
    def test_generate_performance_predictions(self, metadata_generator):
        """Test performance predictions generation."""
        variations = [
            MetadataVariation(
                title="High CTR Title",
                description="Test",
                tags=["test"],
                variation_id="high_ctr",
                strategy="emotional",
                estimated_ctr=0.8,
                seo_score=0.6,
                confidence_score=0.7
            ),
            MetadataVariation(
                title="High SEO Title",
                description="Test",
                tags=["test"],
                variation_id="high_seo",
                strategy="seo_focused",
                estimated_ctr=0.5,
                seo_score=0.9,
                confidence_score=0.7
            )
        ]
        
        predictions = metadata_generator._generate_performance_predictions(variations)
        
        assert isinstance(predictions, dict)
        assert 'best_ctr_variation' in predictions
        assert 'best_seo_variation' in predictions
        assert 'balanced_recommendation' in predictions
        assert 'expected_performance' in predictions
        
        # Should identify best variations correctly
        assert predictions['best_ctr_variation'] == 'high_ctr'
        assert predictions['best_seo_variation'] == 'high_seo'
    
    @pytest.mark.asyncio
    async def test_track_successful_patterns(self, metadata_generator, sample_content_context):
        """Test successful pattern tracking."""
        variations = [
            MetadataVariation(
                title="High Confidence Title",
                description="Test",
                tags=["effective", "tag"],
                variation_id="high_conf",
                strategy="emotional",
                confidence_score=0.8,
                seo_score=0.8,
                estimated_ctr=0.7
            )
        ]
        
        package = MetadataPackage(
            variations=variations,
            recommended_variation="high_conf",
            generation_timestamp=datetime.now(),
            content_analysis={},
            seo_insights={},
            performance_predictions={}
        )
        
        # Mock the memory function by patching the logger (since import will fail)
        with patch('ai_video_editor.modules.intelligence.metadata_generator.logger') as mock_logger:
            await metadata_generator._track_successful_patterns(package, sample_content_context)
            
            # Should log debug message about MCP Memory not being available
            mock_logger.debug.assert_called_with("MCP Memory not available, skipping pattern tracking")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, metadata_generator):
        """Test error handling in metadata generation."""
        # Test with invalid context
        invalid_context = ContentContext(
            project_id="invalid",
            video_files=[],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        with pytest.raises(ContentContextError):
            await metadata_generator.generate_metadata_package(invalid_context)
    
    def test_integration_with_content_context(self, metadata_generator, sample_content_context):
        """Test integration with ContentContext data structures."""
        # Test that all ContentContext data is properly utilized
        content_analysis = metadata_generator._analyze_content_for_metadata(sample_content_context)
        
        # Should extract data from all relevant fields
        assert content_analysis['primary_concept'] in sample_content_context.key_concepts
        assert content_analysis['key_themes'] == sample_content_context.content_themes[:5]
        assert content_analysis['emotional_peaks'] == len(sample_content_context.emotional_markers)
        assert content_analysis['visual_highlights'] == len(sample_content_context.visual_highlights)
        
        # Should use audio analysis data
        assert content_analysis['target_audience'] == 'Beginners'  # From complexity level
        assert 'financial_concepts' in content_analysis  # From audio analysis


if __name__ == "__main__":
    pytest.main([__file__])