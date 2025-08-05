"""
Unit tests for TrendAnalyzer class.

Tests keyword research automation, DDG Search integration, caching behavior,
and ContentContext integration.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from ai_video_editor.core.content_context import (
    ContentContext, ContentType, TrendingKeywords, UserPreferences,
    AudioAnalysisResult, AudioSegment
)
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.exceptions import ContentContextError, APIIntegrationError
from ai_video_editor.modules.intelligence.trend_analyzer import TrendAnalyzer


class TestTrendAnalyzer:
    """Test cases for TrendAnalyzer class."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock CacheManager for testing."""
        cache_manager = Mock(spec=CacheManager)
        cache_manager.get_keyword_research.return_value = None
        cache_manager.cache_keyword_research.return_value = None
        cache_manager.get.return_value = None
        cache_manager.put.return_value = None
        return cache_manager
    
    @pytest.fixture
    def trend_analyzer(self, mock_cache_manager):
        """TrendAnalyzer instance for testing."""
        return TrendAnalyzer(cache_manager=mock_cache_manager)
    
    @pytest.fixture
    def sample_content_context(self):
        """Sample ContentContext for testing."""
        audio_analysis = AudioAnalysisResult(
            transcript_text="Learn about financial education and investment basics",
            segments=[
                AudioSegment(
                    text="Learn about financial education",
                    start=0.0,
                    end=3.0,
                    confidence=0.95,
                    financial_concepts=["financial education"]
                ),
                AudioSegment(
                    text="and investment basics",
                    start=3.0,
                    end=6.0,
                    confidence=0.92,
                    financial_concepts=["investment basics"]
                )
            ],
            overall_confidence=0.94,
            language="en",
            processing_time=2.5,
            model_used="whisper-large",
            financial_concepts=["financial education", "investment basics"]
        )
        
        context = ContentContext(
            project_id="test-project-123",
            video_files=["test_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences(),
            key_concepts=["financial education", "investment", "money management"],
            content_themes=["education", "finance", "personal growth"],
            audio_analysis=audio_analysis
        )
        
        return context
    
    @pytest.fixture
    def mock_ddg_search_results(self):
        """Mock DDG search results."""
        return [
            {
                'title': 'Financial Education - Complete Guide 2024',
                'snippet': 'Learn about financial literacy, investment basics, and money management. '
                          'Discover the latest trends in personal finance education.',
                'url': 'https://example1.com/financial-education'
            },
            {
                'title': 'Investment Basics for Beginners',
                'snippet': 'Start your investment journey with these fundamental concepts. '
                          'Learn about stocks, bonds, and portfolio diversification.',
                'url': 'https://example2.com/investment-basics'
            },
            {
                'title': 'Money Management Tips and Tricks',
                'snippet': 'Effective strategies for managing your personal finances. '
                          'Budgeting, saving, and financial planning made simple.',
                'url': 'https://example3.com/money-management'
            },
            {
                'title': 'Personal Finance Trends 2024',
                'snippet': 'Latest trends in personal finance including digital banking, '
                          'cryptocurrency, and sustainable investing.',
                'url': 'https://example4.com/finance-trends'
            },
            {
                'title': 'Financial Literacy for Young Adults',
                'snippet': 'Essential financial skills every young adult should know. '
                          'Credit scores, student loans, and first-time investing.',
                'url': 'https://example5.com/financial-literacy'
            }
        ]
    
    @pytest.fixture
    def mock_trending_keywords(self):
        """Mock TrendingKeywords result."""
        return TrendingKeywords(
            primary_keywords=["financial education", "investment basics", "money management"],
            long_tail_keywords=["financial education for beginners", "investment basics guide"],
            trending_hashtags=["#FinancialLiteracy", "#InvestmentTips"],
            seasonal_keywords=["2024", "trends"],
            competitor_keywords=["personal finance", "budgeting"],
            search_volume_data={
                "financial education": 12000,
                "investment basics": 8500,
                "money management": 6200
            },
            research_timestamp=datetime.now(),
            keyword_difficulty={
                "financial education": 0.7,
                "investment basics": 0.5,
                "money management": 0.6
            },
            keyword_confidence={
                "financial education": 0.9,
                "investment basics": 0.8,
                "money management": 0.7
            },
            trending_topics=["digital banking", "sustainable investing"],
            competitor_analysis={
                'keywords': ['personal finance', 'budgeting', 'investing'],
                'domains': ['youtube.com', 'medium.com'],
                'analysis_timestamp': datetime.now().isoformat()
            },
            research_quality_score=0.85,
            cache_hit_rate=0.6
        )
    
    @pytest.mark.asyncio
    async def test_analyze_trends_success(self, trend_analyzer, sample_content_context, 
                                        mock_trending_keywords, mock_cache_manager):
        """Test successful trend analysis."""
        # Mock the research_keywords method
        with patch.object(trend_analyzer, 'research_keywords', 
                         new_callable=AsyncMock, return_value=mock_trending_keywords) as mock_research:
            
            result_context = await trend_analyzer.analyze_trends(sample_content_context)
            
            # Verify results
            assert result_context.trending_keywords is not None
            assert result_context.trending_keywords.primary_keywords == mock_trending_keywords.primary_keywords
            assert result_context.trending_keywords.research_quality_score == 0.85
            
            # Verify cache operations
            mock_cache_manager.get_keyword_research.assert_called_once()
            mock_cache_manager.cache_keyword_research.assert_called_once()
            
            # Verify processing metrics updated
            assert "trend_analyzer" in result_context.processing_metrics.module_processing_times
    
    @pytest.mark.asyncio
    async def test_analyze_trends_with_cache_hit(self, trend_analyzer, sample_content_context,
                                               mock_trending_keywords, mock_cache_manager):
        """Test trend analysis with cache hit."""
        # Mock cache hit
        mock_cache_manager.get_keyword_research.return_value = mock_trending_keywords
        
        result_context = await trend_analyzer.analyze_trends(sample_content_context)
        
        # Verify cached result used
        assert result_context.trending_keywords == mock_trending_keywords
        
        # Verify cache was checked but not updated (cache hit)
        mock_cache_manager.get_keyword_research.assert_called_once()
        mock_cache_manager.cache_keyword_research.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_research_keywords_success(self, trend_analyzer, mock_ddg_search_results):
        """Test successful keyword research."""
        concepts = ["financial education", "investment", "money management"]
        content_type = "educational"
        
        # Mock DDG search
        with patch.object(trend_analyzer, '_perform_ddg_search', 
                         new_callable=AsyncMock, return_value=mock_ddg_search_results) as mock_search:
            with patch.object(trend_analyzer, 'analyze_competitors',
                             new_callable=AsyncMock, return_value={'keywords': ['personal finance'], 'domains': ['youtube.com']}) as mock_competitors:
                
                result = await trend_analyzer.research_keywords(concepts, content_type)
                
                # Verify result structure
                assert isinstance(result, TrendingKeywords)
                assert len(result.primary_keywords) > 0
                assert len(result.long_tail_keywords) >= 0
                assert result.research_timestamp is not None
                assert result.research_quality_score > 0.0
                
                # Verify search was performed
                assert mock_search.call_count > 0
                mock_competitors.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_competitors_success(self, trend_analyzer, mock_ddg_search_results):
        """Test successful competitor analysis."""
        primary_keywords = ["financial education", "investment basics"]
        
        # Mock DDG search for competitor analysis
        with patch.object(trend_analyzer, '_perform_ddg_search',
                         new_callable=AsyncMock, return_value=mock_ddg_search_results) as mock_search:
            
            result = await trend_analyzer.analyze_competitors(primary_keywords)
            
            # Verify result structure
            assert 'keywords' in result
            assert 'domains' in result
            assert 'analysis_timestamp' in result
            assert isinstance(result['keywords'], list)
            assert isinstance(result['domains'], list)
            
            # Verify searches were performed
            assert mock_search.call_count > 0
    
    @pytest.mark.asyncio
    async def test_assess_keyword_difficulty(self, trend_analyzer):
        """Test keyword difficulty assessment."""
        keywords = [
            "financial education",  # Medium difficulty
            "best investment app",  # High difficulty (commercial + competitive)
            "how to invest money",  # Lower difficulty (informational)
            "investment"  # High difficulty (single word)
        ]
        
        result = await trend_analyzer.assess_keyword_difficulty(keywords)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert len(result) == len(keywords)
        
        # Verify difficulty scores are in valid range
        for keyword, difficulty in result.items():
            assert 0.0 <= difficulty <= 1.0
        
        # Verify relative difficulties make sense
        assert result["best investment app"] > result["how to invest money"]
        assert result["investment"] > result["how to invest money"]
    
    def test_extract_research_concepts(self, trend_analyzer, sample_content_context):
        """Test concept extraction from ContentContext."""
        concepts = trend_analyzer._extract_research_concepts(sample_content_context)
        
        # Verify concepts extracted
        assert len(concepts) > 0
        assert "financial education" in concepts
        assert "investment" in concepts
        assert "money management" in concepts
        
        # Verify no duplicates
        assert len(concepts) == len(set(concepts))
        
        # Verify reasonable limit
        assert len(concepts) <= 10
    
    def test_generate_search_queries(self, trend_analyzer):
        """Test search query generation."""
        concepts = ["financial education", "investment"]
        content_type = "educational"
        
        queries = trend_analyzer._generate_search_queries(concepts, content_type)
        
        # Verify queries generated
        assert len(queries) > 0
        assert "financial education" in queries
        assert "investment" in queries
        
        # Verify content type modifiers applied
        assert any("tutorial" in query for query in queries)
        assert any("guide" in query for query in queries)
        
        # Verify trending queries included
        current_year = str(datetime.now().year)
        assert any(current_year in query for query in queries)
        assert any("trends" in query for query in queries)
    
    def test_extract_trending_keywords(self, trend_analyzer, mock_ddg_search_results):
        """Test trending keyword extraction."""
        concepts = ["financial", "investment"]
        
        keywords = trend_analyzer._extract_trending_keywords(mock_ddg_search_results, concepts)
        
        # Verify keywords extracted
        assert len(keywords) > 0
        assert isinstance(keywords, list)
        
        # Verify reasonable limit
        assert len(keywords) <= 15
        
        # Verify quality filtering (no single characters, common words)
        for keyword in keywords:
            assert len(keyword) > 2
            assert keyword not in ['the', 'and', 'for', 'with']
    
    def test_extract_long_tail_keywords(self, trend_analyzer, mock_ddg_search_results):
        """Test long-tail keyword extraction."""
        concepts = ["financial", "investment"]
        
        long_tail = trend_analyzer._extract_long_tail_keywords(mock_ddg_search_results, concepts)
        
        # Verify long-tail keywords extracted
        assert isinstance(long_tail, list)
        
        # Verify multi-word phrases
        for phrase in long_tail:
            assert len(phrase.split()) >= 3
            assert len(phrase) <= 50
    
    def test_extract_hashtags(self, trend_analyzer):
        """Test hashtag extraction."""
        mock_results = [
            {
                'title': 'Financial Tips #FinancialLiteracy #MoneyTips',
                'snippet': 'Learn about money management #PersonalFinance'
            }
        ]
        
        hashtags = trend_analyzer._extract_hashtags(mock_results)
        
        # Verify hashtags extracted
        assert '#FinancialLiteracy' in hashtags
        assert '#MoneyTips' in hashtags
        assert '#PersonalFinance' in hashtags
        
        # Verify reasonable limit
        assert len(hashtags) <= 10
    
    def test_extract_seasonal_keywords(self, trend_analyzer):
        """Test seasonal keyword extraction."""
        mock_results = [
            {
                'title': 'Financial Planning for 2024',
                'snippet': 'Spring investment strategies and summer budgeting tips'
            },
            {
                'title': 'Holiday Spending Guide',
                'snippet': 'December financial planning and January savings'
            }
        ]
        
        seasonal = trend_analyzer._extract_seasonal_keywords(mock_results)
        
        # Verify seasonal keywords extracted
        assert '2024' in seasonal
        assert 'spring' in seasonal or 'summer' in seasonal
        assert 'holiday' in seasonal or 'december' in seasonal or 'january' in seasonal
    
    def test_calculate_keyword_confidence(self, trend_analyzer, mock_ddg_search_results):
        """Test keyword confidence calculation."""
        keyword = "financial education"
        
        confidence = trend_analyzer._calculate_keyword_confidence(keyword, mock_ddg_search_results)
        
        # Verify confidence score
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.0  # Should have some confidence for this keyword
    
    def test_estimate_search_volumes(self, trend_analyzer):
        """Test search volume estimation."""
        keywords = [
            "investment",  # Single word - higher volume
            "how to invest money",  # Question - moderate volume
            "best investment app for beginners"  # Long tail - lower volume
        ]
        
        volumes = trend_analyzer._estimate_search_volumes(keywords)
        
        # Verify volumes estimated
        assert len(volumes) == len(keywords)
        
        # Verify all volumes are positive integers
        for keyword, volume in volumes.items():
            assert isinstance(volume, int)
            assert volume > 0
        
        # Verify relative volumes make sense
        assert volumes["investment"] > volumes["best investment app for beginners"]
    
    def test_calculate_research_quality_score(self, trend_analyzer):
        """Test research quality score calculation."""
        # High quality scenario
        high_quality_score = trend_analyzer._calculate_research_quality_score(
            total_results=60, keywords_found=20, cache_hits=8, total_searches=10
        )
        
        # Low quality scenario
        low_quality_score = trend_analyzer._calculate_research_quality_score(
            total_results=5, keywords_found=2, cache_hits=0, total_searches=10
        )
        
        # Verify scores are in valid range
        assert 0.0 <= high_quality_score <= 1.0
        assert 0.0 <= low_quality_score <= 1.0
        
        # Verify high quality scores higher than low quality
        assert high_quality_score > low_quality_score
    
    @pytest.mark.asyncio
    async def test_error_handling_ddg_search_failure(self, trend_analyzer, sample_content_context):
        """Test error handling when DDG search fails."""
        # Mock DDG search to raise exception
        with patch.object(trend_analyzer, '_perform_ddg_search',
                         new_callable=AsyncMock, side_effect=APIIntegrationError("ddg_search", "web_search", "Network error")):
            
            # Should not raise exception, should handle gracefully
            result_context = await trend_analyzer.analyze_trends(sample_content_context)
            
            # Should still return context, possibly with limited results
            assert result_context is not None
            assert result_context.project_id == sample_content_context.project_id
    
    @pytest.mark.asyncio
    async def test_error_handling_content_context_error(self, trend_analyzer):
        """Test ContentContext error handling."""
        # Create invalid context
        invalid_context = ContentContext(
            project_id="",  # Invalid empty project ID
            video_files=[],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        # Should handle gracefully and preserve context
        try:
            await trend_analyzer.analyze_trends(invalid_context)
        except ContentContextError as e:
            assert e.context_state is not None
            assert e.context_state.project_id == invalid_context.project_id
    
    def test_content_type_modifiers(self, trend_analyzer):
        """Test content type specific modifiers."""
        # Test educational modifiers
        educational_queries = trend_analyzer._generate_search_queries(
            ["finance"], "educational"
        )
        assert any("tutorial" in query for query in educational_queries)
        assert any("guide" in query for query in educational_queries)
        
        # Test music modifiers
        music_queries = trend_analyzer._generate_search_queries(
            ["beats"], "music"
        )
        assert any("music" in query for query in music_queries)
        
        # Test general modifiers
        general_queries = trend_analyzer._generate_search_queries(
            ["tips"], "general"
        )
        assert any("best" in query for query in general_queries)
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, trend_analyzer, sample_content_context, 
                                  mock_cache_manager, mock_trending_keywords):
        """Test caching behavior and TTL."""
        # Test cache miss scenario
        mock_cache_manager.get_keyword_research.return_value = None
        
        with patch.object(trend_analyzer, 'research_keywords',
                         new_callable=AsyncMock, return_value=mock_trending_keywords) as mock_research:
            
            await trend_analyzer.analyze_trends(sample_content_context)
            
            # Verify cache operations
            mock_cache_manager.get_keyword_research.assert_called_once()
            mock_cache_manager.cache_keyword_research.assert_called_once()
            
            # Verify cache parameters
            cache_call_args = mock_cache_manager.cache_keyword_research.call_args
            concepts, content_type, result = cache_call_args[0]
            
            assert isinstance(concepts, list)
            assert content_type == "educational"
            assert result == mock_trending_keywords
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, trend_analyzer):
        """Test rate limiting between searches."""
        # Mock multiple search queries
        with patch.object(trend_analyzer, '_perform_ddg_search',
                         new_callable=AsyncMock, return_value=[]) as mock_search:
            with patch('asyncio.sleep') as mock_sleep:
                
                concepts = ["finance", "investment"]
                await trend_analyzer.research_keywords(concepts, "educational")
                
                # Verify sleep was called for rate limiting
                assert mock_sleep.call_count > 0
                
                # Verify sleep duration
                for call in mock_sleep.call_args_list:
                    sleep_duration = call[0][0]
                    assert sleep_duration >= trend_analyzer.search_delay
    
    def test_keyword_extraction_from_text(self, trend_analyzer):
        """Test keyword extraction from text content."""
        text = "Learn about financial education and investment basics for beginners"
        
        keywords = trend_analyzer._extract_keywords_from_text(text)
        
        # Verify keywords extracted
        assert "learn" in keywords
        assert "financial" in keywords
        assert "education" in keywords
        assert "investment" in keywords
        assert "basics" in keywords
        assert "beginners" in keywords
        
        # Verify stop words filtered out
        assert "about" not in keywords
        assert "and" not in keywords
        assert "for" not in keywords
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self, trend_analyzer, sample_content_context):
        """Test performance requirements are met."""
        import time
        
        # Mock fast responses
        with patch.object(trend_analyzer, '_perform_ddg_search',
                         new_callable=AsyncMock, return_value=[]) as mock_search:
            
            start_time = time.time()
            await trend_analyzer.analyze_trends(sample_content_context)
            end_time = time.time()
            
            # Should complete within reasonable time (allowing for test overhead)
            processing_time = end_time - start_time
            assert processing_time < 5.0  # 5 seconds for test environment
    
    def test_integration_with_content_context(self, trend_analyzer, sample_content_context):
        """Test integration with ContentContext system."""
        # Verify context extraction works
        concepts = trend_analyzer._extract_research_concepts(sample_content_context)
        
        # Should extract from multiple sources
        assert len(concepts) > 0
        
        # Should include key concepts
        for concept in sample_content_context.key_concepts:
            assert concept in concepts
        
        # Should include audio analysis concepts
        if sample_content_context.audio_analysis:
            for concept in sample_content_context.audio_analysis.financial_concepts:
                assert concept in concepts


# Integration test fixtures and helpers
@pytest.fixture
def integration_test_context():
    """ContentContext for integration testing."""
    return ContentContext(
        project_id="integration-test-123",
        video_files=["integration_test.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(),
        key_concepts=["machine learning", "artificial intelligence", "data science"],
        content_themes=["technology", "education", "programming"]
    )


class TestTrendAnalyzerIntegration:
    """Integration tests for TrendAnalyzer."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, integration_test_context):
        """Test complete end-to-end workflow."""
        cache_manager = Mock(spec=CacheManager)
        cache_manager.get_keyword_research.return_value = None
        cache_manager.cache_keyword_research.return_value = None
        cache_manager.get.return_value = None
        cache_manager.put.return_value = None
        
        trend_analyzer = TrendAnalyzer(cache_manager=cache_manager)
        
        # Mock DDG search to avoid external dependencies
        mock_results = [
            {
                'title': 'Machine Learning Tutorial 2024',
                'snippet': 'Complete guide to machine learning algorithms and applications',
                'url': 'https://example.com/ml-tutorial'
            }
        ]
        
        with patch.object(trend_analyzer, '_perform_ddg_search',
                         new_callable=AsyncMock, return_value=mock_results):
            
            result_context = await trend_analyzer.analyze_trends(integration_test_context)
            
            # Verify complete workflow
            assert result_context.trending_keywords is not None
            assert len(result_context.trending_keywords.primary_keywords) > 0
            assert result_context.trending_keywords.research_quality_score > 0.0
            
            # Verify processing metrics updated
            assert "trend_analyzer" in result_context.processing_metrics.module_processing_times
    
    def test_mock_search_fallback(self):
        """Test mock search fallback when MCP tools unavailable."""
        cache_manager = Mock(spec=CacheManager)
        trend_analyzer = TrendAnalyzer(cache_manager=cache_manager)
        
        # Test mock search generation
        mock_results = trend_analyzer._generate_mock_search_results("machine learning")
        
        assert len(mock_results) > 0
        assert all('title' in result for result in mock_results)
        assert all('snippet' in result for result in mock_results)
        assert all('url' in result for result in mock_results)
        assert all('machine learning' in result['title'].lower() for result in mock_results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])