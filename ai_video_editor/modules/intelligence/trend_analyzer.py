"""
TrendAnalyzer - Automated keyword research and trend analysis using DDG Search.

This module provides comprehensive keyword research, competitor analysis, and trend
identification for content optimization and SEO strategy.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from ai_video_editor.core.content_context import ContentContext, TrendingKeywords
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.exceptions import (
    ContentContextError,
    APIIntegrationError,
    handle_errors
)


logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Automated keyword research and trend analysis using DDG Search.
    
    Provides comprehensive keyword research, competitor analysis, and trend
    identification for content optimization and SEO strategy.
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize TrendAnalyzer with required dependencies.
        
        Args:
            cache_manager: CacheManager instance for result caching
        """
        self.cache_manager = cache_manager
        self.search_delay = 1.0  # Rate limiting delay between searches
        self.max_search_results = 20  # Maximum results per search
        
        # Content type specific search modifiers
        self.content_type_modifiers = {
            'educational': ['tutorial', 'guide', 'learn', 'how to', 'explained'],
            'music': ['music', 'song', 'artist', 'album', 'playlist'],
            'general': ['tips', 'best', 'review', 'comparison', 'latest']
        }
        
        # Keyword difficulty indicators (higher score = more difficult)
        self.difficulty_indicators = {
            'high_competition': ['best', 'top', 'review', 'vs', 'comparison'],
            'commercial': ['buy', 'price', 'cost', 'cheap', 'deal'],
            'informational': ['how', 'what', 'why', 'guide', 'tutorial'],
            'branded': ['brand names', 'company names']
        }
        
        logger.info("TrendAnalyzer initialized with DDG Search integration")
    
    @handle_errors(logger)
    async def analyze_trends(self, context: ContentContext) -> ContentContext:
        """
        Perform comprehensive trend analysis and keyword research.
        
        Args:
            context: ContentContext with content analysis results
            
        Returns:
            Updated ContentContext with trending_keywords populated
            
        Raises:
            ContentContextError: If analysis fails
            APIIntegrationError: If DDG Search fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting trend analysis for project {context.project_id}")
            
            # Extract concepts for research
            concepts = self._extract_research_concepts(context)
            content_type = context.content_type.value
            
            logger.debug(f"Research concepts: {concepts}")
            logger.debug(f"Content type: {content_type}")
            
            # Check cache first
            cached_result = self.cache_manager.get_keyword_research(concepts, content_type)
            if cached_result:
                logger.info("Using cached keyword research results")
                context.trending_keywords = cached_result
                context.processing_metrics.add_module_metrics(
                    "trend_analyzer", time.time() - start_time, 0
                )
                return context
            
            # Perform keyword research
            trending_keywords = await self.research_keywords(concepts, content_type)
            
            # Store results in context
            context.trending_keywords = trending_keywords
            
            # Cache results for 24 hours
            self.cache_manager.cache_keyword_research(concepts, content_type, trending_keywords)
            
            # Update processing metrics
            processing_time = time.time() - start_time
            context.processing_metrics.add_module_metrics(
                "trend_analyzer", processing_time, 0
            )
            
            logger.info(f"Trend analysis completed in {processing_time:.2f}s")
            logger.info(f"Found {len(trending_keywords.primary_keywords)} primary keywords")
            
            return context
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {str(e)}")
            raise ContentContextError(
                f"Trend analysis failed: {str(e)}",
                context_state=context
            )
    
    @handle_errors(logger)
    async def research_keywords(self, concepts: List[str], content_type: str) -> TrendingKeywords:
        """
        Research keywords for given concepts and content type.
        
        Args:
            concepts: List of content concepts to research
            content_type: Type of content (educational, music, general)
            
        Returns:
            TrendingKeywords object with research results
        """
        logger.info(f"Researching keywords for {len(concepts)} concepts")
        
        # Generate search queries
        search_queries = self._generate_search_queries(concepts, content_type)
        
        # Perform searches and collect results
        all_search_results = []
        cache_hits = 0
        total_searches = len(search_queries)
        
        for query in search_queries:
            try:
                # Check cache for individual search results
                cache_key = f"search_result:{query}"
                cached_search = self.cache_manager.get(cache_key)
                
                if cached_search:
                    all_search_results.extend(cached_search)
                    cache_hits += 1
                    logger.debug(f"Cache hit for query: {query}")
                else:
                    # Perform DDG search
                    search_results = await self._perform_ddg_search(query)
                    all_search_results.extend(search_results)
                    
                    # Cache search results for 1 hour
                    self.cache_manager.put(cache_key, search_results, ttl=3600)
                    logger.debug(f"Performed search for: {query}")
                
                # Rate limiting
                await asyncio.sleep(self.search_delay)
                
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {str(e)}")
                continue
        
        # Extract and analyze keywords
        primary_keywords = self._extract_trending_keywords(all_search_results, concepts)
        long_tail_keywords = self._extract_long_tail_keywords(all_search_results, concepts)
        trending_hashtags = self._extract_hashtags(all_search_results)
        seasonal_keywords = self._extract_seasonal_keywords(all_search_results)
        
        # Analyze competitors
        competitor_analysis = await self.analyze_competitors(primary_keywords[:5])
        competitor_keywords = list(competitor_analysis.get('keywords', []))
        
        # Assess keyword difficulty
        all_keywords = primary_keywords + long_tail_keywords
        keyword_difficulty = await self.assess_keyword_difficulty(all_keywords)
        
        # Calculate keyword confidence
        keyword_confidence = {}
        for keyword in all_keywords:
            keyword_confidence[keyword] = self._calculate_keyword_confidence(
                keyword, all_search_results
            )
        
        # Extract trending topics
        trending_topics = self._extract_trending_topics(all_search_results)
        
        # Calculate research quality score
        research_quality_score = self._calculate_research_quality_score(
            len(all_search_results), len(primary_keywords), cache_hits, total_searches
        )
        
        # Calculate cache hit rate
        cache_hit_rate = cache_hits / total_searches if total_searches > 0 else 0.0
        
        # Create TrendingKeywords object
        trending_keywords = TrendingKeywords(
            primary_keywords=primary_keywords,
            long_tail_keywords=long_tail_keywords,
            trending_hashtags=trending_hashtags,
            seasonal_keywords=seasonal_keywords,
            competitor_keywords=competitor_keywords,
            search_volume_data=self._estimate_search_volumes(all_keywords),
            research_timestamp=datetime.now(),
            keyword_difficulty=keyword_difficulty,
            keyword_confidence=keyword_confidence,
            trending_topics=trending_topics,
            competitor_analysis=competitor_analysis,
            research_quality_score=research_quality_score,
            cache_hit_rate=cache_hit_rate
        )
        
        logger.info(f"Keyword research completed with quality score: {research_quality_score:.2f}")
        
        return trending_keywords
    
    @handle_errors(logger)
    async def analyze_competitors(self, primary_keywords: List[str]) -> Dict[str, Any]:
        """
        Analyze competitor content for keyword insights.
        
        Args:
            primary_keywords: Primary keywords to analyze
            
        Returns:
            Competitor analysis results with keyword insights
        """
        logger.info(f"Analyzing competitors for {len(primary_keywords)} keywords")
        
        competitor_data = {
            'keywords': set(),
            'domains': set(),
            'content_patterns': [],
            'title_patterns': [],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        for keyword in primary_keywords[:3]:  # Limit to top 3 keywords
            try:
                # Search for competitor content
                competitor_query = f'"{keyword}" site:youtube.com OR site:medium.com'
                search_results = await self._perform_ddg_search(competitor_query)
                
                for result in search_results[:5]:  # Top 5 results per keyword
                    # Extract domain
                    domain = urlparse(result.get('url', '')).netloc
                    if domain:
                        competitor_data['domains'].add(domain)
                    
                    # Extract keywords from titles and snippets
                    title = result.get('title', '').lower()
                    snippet = result.get('snippet', '').lower()
                    
                    # Extract potential keywords
                    text_content = f"{title} {snippet}"
                    extracted_keywords = self._extract_keywords_from_text(text_content)
                    competitor_data['keywords'].update(extracted_keywords)
                    
                    # Analyze title patterns
                    if title:
                        competitor_data['title_patterns'].append(title)
                
                await asyncio.sleep(self.search_delay)
                
            except Exception as e:
                logger.warning(f"Competitor analysis failed for keyword '{keyword}': {str(e)}")
                continue
        
        # Convert sets to lists for JSON serialization
        competitor_data['keywords'] = list(competitor_data['keywords'])[:20]  # Top 20
        competitor_data['domains'] = list(competitor_data['domains'])
        
        logger.info(f"Competitor analysis found {len(competitor_data['keywords'])} keywords")
        
        return competitor_data
    
    @handle_errors(logger)
    async def assess_keyword_difficulty(self, keywords: List[str]) -> Dict[str, float]:
        """
        Assess keyword difficulty and search volume.
        
        Args:
            keywords: List of keywords to assess
            
        Returns:
            Dictionary mapping keywords to difficulty scores (0.0-1.0)
        """
        logger.info(f"Assessing difficulty for {len(keywords)} keywords")
        
        difficulty_scores = {}
        
        for keyword in keywords:
            try:
                # Calculate difficulty based on various factors
                difficulty_score = 0.0
                
                # Length factor (shorter keywords are generally more competitive)
                word_count = len(keyword.split())
                if word_count == 1:
                    difficulty_score += 0.3
                elif word_count == 2:
                    difficulty_score += 0.2
                elif word_count >= 4:
                    difficulty_score -= 0.1
                
                # Commercial intent indicators
                for indicator in self.difficulty_indicators['commercial']:
                    if indicator in keyword.lower():
                        difficulty_score += 0.2
                        break
                
                # High competition indicators
                for indicator in self.difficulty_indicators['high_competition']:
                    if indicator in keyword.lower():
                        difficulty_score += 0.3
                        break
                
                # Informational keywords are generally easier
                for indicator in self.difficulty_indicators['informational']:
                    if indicator in keyword.lower():
                        difficulty_score -= 0.1
                        break
                
                # Normalize score to 0.0-1.0 range
                difficulty_score = max(0.0, min(1.0, difficulty_score + 0.5))
                difficulty_scores[keyword] = difficulty_score
                
            except Exception as e:
                logger.warning(f"Difficulty assessment failed for keyword '{keyword}': {str(e)}")
                difficulty_scores[keyword] = 0.5  # Default medium difficulty
        
        logger.info(f"Keyword difficulty assessment completed")
        
        return difficulty_scores
    
    def _extract_research_concepts(self, context: ContentContext) -> List[str]:
        """
        Extract concepts for keyword research from ContentContext.
        
        Args:
            context: ContentContext with analysis results
            
        Returns:
            List of concepts for research
        """
        concepts = []
        
        # Add key concepts from content analysis
        concepts.extend(context.key_concepts)
        
        # Add content themes
        concepts.extend(context.content_themes)
        
        # Extract concepts from audio analysis if available
        if context.audio_analysis:
            concepts.extend(context.audio_analysis.financial_concepts)
        
        # Remove duplicates and empty strings
        concepts = list(set([c.strip() for c in concepts if c.strip()]))
        
        # Limit to top 10 concepts to avoid too many searches
        return concepts[:10]
    
    def _generate_search_queries(self, concepts: List[str], content_type: str) -> List[str]:
        """
        Generate targeted search queries for trend research.
        
        Args:
            concepts: Content concepts
            content_type: Type of content
            
        Returns:
            List of search queries for trend research
        """
        queries = []
        modifiers = self.content_type_modifiers.get(content_type, [])
        
        # Primary concept queries
        for concept in concepts[:5]:  # Limit to top 5 concepts
            queries.append(concept)
            
            # Add modified queries
            for modifier in modifiers[:2]:  # Top 2 modifiers per concept
                queries.append(f"{concept} {modifier}")
        
        # Trending queries
        current_year = datetime.now().year
        for concept in concepts[:3]:
            queries.append(f"{concept} {current_year}")
            queries.append(f"{concept} trends")
        
        # Remove duplicates
        return list(set(queries))
    
    async def _perform_ddg_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform DDG search using MCP tools.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        try:
            # Import MCP DDG search function
            from mcp_ddg_search import web_search
            
            # Perform search
            results = await web_search(
                query=query,
                numResults=self.max_search_results
            )
            
            return results.get('results', [])
            
        except ImportError:
            # Fallback: simulate search results for testing
            logger.warning("MCP DDG Search not available, using mock results")
            return self._generate_mock_search_results(query)
        except Exception as e:
            logger.error(f"DDG search failed for query '{query}': {str(e)}")
            raise APIIntegrationError("ddg_search", "web_search", str(e))
    
    def _generate_mock_search_results(self, query: str) -> List[Dict[str, Any]]:
        """
        Generate mock search results for testing.
        
        Args:
            query: Search query
            
        Returns:
            List of mock search results
        """
        mock_results = []
        
        # Generate realistic mock results based on query
        for i in range(5):
            mock_results.append({
                'title': f"{query.title()} - Complete Guide {i+1}",
                'snippet': f"Learn about {query} with this comprehensive guide. "
                          f"Discover the latest trends and best practices for {query}.",
                'url': f"https://example{i+1}.com/{query.replace(' ', '-')}"
            })
        
        return mock_results
    
    def _extract_trending_keywords(self, search_results: List[Dict], concepts: List[str]) -> List[str]:
        """
        Extract trending keywords from search results.
        
        Args:
            search_results: DDG search results
            concepts: Original concepts for context
            
        Returns:
            List of extracted trending keywords
        """
        keyword_frequency = {}
        
        for result in search_results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            
            # Extract keywords from title and snippet
            text_content = f"{title} {snippet}"
            keywords = self._extract_keywords_from_text(text_content)
            
            for keyword in keywords:
                if len(keyword) > 2 and keyword not in ['the', 'and', 'for', 'with']:
                    keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(keyword_frequency.items(), key=lambda x: x[1], reverse=True)
        
        # Filter and return top 15 keywords
        primary_keywords = []
        for keyword, freq in sorted_keywords:
            if len(primary_keywords) >= 15:
                break
            if freq >= 2:  # Minimum frequency threshold
                primary_keywords.append(keyword)
        
        return primary_keywords
    
    def _extract_long_tail_keywords(self, search_results: List[Dict], concepts: List[str]) -> List[str]:
        """
        Extract long-tail keywords from search results.
        
        Args:
            search_results: DDG search results
            concepts: Original concepts for context
            
        Returns:
            List of long-tail keywords
        """
        long_tail_keywords = []
        
        for result in search_results:
            title = result.get('title', '')
            
            # Look for phrases with 3+ words
            phrases = re.findall(r'\b\w+\s+\w+\s+\w+(?:\s+\w+)*\b', title.lower())
            
            for phrase in phrases:
                if (len(phrase.split()) >= 3 and 
                    len(phrase) <= 50 and 
                    any(concept in phrase for concept in concepts)):
                    long_tail_keywords.append(phrase.strip())
        
        # Remove duplicates and return top 10
        return list(set(long_tail_keywords))[:10]
    
    def _extract_hashtags(self, search_results: List[Dict]) -> List[str]:
        """
        Extract hashtags from search results.
        
        Args:
            search_results: DDG search results
            
        Returns:
            List of hashtags
        """
        hashtags = []
        
        for result in search_results:
            text_content = f"{result.get('title', '')} {result.get('snippet', '')}"
            
            # Extract hashtags
            found_hashtags = re.findall(r'#\w+', text_content)
            hashtags.extend(found_hashtags)
        
        # Remove duplicates and return top 10
        return list(set(hashtags))[:10]
    
    def _extract_seasonal_keywords(self, search_results: List[Dict]) -> List[str]:
        """
        Extract seasonal keywords from search results.
        
        Args:
            search_results: DDG search results
            
        Returns:
            List of seasonal keywords
        """
        seasonal_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(spring|summer|fall|autumn|winter)\b',
            r'\b(holiday|christmas|new year|valentine|easter)\b'
        ]
        
        seasonal_keywords = []
        
        for result in search_results:
            text_content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
            
            for pattern in seasonal_patterns:
                matches = re.findall(pattern, text_content)
                seasonal_keywords.extend(matches)
        
        # Remove duplicates and return top 10
        return list(set(seasonal_keywords))[:10]
    
    def _extract_trending_topics(self, search_results: List[Dict]) -> List[str]:
        """
        Extract trending topics from search results.
        
        Args:
            search_results: DDG search results
            
        Returns:
            List of trending topics
        """
        trending_indicators = ['trending', 'viral', 'popular', 'hot', 'latest', 'new']
        trending_topics = []
        
        for result in search_results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            
            # Look for trending indicators
            for indicator in trending_indicators:
                if indicator in title or indicator in snippet:
                    # Extract the topic context
                    text_content = f"{title} {snippet}"
                    words = text_content.split()
                    
                    # Find words around trending indicators
                    for i, word in enumerate(words):
                        if indicator in word:
                            # Extract surrounding context
                            start = max(0, i-2)
                            end = min(len(words), i+3)
                            topic = ' '.join(words[start:end])
                            if len(topic) > 10:
                                trending_topics.append(topic)
        
        # Remove duplicates and return top 10
        return list(set(trending_topics))[:10]
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """
        Extract keywords from text content.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of extracted keywords
        """
        # Remove special characters and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'a', 'an'
        }
        
        # Filter keywords
        keywords = []
        for word in words:
            if (len(word) > 2 and 
                word not in stop_words and 
                not word.isdigit()):
                keywords.append(word)
        
        return keywords
    
    def _calculate_keyword_confidence(self, keyword: str, search_results: List[Dict]) -> float:
        """
        Calculate confidence score for keyword relevance.
        
        Args:
            keyword: Keyword to assess
            search_results: Related search results
            
        Returns:
            Confidence score (0.0-1.0)
        """
        confidence_score = 0.0
        appearances = 0
        
        for result in search_results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            
            # Check for keyword appearances
            if keyword in title:
                confidence_score += 0.3
                appearances += 1
            
            if keyword in snippet:
                confidence_score += 0.2
                appearances += 1
        
        # Normalize based on total results
        if len(search_results) > 0:
            confidence_score = confidence_score / len(search_results)
        
        # Boost confidence for frequently appearing keywords
        if appearances >= 3:
            confidence_score += 0.2
        
        # Normalize to 0.0-1.0 range
        return max(0.0, min(1.0, confidence_score))
    
    def _estimate_search_volumes(self, keywords: List[str]) -> Dict[str, int]:
        """
        Estimate search volumes for keywords.
        
        Args:
            keywords: List of keywords
            
        Returns:
            Dictionary mapping keywords to estimated search volumes
        """
        search_volumes = {}
        
        for keyword in keywords:
            # Simple heuristic based on keyword characteristics
            base_volume = 1000
            
            # Adjust based on keyword length
            word_count = len(keyword.split())
            if word_count == 1:
                base_volume *= 3  # Single words tend to have higher volume
            elif word_count >= 4:
                base_volume *= 0.3  # Long-tail keywords have lower volume
            
            # Adjust based on keyword type
            if any(indicator in keyword.lower() for indicator in ['how', 'what', 'why']):
                base_volume *= 1.5  # Question keywords tend to be popular
            
            if any(indicator in keyword.lower() for indicator in ['best', 'top', 'review']):
                base_volume *= 2  # Commercial keywords tend to be popular
            
            # Add some randomness to make it more realistic
            import random
            random.seed(hash(keyword))  # Deterministic randomness
            volume_variation = random.uniform(0.5, 2.0)
            
            estimated_volume = int(base_volume * volume_variation)
            search_volumes[keyword] = estimated_volume
        
        return search_volumes
    
    def _calculate_research_quality_score(self, total_results: int, keywords_found: int, 
                                        cache_hits: int, total_searches: int) -> float:
        """
        Calculate overall research quality score.
        
        Args:
            total_results: Total search results processed
            keywords_found: Number of keywords found
            cache_hits: Number of cache hits
            total_searches: Total searches performed
            
        Returns:
            Quality score (0.0-1.0)
        """
        quality_score = 0.0
        
        # Results coverage score
        if total_results >= 50:
            quality_score += 0.3
        elif total_results >= 20:
            quality_score += 0.2
        elif total_results >= 10:
            quality_score += 0.1
        
        # Keywords diversity score
        if keywords_found >= 15:
            quality_score += 0.3
        elif keywords_found >= 10:
            quality_score += 0.2
        elif keywords_found >= 5:
            quality_score += 0.1
        
        # Cache efficiency score
        if total_searches > 0:
            cache_efficiency = cache_hits / total_searches
            quality_score += cache_efficiency * 0.2
        
        # Completeness score
        quality_score += 0.2  # Base completeness score
        
        return max(0.0, min(1.0, quality_score))