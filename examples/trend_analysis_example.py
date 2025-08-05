"""
TrendAnalyzer Example - Demonstrates keyword research and trend analysis.

This example shows how to use the TrendAnalyzer class for automated keyword
research, competitor analysis, and trend identification for content optimization.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from ai_video_editor.core.content_context import (
    ContentContext, ContentType, UserPreferences,
    AudioAnalysisResult, AudioSegment
)
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.modules.intelligence.trend_analyzer import TrendAnalyzer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_content_context() -> ContentContext:
    """
    Create a sample ContentContext for demonstration.
    
    Returns:
        ContentContext with sample financial education content
    """
    # Create sample audio analysis
    audio_analysis = AudioAnalysisResult(
        transcript_text=(
            "Welcome to our comprehensive guide on financial education. "
            "Today we'll cover investment basics, money management strategies, "
            "and personal finance tips that every beginner should know. "
            "We'll explore compound interest, portfolio diversification, "
            "and risk management techniques."
        ),
        segments=[
            AudioSegment(
                text="Welcome to our comprehensive guide on financial education",
                start=0.0,
                end=4.0,
                confidence=0.95,
                financial_concepts=["financial education"]
            ),
            AudioSegment(
                text="Today we'll cover investment basics, money management strategies",
                start=4.0,
                end=8.0,
                confidence=0.92,
                financial_concepts=["investment basics", "money management"]
            ),
            AudioSegment(
                text="and personal finance tips that every beginner should know",
                start=8.0,
                end=12.0,
                confidence=0.94,
                financial_concepts=["personal finance"]
            ),
            AudioSegment(
                text="We'll explore compound interest, portfolio diversification",
                start=12.0,
                end=16.0,
                confidence=0.91,
                financial_concepts=["compound interest", "portfolio diversification"]
            ),
            AudioSegment(
                text="and risk management techniques",
                start=16.0,
                end=19.0,
                confidence=0.93,
                financial_concepts=["risk management"]
            )
        ],
        overall_confidence=0.93,
        language="en",
        processing_time=3.2,
        model_used="whisper-large",
        financial_concepts=[
            "financial education", "investment basics", "money management",
            "personal finance", "compound interest", "portfolio diversification",
            "risk management"
        ],
        explanation_segments=[
            {
                "timestamp": 12.0,
                "concept": "compound interest",
                "explanation_type": "mathematical_concept"
            },
            {
                "timestamp": 14.0,
                "concept": "portfolio diversification",
                "explanation_type": "investment_strategy"
            }
        ]
    )
    
    # Create ContentContext
    context = ContentContext(
        project_id="financial-education-demo",
        video_files=["financial_education_guide.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(
            quality_mode="high",
            max_api_cost=5.0
        ),
        key_concepts=[
            "financial education", "investment basics", "money management",
            "personal finance", "budgeting", "saving strategies"
        ],
        content_themes=[
            "education", "finance", "personal development", "wealth building"
        ],
        audio_analysis=audio_analysis
    )
    
    return context


async def demonstrate_trend_analysis():
    """
    Demonstrate comprehensive trend analysis workflow.
    """
    logger.info("=== TrendAnalyzer Demonstration ===")
    
    # Initialize cache manager
    cache_dir = Path("temp/trend_analysis_cache")
    cache_manager = CacheManager(cache_dir=str(cache_dir))
    
    # Initialize trend analyzer
    trend_analyzer = TrendAnalyzer(cache_manager=cache_manager)
    
    # Create sample content context
    context = create_sample_content_context()
    
    logger.info(f"Analyzing trends for project: {context.project_id}")
    logger.info(f"Content type: {context.content_type.value}")
    logger.info(f"Key concepts: {context.key_concepts}")
    
    try:
        # Perform trend analysis
        logger.info("\n--- Starting Trend Analysis ---")
        start_time = datetime.now()
        
        updated_context = await trend_analyzer.analyze_trends(context)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Trend analysis completed in {processing_time:.2f} seconds")
        
        # Display results
        if updated_context.trending_keywords:
            display_trend_analysis_results(updated_context.trending_keywords)
        
        # Display processing metrics
        display_processing_metrics(updated_context.processing_metrics)
        
        # Display cache statistics
        display_cache_statistics(cache_manager)
        
    except Exception as e:
        logger.error(f"Trend analysis failed: {str(e)}")
        raise


def display_trend_analysis_results(trending_keywords):
    """
    Display trend analysis results in a formatted way.
    
    Args:
        trending_keywords: TrendingKeywords object with results
    """
    logger.info("\n=== TREND ANALYSIS RESULTS ===")
    
    # Primary keywords
    logger.info(f"\n📈 Primary Keywords ({len(trending_keywords.primary_keywords)}):")
    for i, keyword in enumerate(trending_keywords.primary_keywords[:10], 1):
        difficulty = trending_keywords.keyword_difficulty.get(keyword, 0.0)
        confidence = trending_keywords.keyword_confidence.get(keyword, 0.0)
        volume = trending_keywords.search_volume_data.get(keyword, 0)
        
        logger.info(f"  {i:2d}. {keyword}")
        logger.info(f"      Difficulty: {difficulty:.2f} | Confidence: {confidence:.2f} | Volume: {volume:,}")
    
    # Long-tail keywords
    if trending_keywords.long_tail_keywords:
        logger.info(f"\n🎯 Long-tail Keywords ({len(trending_keywords.long_tail_keywords)}):")
        for i, keyword in enumerate(trending_keywords.long_tail_keywords[:5], 1):
            logger.info(f"  {i}. {keyword}")
    
    # Trending hashtags
    if trending_keywords.trending_hashtags:
        logger.info(f"\n#️⃣ Trending Hashtags ({len(trending_keywords.trending_hashtags)}):")
        hashtags_str = " ".join(trending_keywords.trending_hashtags[:10])
        logger.info(f"  {hashtags_str}")
    
    # Seasonal keywords
    if trending_keywords.seasonal_keywords:
        logger.info(f"\n🗓️ Seasonal Keywords ({len(trending_keywords.seasonal_keywords)}):")
        seasonal_str = ", ".join(trending_keywords.seasonal_keywords[:10])
        logger.info(f"  {seasonal_str}")
    
    # Trending topics
    if trending_keywords.trending_topics:
        logger.info(f"\n🔥 Trending Topics ({len(trending_keywords.trending_topics)}):")
        for i, topic in enumerate(trending_keywords.trending_topics[:5], 1):
            logger.info(f"  {i}. {topic}")
    
    # Competitor analysis
    if trending_keywords.competitor_analysis:
        comp_analysis = trending_keywords.competitor_analysis
        logger.info(f"\n🏢 Competitor Analysis:")
        
        if comp_analysis.get('keywords'):
            logger.info(f"  Competitor Keywords: {', '.join(comp_analysis['keywords'][:5])}")
        
        if comp_analysis.get('domains'):
            logger.info(f"  Top Domains: {', '.join(comp_analysis['domains'][:3])}")
    
    # Quality metrics
    logger.info(f"\n📊 Research Quality Metrics:")
    logger.info(f"  Research Quality Score: {trending_keywords.research_quality_score:.2f}")
    logger.info(f"  Cache Hit Rate: {trending_keywords.cache_hit_rate:.1%}")
    logger.info(f"  Research Timestamp: {trending_keywords.research_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")


def display_processing_metrics(processing_metrics):
    """
    Display processing performance metrics.
    
    Args:
        processing_metrics: ProcessingMetrics object
    """
    logger.info("\n=== PROCESSING METRICS ===")
    
    logger.info(f"Total Processing Time: {processing_metrics.total_processing_time:.2f}s")
    
    if processing_metrics.module_processing_times:
        logger.info("Module Processing Times:")
        for module, time_taken in processing_metrics.module_processing_times.items():
            logger.info(f"  {module}: {time_taken:.2f}s")
    
    if processing_metrics.api_calls_made:
        logger.info("API Calls Made:")
        for service, count in processing_metrics.api_calls_made.items():
            logger.info(f"  {service}: {count}")
    
    logger.info(f"Memory Peak Usage: {processing_metrics.memory_peak_usage:,} bytes")
    
    if processing_metrics.fallbacks_used:
        logger.info(f"Fallbacks Used: {', '.join(processing_metrics.fallbacks_used)}")


def display_cache_statistics(cache_manager):
    """
    Display cache performance statistics.
    
    Args:
        cache_manager: CacheManager instance
    """
    logger.info("\n=== CACHE STATISTICS ===")
    
    stats = cache_manager.get_stats()
    
    logger.info(f"Cache Hits: {stats['hits']}")
    logger.info(f"Cache Misses: {stats['misses']}")
    logger.info(f"Hit Rate: {stats['hit_rate']:.1%}")
    logger.info(f"Memory Cache Size: {stats['memory_cache_size']}/{stats['memory_cache_max_size']}")
    logger.info(f"API Cost Saved: ${stats['api_cost_saved']:.2f}")
    
    # Storage usage
    storage_stats = cache_manager.get_storage_usage()
    if 'total_size_mb' in storage_stats:
        logger.info(f"Cache Storage: {storage_stats['total_size_mb']:.1f} MB ({storage_stats['file_count']} files)")


async def demonstrate_keyword_research():
    """
    Demonstrate direct keyword research functionality.
    """
    logger.info("\n=== DIRECT KEYWORD RESEARCH DEMONSTRATION ===")
    
    # Initialize components
    cache_manager = CacheManager(cache_dir="temp/keyword_research_cache")
    trend_analyzer = TrendAnalyzer(cache_manager=cache_manager)
    
    # Research keywords for specific concepts
    concepts = ["cryptocurrency", "blockchain", "digital assets"]
    content_type = "educational"
    
    logger.info(f"Researching keywords for concepts: {concepts}")
    logger.info(f"Content type: {content_type}")
    
    try:
        trending_keywords = await trend_analyzer.research_keywords(concepts, content_type)
        
        logger.info(f"\nResearch completed!")
        logger.info(f"Found {len(trending_keywords.primary_keywords)} primary keywords")
        logger.info(f"Found {len(trending_keywords.long_tail_keywords)} long-tail keywords")
        logger.info(f"Research quality score: {trending_keywords.research_quality_score:.2f}")
        
        # Display top results
        logger.info(f"\nTop Primary Keywords:")
        for i, keyword in enumerate(trending_keywords.primary_keywords[:5], 1):
            difficulty = trending_keywords.keyword_difficulty.get(keyword, 0.0)
            confidence = trending_keywords.keyword_confidence.get(keyword, 0.0)
            logger.info(f"  {i}. {keyword} (Difficulty: {difficulty:.2f}, Confidence: {confidence:.2f})")
        
    except Exception as e:
        logger.error(f"Keyword research failed: {str(e)}")


async def demonstrate_competitor_analysis():
    """
    Demonstrate competitor analysis functionality.
    """
    logger.info("\n=== COMPETITOR ANALYSIS DEMONSTRATION ===")
    
    # Initialize components
    cache_manager = CacheManager(cache_dir="temp/competitor_analysis_cache")
    trend_analyzer = TrendAnalyzer(cache_manager=cache_manager)
    
    # Analyze competitors for specific keywords
    primary_keywords = ["personal finance", "investment guide", "budgeting tips"]
    
    logger.info(f"Analyzing competitors for keywords: {primary_keywords}")
    
    try:
        competitor_analysis = await trend_analyzer.analyze_competitors(primary_keywords)
        
        logger.info(f"\nCompetitor analysis completed!")
        
        if competitor_analysis.get('keywords'):
            logger.info(f"Found {len(competitor_analysis['keywords'])} competitor keywords:")
            for keyword in competitor_analysis['keywords'][:10]:
                logger.info(f"  • {keyword}")
        
        if competitor_analysis.get('domains'):
            logger.info(f"\nTop competitor domains:")
            for domain in competitor_analysis['domains'][:5]:
                logger.info(f"  • {domain}")
        
    except Exception as e:
        logger.error(f"Competitor analysis failed: {str(e)}")


async def demonstrate_keyword_difficulty_assessment():
    """
    Demonstrate keyword difficulty assessment.
    """
    logger.info("\n=== KEYWORD DIFFICULTY ASSESSMENT DEMONSTRATION ===")
    
    # Initialize components
    cache_manager = CacheManager(cache_dir="temp/difficulty_assessment_cache")
    trend_analyzer = TrendAnalyzer(cache_manager=cache_manager)
    
    # Test keywords with different difficulty levels
    test_keywords = [
        "finance",  # High difficulty (single word, competitive)
        "best investment app",  # High difficulty (commercial intent)
        "how to start investing",  # Medium difficulty (informational)
        "beginner investment strategies for millennials",  # Lower difficulty (long-tail)
        "personal finance tutorial",  # Medium difficulty (educational)
        "investment vs savings account comparison"  # Medium difficulty (comparison)
    ]
    
    logger.info(f"Assessing difficulty for {len(test_keywords)} keywords")
    
    try:
        difficulty_scores = await trend_analyzer.assess_keyword_difficulty(test_keywords)
        
        logger.info(f"\nKeyword difficulty assessment completed!")
        
        # Sort by difficulty (highest first)
        sorted_keywords = sorted(difficulty_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"\nKeywords ranked by difficulty (1.0 = most difficult):")
        for i, (keyword, difficulty) in enumerate(sorted_keywords, 1):
            difficulty_level = "High" if difficulty > 0.7 else "Medium" if difficulty > 0.4 else "Low"
            logger.info(f"  {i}. {keyword}")
            logger.info(f"     Difficulty: {difficulty:.2f} ({difficulty_level})")
        
    except Exception as e:
        logger.error(f"Keyword difficulty assessment failed: {str(e)}")


async def main():
    """
    Main demonstration function.
    """
    logger.info("Starting TrendAnalyzer comprehensive demonstration...")
    
    try:
        # Demonstrate main trend analysis workflow
        await demonstrate_trend_analysis()
        
        # Demonstrate specific functionalities
        await demonstrate_keyword_research()
        await demonstrate_competitor_analysis()
        await demonstrate_keyword_difficulty_assessment()
        
        logger.info("\n=== DEMONSTRATION COMPLETED SUCCESSFULLY ===")
        logger.info("The TrendAnalyzer has successfully demonstrated:")
        logger.info("✅ Comprehensive trend analysis with ContentContext integration")
        logger.info("✅ Keyword research with difficulty and confidence scoring")
        logger.info("✅ Competitor analysis and keyword extraction")
        logger.info("✅ Intelligent caching with performance optimization")
        logger.info("✅ Error handling and graceful degradation")
        logger.info("✅ Processing metrics and performance monitoring")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())