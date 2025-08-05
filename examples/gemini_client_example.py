"""
Example usage of the GeminiClient for AI Video Editor.

This example demonstrates how to use the GeminiClient for various
AI Director tasks including content analysis, keyword research,
and structured response generation.
"""

import os
import asyncio
from ai_video_editor.modules.intelligence.gemini_client import (
    GeminiClient,
    GeminiConfig,
    create_financial_analysis_prompt,
    create_keyword_research_prompt,
    create_thumbnail_concept_prompt
)
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences


def main():
    """Demonstrate GeminiClient usage."""
    
    # Initialize cache manager for API response caching
    cache_manager = CacheManager(cache_dir="temp/gemini_cache")
    
    # Create GeminiClient with caching enabled
    client = GeminiClient(
        api_key=os.getenv('GEMINI_API_KEY'),  # Set your API key
        cache_manager=cache_manager,
        enable_caching=True,
        cache_ttl=3600  # 1 hour cache
    )
    
    # Create a sample ContentContext
    context = ContentContext(
        project_id="example_project",
        video_files=["sample_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences()
    )
    
    print("=== GeminiClient Example Usage ===\n")
    
    # Example 1: Basic content generation
    print("1. Basic Content Generation:")
    try:
        response = client.generate_content(
            prompt="Explain compound interest in simple terms for beginners.",
            context=context
        )
        print(f"Response: {response.content[:200]}...")
        print(f"Processing time: {response.processing_time:.2f}s")
        print(f"Model used: {response.model_used}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 2: Structured JSON response
    print("2. Structured JSON Response:")
    try:
        schema = {
            "type": "object",
            "required": ["concepts", "difficulty", "keywords"],
            "properties": {
                "concepts": {"type": "array"},
                "difficulty": {"type": "string"},
                "keywords": {"type": "array"}
            }
        }
        
        structured_response = client.generate_structured_response(
            prompt="Analyze this financial topic: 'Building an Emergency Fund'",
            response_schema=schema,
            context=context
        )
        print(f"Structured response: {structured_response}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 3: Financial content analysis
    print("3. Financial Content Analysis:")
    try:
        transcript = """
        Today we're going to talk about the power of compound interest.
        Albert Einstein supposedly called it the eighth wonder of the world.
        When you invest money, you earn returns not just on your original investment,
        but also on the returns you've already earned.
        """
        
        concepts = ["compound interest", "investment", "returns"]
        analysis_prompt = create_financial_analysis_prompt(transcript, concepts)
        
        response = client.generate_content(
            prompt=analysis_prompt,
            config=GeminiConfig(temperature=0.3),  # Lower temperature for analysis
            context=context
        )
        print(f"Analysis: {response.content[:300]}...")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 4: Keyword research
    print("4. Keyword Research:")
    try:
        keyword_prompt = create_keyword_research_prompt(
            content_summary="Educational video about compound interest and long-term investing",
            target_audience="young adults and beginners"
        )
        
        response = client.generate_content(
            prompt=keyword_prompt,
            context=context
        )
        print(f"Keywords: {response.content[:300]}...")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Example 5: Thumbnail concepts
    print("5. Thumbnail Concept Generation:")
    try:
        thumbnail_prompt = create_thumbnail_concept_prompt(
            visual_highlights=["excited expression", "growth chart", "money visualization"],
            emotional_peaks=["excitement", "curiosity", "achievement"]
        )
        
        response = client.generate_content(
            prompt=thumbnail_prompt,
            context=context
        )
        print(f"Thumbnail concepts: {response.content[:300]}...")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Display usage statistics
    print("=== Usage Statistics ===")
    stats = client.get_usage_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\nCache statistics:")
    cache_stats = cache_manager.get_stats()
    for key, value in cache_stats.items():
        print(f"{key}: {value}")


async def async_example():
    """Demonstrate async usage of GeminiClient."""
    
    print("\n=== Async Example ===")
    
    client = GeminiClient(api_key=os.getenv('GEMINI_API_KEY'))
    
    try:
        # Async content generation
        response = await client.generate_content_async(
            prompt="What are the benefits of starting to invest early?"
        )
        print(f"Async response: {response.content[:200]}...")
        print(f"Processing time: {response.processing_time:.2f}s")
    except Exception as e:
        print(f"Async error: {e}")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv('GEMINI_API_KEY'):
        print("Please set the GEMINI_API_KEY environment variable to run this example.")
        print("You can get an API key from: https://makersuite.google.com/app/apikey")
        exit(1)
    
    # Run synchronous examples
    main()
    
    # Run async example
    asyncio.run(async_example())