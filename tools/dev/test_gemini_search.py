#!/usr/bin/env python3
"""
Test Gemini Flash's Google Search capabilities for research tasks.
"""

import os
import google.generativeai as genai

def test_gemini_search():
    """Test if Gemini Flash can perform Google searches."""
    
    print("Testing Gemini Flash Google Search...")
    print("=" * 40)
    
    # Get API key
    api_key = (os.getenv('AI_VIDEO_EDITOR_GEMINI_API_KEY') or 
               os.getenv('GEMINI_API_KEY') or 
               os.getenv('GOOGLE_API_KEY'))
    
    if not api_key:
        print("❌ No API key found")
        return False
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Test search capability
        search_prompt = """
        I need you to search for the latest information about:
        "OpenCV Python video processing performance optimization 2024"
        
        Please provide:
        1. Latest techniques for video processing optimization
        2. Recent performance benchmarks
        3. Best practices for memory management
        4. Any new OpenCV features for video analysis
        
        Use Google Search to find current information and provide specific details with sources.
        """
        
        print("🔍 Testing search capability...")
        response = model.generate_content(search_prompt)
        
        print("✅ Search response received:")
        print("-" * 40)
        print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        print("-" * 40)
        
        # Check if response contains search-like information
        search_indicators = [
            "according to", "recent", "latest", "2024", "source", 
            "research", "study", "benchmark", "performance"
        ]
        
        response_lower = response.text.lower()
        found_indicators = [indicator for indicator in search_indicators if indicator in response_lower]
        
        if found_indicators:
            print(f"✅ Search indicators found: {found_indicators}")
            print("✅ Gemini Flash appears to have search capabilities")
            return True
        else:
            print("⚠️  Response doesn't show clear search indicators")
            print("⚠️  May be using training data instead of live search")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {str(e)}")
        return False

def test_research_task():
    """Test Gemini Flash for a specific research task relevant to AI Video Editor."""
    
    print("\nTesting Research Task...")
    print("=" * 40)
    
    api_key = (os.getenv('AI_VIDEO_EDITOR_GEMINI_API_KEY') or 
               os.getenv('GEMINI_API_KEY') or 
               os.getenv('GOOGLE_API_KEY'))
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        research_prompt = f"""
        Research the current best practices for:
        "AI-powered video thumbnail generation using Python August 2025"
        
        Today's date is August 5, 2025. I need the most recent information about:
        1. Latest AI models for thumbnail generation (August 2025)
        2. Python libraries and frameworks being used in 2025
        3. Performance benchmarks and optimization techniques from 2025
        4. Integration patterns with video processing pipelines in 2025
        5. Any new developments or breakthroughs in August 2025
        
        Please search for the most recent articles, GitHub repositories, and technical documentation from 2025.
        Provide specific examples and code snippets if available.
        Focus on information from 2025, especially recent months.
        """
        
        print("🔍 Researching AI thumbnail generation...")
        response = model.generate_content(research_prompt)
        
        print("✅ Research response:")
        print("-" * 40)
        print(response.text)
        print("-" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Research test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Gemini Flash Capabilities")
    print("=" * 50)
    
    search_works = test_gemini_search()
    research_works = test_research_task()
    
    print("\n" + "=" * 50)
    if search_works and research_works:
        print("🎉 Gemini Flash research capabilities confirmed!")
        print("✅ Ready for enhanced collaborative development")
    else:
        print("⚠️  Limited search capabilities detected")
        print("✅ Can still use for code generation with existing knowledge")