#!/usr/bin/env python3
"""
Test script to check Gemini Flash access.
"""

import os
from pathlib import Path

def test_gemini_access():
    """Test if we can access Gemini Flash."""
    
    print("Testing Gemini Flash Access...")
    print("=" * 40)
    
    # Check for API key (try multiple common names)
    api_key = (os.getenv('AI_VIDEO_EDITOR_GEMINI_API_KEY') or 
               os.getenv('GEMINI_API_KEY') or 
               os.getenv('GOOGLE_API_KEY'))
    
    if not api_key:
        print("❌ No Gemini API key found")
        print("Please set one of: AI_VIDEO_EDITOR_GEMINI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY")
        return False
    
    print("✅ API key found")
    
    # Check if google-generativeai is installed
    try:
        import google.generativeai as genai
        print("✅ google-generativeai package available")
    except ImportError:
        print("❌ google-generativeai not installed")
        print("Run: pip install google-generativeai")
        return False
    
    # Test API connection
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Simple test
        response = model.generate_content("Hello, this is a test. Respond with 'Gemini Flash 2.5 is working!'")
        print("✅ Gemini Flash connection successful")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini Flash connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_gemini_access()
    if success:
        print("\n🎉 Gemini Flash is ready for collaborative development!")
    else:
        print("\n❌ Setup required before using Gemini Flash")