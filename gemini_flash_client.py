#!/usr/bin/env python3
"""
Simple Gemini Flash 2.5 client for collaborative development.
This allows both Kiro and external Gemini Flash to work together.
"""

import os
import json
import google.generativeai as genai
from typing import Dict, Any, Optional
from pathlib import Path

class GeminiFlashClient:
    """Simple client for Gemini Flash 2.5 integration."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini Flash client."""
        self.api_key = (api_key or 
                       os.getenv('AI_VIDEO_EDITOR_GEMINI_API_KEY') or 
                       os.getenv('GEMINI_API_KEY') or 
                       os.getenv('GOOGLE_API_KEY'))
        if not self.api_key:
            raise ValueError("Gemini API key required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Shared memory file for collaboration
        self.shared_memory_file = Path('.kiro/shared_memory.json')
        self.shared_memory_file.parent.mkdir(exist_ok=True)
    
    def read_shared_memory(self) -> Dict[str, Any]:
        """Read shared memory for architectural patterns and specifications."""
        if self.shared_memory_file.exists():
            with open(self.shared_memory_file, 'r') as f:
                return json.load(f)
        return {}
    
    def write_shared_memory(self, data: Dict[str, Any]) -> None:
        """Write to shared memory for Kiro to review."""
        existing = self.read_shared_memory()
        existing.update(data)
        with open(self.shared_memory_file, 'w') as f:
            json.dump(existing, f, indent=2)
    
    def generate_code(self, task_specification: str, context: Dict[str, Any] = None) -> str:
        """Generate code based on task specification and shared context."""
        
        # Read architectural patterns from shared memory
        shared_context = self.read_shared_memory()
        
        # Construct prompt with architectural guidance
        prompt = f"""
You are Gemini Flash 2.5 working collaboratively with Kiro on the AI Video Editor project.

ARCHITECTURAL CONTEXT:
{json.dumps(shared_context.get('architectural_patterns', {}), indent=2)}

TASK SPECIFICATION:
{task_specification}

ADDITIONAL CONTEXT:
{json.dumps(context or {}, indent=2)}

REQUIREMENTS:
1. Follow ContentContext integration patterns
2. Implement proper error handling
3. Include comprehensive unit tests with mocking
4. Follow performance guidelines
5. Maintain code quality standards

Generate the implementation code following these patterns.
"""
        
        try:
            response = self.model.generate_content(prompt)
            
            # Store implementation details in shared memory for Kiro review
            self.write_shared_memory({
                'last_implementation': {
                    'task': task_specification,
                    'code': response.text,
                    'timestamp': str(Path().stat().st_mtime),
                    'status': 'pending_review'
                }
            })
            
            return response.text
            
        except Exception as e:
            error_msg = f"Gemini Flash generation failed: {str(e)}"
            self.write_shared_memory({
                'last_error': {
                    'task': task_specification,
                    'error': error_msg,
                    'timestamp': str(Path().stat().st_mtime)
                }
            })
            raise Exception(error_msg)

def main():
    """CLI interface for Gemini Flash client."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gemini_flash_client.py 'task specification'")
        sys.exit(1)
    
    task_spec = sys.argv[1]
    
    try:
        client = GeminiFlashClient()
        result = client.generate_code(task_spec)
        print("Generated Code:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        print("Code stored in shared memory for Kiro review.")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()