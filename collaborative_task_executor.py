#!/usr/bin/env python3
"""
Collaborative Task Executor for AI Video Editor
Integrates Kiro's architectural guidance with Gemini Flash's implementation capabilities.
"""

import os
import json
import google.generativeai as genai
from typing import Dict, Any, Optional, List
from pathlib import Path
import re

class CollaborativeTaskExecutor:
    """Manages collaborative development between Kiro and Gemini Flash."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the collaborative executor."""
        self.api_key = (api_key or 
                       os.getenv('AI_VIDEO_EDITOR_GEMINI_API_KEY') or 
                       os.getenv('GEMINI_API_KEY') or 
                       os.getenv('GOOGLE_API_KEY'))
        
        if not self.api_key:
            raise ValueError("Gemini API key required")
        
        genai.configure(api_key=self.api_key)
        # Use Flash for development tasks, but AI Video Editor should use Pro for production
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Paths
        self.shared_memory_file = Path('.kiro/shared_memory.json')
        self.tasks_file = Path('.kiro/specs/ai-video-editor/tasks.md')
        self.project_root = Path('.')
        
        # Ensure directories exist
        self.shared_memory_file.parent.mkdir(parents=True, exist_ok=True)
    
    def read_shared_memory(self) -> Dict[str, Any]:
        """Read shared memory for architectural patterns and context."""
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
    
    def read_tasks_file(self) -> str:
        """Read the current tasks from the spec."""
        if self.tasks_file.exists():
            return self.tasks_file.read_text()
        return ""
    
    def extract_task_details(self, task_number: str) -> Dict[str, Any]:
        """Extract specific task details from the tasks file."""
        tasks_content = self.read_tasks_file()
        
        # Find the specific task - improved pattern
        task_pattern = rf"- \[ \] \*\*{re.escape(task_number)}\s+([^*]+)\*\*(.*?)(?=- \[ \] \*\*\d+(?:\.\d+)?\s+|\Z)"
        match = re.search(task_pattern, tasks_content, re.DOTALL)
        
        if not match:
            return {"error": f"Task {task_number} not found"}
        
        title = match.group(1).strip()
        task_content = match.group(0)
        task_details = match.group(2).strip()
        
        # Extract requirements (lines starting with - )
        requirements = re.findall(r'^\s*-\s+(.+)$', task_details, re.MULTILINE)
        
        # Extract _Requirements: references
        req_refs = re.findall(r'_Requirements?:\s*([^_\n]+)', task_details)
        
        return {
            "task_number": task_number,
            "title": title,
            "content": task_content,
            "details": task_details,
            "requirements": requirements,
            "requirement_references": req_refs
        }
    
    def create_task_specification(self, task_number: str) -> str:
        """Create detailed specification for Gemini Flash based on task and architectural patterns."""
        
        # Get task details
        task_details = self.extract_task_details(task_number)
        if "error" in task_details:
            return f"Error: {task_details['error']}"
        
        # Get architectural context
        shared_context = self.read_shared_memory()
        
        # Read existing project files for context
        existing_files = self._get_relevant_files(task_details)
        
        specification = f"""
# COLLABORATIVE TASK SPECIFICATION

## Task Information
**Task Number**: {task_details['task_number']}
**Title**: {task_details['title']}

## Task Details
{task_details['content']}

## Architectural Patterns (MUST FOLLOW)
{json.dumps(shared_context.get('architectural_patterns', {}), indent=2)}

## Existing Project Context
{existing_files}

## Implementation Requirements
1. **ContentContext Integration**: All modules MUST operate on shared ContentContext object
2. **Error Handling**: Use ContentContextError base class with context preservation
3. **Testing Strategy**: Include comprehensive unit tests with mocking
4. **Performance Guidelines**: Follow memory and processing constraints
5. **Code Quality**: Include type hints, docstrings, and clear documentation

## Integration Points
- Import from existing modules: ai_video_editor.core.content_context
- Follow existing error handling patterns
- Place tests in tests/unit/ directory
- Use existing performance monitoring utilities

## Expected Deliverables
1. Implementation code following architectural patterns
2. Comprehensive unit tests with mocking
3. Example usage code
4. Performance considerations
5. Integration documentation

## Research Requirements
If you need current information about libraries, techniques, or best practices, use Google Search to find:
- Latest documentation and examples
- Performance benchmarks and optimization techniques
- Integration patterns and best practices
- Recent developments in the field

Generate complete, production-ready code that integrates seamlessly with the existing AI Video Editor architecture.
"""
        
        return specification
    
    def _get_relevant_files(self, task_details: Dict[str, Any]) -> str:
        """Get content of relevant existing files for context."""
        relevant_files = []
        
        # Always include core files
        core_files = [
            'ai_video_editor/core/content_context.py',
            'ai_video_editor/utils/error_handling.py',
            'ai_video_editor/utils/performance_benchmarks.py'
        ]
        
        for file_path in core_files:
            if Path(file_path).exists():
                try:
                    content = Path(file_path).read_text()
                    relevant_files.append(f"## {file_path}\n```python\n{content[:1000]}...\n```\n")
                except Exception:
                    continue
        
        return "\n".join(relevant_files) if relevant_files else "No existing files found for context."
    
    def execute_task(self, task_number: str, include_research: bool = True) -> str:
        """Execute a task using Gemini Flash with full specification."""
        
        print(f"🚀 Executing Task {task_number} with Gemini Flash...")
        
        # Create detailed specification
        specification = self.create_task_specification(task_number)
        
        # Add research instruction if needed
        if include_research:
            research_prompt = f"""
Before implementing, please research the latest information about the technologies and techniques required for this task. Use Google Search to find:
1. Current best practices and examples
2. Latest library versions and features
3. Performance optimization techniques
4. Integration patterns

Then implement the task following the specification below:

{specification}
"""
        else:
            research_prompt = specification
        
        try:
            # Generate implementation
            response = self.model.generate_content(research_prompt)
            
            # Store in shared memory for Kiro review
            self.write_shared_memory({
                'current_task_execution': {
                    'task_number': task_number,
                    'specification': specification,
                    'implementation': response.text,
                    'timestamp': str(Path().stat().st_mtime),
                    'status': 'pending_kiro_review',
                    'include_research': include_research
                }
            })
            
            print(f"✅ Task {task_number} implementation completed")
            print(f"📝 Stored in shared memory for Kiro review")
            
            return response.text
            
        except Exception as e:
            error_msg = f"Task {task_number} execution failed: {str(e)}"
            self.write_shared_memory({
                'execution_error': {
                    'task_number': task_number,
                    'error': error_msg,
                    'timestamp': str(Path().stat().st_mtime)
                }
            })
            raise Exception(error_msg)
    
    def list_available_tasks(self) -> List[str]:
        """List all available tasks from the tasks file."""
        tasks_content = self.read_tasks_file()
        
        # Find all task numbers
        task_pattern = r"- \[ \] \*\*(\d+(?:\.\d+)?)\s+[^*]+\*\*"
        matches = re.findall(task_pattern, tasks_content)
        
        return matches
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get current task execution status from shared memory."""
        shared_context = self.read_shared_memory()
        return shared_context.get('current_task_execution', {})

def main():
    """CLI interface for collaborative task execution."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python collaborative_task_executor.py <task_number> [--no-research]")
        print("\nAvailable commands:")
        print("  list - List all available tasks")
        print("  status - Show current task status")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        executor = CollaborativeTaskExecutor()
        
        if command == "list":
            tasks = executor.list_available_tasks()
            print("Available Tasks:")
            for task in tasks:
                print(f"  - {task}")
        
        elif command == "status":
            status = executor.get_task_status()
            if status:
                print("Current Task Status:")
                print(json.dumps(status, indent=2))
            else:
                print("No active task execution")
        
        else:
            # Execute specific task
            task_number = command
            include_research = "--no-research" not in sys.argv
            
            result = executor.execute_task(task_number, include_research)
            
            print("\n" + "="*50)
            print("IMPLEMENTATION RESULT:")
            print("="*50)
            print(result)
            print("="*50)
            print(f"\n✅ Task {task_number} completed and ready for Kiro review")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()