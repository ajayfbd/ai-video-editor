# MCP Tools Utilization Guide for AI Video Editor

## Core Principle: Strategic Tool Integration

This guide provides comprehensive instructions for leveraging all available MCP tools throughout the AI Video Editor development process. Each tool serves specific purposes and should be used strategically to maintain context, optimize development workflow, and ensure high-quality implementation.

## Available MCP Tools and Their Strategic Use

### 1. Context7 - Documentation Research
**Purpose**: Get up-to-date documentation for libraries and frameworks
**When to Use**: 
- Before implementing any new library integration
- When encountering API changes or deprecations
- For researching best practices and performance optimization

**Usage Patterns**:
```
# Research library documentation
Use Context7 to resolve library ID: `/openai/whisper`
Get focused documentation with topic: "python integration and basic usage"
Store key insights in Memory for future reference
```

**Key Libraries to Research**:
- `/openai/whisper` - Audio transcription
- `/context7/googleapis_github_io-python-genai` - Gemini API integration
- OpenCV documentation for video analysis
- Movis documentation from official website
- Matplotlib for chart generation
- Blender Python API for animations

### 2. Memory - Project State Management
**Purpose**: Maintain persistent context and track implementation progress
**When to Use**:
- At the start of each major task to record current state
- After completing significant milestones
- When discovering architectural insights or patterns
- For tracking API usage patterns and optimization opportunities

**Usage Patterns**:
```
# Create entities for major components
Create entity: "Audio Analysis Module" with current implementation status
Add observations: "Whisper integration completed", "Error handling implemented"
Create relations: "Audio Analysis Module" -> "uses" -> "ContentContext System"

# Track progress and decisions
Add observations to existing entities with implementation updates
Store successful patterns and architectural decisions
Maintain relationships between components
```

**Key Entities to Track**:
- AI Video Editor Project (overall status)
- ContentContext System (core architecture)
- Implementation Priority Tasks (task completion status)
- API Integration Patterns (successful approaches)
- Performance Optimization Insights (benchmarks and improvements)

### 3. GitHub - Code Management and Collaboration
**Purpose**: Version control, issue tracking, and progress documentation
**When to Use**:
- For all code commits and version control
- Creating issues for bugs or feature requests
- Tracking implementation progress through commits
- Managing pull requests and code reviews

**Usage Patterns**:
```
# Create issues for major tasks
Create issue: "Implement Whisper Integration for Audio Analysis"
Add labels: "enhancement", "audio-processing", "phase-1"
Assign to current milestone

# Track progress through commits
Commit with descriptive messages referencing requirements
Create pull requests for major feature implementations
Use GitHub for continuous integration and testing automation
```

### 4. Filesystem - Code and Configuration Management
**Purpose**: Reading, writing, and managing project files
**When to Use**:
- Creating new modules and classes
- Reading existing code for integration
- Managing configuration files
- Organizing project structure

**Usage Patterns**:
```
# Read existing code for integration
Read multiple files to understand current architecture
Write new modules with proper integration points
Create configuration files for different environments
Manage test files and mock data
```

### 5. DDG Search - Research and Best Practices
**Purpose**: Web research for current best practices and troubleshooting
**When to Use**:
- Researching latest techniques in video processing
- Finding solutions to specific implementation challenges
- Discovering performance optimization strategies
- Staying updated with industry trends

**Usage Patterns**:
```
# Research specific techniques
Search: "frame by frame video analysis AI computer vision"
Search: "movis python video editing performance optimization"
Fetch URLs for detailed documentation and tutorials
Store findings in Memory for team reference
```

### 6. Browser - Interactive Research and Testing
**Purpose**: Interactive web browsing for complex research tasks
**When to Use**:
- Navigating complex documentation sites
- Testing web-based APIs or services
- Researching competitive analysis
- Validating implementation approaches

### 7. Time - Development Scheduling and Milestones
**Purpose**: Track development timeline and schedule milestones
**When to Use**:
- Setting realistic development timelines
- Tracking task completion rates
- Scheduling integration points
- Managing project deadlines

**Usage Patterns**:
```
# Track development milestones
Get current time for milestone tracking
Calculate relative time for task completion estimates
Schedule integration points and testing phases
```

### 8. Sequential Thinking - Complex Problem Solving
**Purpose**: Break down complex implementation challenges
**When to Use**:
- Planning complex integrations
- Solving architectural challenges
- Analyzing performance bottlenecks
- Making critical design decisions

**Usage Patterns**:
```
# Analyze complex problems
Use thinking tool for multi-step problem analysis
Break down integration challenges into manageable steps
Evaluate different architectural approaches
Document decision-making process
```

## Tool Integration Workflows

### Workflow 1: New Library Integration
1. **Context7**: Research library documentation and best practices
2. **Memory**: Store key insights and integration patterns
3. **Sequential Thinking**: Plan integration approach
4. **Filesystem**: Implement integration code
5. **GitHub**: Commit changes and track progress
6. **Memory**: Update implementation status and lessons learned

### Workflow 2: Performance Optimization
1. **DDG Search**: Research optimization techniques
2. **Context7**: Get specific library performance documentation
3. **Sequential Thinking**: Analyze bottlenecks and solutions
4. **Filesystem**: Implement optimizations
5. **Memory**: Store performance insights and benchmarks
6. **GitHub**: Track optimization improvements

### Workflow 3: API Integration
1. **Context7**: Get latest API documentation
2. **Memory**: Check for existing API patterns
3. **Sequential Thinking**: Plan integration strategy
4. **Filesystem**: Implement API client and error handling
5. **Memory**: Store API usage patterns and optimization opportunities
6. **GitHub**: Version control and testing integration

### Workflow 4: Complex Feature Implementation
1. **Memory**: Review current project state and dependencies
2. **Context7**: Research required libraries and techniques
3. **Sequential Thinking**: Break down feature into implementable tasks
4. **DDG Search**: Research best practices and examples
5. **Filesystem**: Implement feature with proper integration
6. **GitHub**: Track implementation progress
7. **Memory**: Update project state and architectural insights

## Quality Assurance with MCP Tools

### Code Quality
- **Filesystem**: Read existing code patterns for consistency
- **Context7**: Research coding best practices for libraries
- **Memory**: Store and reference successful implementation patterns

### Testing Strategy
- **Context7**: Research testing frameworks and mocking strategies
- **Memory**: Track testing patterns and coverage metrics
- **GitHub**: Manage test automation and continuous integration

### Performance Monitoring
- **Memory**: Track performance benchmarks and optimization insights
- **Time**: Monitor development velocity and bottlenecks
- **Sequential Thinking**: Analyze performance issues systematically

## Error Handling and Recovery

### When Things Go Wrong
1. **Memory**: Check for similar issues and solutions
2. **DDG Search**: Research error messages and solutions
3. **Context7**: Verify API documentation for changes
4. **Sequential Thinking**: Systematically analyze the problem
5. **Memory**: Store solution for future reference

### Continuous Improvement
- **Memory**: Regularly update project insights and patterns
- **GitHub**: Track improvements and refactoring efforts
- **Time**: Monitor development efficiency improvements
- **Sequential Thinking**: Regularly evaluate and optimize workflows

## Best Practices

### Tool Selection
- Always start with **Memory** to check existing knowledge
- Use **Context7** for authoritative documentation
- Use **DDG Search** for broader research and examples
- Use **Sequential Thinking** for complex decision-making

### Context Maintenance
- Update **Memory** after each significant milestone
- Store both successes and failures for learning
- Maintain relationships between project components
- Regular cleanup of outdated information

### Efficiency Optimization
- Batch similar research tasks using **Context7** and **DDG Search**
- Use **Memory** to avoid repeating research
- Leverage **Sequential Thinking** for complex planning sessions
- Maintain consistent **GitHub** workflow for all changes

This guide ensures optimal utilization of all available MCP tools throughout the AI Video Editor development process, maintaining context, optimizing workflow, and ensuring high-quality implementation.