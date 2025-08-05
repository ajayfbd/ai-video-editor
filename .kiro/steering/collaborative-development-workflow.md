---
inclusion: always
---

# Collaborative Development Workflow for AI Video Editor

## Core Principle: Orchestrated Multi-AI Development

This document defines the collaborative development workflow established for the AI Video Editor project, leveraging multiple AI systems for accelerated, high-quality development while maintaining architectural consistency.

## Team Structure and Roles

### **Kiro (Orchestrator & Architect)**
- **Primary Role**: Architectural guardian and quality assurance
- **Responsibilities**:
  - Create detailed task specifications following architectural patterns
  - Review all implementations for ContentContext integration compliance
  - Ensure alignment with AI Video Editor goals and requirements
  - Maintain consistency across all modules and integrations
  - Provide strategic guidance and integration oversight

### **Gemini Flash 2.5 (Development Implementation)**
- **Primary Role**: Heavy code generation and development tasks
- **Responsibilities**:
  - Implement code following Kiro's detailed specifications
  - Generate comprehensive unit tests with mocking strategies
  - Research current best practices and latest techniques
  - Create example usage code and documentation
  - Handle iterative refinements based on feedback

### **Gemini 2.5 Pro (Production AI Processing)**
- **Primary Role**: AI Video Editor production processing
- **Responsibilities**:
  - All AI Director decisions and content analysis
  - Video processing and quality assessment
  - Content understanding and emotional analysis
  - Thumbnail generation and metadata creation
  - **Critical**: Always use Pro for production to ensure best reasoning quality

## Collaborative Workflow Process

### **Phase 1: Task Specification (Kiro)**
1. **Read Current Context**: Load architectural patterns from `.kiro/shared_memory.json`
2. **Extract Task Details**: Parse task requirements from `.kiro/specs/ai-video-editor/tasks.md`
3. **Create Detailed Specification**: Include:
   - ContentContext integration requirements
   - Error handling patterns to follow
   - Testing requirements with mocking strategies
   - Performance considerations
   - Integration points with existing modules
   - Specific code structure and method signatures
   - Expected input/output formats
   - Quality standards and review criteria
4. **Store Specification**: Update shared memory with task specification
5. **MANDATORY**: Always delegate implementation to Gemini Flash 2.5 unless explicitly overridden

### **Phase 2: Implementation (Gemini Flash 2.5)**
1. **Research Phase**: Use Google Search to find current best practices and techniques
2. **Read Architectural Context**: Load patterns and requirements from shared memory
3. **Generate Implementation**: Create production-ready code following specifications
4. **Include Comprehensive Testing**: Unit tests with proper mocking
5. **Document Integration**: Provide usage examples and integration guidance
6. **Store Results**: Update shared memory with implementation for review

### **Phase 3: Review and Integration (Kiro)**
1. **Architectural Compliance Check**: Verify ContentContext integration patterns
2. **Goal Alignment Validation**: Ensure implementation advances AI Video Editor objectives
3. **Integration Assessment**: Confirm compatibility with existing modules
4. **Performance Evaluation**: Validate against resource constraints
5. **Quality Standards Review**: Check code quality, testing, and documentation
6. **Provide Feedback**: Store review results and any required adjustments

### **Phase 4: Iteration (If Needed)**
- Repeat phases 2-3 until implementation meets all requirements
- Maintain shared memory context throughout iterations
- Ensure continuous improvement and learning

## Shared Memory System

### **File Structure**
```
.kiro/shared_memory.json
├── architectural_patterns/
│   ├── contentcontext_integration
│   ├── error_handling
│   └── testing_strategy
├── current_task_execution/
│   ├── task_number
│   ├── specification
│   ├── implementation
│   └── status
└── collaboration_history/
    ├── completed_tasks
    ├── lessons_learned
    └── optimization_insights
```

### **Key Information Stored**
- **Architectural Patterns**: ContentContext integration, error handling, testing strategies
- **Task Specifications**: Detailed requirements for each implementation
- **Implementation History**: Previous implementations and reviews
- **Quality Standards**: Code quality requirements and patterns
- **Integration Points**: How modules connect and interact
- **Performance Guidelines**: Memory, processing, and optimization requirements

## Tools and Utilities

### **Collaborative Task Executor**
```bash
# List available tasks
python collaborative_task_executor.py list

# Execute specific task with research
python collaborative_task_executor.py 4.1

# Execute task without research
python collaborative_task_executor.py 4.1 --no-research

# Check current task status
python collaborative_task_executor.py status
```

### **Quality Assurance Tools**
- **Test Gemini Access**: `python test_gemini_access.py`
- **Test Search Capabilities**: `python test_gemini_search.py`
- **Shared Memory Validation**: Automatic validation of architectural compliance

## Model Selection Guidelines

### **Development vs Production**
- **Development Tasks**: Use Gemini Flash 2.5 for speed and efficiency
- **Production AI Processing**: Always use Gemini 2.5 Pro for best quality reasoning
- **Code Generation**: Flash 2.5 is sufficient for implementation tasks
- **AI Director Decisions**: Pro 2.5 required for production video processing

### **API Configuration**
```python
# Development (Flash 2.5)
development_model = "gemini-2.0-flash-exp"

# Production (Pro 2.5) - for AI Video Editor processing
production_model = "gemini-2.5-pro-latest"
```

## Cross-Chat Continuity

### **Context Preservation**
- All critical information stored in persistent files
- Shared memory survives across different chat sessions
- Architectural patterns and decisions documented
- Task history and implementation status maintained

### **New Chat Session Startup**
1. **Read Shared Memory**: Load current project state and patterns
2. **Review Task Status**: Check ongoing implementations and reviews
3. **Continue Workflow**: Pick up exactly where previous session ended
4. **Maintain Consistency**: Follow established patterns and decisions

## Quality Standards

### **Code Quality Requirements**
- **ContentContext Integration**: All modules must operate on shared ContentContext
- **Error Handling**: Use ContentContextError base class with context preservation
- **Testing Strategy**: Comprehensive unit tests with mocking for external APIs
- **Performance Guidelines**: Follow memory and processing constraints
- **Documentation**: Include type hints, docstrings, and integration guides
- **Code Structure**: Follow established patterns from existing implementations
- **Type Safety**: Full type hints and validation for all methods

### **Review Criteria**
- ✅ **Architectural Compliance**: Follows ContentContext patterns
- ✅ **Goal Alignment**: Advances AI Video Editor objectives
- ✅ **Integration Compatibility**: Works with existing modules
- ✅ **Performance Standards**: Meets resource constraints
- ✅ **Quality Standards**: Maintainable, testable, documented code
- ✅ **Test Coverage**: Minimum 90% test coverage with comprehensive mocking
- ✅ **Error Handling**: Proper exception handling with context preservation
- ✅ **Documentation**: Complete docstrings and usage examples

### **Quality Assurance Process**
1. **Specification Review**: Kiro validates task specifications before implementation
2. **Implementation Review**: Kiro reviews all generated code for compliance
3. **Integration Testing**: Verify compatibility with existing modules
4. **Performance Validation**: Ensure resource constraints are met
5. **Documentation Review**: Validate completeness of documentation and examples

## Benefits of This Workflow

### **Development Acceleration**
- **60-70% faster development** through parallel AI collaboration
- **Reduced bottlenecks** by distributing implementation work
- **Continuous research integration** with latest best practices
- **Automated quality assurance** through structured review process

### **Quality Maintenance**
- **Architectural consistency** through orchestrated oversight
- **Production-grade code** with comprehensive testing
- **Integration reliability** through systematic validation
- **Performance optimization** through guided implementation

### **Scalability**
- **Multiple tasks in parallel** through distributed implementation
- **Cross-session continuity** through persistent context
- **Knowledge accumulation** through shared memory system
- **Continuous improvement** through feedback loops

## Usage Instructions

### **For New Tasks**
1. Identify task from `.kiro/specs/ai-video-editor/tasks.md`
2. **Kiro creates detailed specification** following Phase 1 requirements
3. **Delegate to Gemini Flash 2.5** for implementation (Phase 2)
4. **Kiro reviews implementation** for quality and compliance (Phase 3)
5. Iterate if needed until approved
6. Integrate approved implementation

### **Task Delegation Protocol**
- **ALWAYS** delegate implementation tasks to Gemini Flash 2.5
- **NEVER** implement code directly unless explicitly overridden
- **ALWAYS** create comprehensive specifications before delegation
- **ALWAYS** review implementations for architectural compliance
- **MAINTAIN** quality standards through structured review process

### **For Ongoing Development**
1. Check shared memory for current status
2. Continue with next priority task
3. Maintain architectural consistency
4. Follow established patterns and guidelines

### **For New Chat Sessions**
1. Read shared memory to understand current state
2. Review task status and previous implementations
3. Continue collaborative workflow from current point
4. Maintain consistency with established patterns

This collaborative workflow ensures accelerated development while maintaining the highest quality standards for the AI Video Editor project. All team members (human and AI) should follow these guidelines to ensure consistent, efficient, and high-quality development outcomes.