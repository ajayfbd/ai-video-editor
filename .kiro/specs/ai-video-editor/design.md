# Design Document

## Overview

The AI-Assisted Video Editor is a content creation system where all creative and technical decisions are driven by an AI director. This AI emulates the expertise of a professional video editor, content strategist, and SEO specialist to transform raw footage into engaging, discoverable, and publish-ready content packages. The system is built on a unified ContentContext architecture that ensures deep integration between all processing modules, allowing the AI's vision to be executed seamlessly.

### Core Innovation: AI Director & ContentContext

The system's central innovation is the combination of an **AI Director** (powered by the Gemini API) and the **ContentContext** object. The AI Director analyzes the content to make nuanced decisions about editing, B-roll, and metadata. The ContentContext acts as the "director's notes," flowing through all modules to ensure every component executes on the same unified vision. This ensures that thumbnail hook text, YouTube titles, video cuts, and animated B-roll all derive from the same core creative and strategic decisions.

### Development Philosophy

**AI-First**: The system is not merely a set of tools, but an AI-driven director. Every module is designed to interpret and execute the AI's instructions as stored in the ContentContext.

**Integration through Context**: Every module contributes to and benefits from the shared ContentContext, ensuring a cohesive and intelligently generated final product.

**Quality Through Testing**: Comprehensive unit testing with sophisticated mocking ensures reliability without requiring actual video processing in tests.

**Performance Optimization**: Intelligent caching, batch processing, and parallel execution optimize both performance and API costs.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLI Interface Layer                          │
├─────────────────────────────────────────────────────────────────┤
│                  Workflow Orchestrator                          │
│                 (ContentContext Manager)                        │
├─────────────────────────────────────────────────────────────────┤
│  Input Processing  │  Intelligence Layer │  Output Generation   │
│     Module         │      Module         │      Module          │
├─────────────────────────────────────────────────────────────────┤
│  Audio/Video       │  Keyword Research   │  Integrated          │
│  Analysis          │  Trend Analysis     │  Thumbnail+Metadata  │
├─────────────────────────────────────────────────────────────────┤
│                    ContentContext Storage                       │
├─────────────────────────────────────────────────────────────────┤
│  Local Libraries   │  Cloud Services     │  Caching Layer       │
│  (movis, Blender)  │  (Gemini, Imagen)   │  (Redis/Memory)      │
│   (OpenCV, PyVips) │                     │                      │
└─────────────────────────────────────────────────────────────────┘
```

### ContentContext Flow

```
Input Files → Content Analysis → Intelligence Layer → Parallel Output Generation
     ↓              ↓                    ↓                    ↓
Video/Audio → Concepts/Emotions → Keywords/Trends → Thumbnails + Metadata
     ↓              ↓                    ↓                    ↓
ContentContext → ContentContext → ContentContext → Final Assets
```

## Components and Interfaces

### 1. ContentContext System (Core)

**Purpose**: Central data structure that flows through all modules, enabling deep integration. For detailed data models, see `implementation-details.md`.

### 2. Input Processing Module

**Purpose**: To deconstruct raw video files into a rich, multi-modal `ContentContext` object. This module acts as the AI Director's senses, providing the structured data needed for analysis and decision-making. For a detailed breakdown of the classes and methods (`FinancialContentAnalyzer`, etc.), see `implementation-details.md`.

### 3. Intelligence Layer Module

**Purpose**: To act as the **AI Director**. This module takes the comprehensive `ContentContext` and uses the Gemini API to generate a complete creative and strategic plan for the video. For detailed prompt engineering strategies and class design (`FinancialVideoEditor`), see `implementation-details.md`.

### 4. Output Generation & Composition Module

**Purpose**: To execute the AI Director's plan, generating all the necessary assets and compositing them into the final video product.

**Key Components**:
- **Graphics Generator (`Matplotlib`, `python-pptx`)**: Programmatically creates charts, graphs, and slides as directed by the AI.
- **Animation Engine (`Blender`)**: Renders character animations based on the AI's script.
- **Composition Engine (`movis`)**: The core video engine that assembles the main footage, B-roll, graphics, and audio into a single, polished video file. For more details, see the `movis` documentation: [https://rezoo.github.io/movis/reference/index.html](https://rezoo.github.io/movis/reference/index.html)
- **Thumbnail Generator**: Creates high-CTR thumbnails based on the AI's analysis. For the detailed strategy, see `implementation-details.md`.

### 5. Error Handling and Recovery

This project uses a robust error handling strategy centered on `ContentContext` preservation. For detailed patterns and graceful degradation strategies, see the `error-handling-patterns.md` document in the `.kiro/steering` directory.

### 6. Performance

Performance is critical. For detailed guidelines on resource management, API cost optimization, and performance targets, see the `performance-guidelines.md` document in the `.kiro/steering` directory.

### 7. Testing

All development will be guided by a comprehensive testing strategy focused on unit tests with extensive mocking. For details on mocking strategies and coverage targets, see the `testing-strategy.md` document in the `.kiro/steering` directory.
