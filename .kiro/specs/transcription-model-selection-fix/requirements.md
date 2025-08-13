# Requirements Document

## Introduction

The AI Video Editor's transcription CLI needs enhancements to provide better user experience and accuracy for Sanskrit/Hindi content. Current issues include: automatic model downgrading on CPU, lack of Sanskrit vocabulary support for better word prediction, missing progress indication during transcription, and overly complex command syntax that could be simplified for common use cases.

## Requirements

### Requirement 1

**User Story:** As a user with sufficient CPU resources, I want to use the large Whisper model for transcription even on CPU, so that I can get the highest accuracy transcription possible regardless of processing time.

#### Acceptance Criteria

1. WHEN a user specifies `--model large` with `--device cpu` THEN the system SHALL use the large model without automatic downgrading
2. WHEN a user specifies `--model medium` with `--device cpu` THEN the system SHALL use the medium model without automatic downgrading
3. WHEN automatic model downgrading occurs THEN the system SHALL log a warning message explaining the downgrade
4. WHEN a user wants to override automatic downgrading THEN the system SHALL provide a `--force-model` flag to respect the user's model choice

### Requirement 2

**User Story:** As a user who wants optimal performance, I want the sytem to provide intelligent model selection rsecommendations, so that I can make informed decisions about accuracy vs speed trade-offs.

#### Acceptance Criteria

1. WHEN a user specifies a large model on CPU THEN the system SHALL display a warning about expected processing time
2. WHEN the system detects limited resources THEN it SHALL suggest alternative model sizes with expected performance characteristics
3. WHEN a user runs transcription THEN the system SHALL display estimated processing time based on model size and device
4. WHEN transcription completes THEN the system SHALL report actual processing time and model used

### Requirement 3

**User Story:** As a user processing Hindi/multilingual content, I want consistent model behavior across different languages, so that I get reliable transcription quality regardless of the detected language.

#### Acceptance Criteria

1. WHEN processing Hindi content with large model THEN the system SHALL maintain model size consistency
2. WHEN romanization is enabled THEN the model selection SHALL not affect the romanization quality
3. WHEN language is auto-detected THEN the model downgrading logic SHALL not interfere with language detection accuracy
4. WHEN using initial prompts for Hindi content THEN the large model SHALL be available to better utilize the prompt context

### Requirement 4

**User Story:** As a developer integrating the transcription system, I want clear control over model selection behavior, so that I can programmatically choose between performance and accuracy based on my application's needs.

#### Acceptance Criteria

1. WHEN calling transcription programmatically THEN the API SHALL provide explicit model selection control
2. WHEN model downgrading occurs THEN the API SHALL return information about the actual model used
3. WHEN resource constraints are detected THEN the API SHALL provide callbacks or options for handling the situation
4. WHEN batch processing multiple files THEN the system SHALL maintain consistent model selection across all files

### Requirement 5

**User Story:** As a user transcribing Sanskrit/Hindi religious or classical content, I want to provide Sanskrit vocabulary hints to improve word prediction accuracy, so that technical terms, deity names, and classical concepts are transcribed correctly.

#### Acceptance Criteria

1. WHEN processing Sanskrit/Hindi content THEN the system SHALL accept a vocabulary file or word list to bias recognition
2. WHEN Sanskrit vocabulary is provided THEN the system SHALL use it as initial prompt context for better word prediction
3. WHEN religious or classical terms are detected THEN the system SHALL prioritize Sanskrit vocabulary matches over common words
4. WHEN vocabulary hints are used THEN the system SHALL log which vocabulary source was applied
5. WHEN no vocabulary is provided THEN the system SHALL use built-in Sanskrit/Hindi religious vocabulary as fallback

### Requirement 6

**User Story:** As a user running long transcription jobs, I want to see real-time progress indication, so that I know the system is working and can estimate completion time.

#### Acceptance Criteria

1. WHEN transcription starts THEN the system SHALL display a progress bar showing percentage completion
2. WHEN processing segments THEN the system SHALL update progress based on audio duration processed
3. WHEN transcription is running THEN the system SHALL show elapsed time and estimated time remaining
4. WHEN transcription completes THEN the system SHALL display total processing time and final statistics
5. WHEN progress display is not desired THEN the system SHALL provide a `--quiet` flag to suppress progress output

### Requirement 7

**User Story:** As a user who frequently transcribes similar content, I want simplified command syntax with smart defaults, so that I can run transcription with minimal typing and configuration.

#### Acceptance Criteria

1. WHEN transcribing Hindi content THEN the system SHALL auto-detect language and apply appropriate settings
2. WHEN no model is specified THEN the system SHALL choose optimal model based on available resources
3. WHEN output path is not specified THEN the system SHALL generate output filename based on input filename
4. WHEN common use cases are detected THEN the system SHALL provide preset configurations (e.g., `--preset hindi-religious`)
5. WHEN simplified command is used THEN the system SHALL display what settings were automatically applied