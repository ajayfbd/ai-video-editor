"""
Audio Analysis Module - Whisper Integration for Financial Content Analysis.

This module implements the FinancialContentAnalyzer class that uses OpenAI Whisper
for audio transcription and provides specialized analysis for financial educational content.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

import whisper
import torch
import numpy as np
from whisper.normalizers import EnglishTextNormalizer

from ai_video_editor.core.content_context import ContentContext, EmotionalPeak
from ai_video_editor.core.cache_manager import CacheManager, cached
from ai_video_editor.core.exceptions import (
    ProcessingError, 
    ResourceConstraintError, 
    MemoryConstraintError,
    ContentContextError,
    handle_errors,
    retry_on_error
)
from ai_video_editor.utils.logging_config import get_logger


@dataclass
class TranscriptSegment:
    """Represents a segment of transcribed audio with timing and confidence."""
    text: str
    start: float
    end: float
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptSegment':
        return cls(**data)


@dataclass
class Transcript:
    """Complete transcript with segments and metadata."""
    text: str
    segments: List[TranscriptSegment]
    confidence: float
    language: str
    processing_time: float = 0.0
    model_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'segments': [segment.to_dict() for segment in self.segments],
            'confidence': self.confidence,
            'language': self.language,
            'processing_time': self.processing_time,
            'model_used': self.model_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transcript':
        segments = [TranscriptSegment.from_dict(seg) for seg in data.get('segments', [])]
        return cls(
            text=data['text'],
            segments=segments,
            confidence=data['confidence'],
            language=data['language'],
            processing_time=data.get('processing_time', 0.0),
            model_used=data.get('model_used', '')
        )


@dataclass
class FillerWordSegment:
    """Represents a segment with detected filler words."""
    timestamp: float
    text: str
    original_text: str
    filler_words: List[str]
    confidence: float
    should_remove: bool
    cleaned_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'text': self.text,
            'original_text': self.original_text,
            'filler_words': self.filler_words,
            'confidence': self.confidence,
            'should_remove': self.should_remove,
            'cleaned_text': self.cleaned_text
        }


@dataclass
class AudioEnhancementResult:
    """Results of audio enhancement processing."""
    original_duration: float
    enhanced_duration: float
    filler_words_removed: int
    segments_modified: int
    quality_improvement_score: float
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_duration': self.original_duration,
            'enhanced_duration': self.enhanced_duration,
            'filler_words_removed': self.filler_words_removed,
            'segments_modified': self.segments_modified,
            'quality_improvement_score': self.quality_improvement_score,
            'processing_time': self.processing_time
        }


@dataclass
class FinancialAnalysisResult:
    """Results of financial content analysis."""
    concepts_mentioned: List[str] = field(default_factory=list)
    explanation_segments: List[Dict[str, Any]] = field(default_factory=list)
    data_references: List[Dict[str, Any]] = field(default_factory=list)
    complexity_level: str = "medium"
    filler_words_detected: List[FillerWordSegment] = field(default_factory=list)
    emotional_peaks: List[EmotionalPeak] = field(default_factory=list)
    audio_enhancement: Optional[AudioEnhancementResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'concepts_mentioned': self.concepts_mentioned,
            'explanation_segments': self.explanation_segments,
            'data_references': self.data_references,
            'complexity_level': self.complexity_level,
            'filler_words_detected': [fw.to_dict() for fw in self.filler_words_detected],
            'emotional_peaks': [peak.to_dict() for peak in self.emotional_peaks],
            'audio_enhancement': self.audio_enhancement.to_dict() if self.audio_enhancement else None
        }


class FinancialContentAnalyzer:
    """
    Advanced audio analyzer specialized for financial educational content.
    
    Uses OpenAI Whisper for transcription and provides financial content-specific
    analysis including keyword detection, explanation segment identification,
    and filler word removal.
    """
    
    # Financial keywords for content analysis
    FINANCIAL_KEYWORDS = [
        'investment', 'portfolio', 'stocks', 'bonds', 'returns', 'risk',
        'diversification', 'compound interest', 'inflation', 'budgeting',
        'savings', 'debt', 'credit', 'retirement', 'taxes', 'dividend',
        'asset', 'liability', 'equity', 'cash flow', 'roi', 'yield',
        'market', 'trading', 'broker', 'fund', 'etf', 'mutual fund',
        'insurance', 'mortgage', 'loan', 'interest rate', 'apr',
        'financial planning', 'wealth', 'income', 'expense', 'profit',
        'loss', 'capital', 'valuation', 'analysis', 'strategy'
    ]
    
    # Common filler words to detect and potentially remove
    FILLER_WORDS = [
        'um', 'uh', 'ah', 'er', 'like', 'you know', 'so', 'well',
        'actually', 'basically', 'literally', 'right', 'okay', 'alright'
    ]
    
    # Explanation trigger words
    EXPLANATION_TRIGGERS = [
        'explain', 'means', 'definition', 'example', 'for instance',
        'in other words', 'that is', 'specifically', 'essentially',
        'basically', 'simply put', 'to clarify', 'what this means'
    ]
    
    # Data reference indicators
    DATA_INDICATORS = [
        'chart', 'graph', 'data', 'percentage', 'number', 'statistic',
        'figure', 'table', 'report', 'study', 'research', 'analysis',
        'metric', 'measurement', 'calculation', 'result'
    ]
    
    def __init__(self, cache_dir: Optional[str] = None, cache_manager: Optional[CacheManager] = None):
        """
        Initialize the FinancialContentAnalyzer.
        
        Args:
            cache_dir: Directory for caching models and results
            cache_manager: CacheManager instance for expensive operations
        """
        self.logger = get_logger(__name__)
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'whisper'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache manager
        self.cache_manager = cache_manager or CacheManager(
            cache_dir=str(self.cache_dir.parent / "audio_analysis_cache")
        )
        
        # Model cache
        self._models: Dict[str, whisper.Whisper] = {}
        self.normalizer = EnglishTextNormalizer()
        
        # Quality-focused model preferences (larger models for better accuracy)
        self.model_preferences = {
            'fast': 'base',
            'balanced': 'medium', 
            'high': 'large',
            'turbo': 'turbo'  # Latest and fastest large model
        }
        
        # Enhanced filler word patterns with context awareness
        self.filler_patterns = {
            'basic_fillers': ['um', 'uh', 'ah', 'er', 'hmm'],
            'discourse_markers': ['like', 'you know', 'so', 'well', 'actually', 'basically'],
            'hesitation_markers': ['literally', 'right', 'okay', 'alright', 'I mean'],
            'repetitive_phrases': ['sort of', 'kind of', 'you see', 'as I said']
        }
        
        # Emotional analysis patterns
        self.emotional_patterns = {
            'excitement': {
                'words': ['amazing', 'incredible', 'fantastic', 'wow', 'great', 'excellent', 'outstanding'],
                'intensity_multipliers': {'amazing': 1.0, 'incredible': 0.9, 'fantastic': 0.8, 'wow': 0.7}
            },
            'concern': {
                'words': ['careful', 'warning', 'risk', 'danger', 'problem', 'issue', 'trouble'],
                'intensity_multipliers': {'danger': 1.0, 'warning': 0.8, 'careful': 0.6}
            },
            'curiosity': {
                'words': ['interesting', 'wonder', 'question', 'why', 'how', 'what if', 'curious'],
                'intensity_multipliers': {'wonder': 0.8, 'interesting': 0.6, 'curious': 0.7}
            },
            'confidence': {
                'words': ['definitely', 'certainly', 'absolutely', 'sure', 'confident', 'guaranteed'],
                'intensity_multipliers': {'absolutely': 1.0, 'definitely': 0.9, 'certainly': 0.8}
            }
        }
        
        self.logger.info("FinancialContentAnalyzer initialized with cache manager")
    
    @handle_errors()
    def get_model(self, model_size: str = 'medium') -> whisper.Whisper:
        """
        Load and cache Whisper model.
        
        Args:
            model_size: Size of the model ('tiny', 'base', 'small', 'medium', 'large', 'turbo')
            
        Returns:
            Loaded Whisper model
            
        Raises:
            ResourceConstraintError: If insufficient memory for model
            ProcessingError: If model loading fails
        """
        if model_size not in self._models:
            try:
                self.logger.info(f"Loading Whisper model: {model_size}")
                start_time = time.time()
                
                # Check available memory before loading large models
                if model_size in ['large', 'turbo'] and torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    if gpu_memory < 4_000_000_000:  # Less than 4GB GPU memory
                        self.logger.warning(f"Limited GPU memory, falling back to medium model")
                        model_size = 'medium'
                
                model = whisper.load_model(model_size, download_root=str(self.cache_dir))
                load_time = time.time() - start_time
                
                self._models[model_size] = model
                self.logger.info(f"Model {model_size} loaded in {load_time:.2f}s")
                
                # Log model info
                param_count = sum(np.prod(p.shape) for p in model.parameters())
                self.logger.info(
                    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
                    f"and has {param_count:,} parameters"
                )
                
            except Exception as e:
                raise ProcessingError(
                    f"Failed to load Whisper model {model_size}",
                    reason=str(e)
                )
        
        return self._models[model_size]
    
    @handle_errors()
    @retry_on_error(max_retries=2, delay=1.0)
    def transcribe_audio(
        self, 
        audio_path: str, 
        model_size: str = 'medium',
        language: Optional[str] = None,
        use_cache: bool = True
    ) -> Transcript:
        """
        Transcribe audio file using Whisper with caching support.
        
        Args:
            audio_path: Path to audio file
            model_size: Whisper model size to use
            language: Language code (auto-detect if None)
            use_cache: Whether to use cached results
            
        Returns:
            Transcript object with segments and metadata
            
        Raises:
            ProcessingError: If transcription fails
            ResourceConstraintError: If insufficient resources
        """
        if not os.path.exists(audio_path):
            raise ProcessingError(f"Audio file not found: {audio_path}")
        
        # Check cache first if enabled
        if use_cache:
            cache_key = self.cache_manager._generate_key(
                "transcription", 
                audio_path=audio_path, 
                model_size=model_size, 
                language=language
            )
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                self.logger.info(f"Using cached transcription for: {audio_path}")
                return Transcript.from_dict(cached_result)
        
        try:
            model = self.get_model(model_size)
            
            self.logger.info(f"Transcribing audio: {audio_path}")
            start_time = time.time()
            
            # Transcribe with word-level timestamps
            result = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                verbose=False
            )
            
            processing_time = time.time() - start_time
            
            # Convert to our format
            segments = []
            for segment in result.get('segments', []):
                segments.append(TranscriptSegment(
                    text=segment['text'].strip(),
                    start=segment['start'],
                    end=segment['end'],
                    confidence=segment.get('avg_logprob', 0.0)
                ))
            
            transcript = Transcript(
                text=result['text'].strip(),
                segments=segments,
                confidence=np.mean([seg.confidence for seg in segments]) if segments else 0.0,
                language=result.get('language', 'unknown'),
                processing_time=processing_time,
                model_used=model_size
            )
            
            self.logger.info(
                f"Transcription completed in {processing_time:.2f}s, "
                f"confidence: {transcript.confidence:.3f}"
            )
            
            # Cache the result if caching is enabled
            if use_cache:
                cache_key = self.cache_manager._generate_key(
                    "transcription", 
                    audio_path=audio_path, 
                    model_size=model_size, 
                    language=language
                )
                self.cache_manager.put(
                    cache_key, 
                    transcript.to_dict(), 
                    ttl=86400,  # Cache for 24 hours
                    tags=["transcription", f"model:{model_size}"]
                )
            
            return transcript
            
        except Exception as e:
            raise ProcessingError(
                f"Audio transcription failed for {audio_path}",
                reason=str(e)
            )
    
    @handle_errors()
    def detect_and_analyze_filler_words(self, transcript: Transcript) -> List[FillerWordSegment]:
        """
        Advanced filler word detection with context awareness and removal suggestions.
        
        Args:
            transcript: Transcript to analyze
            
        Returns:
            List of FillerWordSegment objects with detailed analysis
        """
        self.logger.info("Detecting filler words with context awareness")
        
        filler_segments = []
        
        for segment in transcript.segments:
            original_text = segment.text
            text_lower = original_text.lower().strip()
            
            # Skip very short segments
            if len(text_lower.split()) < 2:
                continue
            
            detected_fillers = []
            cleaned_words = []
            
            words = text_lower.split()
            
            for word in words:
                word_clean = word.strip('.,!?;:')
                is_filler = False
                
                # Check against all filler patterns
                for category, fillers in self.filler_patterns.items():
                    if word_clean in fillers:
                        detected_fillers.append(word_clean)
                        is_filler = True
                        break
                
                # Keep non-filler words
                if not is_filler:
                    cleaned_words.append(word)
            
            if detected_fillers:
                cleaned_text = ' '.join(cleaned_words).strip()
                
                # Determine if segment should be removed based on filler density
                filler_density = len(detected_fillers) / len(words)
                should_remove = filler_density > 0.4 or len(detected_fillers) > 3
                
                # Don't remove if it would make the text too short or meaningless
                if len(cleaned_text.split()) < 2:
                    should_remove = False
                    cleaned_text = original_text  # Keep original if cleaning makes it too short
                
                filler_segment = FillerWordSegment(
                    timestamp=segment.start,
                    text=original_text,
                    original_text=original_text,
                    filler_words=detected_fillers,
                    confidence=segment.confidence,
                    should_remove=should_remove,
                    cleaned_text=cleaned_text
                )
                
                filler_segments.append(filler_segment)
        
        self.logger.info(f"Detected {len(filler_segments)} segments with filler words")
        return filler_segments
    
    @handle_errors()
    def enhance_audio_content(self, transcript: Transcript, remove_fillers: bool = True) -> Tuple[Transcript, AudioEnhancementResult]:
        """
        Enhance audio content by removing filler words and improving flow.
        
        Args:
            transcript: Original transcript
            remove_fillers: Whether to remove detected filler words
            
        Returns:
            Tuple of enhanced transcript and enhancement results
        """
        self.logger.info("Enhancing audio content")
        start_time = time.time()
        
        # Detect filler words
        filler_segments = self.detect_and_analyze_filler_words(transcript)
        
        if not remove_fillers:
            # Return original transcript with analysis
            enhancement_result = AudioEnhancementResult(
                original_duration=transcript.segments[-1].end if transcript.segments else 0.0,
                enhanced_duration=transcript.segments[-1].end if transcript.segments else 0.0,
                filler_words_removed=0,
                segments_modified=0,
                quality_improvement_score=0.0,
                processing_time=time.time() - start_time
            )
            return transcript, enhancement_result
        
        # Create enhanced segments
        enhanced_segments = []
        filler_words_removed = 0
        segments_modified = 0
        time_saved = 0.0
        
        # Create lookup for filler segments
        filler_lookup = {fs.timestamp: fs for fs in filler_segments}
        
        for segment in transcript.segments:
            if segment.start in filler_lookup:
                filler_seg = filler_lookup[segment.start]
                
                if filler_seg.should_remove and filler_seg.cleaned_text.strip():
                    # Use cleaned text
                    enhanced_segment = TranscriptSegment(
                        text=filler_seg.cleaned_text,
                        start=segment.start,
                        end=segment.end,
                        confidence=segment.confidence
                    )
                    enhanced_segments.append(enhanced_segment)
                    
                    filler_words_removed += len(filler_seg.filler_words)
                    segments_modified += 1
                    
                    # Estimate time saved (rough approximation)
                    words_removed = len(filler_seg.filler_words)
                    time_saved += words_removed * 0.5  # Assume 0.5 seconds per filler word
                
                elif not filler_seg.should_remove:
                    # Keep original segment
                    enhanced_segments.append(segment)
            else:
                # Keep original segment
                enhanced_segments.append(segment)
        
        # Create enhanced transcript
        enhanced_text = ' '.join([seg.text for seg in enhanced_segments])
        enhanced_transcript = Transcript(
            text=enhanced_text,
            segments=enhanced_segments,
            confidence=transcript.confidence,
            language=transcript.language,
            processing_time=transcript.processing_time,
            model_used=transcript.model_used
        )
        
        # Calculate quality improvement score
        original_duration = transcript.segments[-1].end if transcript.segments else 0.0
        enhanced_duration = max(original_duration - time_saved, 0.0)
        
        quality_improvement_score = min(
            (filler_words_removed * 0.1) + (segments_modified * 0.05), 
            1.0
        )
        
        enhancement_result = AudioEnhancementResult(
            original_duration=original_duration,
            enhanced_duration=enhanced_duration,
            filler_words_removed=filler_words_removed,
            segments_modified=segments_modified,
            quality_improvement_score=quality_improvement_score,
            processing_time=time.time() - start_time
        )
        
        self.logger.info(
            f"Audio enhancement complete: {filler_words_removed} filler words removed, "
            f"{segments_modified} segments modified, {time_saved:.1f}s saved"
        )
        
        return enhanced_transcript, enhancement_result
    
    @handle_errors()
    def analyze_financial_content(self, transcript: Transcript, enhance_audio: bool = True) -> FinancialAnalysisResult:
        """
        Analyze transcript for financial content-specific insights with audio enhancement.
        
        Args:
            transcript: Transcript to analyze
            enhance_audio: Whether to perform audio enhancement
            
        Returns:
            FinancialAnalysisResult with detected concepts and segments
        """
        self.logger.info("Analyzing financial content with enhancement")
        
        result = FinancialAnalysisResult()
        
        # Perform audio enhancement if requested
        working_transcript = transcript
        if enhance_audio:
            enhanced_transcript, enhancement_result = self.enhance_audio_content(transcript)
            working_transcript = enhanced_transcript
            result.audio_enhancement = enhancement_result
            
            # Also get detailed filler word analysis
            result.filler_words_detected = self.detect_and_analyze_filler_words(transcript)
        else:
            # Basic filler word detection for compatibility
            result.filler_words_detected = self.detect_and_analyze_filler_words(transcript)
        
        text_lower = working_transcript.text.lower()
        
        # Identify financial concepts mentioned
        for keyword in self.FINANCIAL_KEYWORDS:
            if keyword.lower() in text_lower:
                result.concepts_mentioned.append(keyword)
        
        # Analyze segments for specific patterns
        for segment in working_transcript.segments:
            segment_text = segment.text.lower()
            
            # Identify explanation segments
            if any(trigger in segment_text for trigger in self.EXPLANATION_TRIGGERS):
                result.explanation_segments.append({
                    'timestamp': segment.start,
                    'text': segment.text,
                    'type': 'explanation',
                    'confidence': segment.confidence
                })
            
            # Identify data references
            if any(indicator in segment_text for indicator in self.DATA_INDICATORS):
                result.data_references.append({
                    'timestamp': segment.start,
                    'text': segment.text,
                    'requires_visual': True,
                    'confidence': segment.confidence
                })
        
        # Assess complexity level
        result.complexity_level = self._assess_complexity(working_transcript.text)
        
        # Detect emotional peaks based on enhanced speech patterns
        result.emotional_peaks = self._detect_enhanced_emotional_peaks(working_transcript)
        
        self.logger.info(
            f"Financial analysis complete: {len(result.concepts_mentioned)} concepts, "
            f"{len(result.explanation_segments)} explanations, "
            f"{len(result.data_references)} data references, "
            f"{len(result.filler_words_detected)} filler segments detected"
        )
        
        return result
    
    @handle_errors()
    def analyze_multi_clip_project(
        self, 
        clip_paths: List[str], 
        project_context: Dict[str, Any],
        model_size: str = 'medium'
    ) -> Dict[str, Any]:
        """
        Analyze multiple clips for cohesive financial education video.
        
        Args:
            clip_paths: List of audio/video file paths
            project_context: Context about the overall project
            model_size: Whisper model size to use
            
        Returns:
            Comprehensive analysis of all clips with cross-clip insights
        """
        self.logger.info(f"Analyzing multi-clip project with {len(clip_paths)} clips")
        
        project_analysis = {
            'clips': [],
            'global_context': project_context,
            'content_flow': [],
            'key_concepts': [],
            'retention_hooks': [],
            'engagement_points': [],
            'total_processing_time': 0.0
        }
        
        start_time = time.time()
        
        # Analyze each clip individually
        for i, clip_path in enumerate(clip_paths):
            try:
                self.logger.info(f"Processing clip {i+1}/{len(clip_paths)}: {clip_path}")
                
                # Transcribe audio
                transcript = self.transcribe_audio(clip_path, model_size)
                
                # Analyze financial content
                financial_analysis = self.analyze_financial_content(transcript)
                
                clip_analysis = {
                    'clip_order': i,
                    'clip_path': clip_path,
                    'transcript': transcript.to_dict(),
                    'financial_analysis': financial_analysis.to_dict(),
                    'duration': transcript.segments[-1].end if transcript.segments else 0.0
                }
                
                project_analysis['clips'].append(clip_analysis)
                
            except Exception as e:
                self.logger.error(f"Failed to process clip {clip_path}: {e}")
                # Continue with other clips
                continue
        
        # Generate cross-clip insights
        if project_analysis['clips']:
            project_analysis['content_flow'] = self._analyze_content_flow(project_analysis['clips'])
            project_analysis['key_concepts'] = self._extract_global_concepts(project_analysis['clips'])
            project_analysis['retention_strategy'] = self._plan_retention_strategy(project_analysis)
        
        project_analysis['total_processing_time'] = time.time() - start_time
        
        self.logger.info(
            f"Multi-clip analysis complete in {project_analysis['total_processing_time']:.2f}s"
        )
        
        return project_analysis
    
    @handle_errors()
    def process_batch_audio_files(
        self, 
        audio_files: List[str], 
        model_size: str = 'medium',
        max_parallel: int = 2,
        use_cache: bool = True
    ) -> List[Transcript]:
        """
        Process multiple audio files in batch with memory management and caching.
        
        Args:
            audio_files: List of audio file paths
            model_size: Whisper model size to use
            max_parallel: Maximum parallel processing (limited for memory)
            use_cache: Whether to use cached results
            
        Returns:
            List of Transcript objects
        """
        self.logger.info(f"Batch processing {len(audio_files)} audio files")
        
        transcripts = []
        
        # Process in smaller batches to manage memory
        batch_size = min(max_parallel, 3)  # Conservative batch size
        
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
            
            batch_transcripts = []
            for audio_file in batch:
                try:
                    transcript = self.transcribe_audio(audio_file, model_size, use_cache=use_cache)
                    batch_transcripts.append(transcript)
                except Exception as e:
                    self.logger.error(f"Failed to process {audio_file}: {e}")
                    # Create empty transcript to maintain order
                    batch_transcripts.append(Transcript(
                        text="", segments=[], confidence=0.0, 
                        language="unknown", model_used=model_size
                    ))
            
            transcripts.extend(batch_transcripts)
            
            # Memory cleanup between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.logger.info(f"Batch processing complete: {len(transcripts)} transcripts")
        return transcripts
    
    @handle_errors()
    def integrate_with_content_context(
        self, 
        context: ContentContext, 
        transcript: Transcript,
        financial_analysis: FinancialAnalysisResult
    ) -> ContentContext:
        """
        Integrate transcription and analysis results into ContentContext.
        
        Args:
            context: ContentContext to update
            transcript: Transcription results
            financial_analysis: Financial content analysis results
            
        Returns:
            Updated ContentContext
        """
        self.logger.info("Integrating results with ContentContext")
        
        # Store transcript
        context.audio_transcript = transcript.text
        
        # Add key concepts
        context.key_concepts.extend(financial_analysis.concepts_mentioned)
        context.key_concepts = list(set(context.key_concepts))  # Remove duplicates
        
        # Add emotional markers from analysis
        for peak in financial_analysis.emotional_peaks:
            context.add_emotional_marker(
                timestamp=peak.timestamp,
                emotion=peak.emotion,
                intensity=peak.intensity,
                confidence=peak.confidence,
                context=peak.context
            )
        
        # Update processing metrics
        context.processing_metrics.add_module_metrics(
            module_name="audio_analysis",
            processing_time=transcript.processing_time,
            memory_used=0  # Would need actual memory tracking
        )
        
        # Store detailed analysis in metadata
        if not context.video_metadata:
            context.video_metadata = {}
        
        context.video_metadata['audio_analysis'] = {
            'transcript': transcript.to_dict(),
            'financial_analysis': financial_analysis.to_dict(),
            'model_used': transcript.model_used,
            'processing_timestamp': time.time()
        }
        
        context.update_processing_stage("audio_analysis_complete")
        
        self.logger.info("ContentContext integration complete")
        return context
    
    def _assess_complexity(self, text: str) -> str:
        """Assess the complexity level of financial content."""
        text_lower = text.lower()
        
        # Advanced financial terms
        advanced_terms = [
            'derivatives', 'options', 'futures', 'hedge', 'arbitrage',
            'volatility', 'beta', 'alpha', 'sharpe ratio', 'var',
            'monte carlo', 'black scholes', 'capm', 'wacc'
        ]
        
        # Intermediate terms
        intermediate_terms = [
            'portfolio', 'diversification', 'asset allocation', 'risk tolerance',
            'compound interest', 'present value', 'future value', 'annuity'
        ]
        
        advanced_count = sum(1 for term in advanced_terms if term in text_lower)
        intermediate_count = sum(1 for term in intermediate_terms if term in text_lower)
        
        if advanced_count > 3:
            return "advanced"
        elif intermediate_count > 5 or advanced_count > 0:
            return "intermediate"
        else:
            return "beginner"
    
    def _detect_emotional_peaks(self, transcript: Transcript) -> List[EmotionalPeak]:
        """Detect emotional peaks based on speech patterns and content (legacy method)."""
        return self._detect_enhanced_emotional_peaks(transcript)
    
    def _detect_enhanced_emotional_peaks(self, transcript: Transcript) -> List[EmotionalPeak]:
        """
        Enhanced emotional peak detection with intensity scoring and context analysis.
        
        Args:
            transcript: Transcript to analyze
            
        Returns:
            List of EmotionalPeak objects with detailed analysis
        """
        peaks = []
        
        for segment in transcript.segments:
            text_lower = segment.text.lower()
            words = text_lower.split()
            
            # Analyze each emotional category
            for emotion_type, patterns in self.emotional_patterns.items():
                emotion_score = 0.0
                matched_words = []
                
                for word in words:
                    word_clean = word.strip('.,!?;:')
                    if word_clean in patterns['words']:
                        matched_words.append(word_clean)
                        
                        # Use intensity multiplier if available
                        multiplier = patterns.get('intensity_multipliers', {}).get(word_clean, 0.5)
                        emotion_score += multiplier
                
                # Create emotional peak if significant emotion detected
                if emotion_score > 0.4:  # Threshold for emotional significance
                    # Normalize intensity (cap at 1.0)
                    intensity = min(emotion_score, 1.0)
                    
                    # Adjust confidence based on transcript confidence and word count
                    confidence_adjustment = len(matched_words) * 0.1
                    adjusted_confidence = min(segment.confidence + confidence_adjustment, 1.0)
                    
                    # Create context with matched emotional words
                    context_info = f"{segment.text} [Emotional words: {', '.join(matched_words)}]"
                    
                    peak = EmotionalPeak(
                        timestamp=segment.start,
                        emotion=emotion_type,
                        intensity=intensity,
                        confidence=adjusted_confidence,
                        context=context_info
                    )
                    
                    peaks.append(peak)
        
        # Sort peaks by timestamp and remove duplicates at same timestamp
        peaks.sort(key=lambda x: x.timestamp)
        
        # Remove overlapping peaks (keep highest intensity)
        filtered_peaks = []
        for peak in peaks:
            # Check if there's already a peak within 2 seconds
            overlapping = [p for p in filtered_peaks if abs(p.timestamp - peak.timestamp) < 2.0]
            
            if not overlapping:
                filtered_peaks.append(peak)
            else:
                # Keep the peak with highest intensity
                max_intensity_peak = max(overlapping + [peak], key=lambda x: x.intensity)
                if max_intensity_peak == peak:
                    # Remove overlapping peaks and add current one
                    filtered_peaks = [p for p in filtered_peaks if p not in overlapping]
                    filtered_peaks.append(peak)
        
        self.logger.info(f"Detected {len(filtered_peaks)} emotional peaks")
        return filtered_peaks
    
    def _analyze_content_flow(self, clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze content flow across multiple clips."""
        flow_analysis = []
        
        for i, clip in enumerate(clips):
            concepts = clip['financial_analysis']['concepts_mentioned']
            
            flow_item = {
                'clip_index': i,
                'concepts_introduced': concepts,
                'concepts_from_previous': [],
                'flow_quality': 'good'
            }
            
            # Check concept continuity with previous clips
            if i > 0:
                prev_concepts = clips[i-1]['financial_analysis']['concepts_mentioned']
                flow_item['concepts_from_previous'] = list(set(concepts) & set(prev_concepts))
                
                # Assess flow quality
                if len(flow_item['concepts_from_previous']) == 0:
                    flow_item['flow_quality'] = 'disconnected'
                elif len(flow_item['concepts_from_previous']) > len(concepts) * 0.5:
                    flow_item['flow_quality'] = 'repetitive'
            
            flow_analysis.append(flow_item)
        
        return flow_analysis
    
    def _extract_global_concepts(self, clips: List[Dict[str, Any]]) -> List[str]:
        """Extract key concepts that appear across multiple clips."""
        concept_counts = {}
        
        for clip in clips:
            concepts = clip['financial_analysis']['concepts_mentioned']
            for concept in concepts:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Return concepts that appear in multiple clips, sorted by frequency
        global_concepts = [
            concept for concept, count in concept_counts.items() 
            if count > 1
        ]
        
        return sorted(global_concepts, key=lambda x: concept_counts[x], reverse=True)
    
    def _plan_retention_strategy(self, project_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan retention strategy based on content analysis."""
        clips = project_analysis['clips']
        
        strategy = {
            'hook_placements': [],
            'concept_reinforcements': [],
            'pacing_recommendations': [],
            'engagement_techniques': []
        }
        
        # Suggest hook placements every 30-45 seconds
        total_duration = sum(clip.get('duration', 0) for clip in clips)
        hook_interval = 35  # seconds
        
        for i in range(0, int(total_duration), hook_interval):
            strategy['hook_placements'].append({
                'timestamp': i,
                'type': 'engagement_hook',
                'suggestion': 'Insert visual or audio hook to maintain attention'
            })
        
        # Identify concepts that need reinforcement
        global_concepts = project_analysis.get('key_concepts', [])
        for concept in global_concepts[:3]:  # Top 3 concepts
            strategy['concept_reinforcements'].append({
                'concept': concept,
                'reinforcement_count': 3,
                'technique': 'repetition_with_examples'
            })
        
        return strategy
    
    def cleanup_models(self):
        """Clean up loaded models to free memory."""
        self.logger.info("Cleaning up Whisper models")
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()