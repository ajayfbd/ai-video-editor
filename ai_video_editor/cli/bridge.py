"""
CLI Bridge - Utilities for converting between CLI interface and core modules.

This module provides bridge functions to convert CLI parameters to ContentContext
and format core module results for CLI output, maintaining backward compatibility.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.modules.content_analysis.content_analyzer import MultiModalContentAnalyzer
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.utils.logging_config import get_logger

logger = get_logger(__name__)


class CLIBridge:
    """Bridge between CLI interface and core analysis modules."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize CLI bridge with core analyzers."""
        self.cache_manager = CacheManager(cache_dir) if cache_dir else None
        self.content_analyzer = MultiModalContentAnalyzer(
            cache_manager=self.cache_manager
        )
        logger.info("CLI Bridge initialized")
    
    def create_context_from_cli_args(
        self, 
        video_file: Path, 
        language: str = "hi",
        model: str = "medium",
        **kwargs
    ) -> ContentContext:
        """
        Create ContentContext from CLI arguments.
        
        Args:
            video_file: Path to video file
            language: Language for transcription
            model: Whisper model size
            **kwargs: Additional CLI arguments
            
        Returns:
            ContentContext ready for analysis
        """
        # Create user preferences from CLI args
        user_prefs = UserPreferences()
        user_prefs.whisper_model_size = model
        user_prefs.transcription_language = language
        
        # Determine content type (can be overridden by analysis)
        content_type = ContentType.GENERAL
        if kwargs.get('preset') in ['hindi-religious', 'sanskrit-classical', 'mythological']:
            content_type = ContentType.MUSIC
        
        # Create context
        context = ContentContext(
            project_id=f"cli_analysis_{int(time.time())}",
            video_files=[str(video_file)],
            content_type=content_type,
            user_preferences=user_prefs
        )
        
        logger.info(f"Created ContentContext for CLI analysis: {video_file}")
        return context
    
    def perform_comprehensive_analysis(
        self,
        context: ContentContext,
        audio_analysis: bool = True,
        video_analysis: bool = True,
        scene_detection: bool = True,
        face_detection: bool = True,
        visual_highlights: bool = True
    ) -> ContentContext:
        """
        Perform comprehensive analysis using core modules.
        
        Args:
            context: ContentContext to analyze
            audio_analysis: Whether to perform audio analysis
            video_analysis: Whether to perform video analysis
            scene_detection: Whether to detect scenes
            face_detection: Whether to detect faces
            visual_highlights: Whether to extract visual highlights
            
        Returns:
            Updated ContentContext with analysis results
        """
        logger.info("Performing comprehensive analysis using core modules")
        
        # Perform individual analyses first
        if audio_analysis and context.video_files:
            logger.info("Performing audio analysis...")
            context = self._perform_audio_analysis(context)
        
        if video_analysis and context.video_files:
            logger.info("Performing video analysis...")
            context = self._perform_video_analysis(context)
        
        # Then perform multi-modal analysis to extract concepts and insights
        logger.info("Performing multi-modal content analysis...")
        context = self.content_analyzer.analyze_content(context)
        
        logger.info("Comprehensive analysis completed")
        return context
    
    def _perform_audio_analysis(self, context: ContentContext) -> ContentContext:
        """Perform audio analysis using FinancialContentAnalyzer."""
        try:
            from ai_video_editor.modules.content_analysis.audio_analyzer import FinancialContentAnalyzer
            
            audio_analyzer = FinancialContentAnalyzer(cache_manager=self.cache_manager)
            
            # Transcribe the first video file
            if context.video_files:
                video_file = context.video_files[0]
                model_size = context.user_preferences.whisper_model_size or "medium"
                language = context.user_preferences.transcription_language
                
                logger.info(f"Transcribing audio from {video_file} using model {model_size}")
                transcript = audio_analyzer.transcribe_audio(
                    video_file, 
                    model_size=model_size,
                    language=language
                )
                
                # Analyze financial content
                logger.info("Analyzing financial content...")
                financial_analysis = audio_analyzer.analyze_financial_content(transcript)
                
                # Store results in context properly
                from ai_video_editor.core.content_context import AudioAnalysisResult, AudioSegment
                
                # Convert transcript segments to AudioSegment objects
                audio_segments = []
                for segment in transcript.segments:
                    audio_segment = AudioSegment(
                        text=segment.text,
                        start=segment.start,
                        end=segment.end,
                        confidence=segment.confidence,
                        cleaned_text=segment.text  # For now, use original text
                    )
                    audio_segments.append(audio_segment)
                
                # Create proper AudioAnalysisResult
                audio_analysis_result = AudioAnalysisResult(
                    transcript_text=transcript.text,
                    segments=audio_segments,
                    overall_confidence=transcript.confidence,
                    language=transcript.language,
                    processing_time=getattr(transcript, 'processing_time', 0.0),
                    model_used=getattr(transcript, 'model_used', 'whisper'),
                    financial_concepts=financial_analysis.concepts_mentioned,
                    explanation_segments=financial_analysis.explanation_segments,
                    data_references=financial_analysis.data_references,
                    complexity_level=financial_analysis.complexity_level,
                    detected_emotions=financial_analysis.emotional_peaks
                )
                
                context.audio_analysis = audio_analysis_result
                
                logger.info(f"Audio analysis completed: {len(transcript.segments)} segments")
            
            return context
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return context
    
    def _perform_video_analysis(self, context: ContentContext) -> ContentContext:
        """Perform video analysis using VideoAnalyzer."""
        try:
            from ai_video_editor.modules.content_analysis.video_analyzer import VideoAnalyzer
            
            video_analyzer = VideoAnalyzer(
                cache_manager=self.cache_manager,
                memory_client=getattr(self.content_analyzer, 'memory_client', None)
            )
            
            # Analyze the first video file
            if context.video_files:
                video_file = context.video_files[0]
                logger.info(f"Analyzing video content from {video_file}")
                context = video_analyzer.analyze_video(video_file, context)
                
                logger.info(f"Video analysis completed: {len(context.visual_highlights)} highlights")
            
            return context
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return context
    
    def format_audio_results_for_cli(self, context: ContentContext) -> Dict[str, Any]:
        """
        Format audio analysis results for CLI output compatibility.
        
        Args:
            context: ContentContext with audio analysis results
            
        Returns:
            Dictionary in CLI-compatible format
        """
        if not context.audio_analysis:
            return {"error": "No audio analysis available", "transcript": None, "statistics": {}}
        
        # Extract transcript data
        transcript_data = {
            "text": context.audio_analysis.transcript_text or "",
            "segments": [],
            "language": getattr(context.audio_analysis, 'language', 'unknown'),
            "model_used": getattr(context.audio_analysis, 'model_used', 'unknown')
        }
        
        # Convert segments if available
        if hasattr(context.audio_analysis, 'segments') and context.audio_analysis.segments:
            for segment in context.audio_analysis.segments:
                transcript_data["segments"].append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": segment.confidence
                })
        
        # Calculate statistics
        total_duration = 0.0
        if transcript_data["segments"]:
            total_duration = max(seg["end"] for seg in transcript_data["segments"])
        
        statistics = {
            "total_duration": total_duration,
            "segment_count": len(transcript_data["segments"]),
            "language": transcript_data["language"],
            "model_used": transcript_data["model_used"],
            "average_confidence": self._calculate_average_confidence(transcript_data["segments"])
        }
        
        # Content analysis
        content_analysis = {
            "themes": self._extract_themes_from_context(context),
            "dominant_theme": self._get_dominant_theme(context),
            "repeated_phrases": [],  # Simplified for CLI
            "word_count": len(transcript_data["text"].split()) if transcript_data["text"] else 0,
            "unique_words": len(set(transcript_data["text"].split())) if transcript_data["text"] else 0,
            "speaking_rate": len(transcript_data["text"].split()) / max(total_duration, 1) if transcript_data["text"] else 0
        }
        
        return {
            "transcript": transcript_data,
            "statistics": statistics,
            "content_analysis": content_analysis
        }
    
    def format_video_results_for_cli(self, context: ContentContext) -> Dict[str, Any]:
        """
        Format video analysis results for CLI output compatibility.
        
        Args:
            context: ContentContext with video analysis results
            
        Returns:
            Dictionary in CLI-compatible format
        """
        # Extract metadata
        metadata = context.video_metadata or {}
        
        # Convert visual highlights to scenes format
        scenes = []
        if context.visual_highlights:
            for i, highlight in enumerate(context.visual_highlights):
                scenes.append({
                    "scene_id": i,
                    "timestamp": highlight.timestamp,
                    "confidence": highlight.confidence,
                    "description": highlight.description,
                    "frame_number": int(highlight.timestamp * metadata.get('fps', 30))
                })
        
        # Convert face detections
        faces = []
        if context.visual_highlights:
            for highlight in context.visual_highlights:
                if highlight.faces:
                    for face in highlight.faces:
                        faces.append({
                            "timestamp": highlight.timestamp,
                            "confidence": face.confidence,
                            "bbox": face.bbox,
                            "expression": face.expression,
                            "frame_number": int(highlight.timestamp * metadata.get('fps', 30))
                        })
        
        # Convert visual highlights
        visual_highlights = []
        if context.visual_highlights:
            for highlight in context.visual_highlights:
                visual_highlights.append({
                    "timestamp": highlight.timestamp,
                    "confidence": highlight.confidence,
                    "description": highlight.description,
                    "highlight_type": "general",
                    "visual_elements": highlight.visual_elements
                })
        
        # Calculate statistics
        statistics = {
            "total_scenes": len(scenes),
            "total_faces": len(faces),
            "total_highlights": len(visual_highlights),
            "analysis_quality": self._determine_analysis_quality(len(scenes), len(faces), len(visual_highlights)),
            "resolution": f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
            "fps": metadata.get('fps', 0),
            "duration": metadata.get('duration', 0)
        }
        
        return {
            "metadata": metadata,
            "scenes": scenes,
            "faces": faces,
            "visual_highlights": visual_highlights,
            "statistics": statistics
        }
    
    def generate_analysis_report_for_cli(
        self, 
        context: ContentContext, 
        video_file: Path, 
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report for CLI output.
        
        Args:
            context: ContentContext with analysis results
            video_file: Original video file path
            output_dir: Output directory for report
            
        Returns:
            Comprehensive analysis report dictionary
        """
        audio_results = self.format_audio_results_for_cli(context)
        video_results = self.format_video_results_for_cli(context)
        
        report = {
            "video_file": str(video_file),
            "analysis_summary": {},
            "recommendations": [],
            "editing_suggestions": [],
            "technical_insights": [],
            "content_insights": []
        }
        
        # Audio analysis summary
        if audio_results.get("statistics"):
            stats = audio_results["statistics"]
            content = audio_results.get("content_analysis", {})
            
            avg_conf = stats.get("average_confidence", 0)
            transcription_quality = "high" if avg_conf > 0.7 else "medium" if avg_conf > 0.5 else "low"
            
            report["analysis_summary"]["audio"] = {
                "transcription_quality": transcription_quality,
                "content_type": content.get("dominant_theme", "general"),
                "duration": stats.get("total_duration", 0),
                "segment_count": stats.get("segment_count", 0),
                "speaking_rate": content.get("speaking_rate", 0),
                "vocabulary_richness": content.get("unique_words", 0) / max(content.get("word_count", 1), 1)
            }
            
            # Generate recommendations based on analysis
            if transcription_quality == "low":
                report["recommendations"].append("Low transcription quality detected - consider using a larger Whisper model")
            
            speaking_rate = content.get("speaking_rate", 0)
            if speaking_rate > 3.0:
                report["editing_suggestions"].append("Fast speaking rate - use quick cuts and dynamic pacing")
            elif speaking_rate < 1.5:
                report["editing_suggestions"].append("Slow speaking rate - use longer shots and gentle transitions")
        
        # Video analysis summary
        if video_results.get("statistics"):
            stats = video_results["statistics"]
            
            report["analysis_summary"]["video"] = {
                "scene_count": stats.get("total_scenes", 0),
                "face_count": stats.get("total_faces", 0),
                "key_frames": stats.get("total_highlights", 0),
                "analysis_quality": stats.get("analysis_quality", "unknown"),
                "resolution": stats.get("resolution", "unknown"),
                "fps": stats.get("fps", 0),
                "duration": stats.get("duration", 0)
            }
            
            # Generate video-based recommendations
            scene_count = stats.get("total_scenes", 0)
            if scene_count > 8:
                report["editing_suggestions"].append("Many scene changes - suitable for fast-paced editing style")
            elif scene_count < 3:
                report["editing_suggestions"].append("Few scene changes - consider adding visual variety with effects")
            
            face_count = stats.get("total_faces", 0)
            if face_count > 5:
                report["editing_suggestions"].append("Multiple faces detected - use close-ups during key moments")
            elif face_count == 0:
                report["recommendations"].append("No faces detected - content may be landscape, animation, or text-based")
        
        # Content insights from key concepts
        if context.key_concepts:
            report["content_insights"].append(f"Key concepts identified: {', '.join(context.key_concepts[:5])}")
        
        # Emotional insights
        if context.emotional_markers:
            high_intensity_peaks = [peak for peak in context.emotional_markers if peak.intensity > 0.7]
            if high_intensity_peaks:
                report["content_insights"].append(f"Detected {len(high_intensity_peaks)} high-intensity emotional peaks")
        
        return report
    
    def _calculate_average_confidence(self, segments: List[Dict]) -> float:
        """Calculate average confidence from segments."""
        if not segments:
            return 0.0
        
        confidences = [seg.get("confidence", 0.0) for seg in segments]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _extract_themes_from_context(self, context: ContentContext) -> Dict[str, int]:
        """Extract themes from context for CLI compatibility."""
        themes = {
            "educational": 0,
            "financial": 0,
            "general": 0,
            "technical": 0
        }
        
        # Count themes based on key concepts
        if context.key_concepts:
            financial_keywords = ['investment', 'portfolio', 'financial', 'money', 'budget', 'savings']
            educational_keywords = ['explain', 'learn', 'understand', 'concept', 'tutorial']
            
            for concept in context.key_concepts:
                concept_lower = concept.lower()
                if any(keyword in concept_lower for keyword in financial_keywords):
                    themes["financial"] += 1
                elif any(keyword in concept_lower for keyword in educational_keywords):
                    themes["educational"] += 1
                else:
                    themes["general"] += 1
        
        return themes
    
    def _get_dominant_theme(self, context: ContentContext) -> str:
        """Get dominant theme from context."""
        themes = self._extract_themes_from_context(context)
        if not any(themes.values()):
            return "general"
        
        return max(themes.keys(), key=lambda k: themes[k])
    
    def _determine_analysis_quality(self, scenes: int, faces: int, highlights: int) -> str:
        """Determine analysis quality based on detection counts."""
        total_detections = scenes + faces + highlights
        
        if total_detections > 15:
            return "high"
        elif total_detections > 8:
            return "medium"
        elif total_detections > 3:
            return "low"
        else:
            return "minimal"