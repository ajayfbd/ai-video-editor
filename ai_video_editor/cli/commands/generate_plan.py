"""Generate intelligent editing plan from video analysis and transcript."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import click

from ai_video_editor.utils.logging_config import get_logger
from .utils import load_json, save_json

logger = get_logger(__name__)


@click.command("generate-plan")
@click.argument("video_file", type=click.Path(exists=True, path_type=Path))
@click.option("--transcript", "transcript_json", type=click.Path(exists=True, path_type=Path),
              help="Transcript JSON file (optional, will transcribe if not provided)")
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True,
              help="Output editing plan JSON path")
@click.option("--style", type=click.Choice(["basic", "dynamic", "devotional"]), default="basic",
              help="Editing style preset")
@click.option("--segment-duration", type=float, default=4.0,
              help="Target duration for each segment in seconds")
@click.option("--add-transitions", is_flag=True, help="Add transition effects between segments")
@click.option("--detect-chorus", is_flag=True, help="Detect and highlight repeated sections")
@click.option("--analyze-scenes", is_flag=True, help="Perform scene detection and analysis")
@click.option("--detect-faces", is_flag=True, help="Detect faces and expressions")
@click.option("--quality-threshold", type=float, default=0.3, help="Scene change detection threshold")
@click.option("--ai-director", is_flag=True, help="Use AI Director for sophisticated analysis and planning")
@click.option("--content-type", type=click.Choice(["educational", "music", "general"]), default="general",
              help="Content type for AI Director optimization")
@click.option("--mock-ai", is_flag=True, help="Use mock AI Director for testing (no API calls)")
def generate_plan_cmd(video_file: Path, transcript_json: Optional[Path], output_path: Path, style: str, 
                      segment_duration: float, add_transitions: bool, detect_chorus: bool,
                      analyze_scenes: bool, detect_faces: bool, quality_threshold: float,
                      ai_director: bool, content_type: str, mock_ai: bool):
    """Generate intelligent editing plan from video analysis and transcript.
    
    Analyzes the actual video content (scenes, faces, motion) combined with 
    transcript timing to create smart editing decisions.
    
    Use --ai-director for sophisticated AI-powered analysis and planning.
    """
    try:
        click.echo(f"[INFO] Analyzing video: {video_file}")
        
        # Check if AI Director mode is requested
        if ai_director:
            if mock_ai:
                click.echo("[INFO] 🤖 Using Mock AI Director for testing (no API calls)")
                ai_plan = _generate_mock_ai_director_plan(
                    video_file, transcript_json, output_path, style, content_type,
                    segment_duration, add_transitions, detect_chorus, analyze_scenes, detect_faces
                )
            else:
                click.echo("[INFO] 🤖 Using AI Director for sophisticated analysis and planning")
                ai_plan = _generate_ai_director_plan(
                    video_file, transcript_json, output_path, style, content_type,
                    segment_duration, add_transitions, detect_chorus, analyze_scenes, detect_faces
                )
            return
        
        # Original implementation for backward compatibility
        click.echo("[INFO] Using basic analysis mode (use --ai-director for enhanced AI planning)")
        
        # Load or generate transcript
        if transcript_json:
            transcript = load_json(transcript_json)
            click.echo(f"[INFO] Using provided transcript: {transcript_json}")
        else:
            click.echo("[INFO] No transcript provided - generating basic plan from video analysis only")
            transcript = {"segments": [], "text": "", "language": "unknown"}
        
        segments = transcript.get("segments", [])
        
        # Perform video analysis
        video_analysis = None
        if analyze_scenes or detect_faces:
            video_analysis = _analyze_video_content(
                video_file, analyze_scenes, detect_faces, quality_threshold
            )
            click.echo(f"[INFO] Video analysis complete")
        
        # Generate editing decisions based on video + transcript
        editing_decisions = _generate_intelligent_editing_decisions(
            video_file, segments, video_analysis, style, segment_duration, 
            add_transitions, detect_chorus
        )
        
        # Generate b-roll plans from video analysis
        broll_plans = _generate_intelligent_broll_plans(
            segments, video_analysis, style
        )
        
        # Create AI plan structure
        total_duration = segments[-1].get("end", 0) if segments else _get_video_duration(video_file)
        ai_plan = {
            "editing_decisions": editing_decisions,
            "broll_plans": broll_plans,
            "metadata_strategy": {
                "style": style,
                "target_duration": total_duration,
                "content_type": "devotional" if style == "devotional" else "general",
                "video_analysis_used": video_analysis is not None
            }
        }
        
        # Save plan
        save_json(ai_plan, output_path)
        
        click.echo(f"[OK] Editing plan written to {output_path}")
        click.echo(f"[INFO] Generated {len(editing_decisions)} editing decisions")
        click.echo(f"[INFO] Generated {len(broll_plans)} b-roll opportunities")
        if video_analysis:
            scene_count = len(video_analysis.get("scenes", []))
            face_count = len(video_analysis.get("faces", []))
            click.echo(f"[INFO] Detected {scene_count} scenes, {face_count} face regions")
        
    except Exception as e:
        logger.error(f"Plan generation failed: {e}")
        click.echo(f"[ERROR] Plan generation failed: {e}")
        sys.exit(1)


def _generate_mock_ai_director_plan(video_file: Path, transcript_json: Optional[Path], output_path: Path,
                                   style: str, content_type: str, segment_duration: float, 
                                   add_transitions: bool, detect_chorus: bool, analyze_scenes: bool, 
                                   detect_faces: bool) -> None:
    """Generate mock AI Director plan for testing without API calls."""
    try:
        from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
        from ai_video_editor.modules.content_analysis.content_analyzer import create_content_analyzer
        from ai_video_editor.core.cache_manager import CacheManager
        
        # Create ContentContext
        content_type_enum = ContentType(content_type)
        user_prefs = UserPreferences()
        user_prefs.editing_style = style
        
        context = ContentContext(
            project_id=f"mock_plan_{int(time.time())}",
            video_files=[str(video_file)],
            content_type=content_type_enum,
            user_preferences=user_prefs
        )
        
        click.echo("[INFO] 🔍 Performing local content analysis...")
        
        # Load transcript if provided
        if transcript_json:
            transcript_data = load_json(transcript_json)
            click.echo(f"[INFO] Using provided transcript: {transcript_json}")
            
            # Convert transcript to ContentContext format
            if transcript_data.get("segments"):
                from ai_video_editor.modules.content_analysis.audio_analyzer import TranscriptSegment, Transcript
                
                segments = []
                for seg in transcript_data["segments"]:
                    segments.append(TranscriptSegment(
                        text=seg.get("text", ""),
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        confidence=seg.get("confidence", 0.8)
                    ))
                
                transcript = Transcript(
                    text=transcript_data.get("text", ""),
                    segments=segments,
                    confidence=transcript_data.get("confidence", 0.8),
                    language=transcript_data.get("language", "unknown"),
                    processing_time=transcript_data.get("processing_time", 0.0),
                    model_used=transcript_data.get("model_used", "unknown")
                )
                
                context.audio_transcript = transcript
        
        # Perform local analysis
        cache_manager = CacheManager()
        analyzer = create_content_analyzer(cache_manager)
        
        # Analyze the content locally
        context = analyzer.analyze_content(context)
        
        click.echo("[INFO] 🤖 Generating mock AI Director editing plan...")
        
        # Generate mock intelligent editing decisions
        editing_decisions = []
        broll_plans = []
        
        # Create intelligent cuts based on content analysis
        if hasattr(context, 'audio_transcript') and context.audio_transcript:
            segments = context.audio_transcript.segments
            for i, segment in enumerate(segments):
                # Create smart cuts at natural breaks
                if segment.confidence > 0.7:  # High confidence segments
                    editing_decisions.append({
                        "decision_id": f"mock_ai_cut_{i:03d}",
                        "decision_type": "cut",
                        "timestamp": segment.start,
                        "duration": segment.end - segment.start,
                        "parameters": {
                            "confidence": 0.9,
                            "reason": f"High-confidence segment: {segment.text[:50]}...",
                            "ai_director": True,
                            "mock_mode": True,
                            "engagement_impact": 0.8
                        }
                    })
                
                # Generate content-aware B-roll
                text = segment.text.lower()
                if any(word in text for word in ["राम", "कृष्ण", "गोपाल", "भगवान"]):
                    broll_plans.append({
                        "broll_id": f"mock_deity_{i:03d}",
                        "timestamp": segment.start,
                        "duration": min(segment.end - segment.start, 3.0),
                        "content_type": "deity_imagery",
                        "description": f"AI-detected deity reference: {segment.text[:50]}...",
                        "parameters": {
                            "confidence": 0.85,
                            "visual_impact": 0.9,
                            "ai_director": True,
                            "mock_mode": True,
                            "style": style
                        }
                    })
                elif any(word in text for word in ["भक्त", "पुकार", "प्रार्थना"]):
                    broll_plans.append({
                        "broll_id": f"mock_devotional_{i:03d}",
                        "timestamp": segment.start,
                        "duration": min(segment.end - segment.start, 2.5),
                        "content_type": "devotional_scene",
                        "description": f"AI-detected devotional content: {segment.text[:50]}...",
                        "parameters": {
                            "confidence": 0.8,
                            "visual_impact": 0.7,
                            "ai_director": True,
                            "mock_mode": True,
                            "style": style
                        }
                    })
        
        # Add emotional peak-based decisions
        if hasattr(context, 'emotional_markers') and context.emotional_markers:
            for i, peak in enumerate(context.emotional_markers):
                if peak.intensity > 0.6:
                    editing_decisions.append({
                        "decision_id": f"mock_emotional_{i:03d}",
                        "decision_type": "emphasis",
                        "timestamp": peak.timestamp,
                        "duration": 1.0,
                        "parameters": {
                            "confidence": 0.9,
                            "reason": f"AI-detected emotional peak: {peak.emotion} ({peak.intensity:.1%})",
                            "ai_director": True,
                            "mock_mode": True,
                            "engagement_impact": peak.intensity
                        }
                    })
        
        # Create comprehensive plan
        total_duration = context.audio_transcript.segments[-1].end if (hasattr(context, 'audio_transcript') and context.audio_transcript and context.audio_transcript.segments) else _get_video_duration(video_file)
        
        cli_plan = {
            "editing_decisions": editing_decisions,
            "broll_plans": broll_plans,
            "metadata_strategy": {
                "style": style,
                "content_type": content_type,
                "ai_director_used": True,
                "mock_mode": True,
                "confidence": 0.85,
                "engagement_score": 0.75,
                "target_duration": total_duration,
                "optimization_focus": "engagement",
                "seo_keywords": ["devotional", "spiritual", "bhajan", "religious"],
                "thumbnail_concepts": ["deity imagery", "devotional scene", "spiritual moment"]
            }
        }
        
        # Save the plan
        save_json(cli_plan, output_path)
        
        # Display results
        click.echo(f"[OK] ✨ Mock AI Director plan written to {output_path}")
        click.echo(f"[INFO] 🎬 Generated {len(cli_plan['editing_decisions'])} AI-powered editing decisions")
        click.echo(f"[INFO] 🎨 Generated {len(cli_plan['broll_plans'])} intelligent B-roll opportunities")
        
        if hasattr(context, 'key_concepts') and context.key_concepts:
            click.echo(f"[INFO] 🔍 Identified {len(context.key_concepts)} key concepts")
        
        if hasattr(context, 'emotional_markers') and context.emotional_markers:
            high_peaks = [p for p in context.emotional_markers if p.intensity > 0.7]
            click.echo(f"[INFO] 😊 Detected {len(high_peaks)} high-intensity emotional peaks")
        
        if hasattr(context, 'visual_highlights') and context.visual_highlights:
            click.echo(f"[INFO] 👁️ Found {len(context.visual_highlights)} visual highlights")
        
        confidence = cli_plan["metadata_strategy"].get("confidence", 0.0)
        engagement = cli_plan["metadata_strategy"].get("engagement_score", 0.0)
        click.echo(f"[INFO] 📊 Plan confidence: {confidence:.1%}, Engagement score: {engagement:.1%}")
        click.echo(f"[INFO] 🧪 Mock mode - no API calls made")
        
    except Exception as e:
        logger.error(f"Mock AI Director plan generation failed: {e}")
        click.echo(f"[ERROR] Mock AI Director planning failed: {e}")
        sys.exit(1)


def _generate_enhanced_local_plan(context, style: str, content_type: str):
    """Generate enhanced local plan when AI Director API is unavailable."""
    from ai_video_editor.modules.intelligence.ai_director import AIDirectorPlan, EditingDecision, BRollPlan, MetadataStrategy
    from datetime import datetime
    
    editing_decisions = []
    broll_plans = []
    
    # Generate intelligent decisions based on local analysis
    if hasattr(context, 'audio_transcript') and context.audio_transcript:
        segments = context.audio_transcript.segments
        for i, segment in enumerate(segments):
            if segment.confidence > 0.7:
                editing_decisions.append(EditingDecision(
                    timestamp=segment.start,
                    decision_type="cut",
                    parameters={
                        "start_time": segment.start,
                        "end_time": segment.end,
                        "segment_text": segment.text[:50] + "...",
                        "local_analysis": True
                    },
                    rationale=f"High-confidence transcript segment: {segment.text[:30]}...",
                    confidence=0.85,
                    priority=7
                ))
                
                # Content-aware B-roll based on local analysis
                text = segment.text.lower()
                if any(word in text for word in ["राम", "कृष्ण", "गोपाल", "भगवान"]):
                    broll_plans.append(BRollPlan(
                        timestamp=segment.start,
                        duration=min(segment.end - segment.start, 3.0),
                        content_type="deity_imagery",
                        description=f"Deity reference detected: {segment.text[:40]}...",
                        visual_elements=["deity_image", "spiritual_background"],
                        animation_style="fade_overlay",
                        priority=8
                    ))
    
    # Add emotional peak decisions
    if hasattr(context, 'emotional_markers') and context.emotional_markers:
        for peak in context.emotional_markers:
            if peak.intensity > 0.6:
                editing_decisions.append(EditingDecision(
                    timestamp=peak.timestamp,
                    decision_type="emphasis",
                    parameters={
                        "emotion": peak.emotion,
                        "intensity": peak.intensity,
                        "local_analysis": True
                    },
                    rationale=f"Emotional peak detected: {peak.emotion} ({peak.intensity:.1%})",
                    confidence=0.8,
                    priority=6
                ))
    
    # Create metadata strategy
    metadata_strategy = MetadataStrategy(
        primary_title=f"Devotional Content - {style.title()} Style",
        title_variations=[
            "Spiritual Journey - Divine Moments",
            "Sacred Verses - Devotional Experience",
            "Divine Blessings - Spiritual Content"
        ],
        description=f"Enhanced {content_type} content with {style} styling and local AI analysis",
        tags=["devotional", "spiritual", "religious", "bhajan", "divine"],
        thumbnail_concepts=["deity imagery", "spiritual moment", "devotional scene"],
        hook_text="Experience divine moments",
        target_keywords=["devotional", "spiritual", "religious content"]
    )
    
    # Create AI Director plan structure
    plan = AIDirectorPlan(
        editing_decisions=editing_decisions,
        broll_plans=broll_plans,
        metadata_strategy=metadata_strategy,
        quality_enhancements=["audio_clarity", "visual_enhancement"],
        pacing_adjustments=[{"type": "natural_flow", "confidence": 0.8}],
        engagement_hooks=[{"type": "emotional_peak", "count": len([p for p in context.emotional_markers if p.intensity > 0.6]) if hasattr(context, 'emotional_markers') else 0}],
        created_at=datetime.now(),
        confidence_score=0.8,
        processing_time=1.0,
        model_used="local_analysis"
    )
    
    return plan


def _generate_ai_director_plan(video_file: Path, transcript_json: Optional[Path], output_path: Path,
                              style: str, content_type: str, segment_duration: float, 
                              add_transitions: bool, detect_chorus: bool, analyze_scenes: bool, 
                              detect_faces: bool) -> None:
    """Generate AI Director plan using the full AI Video Editor pipeline."""
    try:
        from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
        from ai_video_editor.modules.content_analysis.content_analyzer import create_content_analyzer
        from ai_video_editor.modules.intelligence.ai_director import FinancialVideoEditor
        from ai_video_editor.modules.intelligence.gemini_client import GeminiClient
        from ai_video_editor.core.cache_manager import CacheManager
        
        # Create ContentContext
        content_type_enum = ContentType(content_type)
        user_prefs = UserPreferences()
        user_prefs.editing_style = style
        
        context = ContentContext(
            project_id=f"generate_plan_{int(time.time())}",
            video_files=[str(video_file)],
            content_type=content_type_enum,
            user_preferences=user_prefs
        )
        
        click.echo("[INFO] 🔍 Performing comprehensive multi-modal analysis...")
        
        # Load transcript if provided
        if transcript_json:
            transcript_data = load_json(transcript_json)
            click.echo(f"[INFO] Using provided transcript: {transcript_json}")
            
            # Convert transcript to ContentContext format
            if transcript_data.get("segments"):
                from ai_video_editor.modules.content_analysis.audio_analyzer import TranscriptSegment, Transcript
                
                segments = []
                for seg in transcript_data["segments"]:
                    segments.append(TranscriptSegment(
                        text=seg.get("text", ""),
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        confidence=seg.get("confidence", 0.8)
                    ))
                
                transcript = Transcript(
                    text=transcript_data.get("text", ""),
                    segments=segments,
                    confidence=transcript_data.get("confidence", 0.8),
                    language=transcript_data.get("language", "unknown"),
                    processing_time=transcript_data.get("processing_time", 0.0),
                    model_used=transcript_data.get("model_used", "unknown")
                )
                
                context.audio_transcript = transcript
        
        # Perform comprehensive analysis using MultiModalContentAnalyzer
        cache_manager = CacheManager()
        analyzer = create_content_analyzer(cache_manager)
        
        # Analyze the content
        context = analyzer.analyze_content(context)
        
        click.echo("[INFO] 🤖 Generating AI Director editing plan...")
        
        # Initialize AI Director with enhanced configuration
        from ai_video_editor.modules.intelligence.gemini_client import GeminiConfig
        
        # Configure Gemini client for CLI usage
        gemini_config = GeminiConfig(
            model="gemini-2.0-flash-exp",
            temperature=0.7,
            max_output_tokens=4000  # Limit output for faster response
        )
        
        gemini_client = GeminiClient(
            default_config=gemini_config,
            timeout=30.0,  # Shorter timeout for CLI
            max_retries=2,  # Fewer retries for faster feedback
            enable_caching=True
        )
        ai_director = FinancialVideoEditor(gemini_client, cache_manager)
        
        # Generate comprehensive editing plan with timeout and fallback
        import asyncio
        
        async def _run_ai_director():
            try:
                # Try AI Director with timeout
                click.echo("[INFO] ⏱️ Calling AI Director (30s timeout)...")
                ai_plan = await asyncio.wait_for(
                    ai_director.generate_editing_plan(context),
                    timeout=30.0
                )
                click.echo("[INFO] ✅ AI Director plan generated successfully")
                return ai_plan, True
                
            except asyncio.TimeoutError:
                click.echo("[WARNING] ⏰ AI Director timed out, falling back to enhanced local analysis")
                return None, False
                
            except Exception as api_error:
                click.echo(f"[WARNING] 🔄 AI Director API failed: {str(api_error)[:100]}...")
                click.echo("[INFO] 🔧 Falling back to enhanced local analysis")
                return None, False
        
        # Run the AI Director
        ai_plan, api_success = asyncio.run(_run_ai_director())
        
        if not api_success:
            ai_plan = _generate_enhanced_local_plan(context, style, content_type)
        
        # Convert AI Director plan to CLI format
        cli_plan = {
            "editing_decisions": [],
            "broll_plans": [],
            "metadata_strategy": {
                "style": style,
                "content_type": content_type,
                "ai_director_used": True,
                "confidence": getattr(ai_plan, 'confidence', 0.8),
                "engagement_score": getattr(ai_plan, 'engagement_score', 0.0)
            }
        }
        
        # Convert editing decisions
        if hasattr(ai_plan, 'editing_decisions'):
            for i, decision in enumerate(ai_plan.editing_decisions):
                # Calculate duration from parameters or use default
                duration = 0.5  # Default duration for cuts
                if hasattr(decision, 'parameters') and decision.parameters:
                    if 'end_time' in decision.parameters and 'start_time' in decision.parameters:
                        duration = decision.parameters['end_time'] - decision.parameters['start_time']
                    elif 'duration' in decision.parameters:
                        duration = decision.parameters['duration']
                
                cli_decision = {
                    "decision_id": f"ai_director_{i:03d}",
                    "decision_type": decision.decision_type,
                    "timestamp": decision.timestamp,
                    "duration": duration,
                    "parameters": {
                        "confidence": decision.confidence,
                        "reason": decision.rationale,  # Use rationale instead of reason
                        "priority": decision.priority,
                        "ai_director": True,
                        "parameters": decision.parameters
                    }
                }
                cli_plan["editing_decisions"].append(cli_decision)
        
        # Convert B-roll plans
        if hasattr(ai_plan, 'broll_plans'):
            for i, broll in enumerate(ai_plan.broll_plans):
                cli_broll = {
                    "broll_id": f"ai_broll_{i:03d}",
                    "timestamp": broll.timestamp,
                    "duration": broll.duration,
                    "content_type": broll.content_type,
                    "description": broll.description,
                    "parameters": {
                        "visual_elements": broll.visual_elements,
                        "animation_style": broll.animation_style,
                        "priority": broll.priority,
                        "ai_director": True,
                        "style": style
                    }
                }
                cli_plan["broll_plans"].append(cli_broll)
        
        # Add metadata strategy from AI Director
        if hasattr(ai_plan, 'metadata_strategy'):
            metadata = ai_plan.metadata_strategy
            cli_plan["metadata_strategy"].update({
                "target_duration": getattr(metadata, 'target_duration', 0.0),
                "optimization_focus": getattr(metadata, 'optimization_focus', 'engagement'),
                "seo_keywords": getattr(metadata, 'seo_keywords', []),
                "thumbnail_concepts": getattr(metadata, 'thumbnail_concepts', [])
            })
        
        # Save the enhanced plan
        save_json(cli_plan, output_path)
        
        # Display results
        click.echo(f"[OK] ✨ AI Director plan written to {output_path}")
        click.echo(f"[INFO] 🎬 Generated {len(cli_plan['editing_decisions'])} AI-powered editing decisions")
        click.echo(f"[INFO] 🎨 Generated {len(cli_plan['broll_plans'])} intelligent B-roll opportunities")
        
        if hasattr(context, 'key_concepts') and context.key_concepts:
            click.echo(f"[INFO] 🔍 Identified {len(context.key_concepts)} key concepts")
        
        if hasattr(context, 'emotional_markers') and context.emotional_markers:
            high_peaks = [p for p in context.emotional_markers if p.intensity > 0.7]
            click.echo(f"[INFO] 😊 Detected {len(high_peaks)} high-intensity emotional peaks")
        
        if hasattr(context, 'visual_highlights') and context.visual_highlights:
            click.echo(f"[INFO] 👁️ Found {len(context.visual_highlights)} visual highlights")
        
        confidence = cli_plan["metadata_strategy"].get("confidence", 0.0)
        engagement = cli_plan["metadata_strategy"].get("engagement_score", 0.0)
        click.echo(f"[INFO] 📊 Plan confidence: {confidence:.1%}, Engagement score: {engagement:.1%}")
        
    except ImportError as e:
        click.echo(f"[ERROR] Missing AI Director dependencies: {e}")
        click.echo("[INFO] Install required packages or use basic mode without --ai-director")
        sys.exit(1)
    except Exception as e:
        logger.error(f"AI Director plan generation failed: {e}")
        click.echo(f"[ERROR] AI Director planning failed: {e}")
        click.echo("[INFO] Try using basic mode without --ai-director flag")
        sys.exit(1)


def _analyze_video_content(video_file: Path, analyze_scenes: bool, detect_faces: bool, 
                          quality_threshold: float) -> Dict[str, Any]:
    """Analyze video content using the existing video analyzer."""
    try:
        from ai_video_editor.modules.content_analysis.video_analyzer import create_video_analyzer
        from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
        from ai_video_editor.core.cache_manager import CacheManager
        
        # Create context and analyzer
        cache_manager = CacheManager()
        analyzer = create_video_analyzer(cache_manager)
        
        context = ContentContext(
            project_id="plan_generation",
            video_files=[str(video_file)],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        # Perform analysis
        analyzed_context = analyzer.analyze_video(str(video_file), context)
        
        # Extract analysis results
        analysis = {
            "scenes": [],
            "faces": [],
            "visual_highlights": [],
            "metadata": {}
        }
        
        # Extract scene information
        if hasattr(analyzed_context, 'visual_highlights'):
            for highlight in analyzed_context.visual_highlights:
                if hasattr(highlight, 'timestamp'):
                    analysis["scenes"].append({
                        "timestamp": highlight.timestamp,
                        "confidence": getattr(highlight, 'confidence', 0.8),
                        "description": getattr(highlight, 'description', 'Scene change')
                    })
        
        # Extract face detection results
        if hasattr(analyzed_context, 'face_detections'):
            for face in analyzed_context.face_detections:
                analysis["faces"].append({
                    "timestamp": getattr(face, 'timestamp', 0.0),
                    "confidence": getattr(face, 'confidence', 0.8),
                    "bbox": getattr(face, 'bbox', [0, 0, 100, 100]),
                    "expression": getattr(face, 'expression', 'neutral')
                })
        
        # Extract video metadata
        if hasattr(analyzed_context, 'video_metadata'):
            metadata = analyzed_context.video_metadata
            analysis["metadata"] = {
                "duration": getattr(metadata, 'duration', 0.0),
                "fps": getattr(metadata, 'fps', 30.0),
                "width": getattr(metadata, 'width', 1920),
                "height": getattr(metadata, 'height', 1080)
            }
        
        return analysis
        
    except Exception as e:
        logger.warning(f"Video analysis failed: {e}. Using basic analysis.")
        return {"scenes": [], "faces": [], "visual_highlights": [], "metadata": {}}


def _get_video_duration(video_file: Path) -> float:
    """Get video duration using ffprobe."""
    try:
        import subprocess
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', str(video_file)
        ], capture_output=True, text=True)
        return float(result.stdout.strip()) if result.stdout.strip() else 0.0
    except Exception:
        return 0.0


def _generate_intelligent_editing_decisions(video_file: Path, segments: List[Dict[str, Any]], 
                                          video_analysis: Optional[Dict[str, Any]], style: str, 
                                          segment_duration: float, add_transitions: bool, 
                                          detect_chorus: bool) -> List[Dict[str, Any]]:
    """Generate intelligent editing decisions based on video analysis and transcript."""
    decisions = []
    
    # Use video analysis scenes if available, otherwise use transcript segments
    if video_analysis and video_analysis.get("scenes"):
        scenes = video_analysis["scenes"]
        click.echo(f"[INFO] Using {len(scenes)} detected scenes for editing decisions")
        
        for i, scene in enumerate(scenes):
            timestamp = scene.get("timestamp", 0.0)
            confidence = scene.get("confidence", 0.8)
            
            # Find corresponding transcript segment
            segment_text = ""
            if segments:
                for seg in segments:
                    if seg.get("start", 0) <= timestamp <= seg.get("end", 0):
                        segment_text = seg.get("text", "")[:30]
                        break
            
            decision = {
                "decision_id": f"scene_cut_{i:03d}",
                "decision_type": "cut",
                "timestamp": timestamp,
                "duration": 0.5,  # Scene cuts are instantaneous
                "parameters": {
                    "cut_type": "scene_change",
                    "confidence": confidence,
                    "reason": f"Scene change detected: {segment_text}...",
                    "video_analysis": True
                }
            }
            decisions.append(decision)
    
    elif segments:
        # Fall back to transcript-based cuts
        click.echo(f"[INFO] Using {len(segments)} transcript segments for editing decisions")
        
        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            text = segment.get("text", "")
            
            decision = {
                "decision_id": f"transcript_cut_{i:03d}",
                "decision_type": "cut",
                "timestamp": start_time,
                "duration": end_time - start_time,
                "parameters": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "reason": f"Transcript segment: {text[:30]}...",
                    "confidence": 0.8,
                    "video_analysis": False
                }
            }
            decisions.append(decision)
    
    else:
        # Generate basic cuts every segment_duration seconds
        total_duration = _get_video_duration(video_file)
        click.echo(f"[INFO] No transcript or scenes - generating cuts every {segment_duration}s")
        
        current_time = 0.0
        i = 0
        while current_time < total_duration:
            decision = {
                "decision_id": f"timed_cut_{i:03d}",
                "decision_type": "cut",
                "timestamp": current_time,
                "duration": min(segment_duration, total_duration - current_time),
                "parameters": {
                    "cut_type": "timed",
                    "reason": f"Timed cut at {current_time:.1f}s",
                    "confidence": 0.6
                }
            }
            decisions.append(decision)
            current_time += segment_duration
            i += 1
        
        # Add style-specific decisions
        if style == "dynamic":
            # Add zoom/pan effects for longer segments
            if end_time - start_time > segment_duration:
                decisions.append({
                    "decision_id": f"zoom_{i:03d}",
                    "decision_type": "effect",
                    "timestamp": start_time,
                    "duration": end_time - start_time,
                    "parameters": {
                        "effect_type": "zoom_pan",
                        "intensity": 0.3,
                        "reason": "Long segment - add visual interest"
                    }
                })
        
        elif style == "devotional":
            # Add fade effects for devotional content
            if i == 0:  # Fade in at start
                decisions.append({
                    "decision_id": f"fade_in_{i:03d}",
                    "decision_type": "effect", 
                    "timestamp": start_time,
                    "duration": 1.0,
                    "parameters": {
                        "effect_type": "fade_in",
                        "reason": "Devotional opening"
                    }
                })
            
            if i == len(segments) - 1:  # Fade out at end
                decisions.append({
                    "decision_id": f"fade_out_{i:03d}",
                    "decision_type": "effect",
                    "timestamp": end_time - 1.0,
                    "duration": 1.0,
                    "parameters": {
                        "effect_type": "fade_out",
                        "reason": "Devotional closing"
                    }
                })
        
        # Add transitions between segments
        if add_transitions and i < len(segments) - 1:
            decisions.append({
                "decision_id": f"transition_{i:03d}",
                "decision_type": "transition",
                "timestamp": end_time,
                "duration": 0.5,
                "parameters": {
                    "transition_type": "crossfade",
                    "reason": "Smooth segment transition"
                }
            })
    
    # Detect chorus/repeated sections if requested
    if detect_chorus:
        chorus_decisions = _detect_chorus_sections(segments)
        decisions.extend(chorus_decisions)
    
    return decisions


def _generate_intelligent_broll_plans(segments: List[Dict[str, Any]], 
                                    video_analysis: Optional[Dict[str, Any]], 
                                    style: str) -> List[Dict[str, Any]]:
    """Generate intelligent b-roll opportunities from video analysis and segments."""
    broll_plans = []
    
    # Use face detection for close-up opportunities
    if video_analysis and video_analysis.get("faces"):
        faces = video_analysis["faces"]
        for i, face in enumerate(faces):
            timestamp = face.get("timestamp", 0.0)
            confidence = face.get("confidence", 0.8)
            expression = face.get("expression", "neutral")
            
            if confidence > 0.7:  # High confidence face detection
                broll_plans.append({
                    "broll_id": f"face_closeup_{i:03d}",
                    "timestamp": timestamp,
                    "duration": 2.0,
                    "content_type": "face_closeup",
                    "description": f"Face close-up with {expression} expression",
                    "parameters": {
                        "zoom_factor": 1.5,
                        "expression": expression,
                        "confidence": confidence,
                        "style": style
                    }
                })
    
    # Use transcript content for thematic b-roll
    for i, segment in enumerate(segments):
        text = segment.get("text", "").lower()
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        
        # Detect potential b-roll opportunities based on content
        broll_type = None
        description = ""
        
        if any(word in text for word in ["राम", "कृष्ण", "गोपाल", "भगवान", "ram", "krishna", "gopal"]):
            broll_type = "deity_imagery"
            description = "Deity imagery overlay"
        elif any(word in text for word in ["भक्त", "पुकार", "प्रार्थना", "bhakt", "pukar"]):
            broll_type = "devotional_scene"
            description = "Devotional scene overlay"
        elif any(word in text for word in ["आकाश", "पृथ्वी", "प्रकृति", "akash", "prithvi"]):
            broll_type = "nature_scene"
            description = "Nature scene overlay"
        elif any(word in text for word in ["ज्वाला", "राज", "असुर", "jwala", "raj", "asur"]):
            broll_type = "dramatic_effect"
            description = "Dramatic visual effect"
        
        if broll_type:
            broll_plans.append({
                "broll_id": f"content_broll_{i:03d}",
                "timestamp": start_time,
                "duration": min(end_time - start_time, 3.0),  # Max 3 seconds
                "content_type": broll_type,
                "description": f"{description}: {text[:50]}...",
                "parameters": {
                    "opacity": 0.7,
                    "blend_mode": "overlay" if broll_type != "dramatic_effect" else "multiply",
                    "style": style,
                    "text_based": True
                }
            })
    
    # Add scene-based b-roll opportunities
    if video_analysis and video_analysis.get("scenes"):
        scenes = video_analysis["scenes"]
        for i, scene in enumerate(scenes):
            timestamp = scene.get("timestamp", 0.0)
            confidence = scene.get("confidence", 0.8)
            
            if confidence > 0.8:  # High confidence scene changes
                broll_plans.append({
                    "broll_id": f"scene_highlight_{i:03d}",
                    "timestamp": timestamp,
                    "duration": 1.5,
                    "content_type": "scene_transition",
                    "description": f"Scene transition effect at {timestamp:.1f}s",
                    "parameters": {
                        "effect_type": "highlight",
                        "confidence": confidence,
                        "style": style,
                        "scene_based": True
                    }
                })
    
    return broll_plans


def _detect_chorus_sections(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect repeated sections (chorus) in the transcript."""
    decisions = []
    
    # Simple chorus detection based on text similarity
    text_segments = [(i, seg.get("text", "")) for i, seg in enumerate(segments)]
    
    for i, (idx1, text1) in enumerate(text_segments):
        for j, (idx2, text2) in enumerate(text_segments[i+1:], i+1):
            # Check for similar text (simple word overlap)
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if len(words1) > 2 and len(words2) > 2:
                overlap = len(words1.intersection(words2))
                similarity = overlap / min(len(words1), len(words2))
                
                if similarity > 0.6:  # 60% word overlap
                    # Mark as chorus/repeated section
                    decisions.append({
                        "decision_id": f"chorus_{idx1}_{idx2}",
                        "decision_type": "highlight",
                        "timestamp": segments[idx1].get("start", 0.0),
                        "duration": segments[idx1].get("end", 0.0) - segments[idx1].get("start", 0.0),
                        "parameters": {
                            "highlight_type": "chorus",
                            "similar_segment": idx2,
                            "similarity": similarity,
                            "reason": "Repeated section detected"
                        }
                    })
    
    return decisions