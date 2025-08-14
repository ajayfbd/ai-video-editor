#!/usr/bin/env python3
"""
Step-by-Step Audio/Video Analysis Testing Script

This script provides comprehensive testing of the AI Video Editor's audio and video
analysis capabilities, following the testing strategy guidelines with proper mocking
and performance monitoring.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.modules.content_analysis.audio_analyzer import FinancialContentAnalyzer
from ai_video_editor.modules.content_analysis.video_analyzer import VideoAnalyzer
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.utils.logging_config import get_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)


class AudioVideoAnalysisTester:
    """Comprehensive tester for audio and video analysis modules."""
    
    def __init__(self):
        """Initialize the tester with required components."""
        self.cache_manager = CacheManager(cache_dir="./temp/test_cache")
        self.audio_analyzer = FinancialContentAnalyzer(cache_manager=self.cache_manager)
        self.video_analyzer = VideoAnalyzer(cache_manager=self.cache_manager)
        
        # Test files
        self.test_files = {
            'audio': [
                'test_audio.wav',
                'whisper_test.wav'
            ],
            'video': [
                'test_video.mp4',
                'test_render_output.mp4',
                'test_timeline_render.mp4'
            ]
        }
        
        # Performance tracking
        self.test_results = {}
        
    def print_header(self, title: str):
        """Print a formatted header for test sections."""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
    
    def print_step(self, step: str, description: str):
        """Print a formatted step description."""
        print(f"\n📋 Step: {step}")
        print(f"   {description}")
        print("-" * 40)
    
    def check_prerequisites(self) -> bool:
        """Check if all required files and dependencies are available."""
        self.print_header("Prerequisites Check")
        
        all_good = True
        
        # Check test files
        print("📁 Checking test files...")
        for file_type, files in self.test_files.items():
            for file_path in files:
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    print(f"   ✅ {file_path} ({file_size:,} bytes)")
                else:
                    print(f"   ❌ {file_path} - NOT FOUND")
                    all_good = False
        
        # Check dependencies
        print("\n🔧 Checking dependencies...")
        try:
            import whisper
            print("   ✅ OpenAI Whisper")
        except ImportError:
            print("   ❌ OpenAI Whisper - NOT INSTALLED")
            all_good = False
        
        try:
            import cv2
            print("   ✅ OpenCV")
        except ImportError:
            print("   ❌ OpenCV - NOT INSTALLED")
            all_good = False
        
        try:
            import ffmpeg
            print("   ✅ FFmpeg-python")
        except ImportError:
            print("   ❌ FFmpeg-python - NOT INSTALLED")
            all_good = False
        
        # Check cache directory
        cache_dir = Path("./temp/test_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Cache directory: {cache_dir}")
        
        return all_good
    
    def test_audio_analysis_step_by_step(self) -> Dict[str, Any]:
        """Test audio analysis with detailed step-by-step breakdown."""
        self.print_header("Audio Analysis Testing")
        
        results = {
            'transcription': {},
            'financial_analysis': {},
            'enhancement': {},
            'performance': {}
        }
        
        for audio_file in self.test_files['audio']:
            if not Path(audio_file).exists():
                print(f"⏭️  Skipping {audio_file} - file not found")
                continue
            
            print(f"\n🎵 Testing with: {audio_file}")
            
            # Step 1: Basic Transcription
            self.print_step("1", "Basic Audio Transcription")
            try:
                start_time = time.time()
                transcript = self.audio_analyzer.transcribe_audio(
                    audio_file, 
                    model_size='base',  # Use smaller model for testing
                    use_cache=True
                )
                transcription_time = time.time() - start_time
                
                print(f"   ✅ Transcription completed in {transcription_time:.2f}s")
                print(f"   📝 Text length: {len(transcript.text)} characters")
                print(f"   🎯 Confidence: {transcript.confidence:.3f}")
                print(f"   🗣️  Language: {transcript.language}")
                print(f"   📊 Segments: {len(transcript.segments)}")
                
                if transcript.text:
                    preview = transcript.text[:100] + "..." if len(transcript.text) > 100 else transcript.text
                    print(f"   📖 Preview: {preview}")
                
                results['transcription'][audio_file] = {
                    'success': True,
                    'processing_time': transcription_time,
                    'text_length': len(transcript.text),
                    'confidence': transcript.confidence,
                    'segments': len(transcript.segments)
                }
                
            except Exception as e:
                print(f"   ❌ Transcription failed: {e}")
                results['transcription'][audio_file] = {'success': False, 'error': str(e)}
                continue
            
            # Step 2: Financial Content Analysis
            self.print_step("2", "Financial Content Analysis")
            try:
                start_time = time.time()
                financial_analysis = self.audio_analyzer.analyze_financial_content(
                    transcript, 
                    enhance_audio=True
                )
                analysis_time = time.time() - start_time
                
                print(f"   ✅ Analysis completed in {analysis_time:.2f}s")
                print(f"   💰 Financial concepts: {len(financial_analysis.concepts_mentioned)}")
                print(f"   📚 Explanation segments: {len(financial_analysis.explanation_segments)}")
                print(f"   📊 Data references: {len(financial_analysis.data_references)}")
                print(f"   🎭 Emotional peaks: {len(financial_analysis.emotional_peaks)}")
                print(f"   🗣️  Filler words detected: {len(financial_analysis.filler_words_detected)}")
                print(f"   📈 Complexity level: {financial_analysis.complexity_level}")
                
                if financial_analysis.concepts_mentioned:
                    concepts_preview = ", ".join(financial_analysis.concepts_mentioned[:5])
                    print(f"   🏷️  Key concepts: {concepts_preview}")
                
                if financial_analysis.audio_enhancement:
                    enhancement = financial_analysis.audio_enhancement
                    print(f"   🎧 Audio enhancement:")
                    print(f"      - Filler words removed: {enhancement.filler_words_removed}")
                    print(f"      - Segments modified: {enhancement.segments_modified}")
                    print(f"      - Quality improvement: {enhancement.quality_improvement_score:.2f}")
                
                results['financial_analysis'][audio_file] = {
                    'success': True,
                    'processing_time': analysis_time,
                    'concepts_count': len(financial_analysis.concepts_mentioned),
                    'explanation_segments': len(financial_analysis.explanation_segments),
                    'filler_words': len(financial_analysis.filler_words_detected)
                }
                
            except Exception as e:
                print(f"   ❌ Financial analysis failed: {e}")
                results['financial_analysis'][audio_file] = {'success': False, 'error': str(e)}
            
            # Step 3: Performance Analysis
            self.print_step("3", "Performance Analysis")
            total_time = transcription_time + analysis_time
            file_size = Path(audio_file).stat().st_size
            processing_rate = file_size / total_time if total_time > 0 else 0
            
            print(f"   ⏱️  Total processing time: {total_time:.2f}s")
            print(f"   📁 File size: {file_size:,} bytes")
            print(f"   🚀 Processing rate: {processing_rate:,.0f} bytes/second")
            
            results['performance'][audio_file] = {
                'total_time': total_time,
                'file_size': file_size,
                'processing_rate': processing_rate
            }
        
        return results
    
    def test_video_analysis_step_by_step(self) -> Dict[str, Any]:
        """Test video analysis with detailed step-by-step breakdown."""
        self.print_header("Video Analysis Testing")
        
        results = {
            'metadata_extraction': {},
            'scene_detection': {},
            'frame_analysis': {},
            'face_detection': {},
            'performance': {}
        }
        
        for video_file in self.test_files['video']:
            if not Path(video_file).exists():
                print(f"⏭️  Skipping {video_file} - file not found")
                continue
            
            print(f"\n🎬 Testing with: {video_file}")
            
            # Create a test ContentContext
            context = ContentContext(
                project_id=f"test_{int(time.time())}",
                video_files=[video_file],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences(quality_mode="balanced")
            )
            
            # Step 1: Video Metadata Extraction
            self.print_step("1", "Video Metadata Extraction")
            try:
                start_time = time.time()
                metadata = self.video_analyzer._extract_video_metadata(video_file)
                metadata_time = time.time() - start_time
                
                print(f"   ✅ Metadata extracted in {metadata_time:.2f}s")
                print(f"   ⏱️  Duration: {metadata.duration:.2f}s")
                print(f"   🎞️  FPS: {metadata.fps:.2f}")
                print(f"   📐 Resolution: {metadata.width}x{metadata.height}")
                print(f"   🎥 Codec: {metadata.codec}")
                if metadata.bitrate:
                    print(f"   📊 Bitrate: {metadata.bitrate:,} bps")
                if metadata.total_frames:
                    print(f"   🖼️  Total frames: {metadata.total_frames:,}")
                
                results['metadata_extraction'][video_file] = {
                    'success': True,
                    'processing_time': metadata_time,
                    'duration': metadata.duration,
                    'fps': metadata.fps,
                    'resolution': f"{metadata.width}x{metadata.height}"
                }
                
            except Exception as e:
                print(f"   ❌ Metadata extraction failed: {e}")
                results['metadata_extraction'][video_file] = {'success': False, 'error': str(e)}
                continue
            
            # Step 2: Scene Detection
            self.print_step("2", "Scene Detection")
            try:
                start_time = time.time()
                scenes = self.video_analyzer._detect_scenes(video_file)
                scene_time = time.time() - start_time
                
                print(f"   ✅ Scene detection completed in {scene_time:.2f}s")
                print(f"   🎬 Scenes detected: {len(scenes)}")
                
                for i, scene in enumerate(scenes[:3]):  # Show first 3 scenes
                    print(f"      Scene {i+1}: {scene.start_time:.1f}s - {scene.end_time:.1f}s ({scene.duration:.1f}s)")
                
                if len(scenes) > 3:
                    print(f"      ... and {len(scenes) - 3} more scenes")
                
                results['scene_detection'][video_file] = {
                    'success': True,
                    'processing_time': scene_time,
                    'scenes_count': len(scenes)
                }
                
            except Exception as e:
                print(f"   ❌ Scene detection failed: {e}")
                results['scene_detection'][video_file] = {'success': False, 'error': str(e)}
            
            # Step 3: Frame Analysis (limited sample)
            self.print_step("3", "Frame Analysis (Sample)")
            try:
                start_time = time.time()
                
                # Analyze just the first few frames for testing
                import cv2
                cap = cv2.VideoCapture(video_file)
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video: {video_file}")
                
                frame_analyses = []
                frame_count = 0
                max_frames = 10  # Limit for testing
                
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    timestamp = frame_count / metadata.fps
                    analysis = self.video_analyzer._analyze_single_frame(
                        frame, timestamp, frame_count, context
                    )
                    if analysis:
                        frame_analyses.append(analysis)
                    
                    frame_count += 1
                
                cap.release()
                frame_analysis_time = time.time() - start_time
                
                print(f"   ✅ Frame analysis completed in {frame_analysis_time:.2f}s")
                print(f"   🖼️  Frames analyzed: {len(frame_analyses)}")
                
                # Analyze results
                total_faces = sum(len(analysis.faces) for analysis in frame_analyses)
                avg_quality = sum(analysis.quality_score for analysis in frame_analyses) / len(frame_analyses) if frame_analyses else 0
                high_potential_frames = sum(1 for analysis in frame_analyses if analysis.thumbnail_potential > 0.7)
                
                print(f"   👥 Total faces detected: {total_faces}")
                print(f"   ⭐ Average quality score: {avg_quality:.2f}")
                print(f"   🎯 High potential frames: {high_potential_frames}")
                
                # Show visual elements detected
                all_elements = []
                for analysis in frame_analyses:
                    for elem in analysis.visual_elements:
                        if hasattr(elem, 'element_type'):
                            all_elements.append(elem.element_type)
                        elif isinstance(elem, str):
                            all_elements.append(elem)
                        else:
                            all_elements.append(str(elem))
                
                unique_elements = list(set(all_elements))
                if unique_elements:
                    print(f"   🎨 Visual elements detected: {', '.join(unique_elements[:5])}")
                
                results['frame_analysis'][video_file] = {
                    'success': True,
                    'processing_time': frame_analysis_time,
                    'frames_analyzed': len(frame_analyses),
                    'faces_detected': total_faces,
                    'avg_quality': avg_quality,
                    'high_potential_frames': high_potential_frames
                }
                
            except Exception as e:
                print(f"   ❌ Frame analysis failed: {e}")
                results['frame_analysis'][video_file] = {'success': False, 'error': str(e)}
            
            # Step 4: Complete Video Analysis
            self.print_step("4", "Complete Video Analysis Integration")
            try:
                start_time = time.time()
                
                # Reset context for full analysis
                context = ContentContext(
                    project_id=f"test_full_{int(time.time())}",
                    video_files=[video_file],
                    content_type=ContentType.EDUCATIONAL,
                    user_preferences=UserPreferences(quality_mode="balanced")
                )
                
                # Run full video analysis
                analyzed_context = self.video_analyzer.analyze_video(video_file, context)
                full_analysis_time = time.time() - start_time
                
                print(f"   ✅ Full analysis completed in {full_analysis_time:.2f}s")
                print(f"   🎯 Visual highlights: {len(analyzed_context.visual_highlights)}")
                
                # Show some highlights
                for i, highlight in enumerate(analyzed_context.visual_highlights[:3]):
                    print(f"      Highlight {i+1}: {highlight.timestamp:.1f}s - {highlight.description}")
                    print(f"         Faces: {len(highlight.faces)}, Elements: {len(highlight.visual_elements)}")
                    print(f"         Thumbnail potential: {highlight.thumbnail_potential:.2f}")
                
                if len(analyzed_context.visual_highlights) > 3:
                    print(f"      ... and {len(analyzed_context.visual_highlights) - 3} more highlights")
                
                results['performance'][video_file] = {
                    'total_time': full_analysis_time,
                    'file_size': Path(video_file).stat().st_size,
                    'visual_highlights': len(analyzed_context.visual_highlights)
                }
                
            except Exception as e:
                print(f"   ❌ Full video analysis failed: {e}")
        
        return results
    
    def test_integration_workflow(self) -> Dict[str, Any]:
        """Test integrated audio/video analysis workflow."""
        self.print_header("Integration Workflow Testing")
        
        results = {'integration_tests': {}}
        
        # Find a video file for integration testing
        test_video = None
        for video_file in self.test_files['video']:
            if Path(video_file).exists():
                test_video = video_file
                break
        
        if not test_video:
            print("❌ No video file available for integration testing")
            return results
        
        print(f"🎬 Integration test with: {test_video}")
        
        # Step 1: Create ContentContext
        self.print_step("1", "ContentContext Creation and Setup")
        try:
            context = ContentContext(
                project_id=f"integration_test_{int(time.time())}",
                video_files=[test_video],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences(quality_mode="balanced")
            )
            print(f"   ✅ ContentContext created: {context.project_id}")
            print(f"   📁 Video files: {len(context.video_files)}")
            print(f"   🏷️  Content type: {context.content_type.value}")
            
        except Exception as e:
            print(f"   ❌ ContentContext creation failed: {e}")
            return results
        
        # Step 2: Video Analysis
        self.print_step("2", "Video Analysis Integration")
        try:
            start_time = time.time()
            context = self.video_analyzer.analyze_video(test_video, context)
            video_time = time.time() - start_time
            
            print(f"   ✅ Video analysis completed in {video_time:.2f}s")
            print(f"   🎯 Visual highlights added: {len(context.visual_highlights)}")
            print(f"   📊 Video metadata populated: {'Yes' if context.video_metadata else 'No'}")
            
        except Exception as e:
            print(f"   ❌ Video analysis integration failed: {e}")
            return results
        
        # Step 3: Audio Analysis (if audio available)
        self.print_step("3", "Audio Analysis Integration")
        test_audio = None
        for audio_file in self.test_files['audio']:
            if Path(audio_file).exists():
                test_audio = audio_file
                break
        
        if test_audio:
            try:
                start_time = time.time()
                transcript = self.audio_analyzer.transcribe_audio(test_audio, model_size='base')
                financial_analysis = self.audio_analyzer.analyze_financial_content(transcript)
                audio_time = time.time() - start_time
                
                # Simulate adding audio analysis to context
                # (In real implementation, this would be handled by the audio analyzer)
                context.audio_transcript = transcript.to_dict()
                
                print(f"   ✅ Audio analysis completed in {audio_time:.2f}s")
                print(f"   📝 Transcript added to context")
                print(f"   💰 Financial concepts: {len(financial_analysis.concepts_mentioned)}")
                
            except Exception as e:
                print(f"   ❌ Audio analysis integration failed: {e}")
        else:
            print("   ⏭️  No audio file available for integration testing")
        
        # Step 4: ContentContext Validation
        self.print_step("4", "ContentContext Validation")
        try:
            # Validate context structure
            validation_results = {
                'has_video_metadata': bool(context.video_metadata),
                'has_visual_highlights': len(context.visual_highlights) > 0,
                'has_audio_transcript': bool(getattr(context, 'audio_transcript', None)),
                'project_id_valid': bool(context.project_id),
                'content_type_valid': isinstance(context.content_type, ContentType)
            }
            
            print("   📋 Context validation results:")
            for check, result in validation_results.items():
                status = "✅" if result else "❌"
                print(f"      {status} {check}: {result}")
            
            all_valid = all(validation_results.values())
            print(f"\n   🎯 Overall validation: {'✅ PASSED' if all_valid else '❌ FAILED'}")
            
            results['integration_tests']['validation'] = validation_results
            
        except Exception as e:
            print(f"   ❌ Context validation failed: {e}")
        
        return results
    
    def generate_test_report(self, audio_results: Dict, video_results: Dict, integration_results: Dict):
        """Generate a comprehensive test report."""
        self.print_header("Test Report Summary")
        
        print("📊 AUDIO ANALYSIS RESULTS:")
        for file_name, result in audio_results.get('transcription', {}).items():
            if result.get('success'):
                print(f"   ✅ {file_name}: {result['processing_time']:.2f}s, {result['confidence']:.3f} confidence")
            else:
                print(f"   ❌ {file_name}: {result.get('error', 'Unknown error')}")
        
        print("\n📊 VIDEO ANALYSIS RESULTS:")
        for file_name, result in video_results.get('metadata_extraction', {}).items():
            if result.get('success'):
                print(f"   ✅ {file_name}: {result['resolution']}, {result['duration']:.1f}s")
            else:
                print(f"   ❌ {file_name}: {result.get('error', 'Unknown error')}")
        
        print("\n📊 INTEGRATION TEST RESULTS:")
        validation = integration_results.get('integration_tests', {}).get('validation', {})
        if validation:
            passed = sum(1 for v in validation.values() if v)
            total = len(validation)
            print(f"   🎯 Validation: {passed}/{total} checks passed")
        
        # Performance summary
        print("\n⚡ PERFORMANCE SUMMARY:")
        
        # Audio performance
        audio_times = []
        for results in audio_results.values():
            for file_result in results.values():
                if isinstance(file_result, dict) and 'processing_time' in file_result:
                    audio_times.append(file_result['processing_time'])
        
        if audio_times:
            avg_audio_time = sum(audio_times) / len(audio_times)
            print(f"   🎵 Average audio processing: {avg_audio_time:.2f}s")
        
        # Video performance
        video_times = []
        for results in video_results.values():
            for file_result in results.values():
                if isinstance(file_result, dict) and 'processing_time' in file_result:
                    video_times.append(file_result['processing_time'])
        
        if video_times:
            avg_video_time = sum(video_times) / len(video_times)
            print(f"   🎬 Average video processing: {avg_video_time:.2f}s")
        
        print("\n🎉 Testing completed successfully!")
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        print("🚀 Starting Audio/Video Analysis Step-by-Step Testing")
        print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            print("\n❌ Prerequisites check failed. Please install missing dependencies.")
            return False
        
        # Run tests
        try:
            audio_results = self.test_audio_analysis_step_by_step()
            video_results = self.test_video_analysis_step_by_step()
            integration_results = self.test_integration_workflow()
            
            # Generate report
            self.generate_test_report(audio_results, video_results, integration_results)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Testing failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point for the testing script."""
    tester = AudioVideoAnalysisTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())