"""
Workflow Orchestrator Example - Demonstrates end-to-end pipeline management.

This example shows how to use the WorkflowOrchestrator to process videos
through the complete AI Video Editor pipeline with progress tracking,
error recovery, and resource monitoring.
"""

import asyncio
import tempfile
from pathlib import Path
from rich.console import Console

from ai_video_editor.core.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowConfiguration,
    ProcessingMode,
    WorkflowStage
)
from ai_video_editor.core.content_context import ContentType, UserPreferences
from ai_video_editor.core.config import ProjectSettings, VideoQuality
from ai_video_editor.utils.logging_config import setup_logging, get_logger


def create_sample_video_files(temp_dir: Path) -> list[str]:
    """Create sample video files for testing."""
    video_files = []
    
    for i in range(2):
        video_file = temp_dir / f"sample_video_{i}.mp4"
        # In a real scenario, these would be actual video files
        video_file.write_text(f"Mock video content {i} - Educational finance content")
        video_files.append(str(video_file))
    
    return video_files


async def basic_workflow_example():
    """Demonstrate basic workflow orchestration."""
    console = Console()
    console.print("[cyan]🎬 Basic Workflow Orchestrator Example[/cyan]\n")
    
    # Create temporary directory and sample files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        video_files = create_sample_video_files(temp_path)
        
        # Create basic configuration
        config = WorkflowConfiguration(
            processing_mode=ProcessingMode.FAST,
            enable_parallel_processing=False,
            max_memory_usage_gb=4.0,
            timeout_per_stage=120,
            enable_progress_display=True,
            output_directory=temp_path / "output",
            temp_directory=temp_path / "temp"
        )
        
        # Create project settings
        project_settings = ProjectSettings(
            content_type=ContentType.EDUCATIONAL,
            quality=VideoQuality.HIGH,
            auto_enhance=True,
            enable_b_roll_generation=True,
            enable_thumbnail_generation=True
        )
        
        # Create user preferences
        user_preferences = UserPreferences(
            quality_mode="balanced",
            batch_size=2,
            parallel_processing=True,
            max_api_cost=5.0
        )
        
        # Create and run orchestrator
        orchestrator = WorkflowOrchestrator(config=config, console=console)
        
        try:
            console.print("🚀 Starting workflow execution...")
            
            result_context = await orchestrator.process_video(
                input_files=video_files,
                project_settings=project_settings,
                user_preferences=user_preferences
            )
            
            console.print("\n✅ Workflow completed successfully!")
            
            # Display results
            summary = orchestrator.get_processing_summary()
            console.print(f"\n📊 Processing Summary:")
            console.print(f"   Project ID: {summary.get('project_id', 'N/A')}")
            console.print(f"   Input Files: {len(summary.get('input_files', []))}")
            console.print(f"   Content Type: {summary.get('content_type', 'N/A')}")
            console.print(f"   Duration: {summary.get('workflow_duration', 0):.1f}s")
            
            return result_context
            
        except Exception as e:
            console.print(f"\n❌ Workflow failed: {e}")
            raise


async def advanced_workflow_example():
    """Demonstrate advanced workflow features."""
    console = Console()
    console.print("[cyan]🎬 Advanced Workflow Orchestrator Example[/cyan]\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        video_files = create_sample_video_files(temp_path)
        
        # Create advanced configuration
        config = WorkflowConfiguration(
            processing_mode=ProcessingMode.HIGH_QUALITY,
            enable_parallel_processing=True,
            max_concurrent_stages=3,
            enable_caching=True,
            enable_recovery=True,
            checkpoint_frequency=2,
            timeout_per_stage=300,
            max_memory_usage_gb=8.0,
            enable_progress_display=True,
            output_directory=temp_path / "output",
            temp_directory=temp_path / "temp",
            # Skip some stages for demonstration
            skip_stages=[WorkflowStage.BROLL_GENERATION],
            stage_timeouts={
                WorkflowStage.AUDIO_PROCESSING: 180,
                WorkflowStage.VIDEO_PROCESSING: 240
            }
        )
        
        # Create project settings for music content
        project_settings = ProjectSettings(
            content_type=ContentType.MUSIC,
            quality=VideoQuality.ULTRA,
            auto_enhance=True,
            enable_character_animation=False,
            enable_b_roll_generation=False,  # Disabled for music content
            enable_thumbnail_generation=True
        )
        
        # Create optimized user preferences
        user_preferences = UserPreferences(
            quality_mode="high",
            batch_size=3,
            parallel_processing=True,
            enable_aggressive_caching=False,
            max_api_cost=10.0
        )
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator(config=config, console=console)
        
        try:
            console.print("🚀 Starting advanced workflow execution...")
            console.print(f"   Processing Mode: {config.processing_mode.value}")
            console.print(f"   Parallel Processing: {config.enable_parallel_processing}")
            console.print(f"   Max Concurrent Stages: {config.max_concurrent_stages}")
            console.print(f"   Memory Limit: {config.max_memory_usage_gb}GB")
            console.print(f"   Skipped Stages: {[s.value for s in config.skip_stages]}")
            
            result_context = await orchestrator.process_video(
                input_files=video_files,
                project_settings=project_settings,
                user_preferences=user_preferences
            )
            
            console.print("\n✅ Advanced workflow completed successfully!")
            
            # Display detailed results
            summary = orchestrator.get_processing_summary()
            status = orchestrator.get_workflow_status()
            
            console.print(f"\n📊 Detailed Processing Summary:")
            console.print(f"   Project ID: {summary.get('project_id', 'N/A')}")
            console.print(f"   Content Type: {summary.get('content_type', 'N/A')}")
            console.print(f"   Total Duration: {summary.get('workflow_duration', 0):.1f}s")
            
            # Show stage completion status
            console.print(f"\n⏱️  Stage Completion Status:")
            for stage_name, stage_info in status.get('stages', {}).items():
                status_icon = {
                    'completed': '✅',
                    'failed': '❌',
                    'running': '🔄',
                    'pending': '⏳',
                    'skipped': '⏭️'
                }.get(stage_info['status'], '❓')
                
                duration = stage_info.get('duration')
                duration_text = f" ({duration:.1f}s)" if duration else ""
                
                console.print(f"   {status_icon} {stage_name.replace('_', ' ').title()}: {stage_info['status']}{duration_text}")
            
            # Show performance metrics
            if 'processing_metrics' in summary:
                metrics = summary['processing_metrics']
                console.print(f"\n📈 Performance Metrics:")
                console.print(f"   Peak Memory: {metrics.get('memory_peak_usage', 0) / (1024**3):.1f}GB")
                console.print(f"   API Calls: {sum(metrics.get('api_calls_made', {}).values())}")
                console.print(f"   Recovery Actions: {len(metrics.get('recovery_actions', []))}")
            
            return result_context
            
        except Exception as e:
            console.print(f"\n❌ Advanced workflow failed: {e}")
            
            # Show failure details
            status = orchestrator.get_workflow_status()
            failed_stages = [
                stage for stage, info in status.get('stages', {}).items()
                if info.get('error')
            ]
            
            if failed_stages:
                console.print(f"\n💥 Failed Stages:")
                for stage in failed_stages:
                    error = status['stages'][stage]['error']
                    console.print(f"   ❌ {stage.replace('_', ' ').title()}: {error}")
            
            raise


async def error_recovery_example():
    """Demonstrate error recovery capabilities."""
    console = Console()
    console.print("[cyan]🎬 Error Recovery Example[/cyan]\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        video_files = create_sample_video_files(temp_path)
        
        # Create configuration with recovery enabled
        config = WorkflowConfiguration(
            processing_mode=ProcessingMode.BALANCED,
            enable_recovery=True,
            checkpoint_frequency=1,  # Save checkpoint after each stage
            max_memory_usage_gb=2.0,  # Low limit to trigger memory constraints
            timeout_per_stage=30,     # Short timeout to demonstrate timeout recovery
            enable_progress_display=True,
            output_directory=temp_path / "output",
            temp_directory=temp_path / "temp"
        )
        
        # Create project settings
        project_settings = ProjectSettings(
            content_type=ContentType.GENERAL,
            quality=VideoQuality.MEDIUM
        )
        
        # Create user preferences that might trigger memory constraints
        user_preferences = UserPreferences(
            batch_size=4,  # High batch size
            enable_aggressive_caching=False
        )
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator(config=config, console=console)
        
        try:
            console.print("🚀 Starting workflow with error recovery...")
            console.print("   (This example demonstrates recovery from simulated errors)")
            
            result_context = await orchestrator.process_video(
                input_files=video_files,
                project_settings=project_settings,
                user_preferences=user_preferences
            )
            
            console.print("\n✅ Workflow completed with recovery!")
            
            # Show recovery statistics
            status = orchestrator.get_workflow_status()
            recovery_attempts = status.get('recovery_attempts', {})
            
            if recovery_attempts:
                console.print(f"\n🔄 Recovery Statistics:")
                for stage, attempts in recovery_attempts.items():
                    console.print(f"   {stage.replace('_', ' ').title()}: {attempts} recovery attempts")
            else:
                console.print(f"\n✨ No recovery attempts needed - smooth execution!")
            
            return result_context
            
        except Exception as e:
            console.print(f"\n❌ Workflow failed despite recovery attempts: {e}")
            raise


async def resource_monitoring_example():
    """Demonstrate resource monitoring capabilities."""
    console = Console()
    console.print("[cyan]🎬 Resource Monitoring Example[/cyan]\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        video_files = create_sample_video_files(temp_path)
        
        # Create configuration with resource monitoring
        config = WorkflowConfiguration(
            processing_mode=ProcessingMode.BALANCED,
            max_memory_usage_gb=6.0,
            enable_progress_display=True,
            output_directory=temp_path / "output",
            temp_directory=temp_path / "temp"
        )
        
        # Create project settings
        project_settings = ProjectSettings(
            content_type=ContentType.EDUCATIONAL,
            quality=VideoQuality.HIGH
        )
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator(config=config, console=console)
        
        try:
            console.print("🚀 Starting workflow with resource monitoring...")
            
            result_context = await orchestrator.process_video(
                input_files=video_files,
                project_settings=project_settings
            )
            
            console.print("\n✅ Workflow completed with resource monitoring!")
            
            # Show resource usage statistics
            status = orchestrator.get_workflow_status()
            performance_metrics = status.get('performance_metrics', {})
            
            console.print(f"\n📊 Resource Usage Statistics:")
            console.print(f"   Peak Memory: {performance_metrics.get('peak_memory_gb', 0):.1f}GB")
            console.print(f"   Current Memory: {performance_metrics.get('current_memory_gb', 0):.1f}GB")
            console.print(f"   CPU Usage: {performance_metrics.get('cpu_percent', 0):.1f}%")
            
            # Show processing summary
            summary = orchestrator.get_processing_summary()
            if 'processing_metrics' in summary:
                metrics = summary['processing_metrics']
                console.print(f"\n⚡ Processing Efficiency:")
                console.print(f"   Total Processing Time: {summary.get('workflow_duration', 0):.1f}s")
                console.print(f"   Memory Peak Usage: {metrics.get('memory_peak_usage', 0) / (1024**3):.1f}GB")
                console.print(f"   Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.1%}")
            
            return result_context
            
        except Exception as e:
            console.print(f"\n❌ Resource monitoring workflow failed: {e}")
            raise


async def custom_workflow_example():
    """Demonstrate custom workflow configuration."""
    console = Console()
    console.print("[cyan]🎬 Custom Workflow Configuration Example[/cyan]\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        video_files = create_sample_video_files(temp_path)
        
        # Create highly customized configuration
        config = WorkflowConfiguration(
            processing_mode=ProcessingMode.CUSTOM,
            enable_parallel_processing=True,
            max_concurrent_stages=2,
            enable_caching=True,
            enable_recovery=True,
            checkpoint_frequency=1,
            timeout_per_stage=180,
            max_memory_usage_gb=12.0,
            enable_progress_display=True,
            output_directory=temp_path / "output",
            temp_directory=temp_path / "temp",
            # Custom stage configuration
            skip_stages=[
                WorkflowStage.BROLL_GENERATION,
                WorkflowStage.VIDEO_COMPOSITION  # Skip for demonstration
            ],
            stage_timeouts={
                WorkflowStage.AUDIO_PROCESSING: 120,
                WorkflowStage.INTELLIGENCE_LAYER: 240,
                WorkflowStage.THUMBNAIL_GENERATION: 90
            },
            stage_priorities={
                WorkflowStage.THUMBNAIL_GENERATION: 1,  # High priority
                WorkflowStage.METADATA_GENERATION: 1,   # High priority
                WorkflowStage.CONTENT_ANALYSIS: 2       # Medium priority
            }
        )
        
        # Create custom project settings
        project_settings = ProjectSettings(
            content_type=ContentType.EDUCATIONAL,
            quality=VideoQuality.HIGH,
            target_duration=600,  # 10 minutes
            auto_enhance=True,
            enable_b_roll_generation=False,  # Disabled in config
            enable_thumbnail_generation=True,
            enable_analytics=True
        )
        
        # Create custom user preferences
        user_preferences = UserPreferences(
            quality_mode="high",
            thumbnail_resolution=(1920, 1080),
            batch_size=3,
            parallel_processing=True,
            enable_aggressive_caching=True,
            max_api_cost=15.0
        )
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator(config=config, console=console)
        
        try:
            console.print("🚀 Starting custom workflow execution...")
            console.print(f"   Custom Configuration:")
            console.print(f"   • Processing Mode: {config.processing_mode.value}")
            console.print(f"   • Parallel Stages: {config.max_concurrent_stages}")
            console.print(f"   • Memory Limit: {config.max_memory_usage_gb}GB")
            console.print(f"   • Skipped Stages: {len(config.skip_stages)}")
            console.print(f"   • Custom Timeouts: {len(config.stage_timeouts)}")
            console.print(f"   • Stage Priorities: {len(config.stage_priorities)}")
            
            result_context = await orchestrator.process_video(
                input_files=video_files,
                project_settings=project_settings,
                user_preferences=user_preferences
            )
            
            console.print("\n✅ Custom workflow completed successfully!")
            
            # Show detailed execution analysis
            summary = orchestrator.get_processing_summary()
            status = orchestrator.get_workflow_status()
            
            console.print(f"\n📈 Custom Workflow Analysis:")
            console.print(f"   Project ID: {summary.get('project_id', 'N/A')}")
            console.print(f"   Total Duration: {summary.get('workflow_duration', 0):.1f}s")
            
            # Analyze stage execution
            completed_stages = [
                stage for stage, info in status.get('stages', {}).items()
                if info['status'] == 'completed'
            ]
            skipped_stages = [
                stage for stage, info in status.get('stages', {}).items()
                if info['status'] == 'skipped'
            ]
            
            console.print(f"   Completed Stages: {len(completed_stages)}")
            console.print(f"   Skipped Stages: {len(skipped_stages)}")
            
            if skipped_stages:
                console.print(f"   Skipped: {', '.join(s.replace('_', ' ').title() for s in skipped_stages)}")
            
            return result_context
            
        except Exception as e:
            console.print(f"\n❌ Custom workflow failed: {e}")
            raise


async def main():
    """Run all workflow orchestrator examples."""
    # Setup logging
    setup_logging("INFO")
    logger = get_logger(__name__)
    
    console = Console()
    console.print("[bold blue]🎬 AI Video Editor - Workflow Orchestrator Examples[/bold blue]\n")
    
    examples = [
        ("Basic Workflow", basic_workflow_example),
        ("Advanced Workflow", advanced_workflow_example),
        ("Error Recovery", error_recovery_example),
        ("Resource Monitoring", resource_monitoring_example),
        ("Custom Configuration", custom_workflow_example)
    ]
    
    for example_name, example_func in examples:
        try:
            console.print(f"\n{'='*60}")
            console.print(f"Running: {example_name}")
            console.print(f"{'='*60}")
            
            await example_func()
            
            console.print(f"\n✅ {example_name} completed successfully!")
            
        except Exception as e:
            console.print(f"\n❌ {example_name} failed: {e}")
            logger.error(f"{example_name} failed", exc_info=True)
        
        # Small delay between examples
        await asyncio.sleep(1)
    
    console.print(f"\n{'='*60}")
    console.print("[bold green]🎉 All Workflow Orchestrator examples completed![/bold green]")
    console.print(f"{'='*60}")


if __name__ == "__main__":
    # Handle Windows event loop policy
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())