"""Main CLI entry point for AI Video Editor."""

import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from ai_video_editor.utils.logging_config import setup_logging, get_logger
from ai_video_editor.core.config import get_settings, validate_environment, create_default_config
from ai_video_editor.core.exceptions import VideoEditorError, ConfigurationError

console = Console()
logger = get_logger(__name__)


@click.group()
@click.option(
    "--debug", 
    is_flag=True, 
    help="Enable debug logging"
)
@click.option(
    "--config", 
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file"
)
@click.version_option(version="0.1.0", prog_name="AI Video Editor")
@click.pass_context
def cli(ctx: click.Context, debug: bool, config: Optional[Path]):
    """AI Video Editor - Automated video editing with AI assistance."""
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set up logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(log_level)
    
    # Load configuration
    try:
        if config:
            from ai_video_editor.core.config import load_settings
            settings = load_settings(config)
        else:
            settings = get_settings()
        
        ctx.obj["settings"] = settings
        ctx.obj["debug"] = debug
        
        logger.info(f"AI Video Editor v0.1.0 starting...")
        if debug:
            logger.debug(f"Debug mode enabled")
            logger.debug(f"Configuration loaded from: {config or 'environment/defaults'}")
    
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx: click.Context):
    """Check system status and configuration."""
    try:
        env_status = validate_environment()
        
        # Create status table
        table = Table(title="AI Video Editor Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        # Overall status
        overall_status = "✅ Ready" if env_status["valid"] else "❌ Issues Found"
        table.add_row("Overall", overall_status, "")
        
        # API Keys
        for api, available in env_status["api_keys"].items():
            status_icon = "✅" if available else "⚠️"
            status_text = "Configured" if available else "Missing"
            table.add_row(f"{api.title()} API", f"{status_icon} {status_text}", "")
        
        # Directories
        for dir_name, accessible in env_status["directories"].items():
            status_icon = "✅" if accessible else "❌"
            status_text = "Accessible" if accessible else "Not Found"
            table.add_row(f"{dir_name.title()} Directory", f"{status_icon} {status_text}", "")
        
        # System Resources
        memory_gb = env_status["system"]["memory_gb"]
        table.add_row("System Memory", f"✅ {memory_gb}GB", "Available")
        
        console.print(table)
        
        # Print warnings and errors
        if env_status["warnings"]:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in env_status["warnings"]:
                console.print(f"  ⚠️  {warning}")
        
        if env_status["errors"]:
            console.print("\n[red]Errors:[/red]")
            for error in env_status["errors"]:
                console.print(f"  ❌ {error}")
        
        if not env_status["valid"]:
            console.print("\n[red]Please fix the errors above before using the video editor.[/red]")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        console.print(f"[red]Error checking status: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--output", 
    type=click.Path(path_type=Path),
    default=Path(".env"),
    help="Output path for configuration file"
)
def init(output: Path):
    """Initialize configuration file with defaults."""
    try:
        if output.exists():
            if not click.confirm(f"Configuration file {output} already exists. Overwrite?"):
                console.print("Configuration initialization cancelled.")
                return
        
        create_default_config(output)
        console.print(f"[green]Configuration file created at {output}[/green]")
        console.print("\n[yellow]Please edit the configuration file to add your API keys:[/yellow]")
        console.print(f"  - Gemini API key for content analysis")
        console.print(f"  - Imagen API key for thumbnail generation")
        console.print(f"  - Google Cloud Project ID")
        
    except Exception as e:
        logger.error(f"Error creating configuration: {e}")
        console.print(f"[red]Error creating configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("input_files", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory path")
@click.option("--type", "content_type", type=click.Choice(["educational", "music", "general"]), 
              default="general", help="Content type for optimization")
@click.option("--quality", type=click.Choice(["low", "medium", "high", "ultra"]), 
              default="high", help="Output quality")
@click.option("--mode", type=click.Choice(["fast", "balanced", "high_quality"]), 
              default="balanced", help="Processing mode")
@click.option("--parallel/--sequential", default=True, help="Enable parallel processing")
@click.option("--max-memory", type=float, default=8.0, help="Maximum memory usage in GB")
@click.option("--timeout", type=int, default=1800, help="Timeout per stage in seconds")
@click.option("--no-progress", is_flag=True, help="Disable progress display")
@click.pass_context
def process(ctx: click.Context, input_files, output, content_type, quality, mode, 
           parallel, max_memory, timeout, no_progress):
    """Process video files with AI assistance through the complete pipeline."""
    import asyncio
    from ai_video_editor.core.workflow_orchestrator import (
        WorkflowOrchestrator, WorkflowConfiguration, ProcessingMode
    )
    from ai_video_editor.core.config import ProjectSettings, ContentType, VideoQuality
    
    try:
        console.print(f"[green]Starting AI Video Editor pipeline for {len(input_files)} file(s)...[/green]")
        console.print(f"Content type: {content_type}")
        console.print(f"Quality: {quality}")
        console.print(f"Processing mode: {mode}")
        
        # Create project settings
        project_settings = ProjectSettings(
            content_type=ContentType(content_type),
            quality=VideoQuality(quality),
            auto_enhance=True,
            enable_b_roll_generation=True,
            enable_thumbnail_generation=True
        )
        
        # Create workflow configuration
        workflow_config = WorkflowConfiguration(
            processing_mode=ProcessingMode(mode),
            enable_parallel_processing=parallel,
            max_memory_usage_gb=max_memory,
            timeout_per_stage=timeout,
            enable_progress_display=not no_progress,
            output_directory=output,
            temp_directory=Path("temp") / f"workflow_{int(time.time())}"
        )
        
        # Create and run orchestrator
        orchestrator = WorkflowOrchestrator(
            config=workflow_config,
            console=console
        )
        
        # Run the workflow
        async def run_workflow():
            try:
                result_context = await orchestrator.process_video(
                    input_files=list(input_files),
                    project_settings=project_settings
                )
                
                # Display results
                console.print("\n[green]✅ Processing completed successfully![/green]")
                
                # Show processing summary
                summary = orchestrator.get_processing_summary()
                if summary:
                    console.print(f"\n📊 Project ID: {summary.get('project_id', 'N/A')}")
                    console.print(f"⏱️  Total Time: {summary.get('workflow_duration', 0):.1f}s")
                    
                    metrics = summary.get('processing_metrics', {})
                    if metrics:
                        console.print(f"💾 Peak Memory: {metrics.get('memory_peak_usage', 0) / (1024**3):.1f}GB")
                        console.print(f"🔗 API Calls: {sum(metrics.get('api_calls_made', {}).values())}")
                
                return result_context
                
            except Exception as e:
                console.print(f"\n[red]❌ Processing failed: {e}[/red]")
                logger.error(f"Workflow processing failed: {e}")
                raise
        
        # Run the async workflow
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        result = asyncio.run(run_workflow())
        
        if output:
            console.print(f"\n📁 Output saved to: {output}")
        
        console.print("\n[cyan]Use 'ai-video-editor status' to check system status[/cyan]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹️  Processing cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        console.print(f"[red]Error processing files: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output transcript file")
@click.pass_context
def analyze(ctx: click.Context, audio_file, output):
    """Analyze audio content and generate transcript."""
    try:
        console.print(f"[green]Analyzing audio file: {audio_file}[/green]")
        
        # TODO: Implement actual analysis in future tasks
        console.print("[yellow]Audio analysis functionality will be implemented in upcoming tasks.[/yellow]")
        console.print("This will use Whisper Large-V3 for transcription and Gemini API for content analysis.")
        
        if output:
            console.print(f"Transcript will be saved to: {output}")
        
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        console.print(f"[red]Error analyzing audio: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file path")
@click.pass_context
def enhance(ctx: click.Context, input_file, output):
    """Enhance video quality using AI."""
    try:
        console.print(f"[green]Enhancing video: {input_file}[/green]")
        
        # TODO: Implement actual enhancement in future tasks
        console.print("[yellow]Video enhancement functionality will be implemented in upcoming tasks.[/yellow]")
        console.print("This will use OpenCV for automatic color correction and quality improvements.")
        
        if output:
            console.print(f"Enhanced video will be saved to: {output}")
        
    except Exception as e:
        logger.error(f"Error enhancing video: {e}")
        console.print(f"[red]Error enhancing video: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("project_id", required=False)
@click.option("--list", "list_projects", is_flag=True, help="List all active projects")
@click.option("--details", is_flag=True, help="Show detailed status information")
@click.pass_context
def workflow(ctx: click.Context, project_id, list_projects, details):
    """Monitor and manage workflow execution."""
    try:
        if list_projects:
            console.print("[cyan]Active Workflow Projects:[/cyan]")
            console.print("(This will show active workflows when implemented)")
            return
        
        if project_id:
            console.print(f"[cyan]Workflow Status for Project: {project_id}[/cyan]")
            console.print("(This will show specific project status when implemented)")
        else:
            console.print("[yellow]Please specify a project ID or use --list to see all projects[/yellow]")
            console.print("Usage: ai-video-editor workflow <project_id>")
            console.print("       ai-video-editor workflow --list")
        
    except Exception as e:
        logger.error(f"Error checking workflow status: {e}")
        console.print(f"[red]Error checking workflow status: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--stage", type=str, help="Test specific workflow stage")
@click.option("--mock", is_flag=True, help="Use mock data for testing")
@click.pass_context
def test_workflow(ctx: click.Context, stage, mock):
    """Test workflow orchestrator functionality."""
    import asyncio
    from ai_video_editor.core.workflow_orchestrator import (
        WorkflowOrchestrator, WorkflowConfiguration, ProcessingMode
    )
    
    try:
        console.print("[cyan]Testing Workflow Orchestrator...[/cyan]")
        
        # Create test configuration
        config = WorkflowConfiguration(
            processing_mode=ProcessingMode.FAST,
            enable_parallel_processing=False,
            max_memory_usage_gb=4.0,
            timeout_per_stage=60,
            enable_progress_display=True,
            temp_directory=Path("temp") / "test_workflow"
        )
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator(config=config, console=console)
        
        console.print("[green]✅ Workflow orchestrator created successfully[/green]")
        console.print(f"Configuration: {config.processing_mode.value} mode")
        console.print(f"Parallel processing: {config.enable_parallel_processing}")
        console.print(f"Memory limit: {config.max_memory_usage_gb}GB")
        
        if stage:
            console.print(f"[yellow]Stage-specific testing not yet implemented: {stage}[/yellow]")
        
        if mock:
            console.print("[yellow]Mock data testing not yet implemented[/yellow]")
        
        console.print("\n[cyan]Workflow orchestrator test completed successfully![/cyan]")
        
    except Exception as e:
        logger.error(f"Workflow test failed: {e}")
        console.print(f"[red]Workflow test failed: {e}[/red]")
        sys.exit(1)


def main():
    """Main entry point for the CLI application."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(1)
    except VideoEditorError as e:
        logger.error(f"Video editor error: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()