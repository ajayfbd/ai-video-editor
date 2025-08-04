"""Main CLI entry point for AI Video Editor."""

import sys
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
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file path")
@click.option("--type", "content_type", type=click.Choice(["educational", "music", "general"]), 
              default="general", help="Content type for optimization")
@click.option("--quality", type=click.Choice(["low", "medium", "high", "ultra"]), 
              default="high", help="Output quality")
@click.pass_context
def process(ctx: click.Context, input_files, output, content_type, quality):
    """Process video files with AI assistance."""
    try:
        console.print(f"[green]Processing {len(input_files)} file(s)...[/green]")
        console.print(f"Content type: {content_type}")
        console.print(f"Quality: {quality}")
        
        # TODO: Implement actual processing in future tasks
        console.print("[yellow]Processing functionality will be implemented in upcoming tasks.[/yellow]")
        console.print("This is the basic CLI structure for the video editor.")
        
        for i, file_path in enumerate(input_files, 1):
            console.print(f"  {i}. {file_path}")
        
        if output:
            console.print(f"Output will be saved to: {output}")
        
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