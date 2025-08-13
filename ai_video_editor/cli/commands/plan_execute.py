"""Plan execute command - Convert editing decisions to execution timeline."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from ai_video_editor.utils.logging_config import get_logger
from .utils import load_json, save_json

logger = get_logger(__name__)


@click.command("plan-execute")
@click.option("--ai-plan", "ai_plan_path", type=click.Path(exists=True, path_type=Path), 
              help="AI Director plan JSON with editing_decisions, broll_plans")
@click.option("--decisions", "decisions_path", type=click.Path(exists=True, path_type=Path), 
              help="editing_decisions JSON file")
@click.option("--broll", "broll_path", type=click.Path(exists=True, path_type=Path), 
              help="broll_plans JSON file")
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True, 
              help="Output execution_timeline JSON path")
@click.option("--project-id", default="cli_plan_execution", help="Project ID for context")
def plan_execute_cmd(ai_plan_path: Optional[Path], decisions_path: Optional[Path], 
                     broll_path: Optional[Path], output_path: Path, project_id: str):
    """Convert editing decisions and b-roll plans into an execution timeline."""
    try:
        from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
        from ai_video_editor.modules.video_processing.plan_execution import PlanExecutionEngine

        # Load input data
        if ai_plan_path:
            ai_plan = load_json(ai_plan_path)
            editing_decisions = ai_plan.get("editing_decisions", [])
            broll_plans = ai_plan.get("broll_plans", [])
        else:
            if not decisions_path or not broll_path:
                raise click.UsageError("Provide either --ai-plan or both --decisions and --broll.")
            editing_decisions = load_json(decisions_path)
            broll_plans = load_json(broll_path)

        # Create context
        context = ContentContext(
            project_id=project_id,
            video_files=[],
            content_type=ContentType.GENERAL,
            user_preferences=UserPreferences()
        )
        
        # Attach AI Director-like plan structure expected by execution engine
        context.processed_video = {
            "editing_decisions": editing_decisions,
            "broll_plans": broll_plans,
            # metadata_strategy optional
        }

        # Execute plan
        engine = PlanExecutionEngine()
        timeline = engine.execute_plan(context)

        # Save output
        save_json(timeline.to_dict(), output_path)
        click.echo(f"[OK] Execution timeline written to {output_path}")

    except Exception as e:
        logger.error(f"Plan execution failed: {e}")
        click.echo(f"[ERROR] Plan execution failed: {e}")
        sys.exit(1)