"""
B-Roll Generation and Integration System.

This module implements the complete B-roll generation pipeline that creates
actual graphics, charts, animations, and educational slides from AI Director
specifications and integrates them with the VideoComposer.
"""

import logging
import time
import asyncio
import subprocess
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import json
import tempfile

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    animation = None
    np = None
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from ...core.content_context import ContentContext
from ...core.exceptions import ProcessingError, ContentContextError
from ...modules.intelligence.ai_director import BRollPlan
from ...modules.content_analysis.broll_analyzer import BRollOpportunity

logger = logging.getLogger(__name__)


@dataclass
class GeneratedBRollAsset:
    """Represents a generated B-roll asset."""
    
    asset_id: str
    file_path: str
    asset_type: str  # "chart", "animation", "slide", "graphic"
    timestamp: float
    duration: float
    generation_method: str  # "matplotlib", "blender", "pil", "template"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset_id': self.asset_id,
            'file_path': self.file_path,
            'asset_type': self.asset_type,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'generation_method': self.generation_method,
            'metadata': self.metadata
        }


class EnhancedChartGenerator:
    """Enhanced chart generator that works from AI Director specifications."""
    
    def __init__(self, output_dir: str = "output/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - chart generation will be limited")
    
    def generate_from_ai_specification(self, broll_plan: BRollPlan, context: ContentContext) -> str:
        """Generate chart from AI Director specification."""
        if not MATPLOTLIB_AVAILABLE:
            return self._create_placeholder_chart(broll_plan)
        
        try:
            # Extract chart specifications from AI Director plan
            chart_type = self._determine_chart_type(broll_plan)
            data = self._extract_data_from_context(broll_plan, context)
            
            if chart_type == "compound_interest":
                return self._create_compound_interest_chart(broll_plan, data)
            elif chart_type == "portfolio_allocation":
                return self._create_portfolio_chart(broll_plan, data)
            elif chart_type == "risk_return":
                return self._create_risk_return_chart(broll_plan, data)
            elif chart_type == "growth_comparison":
                return self._create_growth_comparison_chart(broll_plan, data)
            else:
                return self._create_generic_chart(broll_plan, data)
                
        except Exception as e:
            logger.error(f"Chart generation failed: {str(e)}")
            return self._create_placeholder_chart(broll_plan)
    
    def _determine_chart_type(self, broll_plan: BRollPlan) -> str:
        """Determine chart type from B-roll plan description."""
        description = broll_plan.description.lower()
        
        if "compound" in description or "growth" in description:
            return "compound_interest"
        elif "portfolio" in description or "allocation" in description or "diversif" in description:
            return "portfolio_allocation"
        elif "risk" in description and "return" in description:
            return "risk_return"
        elif "comparison" in description or "versus" in description:
            return "growth_comparison"
        else:
            return "generic"
    
    def _extract_data_from_context(self, broll_plan: BRollPlan, context: ContentContext) -> Dict[str, Any]:
        """Extract relevant data from context for chart generation."""
        # Default financial data - in a real implementation, this would be extracted
        # from the audio analysis and AI Director specifications
        return {
            "compound_interest": {
                "principal": 10000,
                "rate": 0.07,
                "years": 30
            },
            "portfolio": {
                "Stocks": 60.0,
                "Bonds": 25.0,
                "Real Estate": 10.0,
                "Cash": 5.0
            },
            "risk_return": {
                "Government Bonds": {"risk": 2, "return": 3},
                "Corporate Bonds": {"risk": 4, "return": 5},
                "Index Funds": {"risk": 6, "return": 7},
                "Individual Stocks": {"risk": 8, "return": 9}
            }
        }
    
    def _create_compound_interest_chart(self, broll_plan: BRollPlan, data: Dict[str, Any]) -> str:
        """Create compound interest growth chart."""
        compound_data = data.get("compound_interest", {})
        principal = compound_data.get("principal", 10000)
        rate = compound_data.get("rate", 0.07)
        years = compound_data.get("years", 30)
        
        # Generate data
        time_years = np.arange(0, years + 1)
        amounts = principal * (1 + rate) ** time_years
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(time_years, amounts, 'b-', linewidth=3, marker='o', markersize=4)
        
        ax.set_xlabel('Years', fontsize=14)
        ax.set_ylabel('Investment Value ($)', fontsize=14)
        ax.set_title('Compound Interest Growth', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add annotations
        final_value = amounts[-1]
        ax.annotate(f'Final Value: ${final_value:,.0f}', 
                   xy=(years, final_value), xytext=(years-5, final_value*0.8),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        output_path = self.output_dir / f"compound_interest_{int(broll_plan.timestamp)}.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated compound interest chart: {output_path}")
        return str(output_path)
    
    def _create_portfolio_chart(self, broll_plan: BRollPlan, data: Dict[str, Any]) -> str:
        """Create portfolio allocation pie chart."""
        portfolio_data = data.get("portfolio", {})
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = list(portfolio_data.keys())
        sizes = list(portfolio_data.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors[:len(labels)],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12}
        )
        
        ax.set_title('Portfolio Allocation', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"portfolio_{int(broll_plan.timestamp)}.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated portfolio chart: {output_path}")
        return str(output_path)
    
    def _create_risk_return_chart(self, broll_plan: BRollPlan, data: Dict[str, Any]) -> str:
        """Create risk vs return scatter plot."""
        risk_return_data = data.get("risk_return", {})
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        names = list(risk_return_data.keys())
        risks = [risk_return_data[name]['risk'] for name in names]
        returns = [risk_return_data[name]['return'] for name in names]
        
        scatter = ax.scatter(risks, returns, s=100, alpha=0.7, c=range(len(names)), cmap='viridis')
        
        for i, name in enumerate(names):
            ax.annotate(name, (risks[i], returns[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Risk Level', fontsize=14)
        ax.set_ylabel('Expected Return (%)', fontsize=14)
        ax.set_title('Risk vs Return Analysis', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"risk_return_{int(broll_plan.timestamp)}.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated risk-return chart: {output_path}")
        return str(output_path)
    
    def _create_growth_comparison_chart(self, broll_plan: BRollPlan, data: Dict[str, Any]) -> str:
        """Create growth comparison chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        years = np.arange(0, 31)
        conservative = 10000 * (1.04) ** years
        moderate = 10000 * (1.07) ** years
        aggressive = 10000 * (1.10) ** years
        
        ax.plot(years, conservative, 'g-', linewidth=2, label='Conservative (4%)')
        ax.plot(years, moderate, 'b-', linewidth=2, label='Moderate (7%)')
        ax.plot(years, aggressive, 'r-', linewidth=2, label='Aggressive (10%)')
        
        ax.set_xlabel('Years', fontsize=14)
        ax.set_ylabel('Investment Value ($)', fontsize=14)
        ax.set_title('Investment Strategy Comparison', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"growth_comparison_{int(broll_plan.timestamp)}.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated growth comparison chart: {output_path}")
        return str(output_path)
    
    def _create_generic_chart(self, broll_plan: BRollPlan, data: Dict[str, Any]) -> str:
        """Create generic chart when specific type cannot be determined."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simple bar chart with sample data
        categories = ['Category A', 'Category B', 'Category C', 'Category D']
        values = [23, 45, 56, 78]
        
        bars = ax.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        
        ax.set_title('Financial Data Visualization', fontsize=16, fontweight='bold')
        ax.set_ylabel('Value', fontsize=14)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value}', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"generic_chart_{int(broll_plan.timestamp)}.png"
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Generated generic chart: {output_path}")
        return str(output_path)
    
    def _create_placeholder_chart(self, broll_plan: BRollPlan) -> str:
        """Create placeholder chart when matplotlib is not available."""
        if not PIL_AVAILABLE:
            # Create empty file as last resort
            output_path = self.output_dir / f"placeholder_{int(broll_plan.timestamp)}.txt"
            with open(output_path, 'w') as f:
                f.write(f"Chart placeholder for: {broll_plan.description}")
            return str(output_path)
        
        # Create simple placeholder image with PIL
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Draw placeholder content
        draw.rectangle([50, 50, 750, 550], outline='black', width=2)
        draw.text((400, 200), "Chart Placeholder", fill='black', font=font, anchor='mm')
        draw.text((400, 250), broll_plan.description[:50], fill='gray', font=font, anchor='mm')
        
        output_path = self.output_dir / f"placeholder_{int(broll_plan.timestamp)}.png"
        img.save(str(output_path))
        
        logger.info(f"Generated placeholder chart: {output_path}")
        return str(output_path)


class BlenderRenderingPipeline:
    """Blender animation rendering pipeline."""
    
    def __init__(self, output_dir: str = "output/animations", blender_executable: str = "blender"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.blender_executable = blender_executable
        
        # Check if Blender is available
        self.blender_available = self._check_blender_availability()
    
    def _check_blender_availability(self) -> bool:
        """Check if Blender is available on the system."""
        try:
            result = subprocess.run([self.blender_executable, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Blender not found - animation rendering will use placeholders")
            return False
    
    def render_animation(self, broll_plan: BRollPlan, context: ContentContext) -> str:
        """Render Blender animation from B-roll plan."""
        if not self.blender_available:
            return self._create_placeholder_animation(broll_plan)
        
        try:
            # Generate Blender script
            script_content = self._generate_blender_script(broll_plan, context)
            
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                script_file.write(script_content)
                script_path = script_file.name
            
            # Execute Blender rendering
            output_path = self.output_dir / f"animation_{int(broll_plan.timestamp)}.mp4"
            
            cmd = [
                self.blender_executable,
                "--background",
                "--python", script_path,
                "--render-output", str(output_path.with_suffix('')),
                "--render-format", "FFMPEG",
                "--render-anim"
            ]
            
            logger.info(f"Executing Blender render: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Clean up script file
            Path(script_path).unlink()
            
            if result.returncode == 0:
                logger.info(f"Blender animation rendered: {output_path}")
                return str(output_path)
            else:
                logger.error(f"Blender rendering failed: {result.stderr}")
                return self._create_placeholder_animation(broll_plan)
                
        except Exception as e:
            logger.error(f"Blender rendering error: {str(e)}")
            return self._create_placeholder_animation(broll_plan)
    
    def _generate_blender_script(self, broll_plan: BRollPlan, context: ContentContext) -> str:
        """Generate Blender Python script for animation."""
        animation_type = self._determine_animation_type(broll_plan)
        
        if animation_type == "financial_concept":
            return self._generate_financial_concept_script(broll_plan)
        elif animation_type == "process_flow":
            return self._generate_process_flow_script(broll_plan)
        else:
            return self._generate_generic_animation_script(broll_plan)
    
    def _determine_animation_type(self, broll_plan: BRollPlan) -> str:
        """Determine animation type from B-roll plan."""
        description = broll_plan.description.lower()
        
        if any(concept in description for concept in ["compound", "interest", "growth", "investment"]):
            return "financial_concept"
        elif any(process in description for process in ["step", "process", "how to", "method"]):
            return "process_flow"
        else:
            return "generic"
    
    def _generate_financial_concept_script(self, broll_plan: BRollPlan) -> str:
        """Generate script for financial concept animation."""
        return f'''
import bpy
import bmesh
from mathutils import Vector
import math

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Scene setup
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = {int(broll_plan.duration * 24)}

# Add camera
bpy.ops.object.camera_add(location=(7, -7, 5))
camera = bpy.context.object
camera.rotation_euler = (1.1, 0, 0.785)

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
light = bpy.context.object
light.data.energy = 3

# Create financial visualization
# Main object representing growth
bpy.ops.mesh.primitive_cylinder_add(location=(0, 0, 0))
growth_cylinder = bpy.context.object
growth_cylinder.name = "GrowthVisualization"

# Animate growth
growth_cylinder.scale = (1, 1, 0.1)
growth_cylinder.keyframe_insert(data_path="scale", frame=1)

growth_cylinder.scale = (1, 1, 3)
growth_cylinder.keyframe_insert(data_path="scale", frame={int(broll_plan.duration * 24)})

# Add material
material = bpy.data.materials.new(name="GrowthMaterial")
material.use_nodes = True
material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.2, 0.8, 0.2, 1.0)
growth_cylinder.data.materials.append(material)

# Set render settings
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.fps = 24
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'

print("Financial concept animation setup complete")
'''
    
    def _generate_process_flow_script(self, broll_plan: BRollPlan) -> str:
        """Generate script for process flow animation."""
        return f'''
import bpy
import bmesh
from mathutils import Vector

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Scene setup
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = {int(broll_plan.duration * 24)}

# Add camera
bpy.ops.object.camera_add(location=(10, -10, 8))
camera = bpy.context.object
camera.rotation_euler = (1.0, 0, 0.785)

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

# Create process steps
step_positions = [(-4, 0, 0), (0, 0, 0), (4, 0, 0)]
steps = []

for i, pos in enumerate(step_positions):
    bpy.ops.mesh.primitive_cube_add(location=pos)
    step = bpy.context.object
    step.name = f"Step{{i+1}}"
    steps.append(step)
    
    # Animate appearance
    step.scale = (0, 0, 0)
    step.keyframe_insert(data_path="scale", frame=1 + i * 20)
    
    step.scale = (1, 1, 1)
    step.keyframe_insert(data_path="scale", frame=20 + i * 20)

# Set render settings
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.fps = 24
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'

print("Process flow animation setup complete")
'''
    
    def _generate_generic_animation_script(self, broll_plan: BRollPlan) -> str:
        """Generate generic animation script."""
        return f'''
import bpy
from mathutils import Vector

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Scene setup
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = {int(broll_plan.duration * 24)}

# Add camera
bpy.ops.object.camera_add(location=(7, -7, 5))
camera = bpy.context.object
camera.rotation_euler = (1.1, 0, 0.785)

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

# Create main object
bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
main_object = bpy.context.object

# Simple rotation animation
main_object.rotation_euler = (0, 0, 0)
main_object.keyframe_insert(data_path="rotation_euler", frame=1)

main_object.rotation_euler = (0, 0, 6.28)  # Full rotation
main_object.keyframe_insert(data_path="rotation_euler", frame={int(broll_plan.duration * 24)})

# Set render settings
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.fps = 24
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'

print("Generic animation setup complete")
'''
    
    def _create_placeholder_animation(self, broll_plan: BRollPlan) -> str:
        """Create placeholder animation when Blender is not available."""
        if PIL_AVAILABLE:
            # Create simple animated GIF placeholder
            frames = []
            for i in range(10):
                img = Image.new('RGB', (800, 600), color='lightblue')
                draw = ImageDraw.Draw(img)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                draw.text((400, 250), f"Animation Frame {i+1}", fill='black', font=font, anchor='mm')
                draw.text((400, 300), broll_plan.description[:50], fill='gray', font=font, anchor='mm')
                frames.append(img)
            
            output_path = self.output_dir / f"placeholder_animation_{int(broll_plan.timestamp)}.gif"
            frames[0].save(str(output_path), save_all=True, append_images=frames[1:], duration=500, loop=0)
            
            logger.info(f"Generated placeholder animation: {output_path}")
            return str(output_path)
        else:
            # Create text placeholder
            output_path = self.output_dir / f"placeholder_animation_{int(broll_plan.timestamp)}.txt"
            with open(output_path, 'w') as f:
                f.write(f"Animation placeholder for: {broll_plan.description}")
            return str(output_path)


class EducationalSlideSystem:
    """Enhanced educational slide generation system."""
    
    def __init__(self, output_dir: str = "output/slides"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_educational_slide(self, broll_plan: BRollPlan, context: ContentContext) -> str:
        """Generate educational slide from B-roll plan."""
        slide_type = self._determine_slide_type(broll_plan)
        
        if slide_type == "concept_explanation":
            return self._create_concept_slide(broll_plan)
        elif slide_type == "step_by_step":
            return self._create_step_slide(broll_plan)
        elif slide_type == "comparison":
            return self._create_comparison_slide(broll_plan)
        else:
            return self._create_generic_slide(broll_plan)
    
    def _determine_slide_type(self, broll_plan: BRollPlan) -> str:
        """Determine slide type from B-roll plan."""
        description = broll_plan.description.lower()
        
        if any(concept in description for concept in ["concept", "definition", "explain"]):
            return "concept_explanation"
        elif any(step in description for step in ["step", "process", "how to"]):
            return "step_by_step"
        elif any(comp in description for comp in ["versus", "compare", "difference"]):
            return "comparison"
        else:
            return "generic"
    
    def _create_concept_slide(self, broll_plan: BRollPlan) -> str:
        """Create concept explanation slide."""
        if not PIL_AVAILABLE:
            return self._create_text_slide(broll_plan)
        
        # Create slide image
        img = Image.new('RGB', (1920, 1080), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 72)
            content_font = ImageFont.truetype("arial.ttf", 36)
        except:
            title_font = ImageFont.load_default()
            content_font = ImageFont.load_default()
        
        # Extract concept from description
        concept = self._extract_concept(broll_plan.description)
        
        # Draw background
        draw.rectangle([0, 0, 1920, 200], fill='#2c3e50')
        
        # Draw title
        draw.text((960, 100), concept, fill='white', font=title_font, anchor='mm')
        
        # Draw content
        content_lines = self._wrap_text(broll_plan.description, 80)
        y_start = 300
        
        for i, line in enumerate(content_lines[:8]):  # Limit to 8 lines
            draw.text((960, y_start + i * 60), line, fill='black', font=content_font, anchor='mm')
        
        # Add decorative elements
        draw.rectangle([100, 900, 1820, 920], fill='#3498db')
        
        output_path = self.output_dir / f"concept_slide_{int(broll_plan.timestamp)}.png"
        img.save(str(output_path))
        
        logger.info(f"Generated concept slide: {output_path}")
        return str(output_path)
    
    def _create_step_slide(self, broll_plan: BRollPlan) -> str:
        """Create step-by-step slide."""
        if not PIL_AVAILABLE:
            return self._create_text_slide(broll_plan)
        
        img = Image.new('RGB', (1920, 1080), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 64)
            step_font = ImageFont.truetype("arial.ttf", 32)
        except:
            title_font = ImageFont.load_default()
            step_font = ImageFont.load_default()
        
        # Draw title
        draw.rectangle([0, 0, 1920, 150], fill='#34495e')
        draw.text((960, 75), "Process Steps", fill='white', font=title_font, anchor='mm')
        
        # Create steps from description
        steps = self._extract_steps(broll_plan.description)
        
        y_start = 250
        for i, step in enumerate(steps[:6]):  # Limit to 6 steps
            # Draw step number circle
            circle_center = (200, y_start + i * 120)
            draw.ellipse([circle_center[0]-30, circle_center[1]-30, 
                         circle_center[0]+30, circle_center[1]+30], fill='#3498db')
            draw.text(circle_center, str(i+1), fill='white', font=step_font, anchor='mm')
            
            # Draw step text
            draw.text((300, y_start + i * 120), step, fill='black', font=step_font, anchor='lm')
        
        output_path = self.output_dir / f"step_slide_{int(broll_plan.timestamp)}.png"
        img.save(str(output_path))
        
        logger.info(f"Generated step slide: {output_path}")
        return str(output_path)
    
    def _create_comparison_slide(self, broll_plan: BRollPlan) -> str:
        """Create comparison slide."""
        if not PIL_AVAILABLE:
            return self._create_text_slide(broll_plan)
        
        img = Image.new('RGB', (1920, 1080), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 64)
            content_font = ImageFont.truetype("arial.ttf", 28)
        except:
            title_font = ImageFont.load_default()
            content_font = ImageFont.load_default()
        
        # Draw title
        draw.rectangle([0, 0, 1920, 150], fill='#e74c3c')
        draw.text((960, 75), "Comparison", fill='white', font=title_font, anchor='mm')
        
        # Draw comparison columns
        draw.line([960, 150, 960, 1080], fill='black', width=3)
        
        # Left column
        draw.text((480, 200), "Option A", fill='#2c3e50', font=title_font, anchor='mm')
        # Right column
        draw.text((1440, 200), "Option B", fill='#2c3e50', font=title_font, anchor='mm')
        
        # Add comparison content
        left_content = ["Lower Risk", "Steady Returns", "Conservative"]
        right_content = ["Higher Risk", "Variable Returns", "Aggressive"]
        
        for i, (left, right) in enumerate(zip(left_content, right_content)):
            y_pos = 350 + i * 80
            draw.text((480, y_pos), left, fill='black', font=content_font, anchor='mm')
            draw.text((1440, y_pos), right, fill='black', font=content_font, anchor='mm')
        
        output_path = self.output_dir / f"comparison_slide_{int(broll_plan.timestamp)}.png"
        img.save(str(output_path))
        
        logger.info(f"Generated comparison slide: {output_path}")
        return str(output_path)
    
    def _create_generic_slide(self, broll_plan: BRollPlan) -> str:
        """Create generic slide."""
        if not PIL_AVAILABLE:
            return self._create_text_slide(broll_plan)
        
        img = Image.new('RGB', (1920, 1080), color='#f8f9fa')
        draw = ImageDraw.Draw(img)
        
        try:
            title_font = ImageFont.truetype("arial.ttf", 64)
            content_font = ImageFont.truetype("arial.ttf", 32)
        except:
            title_font = ImageFont.load_default()
            content_font = ImageFont.load_default()
        
        # Draw title area
        draw.rectangle([0, 0, 1920, 200], fill='#2c3e50')
        draw.text((960, 100), "Educational Content", fill='white', font=title_font, anchor='mm')
        
        # Draw content
        content_lines = self._wrap_text(broll_plan.description, 70)
        y_start = 350
        
        for i, line in enumerate(content_lines[:10]):
            draw.text((960, y_start + i * 50), line, fill='#2c3e50', font=content_font, anchor='mm')
        
        output_path = self.output_dir / f"generic_slide_{int(broll_plan.timestamp)}.png"
        img.save(str(output_path))
        
        logger.info(f"Generated generic slide: {output_path}")
        return str(output_path)
    
    def _create_text_slide(self, broll_plan: BRollPlan) -> str:
        """Create text-based slide when PIL is not available."""
        output_path = self.output_dir / f"text_slide_{int(broll_plan.timestamp)}.txt"
        with open(output_path, 'w') as f:
            f.write(f"Educational Slide\n")
            f.write(f"================\n\n")
            f.write(f"Content: {broll_plan.description}\n")
            f.write(f"Duration: {broll_plan.duration}s\n")
            f.write(f"Type: {broll_plan.content_type}\n")
        
        logger.info(f"Generated text slide: {output_path}")
        return str(output_path)
    
    def _extract_concept(self, description: str) -> str:
        """Extract main concept from description."""
        # Simple concept extraction
        words = description.split()
        if len(words) > 0:
            return " ".join(words[:3]).title()
        return "Financial Concept"
    
    def _extract_steps(self, description: str) -> List[str]:
        """Extract steps from description."""
        # Simple step extraction
        sentences = description.split('.')
        steps = []
        for sentence in sentences[:6]:
            if sentence.strip():
                steps.append(sentence.strip())
        
        if not steps:
            steps = ["Step 1: Analyze", "Step 2: Plan", "Step 3: Execute"]
        
        return steps
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines


class BRollGenerationSystem:
    """
    Complete B-roll generation and integration system.
    
    This class coordinates all B-roll generation components and provides
    integration with VideoComposer for seamless video composition.
    """
    
    def __init__(self, output_dir: str = "output/broll"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize generation components
        self.chart_generator = EnhancedChartGenerator(
            output_dir=str(self.output_dir / "charts")
        )
        self.blender_pipeline = BlenderRenderingPipeline(
            output_dir=str(self.output_dir / "animations")
        )
        self.slide_system = EducationalSlideSystem(
            output_dir=str(self.output_dir / "slides")
        )
        
        # Track generated assets
        self.generated_assets: List[GeneratedBRollAsset] = []
        
        # Performance metrics
        self.generation_time = 0.0
        self.total_assets_generated = 0
        
        logger.info("BRollGenerationSystem initialized with all components")
    
    async def generate_all_broll_assets(self, context: ContentContext) -> List[GeneratedBRollAsset]:
        """
        Generate all B-roll assets from AI Director plans in ContentContext.
        
        Args:
            context: ContentContext with AI Director B-roll plans
            
        Returns:
            List of generated B-roll assets with file paths
        """
        start_time = time.time()
        
        if not context.processed_video or 'broll_plans' not in context.processed_video:
            logger.warning("No B-roll plans found in ContentContext")
            return []
        
        broll_plans_data = context.processed_video['broll_plans']
        logger.info(f"Generating B-roll assets for {len(broll_plans_data)} plans")
        
        # Convert dict data to BRollPlan objects
        broll_plans = []
        for plan_data in broll_plans_data:
            plan = BRollPlan(
                timestamp=plan_data['timestamp'],
                duration=plan_data['duration'],
                content_type=plan_data['content_type'],
                description=plan_data['description'],
                visual_elements=plan_data.get('visual_elements', []),
                animation_style=plan_data.get('animation_style', 'fade_in'),
                priority=plan_data.get('priority', 5)
            )
            broll_plans.append(plan)
        
        # Generate assets for each plan
        generated_assets = []
        
        for plan in broll_plans:
            try:
                asset = await self._generate_single_broll_asset(plan, context)
                if asset:
                    generated_assets.append(asset)
                    self.generated_assets.append(asset)
            except Exception as e:
                logger.error(f"Failed to generate asset for plan at {plan.timestamp}s: {str(e)}")
        
        self.generation_time = time.time() - start_time
        self.total_assets_generated = len(generated_assets)
        
        logger.info(f"Generated {len(generated_assets)} B-roll assets in {self.generation_time:.2f}s")
        
        return generated_assets
    
    async def _generate_single_broll_asset(self, broll_plan: BRollPlan, context: ContentContext) -> Optional[GeneratedBRollAsset]:
        """Generate single B-roll asset from plan."""
        try:
            # Determine generation method based on content type
            if broll_plan.content_type in ['chart_or_graph', 'chart', 'graph']:
                file_path = self.chart_generator.generate_from_ai_specification(broll_plan, context)
                generation_method = "matplotlib"
                asset_type = "chart"
                
            elif broll_plan.content_type in ['animation', 'animated_explanation']:
                file_path = await asyncio.get_event_loop().run_in_executor(
                    None, self.blender_pipeline.render_animation, broll_plan, context
                )
                generation_method = "blender"
                asset_type = "animation"
                
            elif broll_plan.content_type in ['slide', 'educational_slide', 'step_by_step_visual']:
                file_path = self.slide_system.generate_educational_slide(broll_plan, context)
                generation_method = "pil"
                asset_type = "slide"
                
            else:
                # Default to slide generation for unknown types
                file_path = self.slide_system.generate_educational_slide(broll_plan, context)
                generation_method = "pil"
                asset_type = "slide"
            
            # Create asset record
            asset = GeneratedBRollAsset(
                asset_id=f"broll_{int(broll_plan.timestamp)}_{asset_type}",
                file_path=file_path,
                asset_type=asset_type,
                timestamp=broll_plan.timestamp,
                duration=broll_plan.duration,
                generation_method=generation_method,
                metadata={
                    'content_type': broll_plan.content_type,
                    'description': broll_plan.description,
                    'visual_elements': broll_plan.visual_elements,
                    'animation_style': broll_plan.animation_style,
                    'priority': broll_plan.priority
                }
            )
            
            logger.info(f"Generated {asset_type} asset: {Path(file_path).name}")
            return asset
            
        except Exception as e:
            logger.error(f"Error generating B-roll asset: {str(e)}")
            return None
    
    def get_asset_for_timestamp(self, timestamp: float, tolerance: float = 0.5) -> Optional[GeneratedBRollAsset]:
        """Get generated asset for specific timestamp."""
        for asset in self.generated_assets:
            if abs(asset.timestamp - timestamp) <= tolerance:
                return asset
        return None
    
    def get_assets_in_range(self, start_time: float, end_time: float) -> List[GeneratedBRollAsset]:
        """Get all assets within time range."""
        assets_in_range = []
        for asset in self.generated_assets:
            if start_time <= asset.timestamp <= end_time:
                assets_in_range.append(asset)
        return assets_in_range
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about B-roll generation."""
        asset_types = {}
        generation_methods = {}
        
        for asset in self.generated_assets:
            asset_types[asset.asset_type] = asset_types.get(asset.asset_type, 0) + 1
            generation_methods[asset.generation_method] = generation_methods.get(asset.generation_method, 0) + 1
        
        return {
            'total_generation_time': self.generation_time,
            'total_assets_generated': self.total_assets_generated,
            'asset_types': asset_types,
            'generation_methods': generation_methods,
            'chart_generator_available': MATPLOTLIB_AVAILABLE,
            'blender_available': self.blender_pipeline.blender_available,
            'pil_available': PIL_AVAILABLE
        }
    
    def cleanup_generated_assets(self):
        """Clean up generated asset files."""
        cleaned_count = 0
        for asset in self.generated_assets:
            try:
                if Path(asset.file_path).exists():
                    Path(asset.file_path).unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to clean up asset {asset.file_path}: {str(e)}")
        
        self.generated_assets.clear()
        logger.info(f"Cleaned up {cleaned_count} generated assets")