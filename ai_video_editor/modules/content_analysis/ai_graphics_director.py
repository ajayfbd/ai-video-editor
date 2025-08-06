"""
AI Graphics Director - AI-powered graphics generation and animation planning.

This module implements the AIGraphicsDirector class that uses AI to determine
and create appropriate graphics for B-roll opportunities, including matplotlib
charts, movis motion graphics, and Blender animation specifications.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import asyncio

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

from ...modules.intelligence.gemini_client import GeminiClient, GeminiConfig
from ...core.content_context import ContentContext
from ...core.exceptions import ProcessingError, GeminiAPIError
from .broll_analyzer import BRollOpportunity

logger = logging.getLogger(__name__)


@dataclass
class GraphicsSpecification:
    """Detailed specifications for graphics generation."""
    
    graphics_type: str
    chart_type: Optional[str] = None
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    animation_style: str = "fade_in"
    duration: float = 5.0
    color_scheme: List[str] = field(default_factory=lambda: ["#1f77b4", "#ff7f0e", "#2ca02c"])
    text_overlays: List[Dict[str, str]] = field(default_factory=list)
    visual_hierarchy: Dict[str, Any] = field(default_factory=dict)
    key_points: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'graphics_type': self.graphics_type,
            'chart_type': self.chart_type,
            'data_points': self.data_points,
            'animation_style': self.animation_style,
            'duration': self.duration,
            'color_scheme': self.color_scheme,
            'text_overlays': self.text_overlays,
            'visual_hierarchy': self.visual_hierarchy,
            'key_points': self.key_points
        }


@dataclass
class FinancialGraphicsGenerator:
    """Generator for financial education graphics and charts."""
    
    output_dir: str = "output/graphics"
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def create_compound_interest_animation(self, 
                                         principal: float = 10000,
                                         rate: float = 0.07,
                                         years: int = 30,
                                         output_filename: str = "compound_interest.mp4") -> str:
        """Create animated compound interest growth chart."""
        if not MATPLOTLIB_AVAILABLE:
            raise ProcessingError("Matplotlib is required for graphics generation")
        
        # Generate data
        time_years = np.arange(0, years + 1)
        amounts = principal * (1 + rate) ** time_years
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up the plot
        ax.set_xlim(0, years)
        ax.set_ylim(0, amounts[-1] * 1.1)
        ax.set_xlabel('Years', fontsize=14)
        ax.set_ylabel('Investment Value ($)', fontsize=14)
        ax.set_title('Compound Interest Growth Over Time', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Create line and point objects for animation
        line, = ax.plot([], [], 'b-', linewidth=3, label=f'{rate*100:.1f}% Annual Return')
        points, = ax.plot([], [], 'ro', markersize=8)
        
        # Add text annotations
        value_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Animation function
        def animate(frame):
            x_data = time_years[:frame+1]
            y_data = amounts[:frame+1]
            
            line.set_data(x_data, y_data)
            points.set_data(x_data, y_data)
            
            if frame < len(amounts):
                value_text.set_text(f'Year {frame}: ${amounts[frame]:,.0f}')
            
            return line, points, value_text
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(time_years), 
            interval=200, blit=True, repeat=True
        )
        
        # Save animation
        output_path = Path(self.output_dir) / output_filename
        anim.save(str(output_path), writer='ffmpeg', fps=5)
        
        plt.close(fig)
        logger.info(f"Compound interest animation saved to {output_path}")
        
        return str(output_path)
    
    def create_diversification_chart(self, 
                                   portfolio_data: Dict[str, float],
                                   output_filename: str = "diversification.png") -> str:
        """Create portfolio diversification pie chart."""
        if not MATPLOTLIB_AVAILABLE:
            raise ProcessingError("Matplotlib is required for graphics generation")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = list(portfolio_data.keys())
        sizes = list(portfolio_data.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors[:len(labels)],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12}
        )
        
        ax.set_title('Portfolio Diversification', fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend(wedges, labels, title="Asset Classes", loc="center left", 
                 bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        # Save chart
        output_path = Path(self.output_dir) / output_filename
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Diversification chart saved to {output_path}")
        return str(output_path)
    
    def create_risk_return_scatter(self,
                                 investments: Dict[str, Dict[str, float]],
                                 output_filename: str = "risk_return.png") -> str:
        """Create risk vs return scatter plot."""
        if not MATPLOTLIB_AVAILABLE:
            raise ProcessingError("Matplotlib is required for graphics generation")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        names = list(investments.keys())
        risks = [investments[name]['risk'] for name in names]
        returns = [investments[name]['return'] for name in names]
        
        # Create scatter plot
        scatter = ax.scatter(risks, returns, s=100, alpha=0.7, c=range(len(names)), cmap='viridis')
        
        # Add labels for each point
        for i, name in enumerate(names):
            ax.annotate(name, (risks[i], returns[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Risk (Standard Deviation)', fontsize=14)
        ax.set_ylabel('Expected Return (%)', fontsize=14)
        ax.set_title('Risk vs Return Analysis', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        output_path = Path(self.output_dir) / output_filename
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Risk-return chart saved to {output_path}")
        return str(output_path)


@dataclass
class EducationalSlideGenerator:
    """Generator for educational slides and concept explanations."""
    
    output_dir: str = "output/slides"
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def create_financial_concept_slide(self,
                                     concept: str,
                                     explanation: str,
                                     output_filename: str = "concept_slide.png") -> str:
        """Create educational slide for financial concept."""
        if not MATPLOTLIB_AVAILABLE:
            raise ProcessingError("Matplotlib is required for slide generation")
        
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Background
        ax.add_patch(plt.Rectangle((0, 0), 10, 10, facecolor='#f8f9fa', alpha=0.9))
        
        # Title
        ax.text(5, 8.5, concept, fontsize=32, fontweight='bold', 
               ha='center', va='center', color='#2c3e50')
        
        # Explanation text (wrap long text)
        words = explanation.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 80:  # Wrap at ~80 characters
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Display explanation text
        y_start = 6.5
        for i, line in enumerate(lines[:4]):  # Limit to 4 lines
            ax.text(5, y_start - i * 0.8, line, fontsize=18, 
                   ha='center', va='center', color='#34495e')
        
        # Add decorative elements
        ax.add_patch(plt.Rectangle((1, 1), 8, 0.2, facecolor='#3498db', alpha=0.8))
        
        plt.tight_layout()
        
        # Save slide
        output_path = Path(self.output_dir) / output_filename
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        logger.info(f"Concept slide saved to {output_path}")
        return str(output_path)


class AIGraphicsDirector:
    """
    AI-powered graphics director for automated B-roll generation.
    
    This class uses Gemini AI to determine appropriate graphics based on content
    analysis and generates matplotlib charts, educational slides, and motion
    graphics specifications.
    """
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "output"):
        """
        Initialize AIGraphicsDirector.
        
        Args:
            api_key: Gemini API key for AI-powered graphics planning
            output_dir: Directory for generated graphics outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize AI client
        if api_key:
            config = GeminiConfig(api_key=api_key)
            self.gemini_client = GeminiClient(config)
        else:
            self.gemini_client = None
            logger.warning("No API key provided - using template-based graphics generation")
        
        # Initialize generators
        self.graphics_generator = FinancialGraphicsGenerator(
            output_dir=str(self.output_dir / "charts")
        )
        self.slide_generator = EducationalSlideGenerator(
            output_dir=str(self.output_dir / "slides")
        )
        
        # Template graphics specifications
        self.template_specs = {
            'compound_interest': {
                'chart_type': 'line_chart',
                'data_structure': 'time_series',
                'key_elements': ['growth_curve', 'time_axis', 'value_labels'],
                'animation': 'progressive_reveal'
            },
            'diversification': {
                'chart_type': 'pie_chart',
                'data_structure': 'categorical',
                'key_elements': ['asset_classes', 'percentages', 'legend'],
                'animation': 'segment_reveal'
            },
            'risk_return': {
                'chart_type': 'scatter_plot',
                'data_structure': 'xy_coordinates',
                'key_elements': ['data_points', 'axes_labels', 'trend_line'],
                'animation': 'point_by_point'
            }
        }
        
        # Processing metrics
        self.generation_time = 0.0
        self.graphics_created = 0
        
        logger.info("AIGraphicsDirector initialized")
    
    async def generate_contextual_graphics(self, 
                                         broll_opportunity: BRollOpportunity,
                                         context: ContentContext) -> List[str]:
        """
        Generate contextual graphics for B-roll opportunity using AI guidance.
        
        Args:
            broll_opportunity: Detected B-roll opportunity
            context: ContentContext with analysis results
            
        Returns:
            List of file paths to generated graphics
        """
        start_time = time.time()
        
        logger.info(f"Generating graphics for {broll_opportunity.opportunity_type}")
        
        try:
            # Get AI-powered graphics specification
            if self.gemini_client:
                graphics_spec = await self._generate_ai_graphics_spec(broll_opportunity, context)
            else:
                graphics_spec = self._generate_template_graphics_spec(broll_opportunity)
            
            # Create graphics based on specification
            graphics_files = await self._create_graphics_from_spec(graphics_spec, broll_opportunity)
            
            self.generation_time += time.time() - start_time
            self.graphics_created += len(graphics_files)
            
            logger.info(f"Generated {len(graphics_files)} graphics in {time.time() - start_time:.2f}s")
            
            return graphics_files
            
        except Exception as e:
            logger.error(f"Error generating graphics: {str(e)}")
            # Fallback to simple template graphics
            return await self._create_fallback_graphics(broll_opportunity)
    
    async def _generate_ai_graphics_spec(self, 
                                       opportunity: BRollOpportunity,
                                       context: ContentContext) -> GraphicsSpecification:
        """Generate AI-powered graphics specification."""
        
        # Create AI prompt for graphics specification
        prompt = f"""
        Create detailed graphics specifications for educational financial content:
        
        Content: {opportunity.content}
        Type: {opportunity.graphics_type}
        Duration: {opportunity.duration} seconds
        Keywords: {', '.join(opportunity.keywords)}
        
        Context Information:
        - Content Type: Financial Education
        - Target Audience: Educational
        - Visual Style: Professional, Clear, Engaging
        
        Please provide specifications for:
        1. Chart/Graphic type (line chart, bar chart, pie chart, infographic, etc.)
        2. Data visualization approach
        3. Animation style and timing
        4. Key visual elements to emphasize
        5. Color scheme (professional financial theme)
        6. Text overlays and callouts
        7. Visual hierarchy
        
        Respond with a JSON object containing these specifications.
        """
        
        try:
            response = await self.gemini_client.generate_content_async(prompt)
            
            # Parse AI response to graphics specification
            spec_data = self._parse_ai_response(response.text)
            
            return GraphicsSpecification(
                graphics_type=spec_data.get('graphics_type', opportunity.graphics_type),
                chart_type=spec_data.get('chart_type'),
                animation_style=spec_data.get('animation_style', opportunity.animation_style),
                duration=opportunity.duration,
                color_scheme=spec_data.get('color_scheme', ["#1f77b4", "#ff7f0e", "#2ca02c"]),
                text_overlays=spec_data.get('text_overlays', []),
                visual_hierarchy=spec_data.get('visual_hierarchy', {}),
                key_points=spec_data.get('key_points', opportunity.keywords)
            )
            
        except Exception as e:
            logger.warning(f"AI graphics specification failed: {str(e)}, using template")
            return self._generate_template_graphics_spec(opportunity)
    
    def _generate_template_graphics_spec(self, opportunity: BRollOpportunity) -> GraphicsSpecification:
        """Generate template-based graphics specification."""
        
        # Map opportunity types to template specifications
        template_mapping = {
            'chart_or_graph': 'compound_interest',
            'animated_explanation': 'diversification',
            'step_by_step_visual': 'risk_return',
            'formula_visualization': 'compound_interest',
            'comparison_table': 'risk_return'
        }
        
        template_key = template_mapping.get(opportunity.graphics_type, 'compound_interest')
        template = self.template_specs[template_key]
        
        return GraphicsSpecification(
            graphics_type=opportunity.graphics_type,
            chart_type=template['chart_type'],
            animation_style=template['animation'],
            duration=opportunity.duration,
            key_points=opportunity.keywords
        )
    
    async def _create_graphics_from_spec(self, 
                                       spec: GraphicsSpecification,
                                       opportunity: BRollOpportunity) -> List[str]:
        """Create actual graphics files from specification."""
        graphics_files = []
        
        try:
            if spec.graphics_type == 'chart_or_graph':
                # Create financial chart based on content
                if 'compound' in opportunity.content.lower():
                    file_path = self.graphics_generator.create_compound_interest_animation(
                        output_filename=f"compound_{int(opportunity.timestamp)}.mp4"
                    )
                    graphics_files.append(file_path)
                
                elif 'diversif' in opportunity.content.lower():
                    portfolio_data = {
                        'Stocks': 60.0,
                        'Bonds': 25.0,
                        'Real Estate': 10.0,
                        'Cash': 5.0
                    }
                    file_path = self.graphics_generator.create_diversification_chart(
                        portfolio_data,
                        output_filename=f"diversification_{int(opportunity.timestamp)}.png"
                    )
                    graphics_files.append(file_path)
                
                elif 'risk' in opportunity.content.lower():
                    investments = {
                        'Government Bonds': {'risk': 2, 'return': 3},
                        'Corporate Bonds': {'risk': 4, 'return': 5},
                        'Index Funds': {'risk': 6, 'return': 7},
                        'Individual Stocks': {'risk': 8, 'return': 9},
                        'Cryptocurrency': {'risk': 10, 'return': 12}
                    }
                    file_path = self.graphics_generator.create_risk_return_scatter(
                        investments,
                        output_filename=f"risk_return_{int(opportunity.timestamp)}.png"
                    )
                    graphics_files.append(file_path)
            
            elif spec.graphics_type == 'animated_explanation':
                # Create educational slide
                concept = self._extract_concept_from_content(opportunity.content)
                file_path = self.slide_generator.create_financial_concept_slide(
                    concept=concept,
                    explanation=opportunity.content[:200],
                    output_filename=f"concept_{int(opportunity.timestamp)}.png"
                )
                graphics_files.append(file_path)
            
            elif spec.graphics_type in ['step_by_step_visual', 'formula_visualization']:
                # Create process/formula visualization
                concept = "Financial Process"
                if 'compound' in opportunity.content.lower():
                    concept = "Compound Interest Formula"
                elif 'diversif' in opportunity.content.lower():
                    concept = "Diversification Strategy"
                
                file_path = self.slide_generator.create_financial_concept_slide(
                    concept=concept,
                    explanation=opportunity.content[:200],
                    output_filename=f"process_{int(opportunity.timestamp)}.png"
                )
                graphics_files.append(file_path)
        
        except Exception as e:
            logger.error(f"Error creating graphics: {str(e)}")
            graphics_files = await self._create_fallback_graphics(opportunity)
        
        return graphics_files
    
    async def _create_fallback_graphics(self, opportunity: BRollOpportunity) -> List[str]:
        """Create simple fallback graphics when other methods fail."""
        try:
            # Create basic concept slide as fallback
            file_path = self.slide_generator.create_financial_concept_slide(
                concept="Financial Concept",
                explanation=opportunity.content[:150],
                output_filename=f"fallback_{int(opportunity.timestamp)}.png"
            )
            return [file_path]
        except Exception as e:
            logger.error(f"Fallback graphics creation failed: {str(e)}")
            return []
    
    def _extract_concept_from_content(self, content: str) -> str:
        """Extract main concept from content text."""
        content_lower = content.lower()
        
        concept_map = {
            'compound interest': 'Compound Interest',
            'diversification': 'Portfolio Diversification',
            'risk': 'Investment Risk',
            'return': 'Investment Returns',
            'asset allocation': 'Asset Allocation',
            'portfolio': 'Portfolio Management',
            'investment': 'Investment Strategy'
        }
        
        for keyword, concept in concept_map.items():
            if keyword in content_lower:
                return concept
        
        return "Financial Concept"
    
    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response to extract graphics specifications."""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                return json.loads(json_text)
            else:
                # Fallback parsing for non-JSON responses
                return self._parse_text_response(response_text)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI response as JSON, using text parsing")
            return self._parse_text_response(response_text)
    
    def _parse_text_response(self, response_text: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails."""
        spec = {}
        
        # Extract key information using keyword matching
        if 'line chart' in response_text.lower():
            spec['chart_type'] = 'line_chart'
        elif 'bar chart' in response_text.lower():
            spec['chart_type'] = 'bar_chart'
        elif 'pie chart' in response_text.lower():
            spec['chart_type'] = 'pie_chart'
        
        if 'fade' in response_text.lower():
            spec['animation_style'] = 'fade_in'
        elif 'slide' in response_text.lower():
            spec['animation_style'] = 'slide_in'
        
        return spec
    
    def create_movis_motion_graphics_plan(self, opportunity: BRollOpportunity) -> Dict[str, Any]:
        """
        Create motion graphics plan compatible with movis composition engine.
        
        Args:
            opportunity: B-roll opportunity to create motion graphics for
            
        Returns:
            Dict with movis-compatible motion graphics specifications
        """
        motion_plan = {
            'timestamp': opportunity.timestamp,
            'duration': opportunity.duration,
            'layers': [],
            'animations': [],
            'effects': []
        }
        
        # Create background layer
        motion_plan['layers'].append({
            'type': 'solid_color',
            'color': (40, 44, 52, 255),  # Professional dark background
            'duration': opportunity.duration
        })
        
        # Create content layers based on opportunity type
        if opportunity.graphics_type == 'chart_or_graph':
            motion_plan['layers'].extend([
                {
                    'type': 'chart_animation',
                    'chart_type': 'line',
                    'data_animation': 'progressive_reveal',
                    'duration': opportunity.duration
                },
                {
                    'type': 'text_overlay',
                    'text': self._extract_concept_from_content(opportunity.content),
                    'position': (960, 100),  # Top center for 1920x1080
                    'font_size': 48,
                    'duration': opportunity.duration
                }
            ])
        
        elif opportunity.graphics_type == 'animated_explanation':
            motion_plan['layers'].extend([
                {
                    'type': 'icon_animation',
                    'icon_set': 'financial_concepts',
                    'animation': 'scale_in',
                    'duration': opportunity.duration * 0.3
                },
                {
                    'type': 'text_reveal',
                    'text': opportunity.content[:100],
                    'animation': 'typewriter',
                    'duration': opportunity.duration * 0.7
                }
            ])
        
        # Add transition animations
        motion_plan['animations'] = [
            {
                'type': 'fade_in',
                'start_time': 0.0,
                'duration': 0.5
            },
            {
                'type': 'fade_out',
                'start_time': opportunity.duration - 0.5,
                'duration': 0.5
            }
        ]
        
        logger.info(f"Created movis motion graphics plan for {opportunity.opportunity_type}")
        return motion_plan
    
    def create_blender_animation_script(self, opportunity: BRollOpportunity) -> str:
        """
        Create Blender animation script for educational content.
        
        Args:
            opportunity: B-roll opportunity to create animation for
            
        Returns:
            Python script for Blender automation
        """
        script_template = f"""
import bpy
import bmesh
from mathutils import Vector

# Clear existing mesh objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Scene setup
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = {int(opportunity.duration * 24)}  # 24 fps

# Create educational visualization for: {opportunity.opportunity_type}
# Content: {opportunity.content[:100]}...

# Add camera
bpy.ops.object.camera_add(location=(7, -7, 5))
camera = bpy.context.object
camera.rotation_euler = (1.1, 0, 0.785)

# Add lighting
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
light = bpy.context.object
light.data.energy = 3

# Create main visualization object
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
main_object = bpy.context.object

# Animation keyframes
main_object.location = (0, 0, 0)
main_object.keyframe_insert(data_path="location", frame=1)

main_object.location = (0, 0, 2)
main_object.keyframe_insert(data_path="location", frame={int(opportunity.duration * 12)})

main_object.location = (0, 0, 0)
main_object.keyframe_insert(data_path="location", frame={int(opportunity.duration * 24)})

# Set render settings
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.fps = 24

# Render animation
output_path = "/tmp/broll_animation_{int(opportunity.timestamp)}"
scene.render.filepath = output_path
bpy.ops.render.render(animation=True)

print(f"Animation rendered to {{output_path}}")
"""
        
        script_path = self.output_dir / f"blender_script_{int(opportunity.timestamp)}.py"
        with open(script_path, 'w') as f:
            f.write(script_template)
        
        logger.info(f"Blender animation script created: {script_path}")
        return str(script_path)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about graphics generation process."""
        return {
            'generation_time': self.generation_time,
            'graphics_created': self.graphics_created,
            'template_specs_available': len(self.template_specs),
            'ai_enabled': self.gemini_client is not None
        }
