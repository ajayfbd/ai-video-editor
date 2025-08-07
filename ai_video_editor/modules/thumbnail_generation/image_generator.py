"""
Thumbnail Image Generator

This module generates thumbnail images using AI (Imagen API) or procedural methods
with PIL/Pillow as fallback. Supports multiple generation strategies and quality levels.
"""

import logging
import asyncio
import time
import os
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available, thumbnail generation will be limited")

from .thumbnail_models import ThumbnailConcept, ThumbnailGenerationStats
from ...core.content_context import ContentContext
from ...core.exceptions import ContentContextError, handle_errors
from ...core.cache_manager import CacheManager


logger = logging.getLogger(__name__)


class ImageGenerationError(ContentContextError):
    """Raised when thumbnail image generation fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="IMAGE_GENERATION_ERROR", **kwargs)


class ThumbnailImageGenerator:
    """
    Generates thumbnail images using AI or procedural methods.
    
    This class handles the actual image generation for thumbnail concepts,
    supporting both AI-powered generation (Imagen API) and procedural
    generation using PIL/Pillow as fallback.
    """
    
    def __init__(self, cache_manager: CacheManager, output_dir: str = "output/thumbnails"):
        """
        Initialize ThumbnailImageGenerator.
        
        Args:
            cache_manager: Cache manager for storing generated images
            output_dir: Directory for saving generated thumbnails
        """
        self.cache_manager = cache_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock Imagen API client (would be real in production)
        self.imagen_client = None  # Would be initialized with actual API client
        
        self.stats = ThumbnailGenerationStats()
        
        # Standard thumbnail dimensions
        self.thumbnail_sizes = {
            "youtube": (1280, 720),
            "youtube_hd": (1920, 1080),
            "square": (1080, 1080),
            "story": (1080, 1920)
        }
        
        # Background styles configuration
        self.background_styles = {
            "dynamic_gradient": {
                "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
                "style": "radial_gradient"
            },
            "question_mark_overlay": {
                "colors": ["#3498DB", "#2C3E50"],
                "style": "question_overlay"
            },
            "clean_professional": {
                "colors": ["#FFFFFF", "#F8F9FA", "#E9ECEF"],
                "style": "minimal"
            },
            "urgent_arrows": {
                "colors": ["#FF4757", "#FF6B6B", "#FFA502"],
                "style": "arrow_overlay"
            },
            "educational_icons": {
                "colors": ["#2E7D32", "#4CAF50", "#81C784"],
                "style": "educational_elements"
            }
        }
        
        logger.info(f"ThumbnailImageGenerator initialized with output dir: {self.output_dir}")
    
    @handle_errors(logger)
    async def generate_thumbnail_image(
        self, 
        concept: ThumbnailConcept, 
        context: ContentContext,
        size: str = "youtube_hd"
    ) -> str:
        """
        Generate thumbnail image using AI or procedural methods.
        
        Args:
            concept: ThumbnailConcept to generate image for
            context: ContentContext for additional insights
            size: Thumbnail size preset ("youtube", "youtube_hd", etc.)
            
        Returns:
            Path to generated thumbnail image
            
        Raises:
            ImageGenerationError: If image generation fails
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(concept, size)
            cached_path = await self._check_cache(cache_key)
            
            if cached_path and os.path.exists(cached_path):
                logger.debug(f"Using cached thumbnail: {cached_path}")
                return cached_path
            
            # Try AI generation first
            image_path = await self._try_ai_generation(concept, context, size)
            
            if not image_path:
                # Fallback to procedural generation
                logger.info("AI generation failed, using procedural generation")
                image_path = await self._generate_procedural_thumbnail(concept, context, size)
                self.stats.add_fallback("ai_to_procedural")
            
            # Cache the result
            if image_path:
                await self._cache_result(cache_key, image_path)
            
            # Update statistics
            processing_time = time.time() - start_time
            method = "ai_generated" if "ai_" in image_path else "procedural"
            cost = 0.05 if method == "ai_generated" else 0.0
            
            self.stats.add_generation(method, processing_time, cost, concept.thumbnail_potential)
            
            logger.info(f"Generated thumbnail in {processing_time:.2f}s using {method}")
            
            return image_path
            
        except Exception as e:
            logger.error(f"Thumbnail image generation failed: {str(e)}")
            raise ImageGenerationError(f"Failed to generate thumbnail image: {str(e)}", context)
    
    @handle_errors(logger)
    async def generate_ai_background(self, concept: ThumbnailConcept) -> Optional[str]:
        """
        Generate background using Imagen API (mocked for now).
        
        Args:
            concept: ThumbnailConcept for background generation
            
        Returns:
            Path to generated background image or None if failed
        """
        try:
            # Mock AI generation (would use real Imagen API in production)
            await asyncio.sleep(0.1)  # Simulate API call
            
            # Create mock AI-generated background
            if PIL_AVAILABLE:
                dimensions = self.thumbnail_sizes["youtube_hd"]
                background_path = self.output_dir / f"ai_bg_{concept.concept_id}.jpg"
                
                # Create gradient background as AI simulation
                image = Image.new("RGB", dimensions, color="#4ECDC4")
                draw = ImageDraw.Draw(image)
                
                # Add gradient effect
                for i in range(dimensions[1]):
                    alpha = i / dimensions[1]
                    color = self._interpolate_color("#4ECDC4", "#45B7D1", alpha)
                    draw.line([(0, i), (dimensions[0], i)], fill=color)
                
                # Add some visual elements based on concept
                self._add_ai_visual_elements(draw, dimensions, concept)
                
                image.save(background_path, "JPEG", quality=95)
                logger.debug(f"Generated AI background: {background_path}")
                
                return str(background_path)
            
            return None
            
        except Exception as e:
            logger.warning(f"AI background generation failed: {str(e)}")
            return None
    
    @handle_errors(logger)
    async def generate_procedural_thumbnail(
        self, 
        concept: ThumbnailConcept, 
        context: ContentContext,
        size: str = "youtube_hd"
    ) -> str:
        """
        Generate thumbnail using procedural methods (PIL/Pillow).
        
        Args:
            concept: ThumbnailConcept to generate
            context: ContentContext for additional insights
            size: Thumbnail size preset
            
        Returns:
            Path to generated thumbnail
            
        Raises:
            ImageGenerationError: If procedural generation fails
        """
        if not PIL_AVAILABLE:
            raise ImageGenerationError("PIL/Pillow not available for procedural generation")
        
        try:
            dimensions = self.thumbnail_sizes.get(size, self.thumbnail_sizes["youtube_hd"])
            
            # Create base image
            image = Image.new("RGB", dimensions, color="#FFFFFF")
            draw = ImageDraw.Draw(image)
            
            # Generate background based on style
            self._generate_background(image, draw, concept.background_style, dimensions)
            
            # Add visual elements
            self._add_visual_elements(image, draw, concept, dimensions)
            
            # Add hook text
            self._add_hook_text(image, draw, concept, dimensions)
            
            # Apply final enhancements
            image = self._apply_enhancements(image, concept)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"thumb_{concept.strategy}_{timestamp}_{concept.concept_id[:8]}.jpg"
            image_path = self.output_dir / filename
            
            image.save(image_path, "JPEG", quality=95)
            
            logger.debug(f"Generated procedural thumbnail: {image_path}")
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Procedural thumbnail generation failed: {str(e)}")
            raise ImageGenerationError(f"Procedural generation failed: {str(e)}")
    
    async def _try_ai_generation(
        self, 
        concept: ThumbnailConcept, 
        context: ContentContext, 
        size: str
    ) -> Optional[str]:
        """Try AI generation with fallback handling."""
        try:
            # Mock AI generation success/failure
            import random
            if random.random() < 0.7:  # 70% success rate simulation
                return await self._mock_ai_generation(concept, size)
            else:
                logger.warning("Mock AI generation failed")
                return None
                
        except Exception as e:
            logger.warning(f"AI generation attempt failed: {str(e)}")
            return None
    
    async def _mock_ai_generation(self, concept: ThumbnailConcept, size: str) -> str:
        """Mock AI generation for testing purposes."""
        if not PIL_AVAILABLE:
            return None
        
        dimensions = self.thumbnail_sizes.get(size, self.thumbnail_sizes["youtube_hd"])
        
        # Create high-quality mock AI image
        image = Image.new("RGB", dimensions, color="#2C3E50")
        draw = ImageDraw.Draw(image)
        
        # Add sophisticated gradient
        for y in range(dimensions[1]):
            for x in range(dimensions[0]):
                # Create complex gradient pattern
                r = int(44 + (x / dimensions[0]) * 50)
                g = int(62 + (y / dimensions[1]) * 80)
                b = int(80 + ((x + y) / (dimensions[0] + dimensions[1])) * 100)
                
                color = (min(255, r), min(255, g), min(255, b))
                draw.point((x, y), fill=color)
        
        # Add AI-style visual elements
        self._add_ai_visual_elements(draw, dimensions, concept)
        
        # Add hook text with AI styling
        self._add_ai_hook_text(image, draw, concept, dimensions)
        
        # Save with AI prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_thumb_{concept.strategy}_{timestamp}_{concept.concept_id[:8]}.jpg"
        image_path = self.output_dir / filename
        
        image.save(image_path, "JPEG", quality=98)
        
        return str(image_path)
    
    def _generate_background(self, image: Image.Image, draw: ImageDraw.Draw, style: str, dimensions: Tuple[int, int]):
        """Generate background based on style."""
        style_config = self.background_styles.get(style, self.background_styles["clean_professional"])
        colors = style_config["colors"]
        style_type = style_config["style"]
        
        if style_type == "radial_gradient":
            self._create_radial_gradient(draw, dimensions, colors)
        elif style_type == "question_overlay":
            self._create_question_overlay(draw, dimensions, colors)
        elif style_type == "minimal":
            self._create_minimal_background(draw, dimensions, colors)
        elif style_type == "arrow_overlay":
            self._create_arrow_overlay(draw, dimensions, colors)
        elif style_type == "educational_elements":
            self._create_educational_background(draw, dimensions, colors)
    
    def _create_radial_gradient(self, draw: ImageDraw.Draw, dimensions: Tuple[int, int], colors: list):
        """Create radial gradient background."""
        center_x, center_y = dimensions[0] // 2, dimensions[1] // 2
        max_radius = min(dimensions) // 2
        
        for radius in range(max_radius, 0, -5):
            alpha = radius / max_radius
            color = self._interpolate_color(colors[0], colors[1], alpha)
            
            bbox = [
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ]
            draw.ellipse(bbox, fill=color)
    
    def _create_question_overlay(self, draw: ImageDraw.Draw, dimensions: Tuple[int, int], colors: list):
        """Create background with question mark overlay."""
        # Fill with base color
        draw.rectangle([0, 0, dimensions[0], dimensions[1]], fill=colors[0])
        
        # Add large question mark
        font_size = min(dimensions) // 4
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        text = "?"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (dimensions[0] - text_width) // 2
        y = (dimensions[1] - text_height) // 2
        
        # Add shadow
        draw.text((x + 3, y + 3), text, fill="#00000040", font=font)
        # Add main text
        draw.text((x, y), text, fill=colors[1], font=font)
    
    def _create_minimal_background(self, draw: ImageDraw.Draw, dimensions: Tuple[int, int], colors: list):
        """Create clean minimal background."""
        draw.rectangle([0, 0, dimensions[0], dimensions[1]], fill=colors[0])
        
        # Add subtle geometric elements
        for i in range(3):
            x = (dimensions[0] // 4) * (i + 1)
            draw.line([(x, 0), (x, dimensions[1])], fill=colors[1], width=2)
    
    def _create_arrow_overlay(self, draw: ImageDraw.Draw, dimensions: Tuple[int, int], colors: list):
        """Create background with arrow overlays."""
        draw.rectangle([0, 0, dimensions[0], dimensions[1]], fill=colors[0])
        
        # Add arrow pointing up
        arrow_points = [
            (dimensions[0] // 2, dimensions[1] // 4),
            (dimensions[0] // 2 - 50, dimensions[1] // 2),
            (dimensions[0] // 2 + 50, dimensions[1] // 2)
        ]
        draw.polygon(arrow_points, fill=colors[1])
    
    def _create_educational_background(self, draw: ImageDraw.Draw, dimensions: Tuple[int, int], colors: list):
        """Create educational background with icons."""
        draw.rectangle([0, 0, dimensions[0], dimensions[1]], fill=colors[0])
        
        # Add educational elements (simplified)
        # Book icon
        book_x, book_y = dimensions[0] // 4, dimensions[1] // 4
        draw.rectangle([book_x, book_y, book_x + 60, book_y + 40], fill=colors[1])
        
        # Chart icon
        chart_x, chart_y = 3 * dimensions[0] // 4 - 30, dimensions[1] // 4
        for i in range(3):
            height = (i + 1) * 15
            draw.rectangle([
                chart_x + i * 20, chart_y + 40 - height,
                chart_x + i * 20 + 15, chart_y + 40
            ], fill=colors[2])
    
    def _add_visual_elements(self, image: Image.Image, draw: ImageDraw.Draw, concept: ThumbnailConcept, dimensions: Tuple[int, int]):
        """Add visual elements based on concept."""
        elements = concept.visual_elements
        
        if "charts" in elements:
            self._add_chart_element(draw, dimensions)
        if "professional_setting" in elements:
            self._add_professional_elements(draw, dimensions)
        if "text_overlay" in elements:
            self._add_text_overlay_elements(draw, dimensions)
    
    def _add_chart_element(self, draw: ImageDraw.Draw, dimensions: Tuple[int, int]):
        """Add chart visual element."""
        chart_x = dimensions[0] - 200
        chart_y = dimensions[1] - 150
        
        # Simple bar chart
        for i in range(4):
            height = (i + 1) * 20
            draw.rectangle([
                chart_x + i * 30, chart_y + 100 - height,
                chart_x + i * 30 + 25, chart_y + 100
            ], fill="#4CAF50")
    
    def _add_professional_elements(self, draw: ImageDraw.Draw, dimensions: Tuple[int, int]):
        """Add professional visual elements."""
        # Add subtle grid lines
        for i in range(0, dimensions[0], 100):
            draw.line([(i, 0), (i, dimensions[1])], fill="#E0E0E0", width=1)
        for i in range(0, dimensions[1], 100):
            draw.line([(0, i), (dimensions[0], i)], fill="#E0E0E0", width=1)
    
    def _add_text_overlay_elements(self, draw: ImageDraw.Draw, dimensions: Tuple[int, int]):
        """Add text overlay visual elements."""
        # Add text box background
        box_x = dimensions[0] // 4
        box_y = 3 * dimensions[1] // 4
        draw.rectangle([box_x, box_y, box_x + 200, box_y + 50], fill="#FFFFFF80")
    
    def _add_hook_text(self, image: Image.Image, draw: ImageDraw.Draw, concept: ThumbnailConcept, dimensions: Tuple[int, int]):
        """Add hook text to thumbnail."""
        text = concept.hook_text
        text_style = concept.text_style
        
        # Determine font size
        base_size = 72 if text_style.get("size") == "large" else 48
        font_size = min(base_size, dimensions[0] // len(text) * 2)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position text
        x = (dimensions[0] - text_width) // 2
        y = dimensions[1] - text_height - 50  # Bottom area
        
        # Add background box for readability
        padding = 20
        box_coords = [
            x - padding, y - padding,
            x + text_width + padding, y + text_height + padding
        ]
        draw.rectangle(box_coords, fill="#00000080")
        
        # Add text with shadow
        shadow_offset = 3
        draw.text((x + shadow_offset, y + shadow_offset), text, fill="#000000", font=font)
        
        # Main text
        color = text_style.get("color", "#FFFFFF")
        draw.text((x, y), text, fill=color, font=font)
    
    def _add_ai_visual_elements(self, draw: ImageDraw.Draw, dimensions: Tuple[int, int], concept: ThumbnailConcept):
        """Add AI-style visual elements."""
        # Add sophisticated geometric patterns
        center_x, center_y = dimensions[0] // 2, dimensions[1] // 2
        
        # Add circular patterns
        for radius in range(50, 200, 30):
            draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], outline="#FFFFFF40", width=2)
        
        # Add connecting lines
        for angle in range(0, 360, 45):
            import math
            end_x = center_x + int(150 * math.cos(math.radians(angle)))
            end_y = center_y + int(150 * math.sin(math.radians(angle)))
            draw.line([(center_x, center_y), (end_x, end_y)], fill="#FFFFFF20", width=1)
    
    def _add_ai_hook_text(self, image: Image.Image, draw: ImageDraw.Draw, concept: ThumbnailConcept, dimensions: Tuple[int, int]):
        """Add hook text with AI styling."""
        text = concept.hook_text
        
        # Use larger, more dramatic font
        font_size = min(96, dimensions[0] // len(text) * 3)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (dimensions[0] - text_width) // 2
        y = (dimensions[1] - text_height) // 2
        
        # Add glowing effect
        for offset in range(5, 0, -1):
            alpha = 255 // (offset + 1)
            glow_color = f"#{255:02x}{255:02x}{255:02x}{alpha:02x}"
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    if dx * dx + dy * dy <= offset * offset:
                        draw.text((x + dx, y + dy), text, fill="#FFFFFF40", font=font)
        
        # Main text with gradient effect
        draw.text((x, y), text, fill="#FFFFFF", font=font)
    
    def _apply_enhancements(self, image: Image.Image, concept: ThumbnailConcept) -> Image.Image:
        """Apply final enhancements to image."""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance color saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        # Apply slight sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        return image
    
    def _interpolate_color(self, color1: str, color2: str, alpha: float) -> str:
        """Interpolate between two hex colors."""
        # Convert hex to RGB
        c1 = tuple(int(color1[i:i+2], 16) for i in (1, 3, 5))
        c2 = tuple(int(color2[i:i+2], 16) for i in (1, 3, 5))
        
        # Interpolate
        r = int(c1[0] + (c2[0] - c1[0]) * alpha)
        g = int(c1[1] + (c2[1] - c1[1]) * alpha)
        b = int(c1[2] + (c2[2] - c1[2]) * alpha)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _generate_cache_key(self, concept: ThumbnailConcept, size: str) -> str:
        """Generate cache key for thumbnail concept."""
        return f"thumbnail_{concept.concept_id}_{size}_{concept.strategy}"
    
    async def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if thumbnail is cached."""
        try:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data and isinstance(cached_data, dict):
                return cached_data.get("image_path")
        except Exception as e:
            logger.debug(f"Cache check failed: {str(e)}")
        
        return None
    
    async def _cache_result(self, cache_key: str, image_path: str):
        """Cache thumbnail generation result."""
        try:
            cache_data = {
                "image_path": image_path,
                "generated_at": datetime.now().isoformat(),
                "file_size": os.path.getsize(image_path) if os.path.exists(image_path) else 0
            }
            
            # Cache for 24 hours
            await self.cache_manager.set(cache_key, cache_data, ttl=86400)
            
        except Exception as e:
            logger.warning(f"Failed to cache thumbnail result: {str(e)}")
    
    async def _generate_procedural_thumbnail(
        self, 
        concept: ThumbnailConcept, 
        context: ContentContext, 
        size: str
    ) -> str:
        """Generate procedural thumbnail (public wrapper)."""
        return await self.generate_procedural_thumbnail(concept, context, size)