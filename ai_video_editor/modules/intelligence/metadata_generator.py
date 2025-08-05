"""
MetadataGenerator - SEO-optimized metadata generation for YouTube content.

This module provides comprehensive metadata generation including titles, descriptions,
and tags optimized for YouTube SEO and discoverability.
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from ai_video_editor.core.content_context import ContentContext, TrendingKeywords
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.exceptions import (
    ContentContextError,
    APIIntegrationError,
    handle_errors
)


logger = logging.getLogger(__name__)


@dataclass
class MetadataVariation:
    """Represents a single metadata variation for A/B testing."""
    title: str
    description: str
    tags: List[str]
    variation_id: str
    strategy: str  # "emotional", "seo_focused", "curiosity_driven", etc.
    confidence_score: float = 0.0
    estimated_ctr: float = 0.0  # Click-through rate estimate
    seo_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'description': self.description,
            'tags': self.tags,
            'variation_id': self.variation_id,
            'strategy': self.strategy,
            'confidence_score': self.confidence_score,
            'estimated_ctr': self.estimated_ctr,
            'seo_score': self.seo_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetadataVariation':
        return cls(**data)


@dataclass
class MetadataPackage:
    """Complete metadata package with multiple variations."""
    variations: List[MetadataVariation]
    recommended_variation: str  # variation_id of recommended option
    generation_timestamp: datetime
    content_analysis: Dict[str, Any]
    seo_insights: Dict[str, Any]
    performance_predictions: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'variations': [var.to_dict() for var in self.variations],
            'recommended_variation': self.recommended_variation,
            'generation_timestamp': self.generation_timestamp.isoformat(),
            'content_analysis': self.content_analysis,
            'seo_insights': self.seo_insights,
            'performance_predictions': self.performance_predictions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetadataPackage':
        variations = [MetadataVariation.from_dict(var) for var in data['variations']]
        return cls(
            variations=variations,
            recommended_variation=data['recommended_variation'],
            generation_timestamp=datetime.fromisoformat(data['generation_timestamp']),
            content_analysis=data['content_analysis'],
            seo_insights=data['seo_insights'],
            performance_predictions=data['performance_predictions']
        )


class MetadataGenerator:
    """
    SEO-optimized metadata generation for YouTube content.
    
    Generates highly optimized titles, descriptions, and tags based on content
    analysis, trending keywords, and SEO best practices.
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize MetadataGenerator with required dependencies.
        
        Args:
            cache_manager: CacheManager instance for result caching
        """
        self.cache_manager = cache_manager
        
        # Title generation strategies
        self.title_strategies = {
            'emotional': {
                'patterns': [
                    "{concept} That Will {emotion_verb} You",
                    "The {adjective} Truth About {concept}",
                    "{number} {concept} Secrets {target_audience} Don't Want You to Know",
                    "Why {concept} Is {comparative} Than You Think"
                ],
                'emotion_verbs': ['Shock', 'Amaze', 'Surprise', 'Transform', 'Change'],
                'adjectives': ['Shocking', 'Amazing', 'Incredible', 'Unbelievable', 'Hidden'],
                'comparatives': ['Easier', 'Harder', 'More Important', 'More Dangerous', 'More Profitable']
            },
            'seo_focused': {
                'patterns': [
                    "{concept}: Complete {year} Guide",
                    "How to {action} {concept} in {timeframe}",
                    "{concept} Tutorial for {target_audience}",
                    "Best {concept} {category} ({year} Updated)"
                ],
                'actions': ['Master', 'Learn', 'Understand', 'Use', 'Start'],
                'timeframes': ['10 Minutes', '30 Days', '2024', 'One Week'],
                'categories': ['Tips', 'Strategies', 'Methods', 'Techniques', 'Tools']
            },
            'curiosity_driven': {
                'patterns': [
                    "What {expert_type} Don't Tell You About {concept}",
                    "The {concept} Method That {result}",
                    "I Tried {concept} for {timeframe} - Here's What Happened",
                    "{concept} vs {alternative}: Which Is Better?"
                ],
                'expert_types': ['Experts', 'Professionals', 'Gurus', 'Teachers', 'Advisors'],
                'results': ['Changed Everything', 'Actually Works', 'Surprised Me', 'Made Me Rich']
            },
            'educational': {
                'patterns': [
                    "{concept} Explained Simply",
                    "Understanding {concept}: A Beginner's Guide",
                    "{concept} Fundamentals Every {target_audience} Should Know",
                    "The Science Behind {concept}"
                ]
            },
            'listicle': {
                'patterns': [
                    "{number} {concept} {category} That Actually Work",
                    "Top {number} {concept} Mistakes to Avoid",
                    "{number} Signs You Need to {action} Your {concept}",
                    "{number} {concept} Hacks for {target_audience}"
                ],
                'numbers': ['5', '7', '10', '15', '20'],
                'categories': ['Tips', 'Tricks', 'Strategies', 'Methods', 'Tools']
            }
        }
        
        # Description templates
        self.description_templates = {
            'educational': """🎯 In this video, you'll learn everything about {main_concept}.

📚 What you'll discover:
{key_points}

⏰ Timestamps:
{timestamps}

🔗 Helpful Resources:
{resources}

💡 Key Takeaways:
{takeaways}

📈 Ready to master {main_concept}? Watch now and transform your understanding!

{tags_section}

#shorts #{main_hashtag} #education #tutorial""",
            
            'financial': """💰 Master {main_concept} with this comprehensive guide!

📊 What's covered:
{key_points}

⏰ Video Timeline:
{timestamps}

💡 Pro Tips Inside:
{pro_tips}

📈 Start your {main_concept} journey today!

{engagement_hooks}

{tags_section}

#finance #{main_hashtag} #investing #money #wealth""",
            
            'general': """🚀 Everything you need to know about {main_concept}!

✅ In this video:
{key_points}

⏰ Chapters:
{timestamps}

🎯 Perfect for:
{target_audience}

💬 What's your experience with {main_concept}? Let me know in the comments!

{tags_section}

#{main_hashtag} #tutorial #howto #tips"""
        }
        
        # Tag categories for comprehensive coverage
        self.tag_categories = {
            'broad': ['tutorial', 'guide', 'tips', 'howto', 'education', 'learning'],
            'specific': [],  # Will be populated from content analysis
            'trending': [],  # Will be populated from trending keywords
            'audience': ['beginner', 'advanced', 'professional', 'student'],
            'format': ['explained', 'simplified', 'complete', 'ultimate'],
            'emotional': ['amazing', 'incredible', 'shocking', 'surprising']
        }
        
        # SEO optimization patterns
        self.seo_patterns = {
            'title_length': (50, 60),  # Optimal character range
            'description_length': (125, 200),  # First fold length
            'tag_count': (10, 15),  # Optimal tag count
            'keyword_density': (0.02, 0.05),  # 2-5% keyword density
            'emotional_words': ['amazing', 'incredible', 'shocking', 'ultimate', 'secret', 'proven']
        }
        
        logger.info("MetadataGenerator initialized with SEO optimization patterns")
    
    @handle_errors(logger)
    async def generate_metadata_package(self, context: ContentContext) -> ContentContext:
        """
        Generate complete metadata package with multiple variations.
        
        Args:
            context: ContentContext with content analysis and trending keywords
            
        Returns:
            Updated ContentContext with metadata_variations populated
            
        Raises:
            ContentContextError: If generation fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting metadata generation for project {context.project_id}")
            
            # Validate required data
            if not context.trending_keywords:
                raise ContentContextError(
                    "Trending keywords required for metadata generation",
                    context_state=context
                )
            
            # Check cache first
            cache_key = f"metadata_package:{context.project_id}"
            cached_package = self.cache_manager.get(cache_key)
            if cached_package:
                logger.info("Using cached metadata package")
                context.metadata_variations = [cached_package]
                return context
            
            # Extract content insights
            content_analysis = self._analyze_content_for_metadata(context)
            
            # Generate multiple metadata variations
            variations = await self._generate_metadata_variations(context, content_analysis)
            
            # Calculate SEO scores and performance predictions
            for variation in variations:
                variation.seo_score = self._calculate_seo_score(variation, context.trending_keywords)
                variation.estimated_ctr = self._estimate_click_through_rate(variation, content_analysis)
                variation.confidence_score = (variation.seo_score + variation.estimated_ctr) / 2
            
            # Select recommended variation
            recommended_variation = max(variations, key=lambda v: v.confidence_score)
            
            # Create metadata package
            metadata_package = MetadataPackage(
                variations=variations,
                recommended_variation=recommended_variation.variation_id,
                generation_timestamp=datetime.now(),
                content_analysis=content_analysis,
                seo_insights=self._generate_seo_insights(variations, context.trending_keywords),
                performance_predictions=self._generate_performance_predictions(variations)
            )
            
            # Store in context
            context.metadata_variations = [metadata_package.to_dict()]
            
            # Cache the package
            self.cache_manager.put(cache_key, metadata_package.to_dict(), ttl=86400)  # 24 hours
            
            # Track successful patterns in memory
            await self._track_successful_patterns(metadata_package, context)
            
            # Update processing metrics
            processing_time = time.time() - start_time
            context.processing_metrics.add_module_metrics(
                "metadata_generator", processing_time, 0
            )
            
            logger.info(f"Metadata generation completed in {processing_time:.2f}s")
            logger.info(f"Generated {len(variations)} variations with recommended: {recommended_variation.strategy}")
            
            return context
            
        except Exception as e:
            logger.error(f"Metadata generation failed: {str(e)}")
            raise ContentContextError(
                f"Metadata generation failed: {str(e)}",
                context_state=context
            )   
 
    async def _generate_metadata_variations(self, context: ContentContext, 
                                           content_analysis: Dict[str, Any]) -> List[MetadataVariation]:
        """
        Generate multiple metadata variations using different strategies.
        
        Args:
            context: ContentContext with analysis results
            content_analysis: Extracted content insights
            
        Returns:
            List of metadata variations
        """
        variations = []
        
        # Generate variations for each strategy
        strategies = ['emotional', 'seo_focused', 'curiosity_driven', 'educational', 'listicle']
        
        for i, strategy in enumerate(strategies):
            try:
                # Generate title
                title = await self._generate_title(strategy, content_analysis, context.trending_keywords)
                
                # Generate description
                description = await self._generate_description(strategy, content_analysis, context)
                
                # Generate tags
                tags = await self._generate_tags(content_analysis, context.trending_keywords)
                
                # Create variation
                variation = MetadataVariation(
                    title=title,
                    description=description,
                    tags=tags,
                    variation_id=f"var_{i+1}_{strategy}",
                    strategy=strategy
                )
                
                variations.append(variation)
                
            except Exception as e:
                logger.warning(f"Failed to generate {strategy} variation: {str(e)}")
                continue
        
        return variations
    
    async def _generate_title(self, strategy: str, content_analysis: Dict[str, Any], 
                            trending_keywords: TrendingKeywords) -> str:
        """
        Generate optimized title using specified strategy.
        
        Args:
            strategy: Title generation strategy
            content_analysis: Content insights
            trending_keywords: Trending keywords data
            
        Returns:
            Generated title
        """
        if strategy not in self.title_strategies:
            strategy = 'seo_focused'  # Fallback
        
        strategy_config = self.title_strategies[strategy]
        patterns = strategy_config['patterns']
        
        # Select primary concept
        main_concept = content_analysis.get('primary_concept', 'Content')
        
        # Get trending keyword for SEO
        primary_keyword = trending_keywords.primary_keywords[0] if trending_keywords.primary_keywords else main_concept
        
        # Select pattern and fill variables
        import random
        pattern = random.choice(patterns)
        
        # Fill pattern variables
        title = pattern.format(
            concept=primary_keyword,
            main_concept=main_concept,
            year=datetime.now().year,
            number=random.choice(['5', '7', '10', '15']),
            timeframe=random.choice(['10 Minutes', '30 Days', 'One Week']),
            target_audience=content_analysis.get('target_audience', 'Everyone'),
            action=random.choice(strategy_config.get('actions', ['Learn', 'Master'])),
            emotion_verb=random.choice(strategy_config.get('emotion_verbs', ['Amaze'])),
            adjective=random.choice(strategy_config.get('adjectives', ['Amazing'])),
            comparative=random.choice(strategy_config.get('comparatives', ['Better'])),
            expert_type=random.choice(strategy_config.get('expert_types', ['Experts'])),
            result=random.choice(strategy_config.get('results', ['Works'])),
            alternative=content_analysis.get('alternative_concept', 'Traditional Methods'),
            category=random.choice(strategy_config.get('categories', ['Tips']))
        )
        
        # Optimize title length
        title = self._optimize_title_length(title)
        
        return title
    
    async def _generate_description(self, strategy: str, content_analysis: Dict[str, Any], 
                                  context: ContentContext) -> str:
        """
        Generate comprehensive description with timestamps and keywords.
        
        Args:
            strategy: Generation strategy
            content_analysis: Content insights
            context: ContentContext with analysis data
            
        Returns:
            Generated description
        """
        # Determine template based on content type
        content_type = context.content_type.value
        template_key = content_type if content_type in self.description_templates else 'general'
        template = self.description_templates[template_key]
        
        # Extract main concept
        main_concept = content_analysis.get('primary_concept', 'this topic')
        main_hashtag = main_concept.lower().replace(' ', '').replace('-', '')
        
        # Generate key points
        key_points = self._generate_key_points(content_analysis, context)
        
        # Generate timestamps
        timestamps = self._generate_timestamps(context)
        
        # Generate additional content based on template
        additional_content = self._generate_template_content(template_key, content_analysis, context)
        
        # Generate tags section
        tags_section = self._generate_tags_section(context.trending_keywords)
        
        # Fill template
        description = template.format(
            main_concept=main_concept,
            main_hashtag=main_hashtag,
            key_points=key_points,
            timestamps=timestamps,
            tags_section=tags_section,
            **additional_content
        )
        
        # Optimize description length and SEO
        description = self._optimize_description_seo(description, context.trending_keywords)
        
        return description
    
    async def _generate_tags(self, content_analysis: Dict[str, Any], 
                           trending_keywords: TrendingKeywords) -> List[str]:
        """
        Generate 10-15 optimized tags combining broad and specific terms.
        
        Args:
            content_analysis: Content insights
            trending_keywords: Trending keywords data
            
        Returns:
            List of optimized tags
        """
        tags = set()
        
        # Add primary keywords (specific terms) - convert to hashtag format
        for keyword in trending_keywords.primary_keywords[:5]:
            # Convert multi-word keywords to hashtag format
            tag = keyword.lower().replace(' ', '').replace('-', '')
            if len(tag) <= 20:  # YouTube tag length limit
                tags.add(tag)
        
        # Add long-tail keywords
        for keyword in trending_keywords.long_tail_keywords[:3]:
            # Convert to hashtag format
            tag = keyword.lower().replace(' ', '').replace('-', '')
            if len(tag) <= 20:  # YouTube tag length limit
                tags.add(tag)
        
        # Add broad category tags
        broad_tags = self.tag_categories['broad']
        content_type = content_analysis.get('content_type', 'general')
        
        if content_type == 'educational':
            broad_tags.extend(['education', 'learning', 'tutorial', 'explained'])
        elif content_type == 'financial':
            broad_tags.extend(['finance', 'money', 'investing', 'wealth'])
        
        # Add 3-4 broad tags
        import random
        for tag in random.sample(broad_tags, min(4, len(broad_tags))):
            tags.add(tag)
        
        # Add audience-specific tags
        target_audience = content_analysis.get('target_audience', '').lower()
        if 'beginner' in target_audience:
            tags.add('beginner')
        elif 'advanced' in target_audience:
            tags.add('advanced')
        
        # Add trending hashtags (without #)
        for hashtag in trending_keywords.trending_hashtags[:2]:
            clean_tag = hashtag.replace('#', '').lower()
            if len(clean_tag) > 2:
                tags.add(clean_tag)
        
        # Add emotional/engagement tags
        emotional_tags = ['tips', 'guide', 'howto', 'explained', 'simple']
        tags.update(random.sample(emotional_tags, min(2, len(emotional_tags))))
        
        # Convert to list and limit to 15 tags
        final_tags = list(tags)[:15]
        
        # Ensure we have at least 10 tags
        while len(final_tags) < 10:
            fallback_tags = ['content', 'video', 'information', 'helpful', 'useful']
            for tag in fallback_tags:
                if tag not in final_tags:
                    final_tags.append(tag)
                    if len(final_tags) >= 10:
                        break
        
        return final_tags
    
    def _analyze_content_for_metadata(self, context: ContentContext) -> Dict[str, Any]:
        """
        Analyze content to extract insights for metadata generation.
        
        Args:
            context: ContentContext with analysis results
            
        Returns:
            Dictionary with content insights
        """
        analysis = {
            'primary_concept': 'Content',
            'target_audience': 'General Audience',
            'content_type': context.content_type.value,
            'key_themes': context.content_themes[:5],
            'emotional_peaks': len(context.emotional_markers),
            'visual_highlights': len(context.visual_highlights),
            'duration_category': 'medium'
        }
        
        # Extract primary concept from key concepts
        if context.key_concepts:
            analysis['primary_concept'] = context.key_concepts[0]
        
        # Determine target audience from content analysis
        if context.audio_analysis:
            complexity = context.audio_analysis.complexity_level
            if complexity == 'beginner':
                analysis['target_audience'] = 'Beginners'
            elif complexity == 'advanced':
                analysis['target_audience'] = 'Advanced Users'
            else:
                analysis['target_audience'] = 'Everyone'
        
        # Analyze emotional content
        if context.emotional_markers:
            top_emotions = {}
            for marker in context.emotional_markers:
                emotion = marker.emotion
                top_emotions[emotion] = top_emotions.get(emotion, 0) + marker.intensity
            
            # Get dominant emotion
            if top_emotions:
                dominant_emotion = max(top_emotions.items(), key=lambda x: x[1])[0]
                analysis['dominant_emotion'] = dominant_emotion
        
        # Extract financial concepts if available
        if context.audio_analysis and context.audio_analysis.financial_concepts:
            analysis['financial_concepts'] = context.audio_analysis.financial_concepts[:3]
            analysis['alternative_concept'] = 'Traditional Finance'
        
        return analysis
    
    def _generate_key_points(self, content_analysis: Dict[str, Any], 
                           context: ContentContext) -> str:
        """Generate key points list for description."""
        points = []
        
        # Add key themes
        for theme in content_analysis.get('key_themes', [])[:4]:
            points.append(f"• {theme.title()}")
        
        # Add financial concepts if available
        if 'financial_concepts' in content_analysis:
            for concept in content_analysis['financial_concepts'][:2]:
                points.append(f"• {concept.title()}")
        
        # Add emotional highlights
        if context.emotional_markers:
            top_emotion = content_analysis.get('dominant_emotion', 'excitement')
            points.append(f"• Key insights that will {top_emotion.lower()} you")
        
        # Ensure we have at least 3 points
        while len(points) < 3:
            fallback_points = [
                "• Essential concepts explained clearly",
                "• Practical tips you can use immediately",
                "• Expert insights and best practices"
            ]
            for point in fallback_points:
                if point not in points:
                    points.append(point)
                    if len(points) >= 3:
                        break
        
        return '\n'.join(points[:5])
    
    def _generate_timestamps(self, context: ContentContext) -> str:
        """Generate timestamps section for description."""
        timestamps = []
        
        # Use emotional markers for timestamps
        if context.emotional_markers:
            for i, marker in enumerate(context.emotional_markers[:5]):
                minutes = int(marker.timestamp // 60)
                seconds = int(marker.timestamp % 60)
                timestamp_str = f"{minutes:02d}:{seconds:02d}"
                
                # Generate descriptive text
                description = f"{marker.context}" if marker.context else f"{marker.emotion.title()} moment"
                timestamps.append(f"{timestamp_str} - {description}")
        
        # Use visual highlights if no emotional markers
        elif context.visual_highlights:
            for i, highlight in enumerate(context.visual_highlights[:5]):
                minutes = int(highlight.timestamp // 60)
                seconds = int(highlight.timestamp % 60)
                timestamp_str = f"{minutes:02d}:{seconds:02d}"
                timestamps.append(f"{timestamp_str} - {highlight.description}")
        
        # Fallback timestamps
        if not timestamps:
            timestamps = [
                "00:00 - Introduction",
                "02:30 - Main Content",
                "08:45 - Key Insights",
                "12:00 - Conclusion"
            ]
        
        return '\n'.join(timestamps[:6])
    
    def _generate_template_content(self, template_key: str, content_analysis: Dict[str, Any], 
                                 context: ContentContext) -> Dict[str, str]:
        """Generate additional content based on template type."""
        content = {}
        
        if template_key == 'financial':
            content['pro_tips'] = self._generate_pro_tips(content_analysis)
            content['engagement_hooks'] = self._generate_engagement_hooks(content_analysis)
        elif template_key == 'educational':
            content['resources'] = self._generate_resources(content_analysis)
            content['takeaways'] = self._generate_takeaways(content_analysis)
        else:
            content['target_audience'] = content_analysis.get('target_audience', 'Everyone')
        
        return content
    
    def _generate_pro_tips(self, content_analysis: Dict[str, Any]) -> str:
        """Generate pro tips section."""
        tips = [
            "• Start with small amounts to minimize risk",
            "• Always do your own research before investing",
            "• Diversification is key to long-term success"
        ]
        
        if 'financial_concepts' in content_analysis:
            for concept in content_analysis['financial_concepts'][:2]:
                tips.append(f"• Master {concept.lower()} for better results")
        
        return '\n'.join(tips[:4])
    
    def _generate_engagement_hooks(self, content_analysis: Dict[str, Any]) -> str:
        """Generate engagement hooks."""
        hooks = [
            "💬 What's your biggest financial goal? Share below!",
            "🔔 Subscribe for more financial education content!",
            "👍 Like if this helped you understand the concept better!"
        ]
        
        return '\n'.join(hooks)
    
    def _generate_resources(self, content_analysis: Dict[str, Any]) -> str:
        """Generate resources section."""
        resources = [
            "• Free calculator: [Link in description]",
            "• Recommended reading: [Book suggestions]",
            "• Related videos: [Playlist link]"
        ]
        
        return '\n'.join(resources)
    
    def _generate_takeaways(self, content_analysis: Dict[str, Any]) -> str:
        """Generate key takeaways."""
        takeaways = [
            "• Understanding the fundamentals is crucial",
            "• Practice makes perfect - start today",
            "• Knowledge is your best investment"
        ]
        
        return '\n'.join(takeaways)
    
    def _generate_tags_section(self, trending_keywords: TrendingKeywords) -> str:
        """Generate tags section for description."""
        tags = []
        
        # Add primary keywords as hashtags
        for keyword in trending_keywords.primary_keywords[:3]:
            hashtag = keyword.replace(' ', '').replace('-', '').lower()
            tags.append(f"#{hashtag}")
        
        # Add trending hashtags
        for hashtag in trending_keywords.trending_hashtags[:2]:
            if not hashtag.startswith('#'):
                hashtag = f"#{hashtag}"
            tags.append(hashtag.lower())
        
        return ' '.join(tags)
    
    def _optimize_title_length(self, title: str) -> str:
        """Optimize title length for YouTube SEO."""
        min_length, max_length = self.seo_patterns['title_length']
        
        if len(title) <= max_length:
            return title
        
        # Truncate while preserving meaning
        words = title.split()
        optimized_title = ""
        
        for word in words:
            test_title = f"{optimized_title} {word}".strip()
            if len(test_title) <= max_length:
                optimized_title = test_title
            else:
                break
        
        # Ensure minimum length
        if len(optimized_title) < min_length and len(title) >= min_length:
            return title[:max_length-3] + "..."
        
        return optimized_title
    
    def _optimize_description_seo(self, description: str, trending_keywords: TrendingKeywords) -> str:
        """Optimize description for SEO."""
        # Ensure primary keyword appears in first 125 characters
        if trending_keywords.primary_keywords:
            primary_keyword = trending_keywords.primary_keywords[0]
            
            # Check if keyword is in first 125 characters
            first_fold = description[:125]
            if primary_keyword.lower() not in first_fold.lower():
                # Prepend keyword mention
                keyword_intro = f"Learn about {primary_keyword} in this comprehensive guide. "
                description = keyword_intro + description
        
        return description
    
    def _calculate_seo_score(self, variation: MetadataVariation, 
                           trending_keywords: TrendingKeywords) -> float:
        """Calculate SEO score for metadata variation."""
        score = 0.0
        
        # Title optimization score
        title_length = len(variation.title)
        min_len, max_len = self.seo_patterns['title_length']
        if min_len <= title_length <= max_len:
            score += 0.2
        
        # Keyword presence in title
        if trending_keywords.primary_keywords:
            primary_keyword = trending_keywords.primary_keywords[0].lower()
            if primary_keyword in variation.title.lower():
                score += 0.3
        
        # Description optimization
        desc_length = len(variation.description)
        if desc_length >= 125:  # Minimum for first fold
            score += 0.2
        
        # Tag optimization
        tag_count = len(variation.tags)
        min_tags, max_tags = self.seo_patterns['tag_count']
        if min_tags <= tag_count <= max_tags:
            score += 0.2
        
        # Trending keyword coverage in tags
        trending_coverage = 0
        for keyword in trending_keywords.primary_keywords[:5]:
            if any(keyword.lower() in tag for tag in variation.tags):
                trending_coverage += 1
        
        if trending_coverage > 0:
            score += 0.1 * (trending_coverage / 5)
        
        return min(1.0, score)
    
    def _estimate_click_through_rate(self, variation: MetadataVariation, 
                                   content_analysis: Dict[str, Any]) -> float:
        """Estimate click-through rate based on title and thumbnail potential."""
        ctr_score = 0.0
        
        # Emotional words boost CTR
        emotional_words = self.seo_patterns['emotional_words']
        for word in emotional_words:
            if word.lower() in variation.title.lower():
                ctr_score += 0.1
        
        # Numbers in title boost CTR
        if re.search(r'\d+', variation.title):
            ctr_score += 0.15
        
        # Question format boost
        if '?' in variation.title:
            ctr_score += 0.1
        
        # Strategy-specific boosts
        strategy_boosts = {
            'emotional': 0.2,
            'curiosity_driven': 0.25,
            'listicle': 0.15,
            'seo_focused': 0.1,
            'educational': 0.05
        }
        
        ctr_score += strategy_boosts.get(variation.strategy, 0.0)
        
        # Content quality indicators
        if content_analysis.get('emotional_peaks', 0) > 3:
            ctr_score += 0.1
        
        if content_analysis.get('visual_highlights', 0) > 5:
            ctr_score += 0.1
        
        return min(1.0, ctr_score)
    
    def _generate_seo_insights(self, variations: List[MetadataVariation], 
                             trending_keywords: TrendingKeywords) -> Dict[str, Any]:
        """Generate SEO insights for the metadata package."""
        insights = {
            'keyword_coverage': {},
            'optimization_suggestions': [],
            'trending_alignment': 0.0,
            'competition_analysis': {}
        }
        
        # Analyze keyword coverage across variations
        all_keywords = set()
        for variation in variations:
            title_words = set(variation.title.lower().split())
            tag_words = set(variation.tags)
            all_keywords.update(title_words)
            all_keywords.update(tag_words)
        
        # Check coverage of trending keywords
        covered_keywords = 0
        for keyword in trending_keywords.primary_keywords[:10]:
            if any(keyword.lower() in word for word in all_keywords):
                covered_keywords += 1
        
        insights['trending_alignment'] = covered_keywords / min(10, len(trending_keywords.primary_keywords))
        
        # Generate optimization suggestions
        if insights['trending_alignment'] < 0.7:
            insights['optimization_suggestions'].append(
                "Consider incorporating more trending keywords in titles and tags"
            )
        
        avg_seo_score = sum(v.seo_score for v in variations) / len(variations)
        if avg_seo_score < 0.6:
            insights['optimization_suggestions'].append(
                "Optimize title lengths and keyword placement for better SEO"
            )
        
        return insights
    
    def _generate_performance_predictions(self, variations: List[MetadataVariation]) -> Dict[str, Any]:
        """Generate performance predictions for variations."""
        predictions = {
            'best_ctr_variation': '',
            'best_seo_variation': '',
            'balanced_recommendation': '',
            'expected_performance': {}
        }
        
        # Find best variations
        best_ctr = max(variations, key=lambda v: v.estimated_ctr)
        best_seo = max(variations, key=lambda v: v.seo_score)
        best_overall = max(variations, key=lambda v: v.confidence_score)
        
        predictions['best_ctr_variation'] = best_ctr.variation_id
        predictions['best_seo_variation'] = best_seo.variation_id
        predictions['balanced_recommendation'] = best_overall.variation_id
        
        # Performance expectations
        for variation in variations:
            predictions['expected_performance'][variation.variation_id] = {
                'estimated_ctr': f"{variation.estimated_ctr:.1%}",
                'seo_score': f"{variation.seo_score:.1%}",
                'confidence': f"{variation.confidence_score:.1%}",
                'strategy': variation.strategy
            }
        
        return predictions
    
    async def _track_successful_patterns(self, metadata_package: MetadataPackage, 
                                       context: ContentContext):
        """Track successful metadata patterns in memory for future optimization."""
        try:
            # Store successful patterns for learning
            pattern_data = {
                'content_type': context.content_type.value,
                'successful_strategies': [v.strategy for v in metadata_package.variations 
                                        if v.confidence_score > 0.7],
                'high_performing_titles': [v.title for v in metadata_package.variations 
                                         if v.estimated_ctr > 0.6],
                'effective_tags': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Collect effective tags
            for variation in metadata_package.variations:
                if variation.seo_score > 0.7:
                    pattern_data['effective_tags'].extend(variation.tags[:5])
            
            # Remove duplicates
            pattern_data['effective_tags'] = list(set(pattern_data['effective_tags']))
            
            # Store in memory using MCP Memory tool
            try:
                from mcp_memory_add_observations import add_observations
                
                observations = [
                    f"Successful metadata patterns for {context.content_type.value} content",
                    f"High-performing strategies: {', '.join(pattern_data['successful_strategies'])}",
                    f"Effective tags identified: {', '.join(pattern_data['effective_tags'][:10])}",
                    f"Generated {len(metadata_package.variations)} variations with avg confidence: {sum(v.confidence_score for v in metadata_package.variations) / len(metadata_package.variations):.2f}"
                ]
                
                await add_observations([{
                    "entityName": "MetadataGenerator Patterns",
                    "contents": observations
                }])
            except ImportError:
                # MCP Memory not available, skip tracking
                logger.debug("MCP Memory not available, skipping pattern tracking")
            
            logger.info("Successfully tracked metadata patterns in memory")
            
        except Exception as e:
            logger.warning(f"Failed to track successful patterns: {str(e)}")