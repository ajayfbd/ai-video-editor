# tests/data/sample_data.py
"""
Sample data definitions for consistent testing across the AI Video Editor.
Provides standardized test data for all modules.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any

# ============================================================================
# SAMPLE VIDEO PROPERTIES
# ============================================================================

EDUCATIONAL_VIDEO_PROPERTIES = {
    "file_path": "mock://educational_video_15min.mp4",
    "duration": 900.0,  # 15 minutes
    "resolution": (1920, 1080),
    "fps": 30,
    "bitrate": 5000000,
    "codec": "h264",
    "file_size": 562500000,  # ~562MB
    "format": "mp4",
    "has_audio": True,
    "audio_codec": "aac",
    "audio_bitrate": 128000,
    "creation_date": "2025-01-01T10:00:00Z",
    "content_type": "educational",
    "expected_processing_time": 600  # 10 minutes max
}

MUSIC_VIDEO_PROPERTIES = {
    "file_path": "mock://music_video_5min.mp4",
    "duration": 300.0,  # 5 minutes
    "resolution": (1920, 1080),
    "fps": 60,  # Higher fps for music
    "bitrate": 8000000,
    "codec": "h264",
    "file_size": 300000000,  # ~300MB
    "format": "mp4",
    "has_audio": True,
    "audio_codec": "aac",
    "audio_bitrate": 320000,  # Higher quality audio
    "creation_date": "2025-01-01T11:00:00Z",
    "content_type": "music",
    "expected_processing_time": 300  # 5 minutes max
}

GENERAL_VIDEO_PROPERTIES = {
    "file_path": "mock://general_video_3min.mp4",
    "duration": 180.0,  # 3 minutes
    "resolution": (1920, 1080),
    "fps": 30,
    "bitrate": 4000000,
    "codec": "h264",
    "file_size": 90000000,  # ~90MB
    "format": "mp4",
    "has_audio": True,
    "audio_codec": "aac",
    "audio_bitrate": 128000,
    "creation_date": "2025-01-01T12:00:00Z",
    "content_type": "general",
    "expected_processing_time": 180  # 3 minutes max
}

# ============================================================================
# SAMPLE AUDIO TRANSCRIPTS
# ============================================================================

EDUCATIONAL_TRANSCRIPT = {
    "full_text": "Welcome to financial education. Today we're going to learn about compound interest and how it can transform your financial future. The key concept here is that your money grows exponentially over time. Let me show you some real examples. If you invest $1000 at 7% annual return, after 10 years you'll have $1967. But after 30 years, you'll have $7612. This is the power of compound interest working for you.",
    "segments": [
        {
            "text": "Welcome to financial education.",
            "start_time": 0.0,
            "end_time": 2.5,
            "confidence": 0.95,
            "speaker": "main",
            "emotional_markers": [
                {"emotion": "welcoming", "intensity": 0.7, "timestamp": 1.2}
            ]
        },
        {
            "text": "Today we're going to learn about compound interest",
            "start_time": 2.5,
            "end_time": 6.2,
            "confidence": 0.92,
            "speaker": "main",
            "emotional_markers": [
                {"emotion": "excitement", "intensity": 0.8, "timestamp": 4.5}
            ]
        },
        {
            "text": "and how it can transform your financial future.",
            "start_time": 6.2,
            "end_time": 9.8,
            "confidence": 0.89,
            "speaker": "main",
            "emotional_markers": [
                {"emotion": "confidence", "intensity": 0.9, "timestamp": 8.0}
            ]
        },
        {
            "text": "The key concept here is that your money grows exponentially over time.",
            "start_time": 9.8,
            "end_time": 14.5,
            "confidence": 0.94,
            "speaker": "main",
            "emotional_markers": [
                {"emotion": "emphasis", "intensity": 0.85, "timestamp": 12.0}
            ]
        }
    ],
    "key_concepts_identified": [
        "compound interest", "financial education", "investment strategy",
        "exponential growth", "long-term planning", "financial transformation"
    ],
    "filler_words": ["um", "uh", "you know"],
    "filler_word_count": 2,
    "speech_pace": "moderate",
    "overall_confidence": 0.925,
    "language": "en-US",
    "emotional_peaks": [
        {"timestamp": 4.5, "emotion": "excitement", "intensity": 0.8},
        {"timestamp": 8.0, "emotion": "confidence", "intensity": 0.9},
        {"timestamp": 12.0, "emotion": "emphasis", "intensity": 0.85}
    ]
}

MUSIC_TRANSCRIPT = {
    "full_text": "This beat is incredible. The way the bass drops here creates such an amazing atmosphere. You can feel the energy building up throughout the entire track. The production quality is just phenomenal.",
    "segments": [
        {
            "text": "This beat is incredible.",
            "start_time": 0.0,
            "end_time": 2.0,
            "confidence": 0.88,
            "speaker": "main",
            "emotional_markers": [
                {"emotion": "amazement", "intensity": 0.9, "timestamp": 1.0}
            ]
        },
        {
            "text": "The way the bass drops here creates such an amazing atmosphere.",
            "start_time": 2.0,
            "end_time": 6.5,
            "confidence": 0.85,
            "speaker": "main",
            "emotional_markers": [
                {"emotion": "excitement", "intensity": 0.95, "timestamp": 4.0}
            ]
        }
    ],
    "key_concepts_identified": [
        "beat", "bass drop", "atmosphere", "energy", "production quality"
    ],
    "filler_words": ["like", "you know", "just"],
    "filler_word_count": 1,
    "speech_pace": "fast",
    "overall_confidence": 0.865,
    "language": "en-US",
    "emotional_peaks": [
        {"timestamp": 1.0, "emotion": "amazement", "intensity": 0.9},
        {"timestamp": 4.0, "emotion": "excitement", "intensity": 0.95}
    ]
}

# ============================================================================
# SAMPLE API RESPONSES
# ============================================================================

GEMINI_EDUCATIONAL_RESPONSE = {
    "content_analysis": {
        "key_concepts": ["financial literacy", "compound interest", "investment strategy", "exponential growth"],
        "content_themes": ["education", "finance", "personal development", "mathematics"],
        "educational_opportunities": ["mathematical visualization", "chart generation", "calculator demonstration"],
        "content_type_suggestion": "educational",
        "complexity_level": "intermediate",
        "target_audience": "young adults"
    },
    "trending_keywords": {
        "primary_keywords": ["financial education", "compound interest explained", "investment for beginners"],
        "long_tail_keywords": ["how compound interest works", "why start investing early", "financial literacy basics"],
        "trending_hashtags": ["#FinancialLiteracy", "#CompoundInterest", "#InvestmentTips", "#MoneyManagement"],
        "seasonal_keywords": ["2025 investment trends", "new year financial goals", "retirement planning"],
        "search_volume_data": {
            "financial education": 15000,
            "compound interest": 12000,
            "investment for beginners": 8500
        },
        "competition_level": {
            "financial education": "medium",
            "compound interest": "high",
            "investment for beginners": "high"
        }
    },
    "emotional_analysis": {
        "peaks": [
            {"timestamp": 4.5, "emotion": "excitement", "intensity": 0.8, "context": "introducing compound interest"},
            {"timestamp": 8.0, "emotion": "confidence", "intensity": 0.9, "context": "explaining transformation"},
            {"timestamp": 12.0, "emotion": "emphasis", "intensity": 0.85, "context": "key concept explanation"}
        ],
        "overall_tone": "educational_enthusiastic",
        "engagement_level": 0.87,
        "clarity_score": 0.92
    },
    "competitor_insights": {
        "successful_titles": [
            "How I Made $10,000 From Compound Interest (Real Numbers)",
            "The Compound Interest Mistake That Cost Me $50,000",
            "Why I Started Investing at 18 (Compound Interest Explained)"
        ],
        "thumbnail_patterns": [
            {"pattern": "shocked_face_with_money", "success_rate": 0.85, "ctr_average": 0.12},
            {"pattern": "before_after_charts", "success_rate": 0.78, "ctr_average": 0.10},
            {"pattern": "red_arrow_pointing_up", "success_rate": 0.72, "ctr_average": 0.09}
        ],
        "common_hooks": [
            "This changed my life",
            "I wish I knew this at 18",
            "The secret banks don't want you to know"
        ]
    },
    "engagement_predictions": {
        "estimated_ctr": 0.12,
        "estimated_watch_time": 0.68,
        "estimated_retention": 0.75,
        "virality_score": 0.45,
        "educational_value": 0.89,
        "shareability": 0.67
    }
}

IMAGEN_THUMBNAIL_RESPONSE = {
    "image_url": "https://mock-imagen-api.com/generated-background-financial-education.jpg",
    "image_data": b"mock_image_data_financial_education_background",
    "generation_metadata": {
        "prompt_used": "Professional financial education background with charts, graphs, and money symbols, modern clean design, blue and gold color scheme",
        "style": "modern_professional",
        "resolution": "1920x1080",
        "generation_time": 2.8,
        "cost": 0.06,
        "model_version": "imagen-3.0",
        "safety_rating": "safe"
    },
    "quality_metrics": {
        "overall_quality": 0.92,
        "concept_alignment": 0.88,
        "visual_appeal": 0.90,
        "text_readability": 0.85,
        "brand_consistency": 0.87
    },
    "alternative_concepts": [
        {
            "concept": "calculator_and_growth_chart",
            "estimated_quality": 0.89,
            "estimated_cost": 0.05
        },
        {
            "concept": "money_tree_growing",
            "estimated_quality": 0.85,
            "estimated_cost": 0.05
        }
    ]
}

# ============================================================================
# EXPECTED OUTPUTS
# ============================================================================

EXPECTED_THUMBNAIL_CONCEPTS = [
    {
        "concept_id": "financial_education_excited",
        "emotional_theme": "excitement",
        "visual_elements": ["upward_chart", "money_symbols", "professional_background"],
        "text_overlay": "Transform Your Financial Future!",
        "background_type": "ai_generated",
        "target_keywords": ["financial education", "compound interest"],
        "ctr_prediction": 0.12,
        "alignment_score": 0.89
    },
    {
        "concept_id": "compound_interest_explanation",
        "emotional_theme": "confidence",
        "visual_elements": ["mathematical_formula", "growth_visualization", "clean_design"],
        "text_overlay": "The Secret to Wealth Building",
        "background_type": "procedural",
        "target_keywords": ["compound interest", "wealth building"],
        "ctr_prediction": 0.10,
        "alignment_score": 0.85
    }
]

EXPECTED_METADATA_SETS = [
    {
        "platform": "youtube",
        "title": "How Compound Interest Can Transform Your Financial Future (Real Examples)",
        "description": "Learn the power of compound interest with real examples and calculations. Discover why starting early is crucial for building wealth and how your money can grow exponentially over time.\n\n🔥 Key Topics Covered:\n• What is compound interest\n• Real calculation examples\n• Why time matters more than amount\n• How to get started today\n\n💰 Timestamps:\n0:00 Introduction to Financial Education\n2:30 Compound Interest Explained\n6:15 Real Examples and Calculations\n9:45 Getting Started Guide\n\n#FinancialEducation #CompoundInterest #InvestmentTips #MoneyManagement #WealthBuilding",
        "tags": [
            "financial education", "compound interest", "investment tips", "money management",
            "wealth building", "personal finance", "investing for beginners", "financial literacy"
        ],
        "hashtags": [],
        "thumbnail_alignment_score": 0.89,
        "estimated_performance": {
            "ctr": 0.12,
            "watch_time": 0.68,
            "engagement": 0.75
        }
    },
    {
        "platform": "instagram",
        "title": "",  # Instagram doesn't use titles
        "description": "🚀 The power of compound interest explained! Your money can grow exponentially over time - here's how to make it work for you.\n\n💡 Key takeaway: Starting early beats investing more later!\n\n📊 Real example: $1000 at 7% becomes $7612 in 30 years\n\n👆 Save this post and start your wealth building journey today!\n\n#FinancialEducation #CompoundInterest #InvestmentTips #MoneyManagement #WealthBuilding #PersonalFinance #FinancialLiteracy #InvestingTips #MoneyTips #FinancialFreedom",
        "tags": [],
        "hashtags": [
            "#FinancialEducation", "#CompoundInterest", "#InvestmentTips", "#MoneyManagement",
            "#WealthBuilding", "#PersonalFinance", "#FinancialLiteracy", "#InvestingTips",
            "#MoneyTips", "#FinancialFreedom"
        ],
        "thumbnail_alignment_score": 0.87,
        "estimated_performance": {
            "engagement_rate": 0.08,
            "reach": 0.15,
            "saves": 0.12
        }
    }
]

# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

PERFORMANCE_BENCHMARKS = {
    "educational_content": {
        "max_processing_time": 600,  # 10 minutes
        "max_memory_usage": 16_000_000_000,  # 16GB
        "target_api_cost": 1.50,
        "expected_quality_score": 0.85
    },
    "music_content": {
        "max_processing_time": 300,  # 5 minutes
        "max_memory_usage": 12_000_000_000,  # 12GB
        "target_api_cost": 1.00,
        "expected_quality_score": 0.80
    },
    "general_content": {
        "max_processing_time": 180,  # 3 minutes
        "max_memory_usage": 8_000_000_000,  # 8GB
        "target_api_cost": 0.75,
        "expected_quality_score": 0.75
    },
    "context_operations": {
        "max_creation_time": 0.001,  # 1ms
        "max_validation_time": 0.01,  # 10ms
        "max_serialization_time": 0.1,  # 100ms
        "max_context_size": 500_000_000  # 500MB
    }
}

# ============================================================================
# ERROR SCENARIOS
# ============================================================================

ERROR_SCENARIOS = {
    "api_failures": {
        "gemini_timeout": {
            "error_type": "timeout",
            "expected_fallback": "cached_analysis",
            "recovery_time": 5.0
        },
        "imagen_rate_limit": {
            "error_type": "rate_limit",
            "expected_fallback": "procedural_generation",
            "recovery_time": 60.0
        },
        "whisper_service_unavailable": {
            "error_type": "service_unavailable",
            "expected_fallback": "basic_transcription",
            "recovery_time": 30.0
        }
    },
    "resource_constraints": {
        "low_memory": {
            "error_type": "resource_constraint",
            "available_memory": 4_000_000_000,  # 4GB
            "expected_behavior": "quality_reduction",
            "processing_adjustment": "batch_size_1"
        },
        "high_cpu_load": {
            "error_type": "resource_constraint",
            "cpu_usage": 90,
            "expected_behavior": "parallel_processing_disabled",
            "processing_adjustment": "sequential_processing"
        }
    },
    "data_corruption": {
        "invalid_video_file": {
            "error_type": "file_corruption",
            "expected_behavior": "graceful_error",
            "user_message": "Video file appears to be corrupted"
        },
        "malformed_context": {
            "error_type": "data_validation",
            "expected_behavior": "context_repair",
            "recovery_strategy": "partial_processing"
        }
    }
}


def get_sample_data(data_type: str, content_type: str = "educational") -> Dict[str, Any]:
    """
    Get sample data for testing.
    
    Args:
        data_type: Type of data to retrieve ('video_properties', 'transcript', 'api_response', etc.)
        content_type: Content type ('educational', 'music', 'general')
    
    Returns:
        Dictionary containing the requested sample data
    """
    data_map = {
        "video_properties": {
            "educational": EDUCATIONAL_VIDEO_PROPERTIES,
            "music": MUSIC_VIDEO_PROPERTIES,
            "general": GENERAL_VIDEO_PROPERTIES
        },
        "transcript": {
            "educational": EDUCATIONAL_TRANSCRIPT,
            "music": MUSIC_TRANSCRIPT,
            "general": EDUCATIONAL_TRANSCRIPT  # Use educational as fallback
        },
        "gemini_response": {
            "educational": GEMINI_EDUCATIONAL_RESPONSE,
            "music": GEMINI_EDUCATIONAL_RESPONSE,  # Would have music-specific version
            "general": GEMINI_EDUCATIONAL_RESPONSE
        },
        "imagen_response": {
            "educational": IMAGEN_THUMBNAIL_RESPONSE,
            "music": IMAGEN_THUMBNAIL_RESPONSE,
            "general": IMAGEN_THUMBNAIL_RESPONSE
        },
        "expected_thumbnails": {
            "educational": EXPECTED_THUMBNAIL_CONCEPTS,
            "music": EXPECTED_THUMBNAIL_CONCEPTS,
            "general": EXPECTED_THUMBNAIL_CONCEPTS
        },
        "expected_metadata": {
            "educational": EXPECTED_METADATA_SETS,
            "music": EXPECTED_METADATA_SETS,
            "general": EXPECTED_METADATA_SETS
        },
        "performance_benchmarks": PERFORMANCE_BENCHMARKS,
        "error_scenarios": ERROR_SCENARIOS
    }
    
    if data_type not in data_map:
        raise ValueError(f"Unknown data type: {data_type}")
    
    if data_type in ["performance_benchmarks", "error_scenarios"]:
        return data_map[data_type]
    
    if content_type not in data_map[data_type]:
        raise ValueError(f"Unknown content type: {content_type}")
    
    return data_map[data_type][content_type]