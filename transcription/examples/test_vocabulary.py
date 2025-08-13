#!/usr/bin/env python3
"""
Test script to demonstrate the comprehensive Sanskrit/Hindi vocabulary system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from ai_video_editor.utils.sanskrit_hindi_vocab import sanskrit_hindi_vocab

def main():
    print("=== Sanskrit/Hindi Vocabulary System ===\n")
    
    # Show vocabulary statistics
    stats = sanskrit_hindi_vocab.get_vocabulary_stats()
    print("📊 Vocabulary Statistics:")
    for category, count in stats.items():
        print(f"  {category.replace('_', ' ').title()}: {count} terms")
    print()
    
    # Show sample vocabulary for different presets
    presets = ["religious", "classical", "mythological", "comprehensive"]
    
    for preset in presets:
        print(f"🔤 Sample {preset.title()} Vocabulary (first 20 terms):")
        vocab = sanskrit_hindi_vocab.get_vocabulary_prompt(preset, 20)
        terms = vocab.split(", ")
        for i, term in enumerate(terms, 1):
            print(f"  {i:2d}. {term}")
        print()
    
    # Show contextual vocabulary example
    print("🎯 Contextual Vocabulary Example:")
    print("If audio contains: ['भगवान', 'राम', 'प्रह्लाद']")
    contextual = sanskrit_hindi_vocab.get_contextual_vocabulary(['भगवान', 'राम', 'प्रह्लाद'], 15)
    terms = contextual.split(", ")
    for i, term in enumerate(terms, 1):
        print(f"  {i:2d}. {term}")
    print()
    
    # Show all categories
    print("📚 All Available Categories:")
    categories = sanskrit_hindi_vocab.get_all_categories()
    for category, words in categories.items():
        print(f"  {category.replace('_', ' ').title()}: {words[:5]}... ({len(words)} total)")
    print()
    
    print("✨ This vocabulary system provides comprehensive coverage for:")
    print("  • Religious and devotional content")
    print("  • Classical Sanskrit texts")
    print("  • Mythological stories (Ramayana, Mahabharata)")
    print("  • Philosophical and spiritual concepts")
    print("  • Ritual and ceremonial terms")
    print("  • Common Hindi words")
    print()
    print("🚀 No need for external vocabulary files - everything is built-in!")

if __name__ == "__main__":
    main()