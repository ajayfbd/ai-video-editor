#!/usr/bin/env python3
"""
Documentation Analysis Script
Analyzes existing documentation structure and identifies redundancies
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import json

@dataclass
class DocumentSection:
    file_path: str
    heading: str
    content: str
    level: int
    line_start: int
    line_end: int

@dataclass
class ContentOverlap:
    source_files: List[str]
    overlapping_content: str
    similarity_score: float
    merge_recommendation: str
    priority: int

@dataclass
class RedundancyReport:
    total_files_analyzed: int
    total_redundancies: int
    high_priority_merges: List[ContentOverlap]
    content_consolidation_opportunities: List[Dict]
    estimated_reduction_percentage: float

class DocumentationAnalyzer:
    def __init__(self):
        self.documents = {}
        self.sections = []
        self.redundancies = []
        
    def analyze_file(self, file_path: str, content: str):
        """Analyze a single documentation file."""
        self.documents[file_path] = content
        
        # Extract sections
        sections = self._extract_sections(file_path, content)
        self.sections.extend(sections)
        
    def _extract_sections(self, file_path: str, content: str) -> List[DocumentSection]:
        """Extract sections from markdown content."""
        sections = []
        lines = content.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            # Check for markdown headers
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if header_match:
                # Save previous section
                if current_section:
                    current_section.line_end = i - 1
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                heading = header_match.group(2).strip()
                current_section = DocumentSection(
                    file_path=file_path,
                    heading=heading,
                    content="",
                    level=level,
                    line_start=i,
                    line_end=i
                )
            elif current_section:
                current_section.content += line + '\n'
        
        # Add final section
        if current_section:
            current_section.line_end = len(lines) - 1
            sections.append(current_section)
            
        return sections
    
    def find_redundancies(self) -> List[ContentOverlap]:
        """Find redundant content across documents."""
        redundancies = []
        
        # Group sections by similar headings
        heading_groups = defaultdict(list)
        for section in self.sections:
            normalized_heading = self._normalize_heading(section.heading)
            heading_groups[normalized_heading].append(section)
        
        # Find overlapping content
        for heading, sections in heading_groups.items():
            if len(sections) > 1:
                overlap = self._analyze_content_overlap(sections)
                if overlap:
                    redundancies.append(overlap)
        
        # Find similar content blocks
        content_overlaps = self._find_content_similarities()
        redundancies.extend(content_overlaps)
        
        return redundancies
    
    def _normalize_heading(self, heading: str) -> str:
        """Normalize heading for comparison."""
        # Remove common variations
        heading = heading.lower()
        heading = re.sub(r'[^\w\s]', '', heading)
        heading = re.sub(r'\s+', ' ', heading).strip()
        
        # Normalize common terms
        replacements = {
            'getting started': 'getting_started',
            'quick start': 'quick_start',
            'installation': 'install',
            'configuration': 'config',
            'troubleshooting': 'troubleshoot',
        }
        
        for old, new in replacements.items():
            heading = heading.replace(old, new)
            
        return heading
    
    def _analyze_content_overlap(self, sections: List[DocumentSection]) -> ContentOverlap:
        """Analyze overlap between sections with similar headings."""
        if len(sections) < 2:
            return None
            
        # Calculate content similarity
        contents = [section.content for section in sections]
        similarity = self._calculate_similarity(contents)
        
        if similarity > 0.3:  # 30% similarity threshold
            return ContentOverlap(
                source_files=[section.file_path for section in sections],
                overlapping_content=sections[0].heading,
                similarity_score=similarity,
                merge_recommendation=self._generate_merge_recommendation(sections),
                priority=self._calculate_priority(similarity, sections)
            )
        
        return None
    
    def _find_content_similarities(self) -> List[ContentOverlap]:
        """Find similar content blocks across different sections."""
        similarities = []
        
        # Look for specific patterns
        patterns = {
            'installation_commands': [
                r'pip install -r requirements\.txt',
                r'python -m ai_video_editor\.cli\.main init',
                r'\.env.*API.*key'
            ],
            'cli_examples': [
                r'python -m ai_video_editor\.cli\.main process',
                r'--type (educational|music|general)',
                r'--quality (low|medium|high|ultra)'
            ],
            'api_configuration': [
                r'AI_VIDEO_EDITOR_GEMINI_API_KEY',
                r'AI_VIDEO_EDITOR_IMAGEN_API_KEY',
                r'AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT'
            ],
            'system_requirements': [
                r'Python 3\.9\+',
                r'8GB.*RAM',
                r'Internet connection'
            ]
        }
        
        for pattern_name, pattern_list in patterns.items():
            matching_sections = []
            for section in self.sections:
                if any(re.search(pattern, section.content, re.IGNORECASE) for pattern in pattern_list):
                    matching_sections.append(section)
            
            if len(matching_sections) > 1:
                similarities.append(ContentOverlap(
                    source_files=[s.file_path for s in matching_sections],
                    overlapping_content=pattern_name,
                    similarity_score=0.8,  # High similarity for pattern matches
                    merge_recommendation=f"Consolidate {pattern_name} information into single authoritative source",
                    priority=90
                ))
        
        return similarities
    
    def _calculate_similarity(self, contents: List[str]) -> float:
        """Calculate similarity between content blocks."""
        if len(contents) < 2:
            return 0.0
        
        # Simple word-based similarity
        words_sets = [set(re.findall(r'\w+', content.lower())) for content in contents]
        
        # Calculate Jaccard similarity
        intersection = set.intersection(*words_sets)
        union = set.union(*words_sets)
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
    
    def _generate_merge_recommendation(self, sections: List[DocumentSection]) -> str:
        """Generate merge recommendation for overlapping sections."""
        files = [Path(s.file_path).name for s in sections]
        
        if 'README.md' in files and 'quick-guide.md' in files:
            return "Merge README.md and quick-guide.md into streamlined quick-start.md"
        elif any('user-guide' in f for f in files):
            return "Consolidate user guide sections into comprehensive single document"
        elif any('status' in s.heading.lower() for s in sections):
            return "Merge project status documents into single authoritative status report"
        else:
            return f"Consolidate duplicate content from {', '.join(files)}"
    
    def _calculate_priority(self, similarity: float, sections: List[DocumentSection]) -> int:
        """Calculate merge priority (0-100)."""
        base_priority = int(similarity * 100)
        
        # Boost priority for specific cases
        files = [Path(s.file_path).name for s in sections]
        
        if 'README.md' in files and 'quick-guide.md' in files:
            base_priority += 20
        elif any('status' in f.lower() for f in files):
            base_priority += 15
        elif len(sections) > 2:
            base_priority += 10
            
        return min(base_priority, 100)
    
    def generate_report(self) -> RedundancyReport:
        """Generate comprehensive redundancy report."""
        redundancies = self.find_redundancies()
        
        # Sort by priority
        redundancies.sort(key=lambda x: x.priority, reverse=True)
        
        # Calculate estimated reduction
        total_content = sum(len(content) for content in self.documents.values())
        redundant_content = sum(
            len(r.overlapping_content) * (len(r.source_files) - 1) 
            for r in redundancies
        )
        reduction_percentage = (redundant_content / total_content) * 100 if total_content > 0 else 0
        
        # Identify consolidation opportunities
        consolidation_opportunities = self._identify_consolidation_opportunities()
        
        return RedundancyReport(
            total_files_analyzed=len(self.documents),
            total_redundancies=len(redundancies),
            high_priority_merges=redundancies[:10],  # Top 10 priorities
            content_consolidation_opportunities=consolidation_opportunities,
            estimated_reduction_percentage=reduction_percentage
        )
    
    def _identify_consolidation_opportunities(self) -> List[Dict]:
        """Identify specific consolidation opportunities."""
        opportunities = []
        
        # Quick start consolidation
        quick_start_files = [f for f in self.documents.keys() 
                           if any(term in f.lower() for term in ['readme', 'quick', 'getting-started'])]
        if len(quick_start_files) > 1:
            opportunities.append({
                'type': 'quick_start_consolidation',
                'files': quick_start_files,
                'recommendation': 'Create single quick-start.md combining README.md and quick-guide.md',
                'priority': 95,
                'estimated_reduction': '40-50%'
            })
        
        # User guide consolidation
        user_guide_files = [f for f in self.documents.keys() if 'user-guide' in f]
        if len(user_guide_files) > 2:
            opportunities.append({
                'type': 'user_guide_consolidation',
                'files': user_guide_files,
                'recommendation': 'Merge overlapping user guide sections',
                'priority': 85,
                'estimated_reduction': '30-40%'
            })
        
        # Status document consolidation
        status_files = [f for f in self.documents.keys() 
                       if any(term in f.upper() for term in ['STATUS', 'ANALYSIS', 'SUMMARY'])]
        if len(status_files) > 1:
            opportunities.append({
                'type': 'status_consolidation',
                'files': status_files,
                'recommendation': 'Merge all status documents into single project-status.md',
                'priority': 90,
                'estimated_reduction': '60-70%'
            })
        
        # CLI reference consolidation
        cli_files = [f for f in self.documents.keys() if 'cli' in f.lower()]
        cli_sections = [s for s in self.sections if 'cli' in s.heading.lower() or 'command' in s.heading.lower()]
        if len(cli_files) > 1 or len(cli_sections) > 3:
            opportunities.append({
                'type': 'cli_consolidation',
                'files': cli_files,
                'recommendation': 'Consolidate CLI information into single authoritative reference',
                'priority': 75,
                'estimated_reduction': '25-35%'
            })
        
        return opportunities

def main():
    """Main analysis function."""
    analyzer = DocumentationAnalyzer()
    
    # Documentation files to analyze
    doc_files = [
        'README.md',
        'quick-guide.md',
        'COMPREHENSIVE_PROJECT_ANALYSIS.md',
        'CONSOLIDATED_TASK_STATUS.md',
        'TEST_FIXES_SUMMARY.md',
        'docs/README.md',
        'docs/user-guide/README.md',
        'docs/user-guide/getting-started.md',
        'docs/user-guide/quick-guide.md',
        'docs/user-guide/cli-reference.md',
        'docs/api/README.md',
        'docs/developer/architecture.md',
        'docs/examples/README.md',
        'docs/tutorials/README.md',
        'docs/tutorials/workflows/educational-content.md',
        'docs/support/troubleshooting.md',
        'docs/support/faq.md',
        'docs/support/performance.md'
    ]
    
    # Read and analyze each file
    for file_path in doc_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                analyzer.analyze_file(file_path, content)
                print(f"✅ Analyzed: {file_path}")
            except Exception as e:
                print(f"❌ Error analyzing {file_path}: {e}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    # Generate report
    print("\n🔍 Generating redundancy report...")
    report = analyzer.generate_report()
    
    # Save detailed report
    report_data = {
        'summary': {
            'total_files_analyzed': report.total_files_analyzed,
            'total_redundancies': report.total_redundancies,
            'estimated_reduction_percentage': report.estimated_reduction_percentage
        },
        'high_priority_merges': [
            {
                'source_files': overlap.source_files,
                'overlapping_content': overlap.overlapping_content,
                'similarity_score': overlap.similarity_score,
                'merge_recommendation': overlap.merge_recommendation,
                'priority': overlap.priority
            }
            for overlap in report.high_priority_merges
        ],
        'consolidation_opportunities': report.content_consolidation_opportunities
    }
    
    with open('documentation_redundancy_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Print summary
    print(f"\n📊 Documentation Analysis Complete")
    print(f"Files analyzed: {report.total_files_analyzed}")
    print(f"Redundancies found: {report.total_redundancies}")
    print(f"Estimated reduction: {report.estimated_reduction_percentage:.1f}%")
    print(f"\n📄 Detailed report saved to: documentation_redundancy_report.json")
    
    return report

if __name__ == "__main__":
    main()