#!/usr/bin/env python3
"""
Plot Catalog Generator for OmniQ Framework

This script creates a comprehensive catalog of all generated plots,
organizing them by type and providing detailed descriptions for easy reference.
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class PlotCatalogGenerator:
    def __init__(self):
        """Initialize the plot catalog generator."""
        self.output_dirs = {
            "plots/research_plots": "Comprehensive Research Analysis",
            "plots/advanced_research_plots": "Advanced Statistical Analysis",
            "plots/publication_plots": "Publication-Ready Figures"
        }
        
        self.plot_categories = {
            "performance": "Performance Analysis",
            "efficiency": "Efficiency Analysis",
            "architecture": "Architecture Comparison",
            "statistical": "Statistical Analysis",
            "publication": "Publication Figures"
        }
        
    def categorize_plot(self, filename):
        """Categorize plot based on filename."""
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['accuracy', 'performance']):
            return "performance"
        elif any(word in filename_lower for word in ['efficiency', 'parameter', 'memory']):
            return "efficiency"
        elif any(word in filename_lower for word in ['architecture', 'fusion', 'comparison']):
            return "architecture"
        elif any(word in filename_lower for word in ['correlation', 'statistical', 'pca']):
            return "statistical"
        elif any(word in filename_lower for word in ['main_results', 'frontier', 'publication']):
            return "publication"
        else:
            return "other"
    
    def get_plot_description(self, filename):
        """Get detailed description for each plot."""
        descriptions = {
            # Combined plots
            "performance_comparison.png": "Multi-panel performance analysis showing accuracy, parameter efficiency, memory usage, and overall efficiency",
            "architectural_analysis.png": "Architecture comparison including complexity analysis, fusion mechanisms, and resource utilization",
            "efficiency_analysis.png": "Comprehensive efficiency analysis with rankings and trade-offs",
            "main_results_figure.png": "Publication-ready main results figure with 6 key analysis panels",
            "efficiency_frontier.png": "Pareto efficiency frontier showing optimal performance-resource trade-offs",
            "fusion_comparison.png": "Detailed comparison of fusion mechanisms (Transformer vs Mamba vs Baseline)",
            "correlation_analysis.png": "Statistical correlation analysis with PCA and significance testing",
            "fusion_mechanism_study.png": "In-depth fusion mechanism analysis including computational complexity",
            "lora_impact_analysis.png": "LoRA training strategy impact on performance and efficiency",
            
            # Individual plots
            "individual_accuracy_comparison.png": "Standalone accuracy comparison chart (Top-1 and Top-5)",
            "individual_parameter_efficiency.png": "Parameter efficiency scatter plot (Accuracy vs Parameters)",
            "individual_memory_efficiency.png": "Memory efficiency analysis with latency bubble sizes",
            "individual_efficiency_score.png": "Overall efficiency score ranking bar chart",
            "individual_accuracy.png": "Clean accuracy comparison for publications",
            "individual_param_efficiency.png": "Parameter efficiency bar chart for publications",
            "individual_memory_analysis.png": "Memory vs accuracy scatter plot for publications",
            "individual_latency.png": "Inference latency comparison bar chart",
            
            # Reports
            "research_summary_report.txt": "Comprehensive text summary of experimental findings and rankings",
            "statistical_analysis_report.txt": "Advanced statistical analysis report with correlations and recommendations",
            "results_table.csv": "Experimental results in CSV format for data analysis",
            "results_table.tex": "LaTeX table formatted for academic publications"
        }
        
        # Handle PDF versions
        pdf_name = filename.replace('.pdf', '.png')
        if pdf_name in descriptions:
            return descriptions[pdf_name] + " (PDF version)"
        
        return descriptions.get(filename, "Generated analysis visualization")
    
    def get_usage_recommendations(self, filename):
        """Get usage recommendations for each plot type."""
        usage_map = {
            "performance": "Use for technical reports, performance analysis sections, and comparative studies",
            "efficiency": "Ideal for resource optimization discussions and efficiency comparisons",
            "architecture": "Perfect for technical architecture sections and fusion mechanism analysis",
            "statistical": "Use in methodology sections and for detailed statistical analysis",
            "publication": "Optimized for academic papers, conference presentations, and publications",
            "individual": "Use when you need focused, single-topic visualizations for specific discussions"
        }
        
        category = self.categorize_plot(filename)
        if "individual" in filename:
            return usage_map.get("individual", usage_map.get(category, "General analysis use"))
        return usage_map.get(category, "General analysis use")
    
    def scan_output_directories(self):
        """Scan all output directories and catalog files."""
        catalog = {
            "metadata": {
                "generated_on": datetime.now().isoformat(),
                "total_files": 0,
                "total_directories": 0
            },
            "directories": {},
            "categories": {},
            "file_index": {}
        }
        
        total_files = 0
        
        for dir_name, dir_description in self.output_dirs.items():
            dir_path = Path(dir_name)
            
            if not dir_path.exists():
                continue
                
            files = list(dir_path.glob("*"))
            total_files += len(files)
            
            catalog["directories"][dir_name] = {
                "description": dir_description,
                "file_count": len(files),
                "files": []
            }
            
            for file_path in sorted(files):
                file_info = {
                    "name": file_path.name,
                    "size_kb": round(file_path.stat().st_size / 1024, 2) if file_path.exists() else 0,
                    "type": file_path.suffix.lower(),
                    "category": self.categorize_plot(file_path.name),
                    "description": self.get_plot_description(file_path.name),
                    "usage": self.get_usage_recommendations(file_path.name),
                    "path": str(file_path)
                }
                
                catalog["directories"][dir_name]["files"].append(file_info)
                catalog["file_index"][file_path.name] = file_info
                
                # Add to category index
                category = file_info["category"]
                if category not in catalog["categories"]:
                    catalog["categories"][category] = []
                catalog["categories"][category].append(file_info)
        
        catalog["metadata"]["total_files"] = total_files
        catalog["metadata"]["total_directories"] = len([d for d in self.output_dirs.keys() if Path(d).exists()])
        
        return catalog
    
    def generate_markdown_catalog(self, catalog):
        """Generate a comprehensive markdown catalog."""
        md_content = []
        
        # Header
        md_content.append("# OmniQ Framework - Plot Catalog")
        md_content.append("")
        md_content.append("This catalog provides a comprehensive overview of all generated research plots and analyses.")
        md_content.append("")
        
        # Metadata
        md_content.append("## Overview")
        md_content.append("")
        md_content.append(f"- **Generated on**: {catalog['metadata']['generated_on']}")
        md_content.append(f"- **Total files**: {catalog['metadata']['total_files']}")
        md_content.append(f"- **Output directories**: {catalog['metadata']['total_directories']}")
        md_content.append("")
        
        # Quick reference table
        md_content.append("## Quick Reference")
        md_content.append("")
        md_content.append("| Plot Type | Best For | Key Files |")
        md_content.append("|-----------|----------|-----------|")
        md_content.append("| **Combined Figures** | Comprehensive analysis | `*_comparison.pdf`, `*_analysis.pdf` |")
        md_content.append("| **Individual Plots** | Focused discussions | `individual_*.pdf` |")
        md_content.append("| **Publication Ready** | Papers & presentations | `main_results_figure.pdf`, `efficiency_frontier.pdf` |")
        md_content.append("| **Statistical Analysis** | Methodology sections | `correlation_analysis.pdf`, `*_report.txt` |")
        md_content.append("")
        
        # Directory breakdown
        md_content.append("## Directory Structure")
        md_content.append("")
        
        for dir_name, dir_info in catalog["directories"].items():
            md_content.append(f"### {dir_name}/")
            md_content.append("")
            md_content.append(f"**Purpose**: {dir_info['description']}")
            md_content.append("")
            md_content.append(f"**Files**: {dir_info['file_count']}")
            md_content.append("")
            
            # Group files by type
            file_types = {}
            for file_info in dir_info["files"]:
                file_type = file_info["type"]
                if file_type not in file_types:
                    file_types[file_type] = []
                file_types[file_type].append(file_info)
            
            for file_type, files in file_types.items():
                if file_type in ['.png', '.pdf']:
                    md_content.append(f"#### {file_type.upper()} Files")
                    md_content.append("")
                    for file_info in files:
                        md_content.append(f"- **{file_info['name']}** ({file_info['size_kb']} KB)")
                        md_content.append(f"  - *Description*: {file_info['description']}")
                        md_content.append(f"  - *Usage*: {file_info['usage']}")
                        md_content.append("")
                elif file_type in ['.txt', '.csv', '.tex']:
                    md_content.append(f"#### {file_type.upper()} Files")
                    md_content.append("")
                    for file_info in files:
                        md_content.append(f"- **{file_info['name']}** ({file_info['size_kb']} KB)")
                        md_content.append(f"  - *Description*: {file_info['description']}")
                        md_content.append("")
        
        # Category breakdown
        md_content.append("## Plots by Category")
        md_content.append("")
        
        for category, category_name in self.plot_categories.items():
            if category in catalog["categories"]:
                files = catalog["categories"][category]
                md_content.append(f"### {category_name}")
                md_content.append("")
                
                for file_info in files:
                    md_content.append(f"- **{file_info['name']}** - {file_info['description']}")
                md_content.append("")
        
        # Usage guide
        md_content.append("## Usage Guide")
        md_content.append("")
        md_content.append("### For Academic Papers")
        md_content.append("1. Use `publication_plots/main_results_figure.pdf` as your main results figure")
        md_content.append("2. Include `publication_plots/results_table.tex` in your LaTeX document")
        md_content.append("3. Reference `publication_plots/efficiency_frontier.pdf` for performance analysis")
        md_content.append("")
        md_content.append("### For Technical Reports")
        md_content.append("1. Use combined figures from `research_plots/` for comprehensive analysis")
        md_content.append("2. Include text summaries from `*_report.txt` files")
        md_content.append("3. Reference individual plots for focused discussions")
        md_content.append("")
        md_content.append("### For Presentations")
        md_content.append("1. Use individual plots for clear, focused slides")
        md_content.append("2. Start with `individual_accuracy.pdf` for performance overview")
        md_content.append("3. Use `individual_efficiency_score.pdf` for efficiency comparison")
        md_content.append("")
        
        return "\n".join(md_content)
    
    def generate_catalog(self):
        """Generate complete plot catalog."""
        print("Scanning output directories...")
        catalog = self.scan_output_directories()

        print("Generating catalog documentation...")

        # Save JSON catalog
        json_path = Path("plot_catalog.json")
        with open(json_path, 'w') as f:
            json.dump(catalog, f, indent=2)

        # Save markdown catalog
        md_content = self.generate_markdown_catalog(catalog)
        md_path = Path("PLOT_CATALOG.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"Catalog generated:")
        print(f"  {json_path} - Machine-readable catalog")
        print(f"  {md_path} - Human-readable catalog")
        print(f"  {catalog['metadata']['total_files']} files cataloged")
        print(f"  {catalog['metadata']['total_directories']} directories scanned")

        return catalog

def main():
    """Main function to generate plot catalog."""
    generator = PlotCatalogGenerator()
    catalog = generator.generate_catalog()

    print("\nCatalog Summary:")
    for category, files in catalog["categories"].items():
        if files:
            print(f"  {category.title()}: {len(files)} files")

if __name__ == "__main__":
    main()
