#!/usr/bin/env python3
"""
Master Script for OmniQ Research Plot Generation

This script runs all plotting and analysis tools to generate comprehensive
research visualizations and reports for the OmniQ framework.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    """Run a plotting script and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print('='*60)
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"SUCCESS: {description} completed in {end_time-start_time:.1f}s")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"ERROR: {description} failed")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {description} took too long (>5 minutes)")
        return False
    except Exception as e:
        print(f"EXCEPTION: {description} crashed: {e}")
        return False
    
    return True

def check_requirements():
    """Check if required packages are installed."""
    required_packages = {
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn'  # sklearn is the import name for scikit-learn
    }

    missing_packages = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        print(" Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        return False

    return True

def check_data_file():
    """Check if the results CSV file exists."""
    csv_path = Path("results/summary.csv")
    if not csv_path.exists():
        print(f" Data file not found: {csv_path}")
        print("Please ensure you have experimental results in results/summary.csv")
        return False
    
    print(f"Data file found: {csv_path}")
    return True

def create_output_summary():
    """Create a summary of all generated outputs."""
    output_dirs = [
        "research_plots",
        "advanced_research_plots", 
        "publication_plots"
    ]
    
    summary_path = Path("PLOTTING_SUMMARY.md")
    
    with open(summary_path, 'w') as f:
        f.write("# OmniQ Research Plots Summary\n\n")
        f.write("This document summarizes all generated research plots and analyses.\n\n")
        
        f.write("## Generated Output Directories\n\n")
        
        total_files = 0
        for output_dir in output_dirs:
            dir_path = Path(output_dir)
            if dir_path.exists():
                files = list(dir_path.glob("*"))
                total_files += len(files)
                
                f.write(f"### {output_dir}/\n")
                f.write(f"**Purpose**: {get_dir_description(output_dir)}\n\n")
                f.write(f"**Files generated**: {len(files)}\n\n")
                
                for file in sorted(files):
                    f.write(f"- `{file.name}` - {get_file_description(file.name)}\n")
                f.write("\n")
        
        f.write(f"## Summary Statistics\n\n")
        f.write(f"- **Total files generated**: {total_files}\n")
        f.write(f"- **Output directories**: {len(output_dirs)}\n")
        f.write(f"- **Generated on**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Usage Recommendations\n\n")
        f.write("### For Research Papers\n")
        f.write("- Use `publication_plots/main_results_figure.pdf` as main results figure\n")
        f.write("- Include `publication_plots/results_table.tex` in LaTeX documents\n")
        f.write("- Reference `publication_plots/efficiency_frontier.pdf` for performance analysis\n\n")
        
        f.write("### For Technical Reports\n")
        f.write("- Use plots from `research_plots/` for comprehensive analysis\n")
        f.write("- Include summary from `research_plots/research_summary_report.txt`\n")
        f.write("- Reference architectural analysis plots for technical details\n\n")
        
        f.write("### For Advanced Analysis\n")
        f.write("- Use `advanced_research_plots/` for statistical analysis\n")
        f.write("- Reference correlation analysis for feature relationships\n")
        f.write("- Use LoRA impact analysis for training strategy insights\n\n")
    
    print(f" Summary created: {summary_path}")

def get_dir_description(dir_name):
    """Get description for output directory."""
    descriptions = {
        "research_plots": "Comprehensive research analysis with performance, architectural, and efficiency plots",
        "advanced_research_plots": "Advanced statistical analysis including correlations, PCA, and fusion studies", 
        "publication_plots": "Publication-ready figures optimized for papers and presentations"
    }
    return descriptions.get(dir_name, "Generated plots and analysis")

def get_file_description(filename):
    """Get description for specific files."""
    descriptions = {
        "performance_comparison.png": "Multi-panel performance analysis",
        "performance_comparison.pdf": "Multi-panel performance analysis (PDF)",
        "architectural_analysis.png": "Architecture comparison and complexity analysis",
        "architectural_analysis.pdf": "Architecture comparison and complexity analysis (PDF)",
        "efficiency_analysis.png": "Detailed efficiency metrics and rankings",
        "efficiency_analysis.pdf": "Detailed efficiency metrics and rankings (PDF)",
        "research_summary_report.txt": "Comprehensive text summary of findings",
        "correlation_analysis.png": "Statistical correlations and PCA analysis",
        "correlation_analysis.pdf": "Statistical correlations and PCA analysis (PDF)",
        "fusion_mechanism_study.png": "Detailed fusion mechanism comparison",
        "fusion_mechanism_study.pdf": "Detailed fusion mechanism comparison (PDF)",
        "lora_impact_analysis.png": "LoRA training strategy analysis",
        "lora_impact_analysis.pdf": "LoRA training strategy analysis (PDF)",
        "statistical_analysis_report.txt": "Advanced statistical analysis report",
        "main_results_figure.png": "Main publication figure with key results",
        "main_results_figure.pdf": "Main publication figure with key results (PDF)",
        "efficiency_frontier.png": "Pareto efficiency frontier analysis",
        "efficiency_frontier.pdf": "Pareto efficiency frontier analysis (PDF)",
        "fusion_comparison.png": "Fusion mechanism performance comparison",
        "fusion_comparison.pdf": "Fusion mechanism performance comparison (PDF)",
        "results_table.csv": "Experimental results in CSV format",
        "results_table.tex": "LaTeX table for publication"
    }
    return descriptions.get(filename, "Generated analysis file")

def main():
    """Main function to run all plotting scripts."""
    print(" OmniQ Research Plot Generation Suite")
    print("="*60)
    
    # Check prerequisites
    print(" Checking prerequisites...")
    if not check_requirements():
        print(" Prerequisites not met. Please install required packages.")
        return False
    
    if not check_data_file():
        print(" Data file not found. Please ensure experimental results exist.")
        return False
    
    print(" All prerequisites met!")
    
    # Define scripts to run
    scripts = [
        ("generate_research_plots.py", "Research Analysis Plots"),
        ("advanced_research_analysis.py", "Advanced Statistical Analysis"),
        ("publication_plots.py", "Publication-Ready Figures")
    ]
    
    # Run all scripts
    success_count = 0
    total_scripts = len(scripts)
    
    for script_name, description in scripts:
        if run_script(script_name, description):
            success_count += 1
        else:
            print(f"  Continuing with remaining scripts...")
    
    # Generate summary
    print(f"\n{'='*60}")
    print(" FINAL SUMMARY")
    print('='*60)
    print(f"Successful: {success_count}/{total_scripts} scripts")
    print(f" Failed: {total_scripts - success_count}/{total_scripts} scripts")
    
    if success_count > 0:
        print("\n Creating output summary...")
        create_output_summary()
        

    else:
        print("\n No plots were generated successfully.")
        print("Please check error messages above and resolve issues.")
    
    return success_count == total_scripts

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
