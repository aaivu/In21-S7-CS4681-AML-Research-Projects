#!/usr/bin/env python3
"""
Complete OmniQ Research Analysis Generator

This master script generates all research plots, analyses, and documentation
for the OmniQ framework, including both individual and combined visualizations.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_script_with_output(script_name, description):
    """Run a script and capture its output."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"Script: {script_name}")
    print('='*70)
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"SUCCESS: Completed in {end_time-start_time:.1f}s")
            # Print key output lines
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-10:]:  # Show last 10 lines
                if line.strip() and not line.startswith('['):
                    print(f"   {line}")
            return True
        else:
            print(f"ERROR: {description} failed")
            print("Error details:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {description} took too long")
        return False
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False

def check_data_and_dependencies():
    """Check prerequisites."""
    print("Checking prerequisites...")
    
    # Check data file
    csv_path = Path("results/summary.csv")
    if not csv_path.exists():
        print(f"Data file not found: {csv_path}")
        return False
    
    # Check key dependencies
    try:
        import matplotlib
        import seaborn
        import pandas
        import numpy
        print("All dependencies available")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def count_generated_files():
    """Count all generated files."""
    output_dirs = ["plots/research_plots", "plots/advanced_research_plots", "plots/publication_plots"]
    total_files = 0
    file_breakdown = {}

    for dir_name in output_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            file_breakdown[dir_name] = len(files)
            total_files += len(files)
        else:
            file_breakdown[dir_name] = 0

    return total_files, file_breakdown

def generate_final_summary():
    """Generate a comprehensive final summary."""
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*70)
    
    total_files, breakdown = count_generated_files()
    
    print(f"\n Generated Files Summary:")
    print(f"   Total files: {total_files}")
    for dir_name, count in breakdown.items():
        print(f"   {dir_name}: {count} files")
    
    print(f"\n Output Structure:")
    output_dirs = ["plots/research_plots", "plots/advanced_research_plots", "plots/publication_plots"]
    
    for dir_name in output_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"\n   {dir_name}/")
            files = sorted(dir_path.glob("*"))
            
            # Group by type
            png_files = [f for f in files if f.suffix == '.png']
            pdf_files = [f for f in files if f.suffix == '.pdf']
            other_files = [f for f in files if f.suffix not in ['.png', '.pdf']]
            
            if png_files:
                print(f"     PNG files ({len(png_files)}):")
                for f in png_files[:5]:  # Show first 5
                    print(f"       - {f.name}")
                if len(png_files) > 5:
                    print(f"       ... and {len(png_files)-5} more")
            
            if pdf_files:
                print(f"     PDF files ({len(pdf_files)}):")
                for f in pdf_files[:3]:  # Show first 3
                    print(f"       - {f.name}")
                if len(pdf_files) > 3:
                    print(f"       ... and {len(pdf_files)-3} more")
            
            if other_files:
                print(f"     Other files ({len(other_files)}):")
                for f in other_files:
                    print(f"       - {f.name}")
    
    print(f"\n Key Outputs for Different Uses:")
    print(f"    For Academic Papers:")
    print(f"     - plots/publication_plots/main_results_figure.pdf")
    print(f"     - plots/publication_plots/results_table.tex")
    print(f"     - plots/publication_plots/efficiency_frontier.pdf")

    print(f"    For Technical Reports:")
    print(f"     - plots/research_plots/performance_comparison.pdf")
    print(f"     - plots/research_plots/architectural_analysis.pdf")
    print(f"     - plots/research_plots/research_summary_report.txt")

    print(f"    For Detailed Analysis:")
    print(f"     - plots/advanced_research_plots/correlation_analysis.pdf")
    print(f"     - plots/advanced_research_plots/statistical_analysis_report.txt")

    print(f"    For Individual Plots:")
    print(f"     - plots/research_plots/individual_*.pdf")
    print(f"     - plots/publication_plots/individual_*.pdf")
    
    print(f"\n Documentation:")
    if Path("PLOT_CATALOG.md").exists():
        print(f"     - PLOT_CATALOG.md (Complete catalog)")
    if Path("plot_catalog.json").exists():
        print(f"     - plot_catalog.json (Machine-readable)")


def main():
    """Main function to run complete analysis."""

    
    # Check prerequisites
    if not check_data_and_dependencies():
        print("\n Prerequisites not met. Please resolve issues above.")
        return False
    
    # Define analysis scripts
    analysis_scripts = [
        ("tools/generate_research_plots.py", "Research Analysis Plots (Combined + Individual)"),
        ("tools/advanced_research_analysis.py", "Advanced Statistical Analysis"),
        ("tools/publication_plots.py", "Publication-Ready Figures (Combined + Individual)"),
        ("tools/plot_catalog_generator.py", "Comprehensive Plot Catalog")
    ]
    
    # Run all analysis scripts
    success_count = 0
    total_scripts = len(analysis_scripts)
    
    for script_name, description in analysis_scripts:
        if run_script_with_output(script_name, description):
            success_count += 1
        else:
            print(f"   Continuing with remaining scripts...")
    
    # Generate final summary
    if success_count > 0:
        generate_final_summary()
        
        print(f"\n Analysis Generation Complete!")
        print(f"  {success_count}/{total_scripts} scripts completed successfully")
        
        if success_count == total_scripts:
            print(f"   All analyses generated successfully!")
        else:
            print(f"    {total_scripts - success_count} scripts had issues")
        
        return True
    else:
        print(f"\n No analyses were generated successfully.")
        print(f"Please check error messages above and resolve issues.")
        return False

if __name__ == "__main__":
    success = main()
    
    print(f"\n{'='*70}")
    if success:
        print(" COMPLETE SUCCESS: All research analyses generated!")
        print("Check the output directories and PLOT_CATALOG.md for details.")
    else:
        print(" PARTIAL/FAILED: Some issues occurred during generation.")
        print("Please review error messages and try again.")
    print("="*70)
    
    sys.exit(0 if success else 1)
