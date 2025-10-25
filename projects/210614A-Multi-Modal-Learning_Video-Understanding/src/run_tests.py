#!/usr/bin/env python3
"""
Test runner for EgoVLP Multi-Scale Enhancement unit tests.

This script runs the comprehensive test suite for all multi-scale components
and provides detailed reporting on test coverage and results.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --component multiscale    # Run specific component tests
    python run_tests.py --verbose          # Detailed output
    python run_tests.py --coverage         # Run with coverage analysis
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_pytest_command(test_files, verbose=False, coverage=False):
    """Run pytest with specified options."""
    cmd = ['python', '-m', 'pytest']
    
    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
    
    if coverage:
        cmd.extend(['--cov=model', '--cov=trainer', '--cov=logger'])
        cmd.append('--cov-report=term-missing')
    
    cmd.extend(test_files)
    
    return subprocess.run(cmd, capture_output=True, text=True)


def check_dependencies():
    """Check if required test dependencies are available."""
    required = ['pytest', 'torch']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Install with: pip install pytest torch")
        return False
    
    # Check for optional coverage package
    try:
        __import__('pytest_cov')
    except ImportError:
        print("⚠ Optional package 'pytest-cov' not found. Install for coverage analysis:")
        print("pip install pytest-cov")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Run EgoVLP multi-scale tests')
    parser.add_argument('--component', choices=['multiscale', 'temporal', 'all'], 
                       default='all', help='Test component to run')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Run with coverage analysis')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path('tests').exists():
        print("❌ Tests directory not found. Run from project root.")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    print("🚀 EgoVLP Multi-Scale Enhancement Test Suite")
    print("=" * 50)
    
    # Determine which tests to run
    test_files = []
    
    if args.component == 'multiscale' or args.component == 'all':
        if Path('tests/test_multiscale.py').exists():
            test_files.append('tests/test_multiscale.py')
        else:
            print("⚠ test_multiscale.py not found")
    
    if args.component == 'temporal' or args.component == 'all':
        if Path('tests/test_temporal_sampler.py').exists():
            test_files.append('tests/test_temporal_sampler.py')
    
    # Add any other existing test files
    if args.component == 'all':
        for test_file in Path('tests').glob('test_*.py'):
            test_path = str(test_file)
            if test_path not in test_files:
                test_files.append(test_path)
    
    if not test_files:
        print("❌ No test files found to run")
        return 1
    
    print(f"📋 Running tests for: {args.component}")
    print(f"📁 Test files: {len(test_files)}")
    for test_file in test_files:
        print(f"   • {test_file}")
    
    if args.dry_run:
        print("\n🔍 Dry run - would execute:")
        cmd_preview = ['python', '-m', 'pytest']
        if args.verbose:
            cmd_preview.append('-v')
        if args.coverage:
            cmd_preview.extend(['--cov=model', '--cov=trainer', '--cov=logger'])
        cmd_preview.extend(test_files)
        print(f"   {' '.join(cmd_preview)}")
        return 0
    
    print("\n🧪 Executing tests...")
    print("-" * 30)
    
    # Run tests
    result = run_pytest_command(test_files, args.verbose, args.coverage)
    
    # Print results
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Summary
    print("\n" + "=" * 50)
    if result.returncode == 0:
        print("✅ All tests passed!")
        print("\n📊 Test Summary:")
        print("   • MultiScaleVideoEncoder: Input shapes, fusion weights, gradients")
        print("   • TemporalConsistencyLoss: Loss behavior, lambda scheduling")
        print("   • TemporalPairBatchSampler: Batch composition, temporal pairs")
        print("   • TemperatureScheduler: Cosine curve, edge cases")
        print("   • Integration tests: End-to-end training step")
        
        if args.coverage:
            print("\n📈 Coverage analysis completed")
            print("   Check coverage report above for detailed metrics")
        
        print("\n🎯 Next steps:")
        print("   • Run full training pipeline: python run/train_egoclip.py")
        print("   • Test evaluation: python run/test_egoclip.py")
        print("   • Monitor with: python logging_demo.py")
        
    else:
        print("❌ Some tests failed!")
        print(f"   Exit code: {result.returncode}")
        print("\n🔧 Troubleshooting:")
        print("   • Check that all dependencies are installed")
        print("   • Ensure CUDA is available if testing GPU functionality")
        print("   • Run individual test files to isolate issues")
        print("   • Use --verbose for detailed error messages")
    
    return result.returncode


if __name__ == '__main__':
    exit(main())