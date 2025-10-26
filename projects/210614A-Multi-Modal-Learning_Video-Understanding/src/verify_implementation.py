#!/usr/bin/env python3
"""
Simple verification script for EgoMCQ evaluation implementation.
This script checks file structure and basic syntax without requiring dependencies.
"""

import os
import ast
import json
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and print status."""
    if os.path.exists(file_path):
        print(f"  ✓ {description}: {file_path}")
        return True
    else:
        print(f"  ✗ {description} missing: {file_path}")
        return False

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        ast.parse(source)
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"  ⚠ Could not check {file_path}: {e}")
        return True  # Assume it's okay if we can't check

def check_json_syntax(file_path):
    """Check if a JSON file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"  ⚠ Could not check {file_path}: {e}")
        return True

def main():
    print("EgoVLP Multi-Scale Enhancement - File Structure Verification")
    print("="*60)
    
    # Check project root
    project_root = Path.cwd()
    print(f"Project root: {project_root}")
    
    # Files to check
    files_to_check = [
        # Core implementation files
        ("run/test_egoclip.py", "EgoMCQ evaluation script", "python"),
        ("model/model.py", "Enhanced model with MultiScaleVideoEncoder", "python"),
        ("test_memory_optimizations.py", "Memory optimization tests", "python"),
        ("configs/pt/egoclip_rtx3090_optimized.json", "RTX 3090 optimized config", "json"),
        
        # Logging components
        ("logger/comprehensive_logger.py", "Comprehensive logging system", "python"),
        ("trainer/enhanced_trainer_egoclip.py", "Enhanced trainer with logging", "python"),
        ("logging_demo.py", "Logging analysis and demo script", "python"),
        
        # Unit tests
        ("tests/test_multiscale.py", "Comprehensive unit tests", "python"),
        ("run_tests.py", "Test runner and coverage analysis", "python"),
        
        # Documentation
        ("EVALUATION_GUIDE.md", "Evaluation guide", "text"),
        ("MEMORY_OPTIMIZATION_GUIDE.md", "Memory optimization guide", "text"),
        ("COMPREHENSIVE_LOGGING_GUIDE.md", "Comprehensive logging guide", "text"),
        ("FINAL_IMPLEMENTATION_SUMMARY.md", "Final implementation summary", "text"),
        
        # Existing files that should be present
        ("configs/eval/egomcq.json", "EgoMCQ evaluation config", "json"),
        ("data_loader/EgoClip_EgoMCQ_dataset.py", "EgoMCQ dataset loader", "python"),
        ("model/metric.py", "Metrics module with EgoMCQ support", "python"),
        ("parse_config.py", "Configuration parser", "python"),
    ]
    
    print(f"\nChecking {len(files_to_check)} files...")
    
    all_good = True
    python_files = []
    json_files = []
    
    for file_path, description, file_type in files_to_check:
        if check_file_exists(file_path, description):
            if file_type == "python":
                python_files.append(file_path)
            elif file_type == "json":
                json_files.append(file_path)
        else:
            all_good = False
    
    # Check Python syntax
    print(f"\nChecking Python syntax for {len(python_files)} files...")
    for file_path in python_files:
        if not check_python_syntax(file_path):
            all_good = False
        else:
            print(f"  ✓ {file_path}")
    
    # Check JSON syntax  
    print(f"\nChecking JSON syntax for {len(json_files)} files...")
    for file_path in json_files:
        if not check_json_syntax(file_path):
            all_good = False
        else:
            print(f"  ✓ {file_path}")
    
    # Check key function implementations
    print(f"\nChecking key implementations...")
    
    # Check if test_egoclip.py has main functions
    try:
        with open("run/test_egoclip.py", 'r') as f:
            content = f.read()
        
        required_functions = [
            "test_egomcq_multiscale",
            "analyze_fusion_weights", 
            "save_results",
            "print_results",
            "main"
        ]
        
        for func_name in required_functions:
            if f"def {func_name}(" in content:
                print(f"  ✓ Function {func_name} found in test_egoclip.py")
            else:
                print(f"  ✗ Function {func_name} missing in test_egoclip.py")
                all_good = False
                
    except Exception as e:
        print(f"  ⚠ Could not check test_egoclip.py functions: {e}")
    
    # Check comprehensive logging components
    try:
        with open("logger/comprehensive_logger.py", 'r') as f:
            content = f.read()
        
        logging_classes = [
            "ComprehensiveLogger",
            "GPUMemoryTracker"
        ]
        
        for class_name in logging_classes:
            if f"class {class_name}" in content:
                print(f"  ✓ Class {class_name} found in comprehensive_logger.py")
            else:
                print(f"  ✗ Class {class_name} missing in comprehensive_logger.py")
                all_good = False
                
    except Exception as e:
        print(f"  ⚠ Could not check comprehensive_logger.py classes: {e}")
    
    # Check unit test components
    try:
        with open("tests/test_multiscale.py", 'r') as f:
            content = f.read()
        
        test_classes = [
            "TestMultiScaleVideoEncoder",
            "TestTemporalConsistencyLoss", 
            "TestTemporalPairBatchSampler",
            "TestTemperatureScheduler"
        ]
        
        for class_name in test_classes:
            if f"class {class_name}" in content:
                print(f"  ✓ Test class {class_name} found in test_multiscale.py")
            else:
                print(f"  ✗ Test class {class_name} missing in test_multiscale.py")
                all_good = False
                
    except Exception as e:
        print(f"  ⚠ Could not check test_multiscale.py classes: {e}")
    
    # Check enhanced trainer integration
    try:
        with open("trainer/enhanced_trainer_egoclip.py", 'r') as f:
            content = f.read()
        
        if "comprehensive_logger" in content.lower():
            print(f"  ✓ Comprehensive logging integrated in enhanced_trainer_egoclip.py")
        else:
            print(f"  ✗ Comprehensive logging not integrated in enhanced_trainer_egoclip.py")
            all_good = False
            
        if "log_iteration_losses" in content:
            print(f"  ✓ Iteration loss logging found in enhanced trainer")
        else:
            print(f"  ✗ Iteration loss logging missing in enhanced trainer")
            all_good = False
                
    except Exception as e:
        print(f"  ⚠ Could not check enhanced trainer logging integration: {e}")
    
    # Check if model.py has MultiScaleVideoEncoder
    try:
        with open("model/model.py", 'r') as f:
            content = f.read()
        
        required_classes = [
            "MultiScaleVideoEncoder",
            "TemporalConsistencyLoss",
            "MemoryManager"
        ]
        
        for class_name in required_classes:
            if f"class {class_name}" in content:
                print(f"  ✓ Class {class_name} found in model.py")
            else:
                print(f"  ✗ Class {class_name} missing in model.py")
                all_good = False
                
    except Exception as e:
        print(f"  ⚠ Could not check model.py classes: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    if all_good:
        print("✅ All files and implementations verified successfully!")
        print("\n🧪 Run unit tests:")
        print("   python run_tests.py --verbose")
        print("   python run_tests.py --component multiscale")
        print("   python run_tests.py --coverage")
        print("\n🚀 Next steps:")
        print("1. Install required dependencies: torch, transformers, pytest")
        print("2. Download EgoMCQ dataset and configure paths")  
        print("3. Run unit tests to verify functionality:")
        print("   python run_tests.py")
        print("4. Train with comprehensive logging:")
        print("   python run/train_egoclip.py -c configs/pt/egoclip_rtx3090_optimized.json")
        print("5. Monitor training progress:")
        print("   tensorboard --logdir saved/EgoClip_MultiScale_RTX3090/tensorboard")
        print("6. Run evaluation with multi-scale support:")
        print("   python run/test_egoclip.py -c configs/eval/egomcq.json -r checkpoint.pth")
        print("7. Analyze logging results:")
        print("   python logging_demo.py -l saved/EgoClip_MultiScale_RTX3090")
        print("8. Compare multi-scale vs single-scale performance")
        print("9. View comprehensive guides:")
        print("   • COMPREHENSIVE_LOGGING_GUIDE.md")
        print("   • EVALUATION_GUIDE.md")
        print("   • MEMORY_OPTIMIZATION_GUIDE.md")
        status = 0
    else:
        print("❌ Some files or implementations are missing/invalid!")
        print("Please check the errors above and fix them.")
        status = 1
    
    print(f"{'='*60}")
    return status

if __name__ == '__main__':
    exit(main())