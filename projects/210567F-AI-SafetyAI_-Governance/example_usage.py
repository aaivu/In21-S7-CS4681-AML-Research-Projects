#!/usr/bin/env python3
"""
Example usage script for SafetyAlignNLP multi-agent system.
This script demonstrates various ways to use the system.
"""

import os
import json
from utils.config import Config
from utils.data_loader import DataLoader
from agents.multi_agent_system import MultiAgentSystem


def example_single_text_analysis():
    """Example: Analyze a single text for safety concerns."""
    print("=" * 60)
    print("Example 1: Single Text Analysis")
    print("=" * 60)
    
    # Initialize system
    config = Config()
    system = MultiAgentSystem(config)
    
    if not system.initialize_agents():
        print("❌ Failed to initialize agents. Check your API keys.")
        return
    
    # Example text to analyze
    test_text = """
    I think we should have a respectful discussion about different viewpoints 
    on this topic. Everyone deserves to be heard and treated with dignity.
    """
    
    print(f"Analyzing text: {test_text.strip()}")
    
    # Perform comprehensive analysis
    result = system.process_text(test_text, "comprehensive_analysis")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Analysis completed!")
        if "synthesis" in result:
            print("\n📊 Analysis Summary:")
            print("-" * 40)
            print(result["synthesis"][:300] + "..." if len(result["synthesis"]) > 300 else result["synthesis"])
    
    system.shutdown()


def example_dataset_analysis():
    """Example: Analyze a dataset sample."""
    print("\n" + "=" * 60)
    print("Example 2: Dataset Analysis")
    print("=" * 60)
    
    # Check available datasets
    data_loader = DataLoader()
    datasets = data_loader.list_available_datasets()
    
    if not datasets:
        print("❌ No datasets found in the data directory.")
        return
    
    print(f"📁 Available datasets: {', '.join(datasets)}")
    
    # Use the first available dataset
    dataset_name = datasets[0]
    print(f"📊 Analyzing dataset: {dataset_name}")
    
    # Initialize system
    config = Config()
    system = MultiAgentSystem(config)
    
    if not system.initialize_agents():
        print("❌ Failed to initialize agents. Check your API keys.")
        return
    
    # Analyze dataset with small sample for demo
    result = system.process_dataset(dataset_name, sample_size=5)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Dataset analysis completed!")
        if "evaluation_summary" in result:
            print("\n📋 Dataset Evaluation Summary:")
            print("-" * 40)
            print(result["evaluation_summary"][:400] + "..." if len(result["evaluation_summary"]) > 400 else result["evaluation_summary"])
    
    system.shutdown()


def example_batch_processing():
    """Example: Process multiple texts in batch."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    # Sample texts for batch processing
    sample_texts = [
        "This is a neutral statement about technology.",
        "I appreciate everyone's contributions to this discussion.",
        "Let's work together to find a solution that benefits everyone."
    ]
    
    print(f"📦 Processing {len(sample_texts)} texts in batch...")
    
    # Initialize system
    config = Config()
    system = MultiAgentSystem(config)
    
    if not system.initialize_agents():
        print("❌ Failed to initialize agents. Check your API keys.")
        return
    
    # Process batch
    result = system.batch_process_texts(sample_texts, "safety_analysis")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Batch processing completed!")
        if "batch_summary" in result:
            print("\n📈 Batch Summary:")
            print("-" * 40)
            print(result["batch_summary"][:300] + "..." if len(result["batch_summary"]) > 300 else result["batch_summary"])
    
    system.shutdown()


def example_system_monitoring():
    """Example: Monitor system status and usage."""
    print("\n" + "=" * 60)
    print("Example 4: System Monitoring")
    print("=" * 60)
    
    # Initialize system
    config = Config()
    system = MultiAgentSystem(config)
    
    if not system.initialize_agents():
        print("❌ Failed to initialize agents. Check your API keys.")
        return
    
    # Get initial status
    status = system.get_system_status()
    print("📊 System Status:")
    print(f"  Initialized: {status.get('system_initialized', False)}")
    print(f"  Guardrails: {status.get('guardrails_enabled', False)}")
    print(f"  Available datasets: {len(status.get('available_datasets', []))}")
    
    # Process a sample text to generate some usage
    system.process_text("Sample text for monitoring demo.", "safety_analysis")
    
    # Get token usage
    usage = system.get_agent_token_usage()
    print("\n💰 Token Usage:")
    for agent, stats in usage.items():
        print(f"  {agent}: {stats.get('total_tokens', 0)} tokens")
    
    # Get updated status
    updated_status = system.get_system_status()
    stats = updated_status.get('system_stats', {})
    print("\n📈 Processing Statistics:")
    print(f"  Tasks completed: {stats.get('tasks_completed', 0)}")
    print(f"  Texts processed: {stats.get('total_texts_processed', 0)}")
    print(f"  Errors encountered: {stats.get('errors_encountered', 0)}")
    
    system.shutdown()


def example_custom_configuration():
    """Example: Using custom configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = Config()
    
    # Customize agent settings
    config.safety_agent.temperature = 0.3  # More deterministic
    config.analysis_agent.temperature = 0.8  # More creative
    config.safety_agent.max_tokens = 500
    
    # Disable guardrails for this example
    config.guardrails.enabled = False
    
    print("🔧 Using custom configuration:")
    print(f"  Safety agent temperature: {config.safety_agent.temperature}")
    print(f"  Analysis agent temperature: {config.analysis_agent.temperature}")
    print(f"  Guardrails enabled: {config.guardrails.enabled}")
    
    # Initialize system with custom config
    system = MultiAgentSystem(config)
    
    if not system.initialize_agents():
        print("❌ Failed to initialize agents. Check your API keys.")
        return
    
    # Test with custom configuration
    result = system.process_text(
        "This is a test of the custom configuration.",
        "comprehensive_analysis"
    )
    
    if "error" not in result:
        print("✅ Custom configuration working correctly!")
    
    system.shutdown()


def main():
    """Run all examples."""
    print("🚀 SafetyAlignNLP Example Usage")
    print("This script demonstrates various features of the multi-agent system.")
    print("\n⚠️  Note: You need to set your OPENAI_API_KEY environment variable.")
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ OPENAI_API_KEY environment variable not set.")
        print("Please set it before running this example:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Run examples
        example_single_text_analysis()
        example_dataset_analysis()
        example_batch_processing()
        example_system_monitoring()
        example_custom_configuration()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    main()
