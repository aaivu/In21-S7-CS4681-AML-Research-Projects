#!/usr/bin/env python3
"""
SafetyAlignNLP - Multi-Agent System for Safety Alignment and NLP Analysis

This is the main entry point for the multi-agent system designed to analyze
adversarial text data for safety concerns and patterns.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from utils.config import Config
from utils.data_loader import DataLoader
from utils.safety import (
    get_global_safety_layer, wrap_agent_with_safety, 
    set_global_harm_threshold, get_global_safety_statistics
)
from agents.multi_agent_system import MultiAgentSystem
from agents.overseer import OverseerAgent


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('multi_agent_system.log')
        ]
    )


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    SafetyAlignNLP                            ║
║              Multi-Agent Safety Analysis System             ║
║                                                              ║
║  Powered by LangChain, HuggingFace, and NeMo Guardrails    ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def analyze_single_text(system, text: str, task_type: str = "comprehensive_analysis") -> Dict[str, Any]:
    """Analyze a single text input with TSDI safety layer."""
    print(f"\n🔍 Analyzing text (length: {len(text)} characters)...")
    print(f"📝 Preview: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Wrap system with safety layer if it's MultiAgentSystem
    if isinstance(system, MultiAgentSystem):
        wrapped_system = wrap_agent_with_safety(system)
        result = wrapped_system.process_text(text, task_type)
    else:
        # For OverseerAgent, it already has safety mechanisms
        result = system.process(text)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return result
    elif result.get('status') == 'blocked':
        print(f"🚫 Content blocked by TSDI safety layer: {result.get('message', 'Safety violation')}")
        if 'safety_details' in result:
            safety_details = result['safety_details']
            if 'harm_scores' in safety_details:
                print(f"🛡️  Harm scores: {safety_details['harm_scores']}")
        return result
    
    print("✅ Analysis completed successfully!")
    
    # Display safety information
    if result.get('safety_verified'):
        print("🛡️  Content verified safe by TSDI layer")
        if 'harm_scores' in result:
            max_harm = max(result['harm_scores'].values()) if result['harm_scores'] else 0
            print(f"📊 Max harm score: {max_harm:.3f}")
    
    # Display summary
    if "synthesis" in result:
        print("\n📊 Analysis Summary:")
        print("-" * 50)
        print(result["synthesis"][:500] + "..." if len(result["synthesis"]) > 500 else result["synthesis"])
    
    return result


def analyze_dataset(system: MultiAgentSystem, dataset_name: str, sample_size: int = 20) -> Dict[str, Any]:
    """Analyze a dataset with TSDI safety layer."""
    print(f"\n📊 Analyzing dataset: {dataset_name}")
    print(f"📈 Sample size: {sample_size}")
    
    # Wrap system with safety layer
    wrapped_system = wrap_agent_with_safety(system)
    result = wrapped_system.process_dataset(dataset_name, sample_size)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return result
    elif result.get('status') == 'blocked':
        print(f"🚫 Dataset processing blocked by TSDI safety layer: {result.get('message', 'Safety violation')}")
        return result
    
    print("✅ Dataset analysis completed successfully!")
    
    # Display safety statistics
    safety_stats = get_global_safety_statistics()
    print(f"🛡️  TSDI Safety Stats: {safety_stats['safety_rate']:.2%} safe, {safety_stats['total_blocked']} blocked")
    
    # Display summary
    if "evaluation_summary" in result:
        print("\n📋 Dataset Evaluation Summary:")
        print("-" * 50)
        print(result["evaluation_summary"][:500] + "..." if len(result["evaluation_summary"]) > 500 else result["evaluation_summary"])
    
    return result


def interactive_mode(system):
    """Run the system in interactive mode."""
    print("\n🎯 Entering interactive mode. Type 'help' for commands, 'quit' to exit.")
    
    # Determine system type
    is_overseer = isinstance(system, OverseerAgent)
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif command.lower() == 'help':
                if is_overseer:
                    print("""
Available commands (Overseer Mode):
  query <text>             - Process query through overseer with safety scanning
  summarize <text>         - Summarize text
  translate <text> to <lang> - Translate text to target language
  ask <question>           - Ask a question
  safety <text>            - Check text safety only
  safety-stats             - Show safety statistics
  tsdi-stats               - Show TSDI safety layer statistics
  status                   - Show system status
  reset                    - Reset safety log
  help                     - Show this help message
  quit/exit/q              - Exit the program
                    """)
                else:
                    print("""
Available commands (Multi-Agent Mode):
  analyze <text>           - Analyze a text string
  dataset <name> [size]    - Analyze a dataset (optional sample size)
  list                     - List available datasets
  status                   - Show system status
  usage                    - Show token usage statistics
  tsdi-stats               - Show TSDI safety layer statistics
  reset                    - Reset conversation history
  help                     - Show this help message
  quit/exit/q              - Exit the program
                    """)
            
            elif command.lower() == 'list' and not is_overseer:
                data_loader = DataLoader()
                datasets = data_loader.list_available_datasets()
                print(f"\n📁 Available datasets: {', '.join(datasets)}")
            
            elif command.lower() == 'status':
                if is_overseer:
                    status = system.get_agent_status()
                    print(f"\n📊 Overseer Status:")
                    print(f"  Available task agents: {', '.join(status.get('available_task_agents', []))}")
                    print(f"  Safety tools: {status.get('safety_tools_count', 0)}")
                    stats = status.get('safety_statistics', {})
                    print(f"  Total requests: {stats.get('total_requests', 0)}")
                    print(f"  Blocked requests: {stats.get('blocked_requests', 0)}")
                    print(f"  Safety rate: {stats.get('safety_rate', 0):.2%}")
                else:
                    status = system.get_system_status()
                    print(f"\n📊 System Status:")
                    print(f"  Initialized: {status.get('system_initialized', False)}")
                    print(f"  Guardrails: {status.get('guardrails_enabled', False)}")
                    print(f"  Tasks completed: {status.get('system_stats', {}).get('tasks_completed', 0)}")
                    print(f"  Texts processed: {status.get('system_stats', {}).get('total_texts_processed', 0)}")
            
            elif command.lower() == 'usage' and not is_overseer:
                usage = system.get_agent_token_usage()
                print("\n💰 Token Usage:")
                for agent, stats in usage.items():
                    print(f"  {agent}: {stats.get('total_tokens', 0)} tokens")
            
            elif command.lower() == 'safety-stats' and is_overseer:
                stats = system.get_safety_statistics()
                print("\n🛡️  OverseerAgent Safety Statistics:")
                print(f"  Total requests: {stats.get('total_requests', 0)}")
                print(f"  Blocked requests: {stats.get('blocked_requests', 0)}")
                print(f"  Safety rate: {stats.get('safety_rate', 0):.2%}")
                print(f"  Recent safety events: {stats.get('total_safety_events', 0)}")
                
                # Also show TSDI global stats
                tsdi_stats = get_global_safety_statistics()
                print(f"\n🛡️  TSDI Global Safety Statistics:")
                print(f"  Total processed: {tsdi_stats.get('total_processed', 0)}")
                print(f"  Total blocked: {tsdi_stats.get('total_blocked', 0)}")
                print(f"  Safety rate: {tsdi_stats.get('safety_rate', 0):.2%}")
                print(f"  Bias corrections: {tsdi_stats.get('bias_corrections', 0)}")
                print(f"  Harm threshold: {tsdi_stats.get('harm_threshold', 0.2)}")
            
            elif command.lower() == 'tsdi-stats':
                tsdi_stats = get_global_safety_statistics()
                print("\n🛡️  TSDI Safety Statistics:")
                print(f"  Total processed: {tsdi_stats.get('total_processed', 0)}")
                print(f"  Total blocked: {tsdi_stats.get('total_blocked', 0)}")
                print(f"  Safety rate: {tsdi_stats.get('safety_rate', 0):.2%}")
                print(f"  Bias corrections: {tsdi_stats.get('bias_corrections', 0)}")
                print(f"  Harm threshold: {tsdi_stats.get('harm_threshold', 0.2)}")
                print(f"  Recent events: {len(tsdi_stats.get('recent_events', []))}")
            
            elif command.lower() == 'reset':
                if is_overseer:
                    system.reset_safety_log()
                    print("🔄 Safety log reset")
                else:
                    system.reset_all_conversations()
                    print("🔄 Conversation history reset for all agents")
            
            # Overseer-specific commands
            elif command.startswith('query ') and is_overseer:
                query = command[6:].strip()
                if query:
                    result = system.process(query)
                    if result.get('status') == 'completed':
                        print(f"\n✅ Response: {result.get('response', 'No response')}")
                    elif result.get('status') == 'blocked':
                        print(f"\n🚫 Request blocked: {result.get('response', 'Safety violation')}")
                    else:
                        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                else:
                    print("❌ Please provide a query")
            
            elif command.startswith('summarize ') and is_overseer:
                text = command[10:].strip()
                if text:
                    result = system.process({"query": f"Summarize: {text}", "task_type": "summarization"})
                    if result.get('status') == 'completed':
                        print(f"\n📝 Summary: {result.get('response', 'No response')}")
                    else:
                        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                else:
                    print("❌ Please provide text to summarize")
            
            elif command.startswith('translate ') and is_overseer:
                parts = command[10:].strip().split(' to ')
                if len(parts) == 2:
                    text, target_lang = parts
                    query = f"Translate '{text}' to {target_lang}"
                    result = system.process({"query": query, "task_type": "translation"})
                    if result.get('status') == 'completed':
                        print(f"\n🌐 Translation: {result.get('response', 'No response')}")
                    else:
                        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                else:
                    print("❌ Format: translate <text> to <language>")
            
            elif command.startswith('ask ') and is_overseer:
                question = command[4:].strip()
                if question:
                    result = system.process({"query": question, "task_type": "question_answering"})
                    if result.get('status') == 'completed':
                        print(f"\n💡 Answer: {result.get('response', 'No response')}")
                    else:
                        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
                else:
                    print("❌ Please provide a question")
            
            elif command.startswith('safety ') and is_overseer:
                text = command[7:].strip()
                if text:
                    safety_result = system._perform_safety_scan(text)
                    print(f"\n🛡️  Safety Check:")
                    print(f"  Safe: {safety_result.get('safe', False)}")
                    print(f"  Risk Level: {safety_result.get('risk_level', 'Unknown')}")
                    print(f"  Method: {safety_result.get('method', 'Unknown')}")
                else:
                    print("❌ Please provide text to check")
            
            # Multi-agent system commands
            elif command.startswith('analyze ') and not is_overseer:
                text = command[8:].strip()
                if text:
                    analyze_single_text(system, text)
                else:
                    print("❌ Please provide text to analyze")
            
            elif command.startswith('dataset ') and not is_overseer:
                parts = command[8:].strip().split()
                if parts:
                    dataset_name = parts[0]
                    sample_size = int(parts[1]) if len(parts) > 1 else 20
                    analyze_dataset(system, dataset_name, sample_size)
                else:
                    print("❌ Please provide dataset name")
            
            else:
                print("❓ Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SafetyAlignNLP - Multi-Agent Safety Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --interactive                    # Interactive mode
  python main.py --text "Your text here"         # Analyze single text
  python main.py --dataset beaver_first1000-qa   # Analyze dataset
  python main.py --batch-file texts.json         # Batch process from file
        """
    )
    
    # Mode selection
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--overseer', action='store_true',
                       help='Use OverseerAgent for query routing with safety scanning')
    parser.add_argument('--query', '-q', type=str,
                       help='Process a query through OverseerAgent')
    parser.add_argument('-t', '--text', type=str,
                       help='Analyze a single text string')
    parser.add_argument('-d', '--dataset', type=str,
                       help='Analyze a dataset by name')
    parser.add_argument('-b', '--batch-file', type=str,
                       help='Process texts from JSON file')
    parser.add_argument('-s', '--sample-size', type=int, default=20,
                       help='Sample size for dataset analysis (default: 20)')
    parser.add_argument('--task-type', type=str, 
                       choices=['safety_analysis', 'pattern_analysis', 'comprehensive_analysis'],
                       default='comprehensive_analysis',
                       help='Type of analysis to perform (default: comprehensive_analysis)')
    parser.add_argument('-o', '--output', type=str,
                       help='Output file for results (JSON format)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--no-guardrails', action='store_true',
                       help='Disable NeMo Guardrails')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--test', action='store_true',
                       help='Run comprehensive system tests')
    parser.add_argument('--web', action='store_true',
                       help='Start web interface')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host for web interface (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port for web interface (default: 8000)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Print banner
    print_banner()
    
    try:
        # Load configuration
        if args.config:
            # In a real implementation, you'd load from file
            config = Config()
            logger.info(f"Using configuration from: {args.config}")
        else:
            config = Config()
        
        # Disable guardrails if requested
        if args.no_guardrails:
            config.guardrails.enabled = False
            logger.info("Guardrails disabled by user request")
        
        # Initialize TSDI safety layer
        print("🛡️  Initializing TSDI Safety Layer...")
        safety_layer = get_global_safety_layer()
        print(f"✅ TSDI Safety Layer initialized with threshold {safety_layer.harm_threshold}")
        
        # Initialize system based on mode
        if args.overseer or args.query:
            print("🚀 Initializing OverseerAgent with safety scanning...")
            system = OverseerAgent(config)
            print("✅ OverseerAgent initialized successfully!")
            
            # Check overseer status
            status = system.get_agent_status()
            print(f"📊 Available task agents: {', '.join(status.get('available_task_agents', []))}")
            print(f"🛡️  Safety tools: {status.get('safety_tools_count', 0)}")
        else:
            print("🚀 Initializing multi-agent system...")
            system = MultiAgentSystem(config)
            
            if not system.initialize_agents():
                print("❌ Failed to initialize agents. Check your configuration and API keys.")
                return 1
            
            print("✅ System initialized successfully!")
            
            # Check system status
            status = system.get_system_status()
            print(f"📊 Available datasets: {len(status.get('available_datasets', []))}")
            print(f"🛡️  Guardrails enabled: {status.get('guardrails_enabled', False)}")
        
        print(f"🛡️  TSDI Safety Layer active - All executions will be wrapped with bias reduction and harm scoring")
        
        # Execute based on mode
        result = None
        
        if args.test:
            # Run system tests
            print("🧪 Running comprehensive system tests...")
            try:
                from tests.test_system import run_system_tests
                success = run_system_tests()
                return 0 if success else 1
            except ImportError as e:
                print(f"❌ Failed to import test module: {e}")
                return 1
        
        elif args.web:
            # Start web interface
            print("🌐 Starting web interface...")
            try:
                from app import run_web_app
                run_web_app(host=args.host, port=args.port, reload=False)
                return 0
            except ImportError as e:
                print(f"❌ Failed to import web app: {e}")
                return 1
            except KeyboardInterrupt:
                print("\n⏹️  Web interface stopped by user")
                return 0
        
        elif args.interactive:
            interactive_mode(system)
            return 0
        
        elif args.query:
            if not isinstance(system, OverseerAgent):
                print("❌ Query mode requires --overseer flag")
                return 1
            
            print(f"🔍 Processing query: {args.query}")
            result = system.process(args.query)
            
            if result.get('status') == 'completed':
                print(f"✅ Response: {result.get('response', 'No response')}")
            elif result.get('status') == 'blocked':
                print(f"🚫 Request blocked: {result.get('response', 'Safety violation')}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        elif args.text:
            if isinstance(system, OverseerAgent):
                result = system.process(args.text)
                if result.get('status') == 'completed':
                    print(f"✅ Response: {result.get('response', 'No response')}")
                elif result.get('status') == 'blocked':
                    print(f"🚫 Request blocked: {result.get('response', 'Safety violation')}")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
            else:
                result = analyze_single_text(system, args.text, args.task_type)
        
        elif args.dataset:
            if isinstance(system, OverseerAgent):
                print("❌ Dataset analysis not supported in overseer mode. Use multi-agent mode.")
                return 1
            result = analyze_dataset(system, args.dataset, args.sample_size)
        
        elif args.batch_file:
            if not Path(args.batch_file).exists():
                print(f"❌ Batch file not found: {args.batch_file}")
                return 1
            
            with open(args.batch_file, 'r') as f:
                batch_data = json.load(f)
            
            if isinstance(batch_data, list):
                texts = batch_data
            elif isinstance(batch_data, dict) and 'texts' in batch_data:
                texts = batch_data['texts']
            else:
                print("❌ Invalid batch file format. Expected list of texts or {'texts': [...]}")
                return 1
            
            print(f"📦 Processing {len(texts)} texts from batch file...")
            if isinstance(system, OverseerAgent):
                print("❌ Batch processing not supported in overseer mode. Use multi-agent mode.")
                return 1
            
            # Wrap system with safety layer for batch processing
            wrapped_system = wrap_agent_with_safety(system)
            result = wrapped_system.batch_process_texts(texts, args.task_type)
        
        else:
            print("❓ No mode specified. Use --interactive, --text, --dataset, or --batch-file")
            print("   Run with --help for more information.")
            return 1
        
        # Save results if output file specified
        if result and args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {args.output}")
        
        # Show final statistics
        if isinstance(system, OverseerAgent):
            stats = system.get_safety_statistics()
            print(f"\n📈 OverseerAgent Session Summary:")
            print(f"  Total requests: {stats.get('total_requests', 0)}")
            print(f"  Blocked requests: {stats.get('blocked_requests', 0)}")
            print(f"  Safety rate: {stats.get('safety_rate', 0):.2%}")
        else:
            final_status = system.get_system_status()
            stats = final_status.get('system_stats', {})
            print(f"\n📈 Multi-Agent Session Summary:")
            print(f"  Tasks completed: {stats.get('tasks_completed', 0)}")
            print(f"  Texts processed: {stats.get('total_texts_processed', 0)}")
            print(f"  Errors encountered: {stats.get('errors_encountered', 0)}")
            
            # Cleanup
            system.shutdown()
        
        # Show TSDI safety layer statistics
        tsdi_stats = get_global_safety_statistics()
        print(f"\n🛡️  TSDI Safety Layer Summary:")
        print(f"  Total processed: {tsdi_stats.get('total_processed', 0)}")
        print(f"  Total blocked: {tsdi_stats.get('total_blocked', 0)}")
        print(f"  Safety rate: {tsdi_stats.get('safety_rate', 0):.2%}")
        print(f"  Bias corrections: {tsdi_stats.get('bias_corrections', 0)}")
        print(f"  Harm threshold: {tsdi_stats.get('harm_threshold', 0.2)}")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
