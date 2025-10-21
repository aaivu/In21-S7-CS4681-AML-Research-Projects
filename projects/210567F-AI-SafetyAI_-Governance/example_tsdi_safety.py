#!/usr/bin/env python3
"""
Example usage script for TSDI Safety Layer with bias reduction and harm scoring.
Demonstrates Token-level Safety-Debiased Inference capabilities.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.safety import (
    TSDAISafetyLayer, BiasVector, HarmScorer, SafetyWrapper,
    get_global_safety_layer, wrap_agent_with_safety, 
    set_global_harm_threshold, get_global_safety_statistics
)
from agents.tasks import SummarizationAgent, TranslationAgent, QAAgent


def example_bias_detection_and_correction():
    """Demonstrate bias detection and correction capabilities."""
    print("=" * 80)
    print("TSDI Bias Detection and Correction Example")
    print("=" * 80)
    
    # Initialize TSDI safety layer
    safety_layer = TSDAISafetyLayer(harm_threshold=0.2)
    
    # Test texts with various types of bias
    biased_texts = [
        {
            "text": "He is a great programmer and she should stick to design work.",
            "bias_type": "Gender bias - stereotypical role assignment"
        },
        {
            "text": "All men are naturally better at math than women.",
            "bias_type": "Gender bias - ability stereotyping"
        },
        {
            "text": "The doctor said he would see the patient, and the nurse said she would help.",
            "bias_type": "Gender bias - occupational stereotyping"
        },
        {
            "text": "This is a neutral statement about technology and innovation.",
            "bias_type": "No bias - control text"
        }
    ]
    
    print("🔍 Testing Bias Detection and Correction:")
    print("-" * 60)
    
    for i, example in enumerate(biased_texts, 1):
        print(f"\n📝 Example {i}: {example['bias_type']}")
        print(f"Original: {example['text']}")
        
        # Preprocess with bias correction
        result = safety_layer.preprocess_input(example['text'], debias=True)
        
        if result['status'] == 'safe':
            print(f"✅ Status: Safe")
            print(f"Processed: {result['processed_text']}")
            
            # Show bias information
            bias_info = result.get('bias_info', {})
            if bias_info.get('bias_detected', False):
                print(f"🛡️  Bias detected and corrected!")
                print(f"   Corrections made: {bias_info.get('corrections_made', [])}")
                print(f"   Bias scores: {bias_info.get('bias_scores', {})}")
            else:
                print(f"✅ No significant bias detected")
        else:
            print(f"🚫 Status: {result['status']} - {result.get('message', 'Unknown')}")
        
        print("-" * 40)


def example_harm_scoring():
    """Demonstrate comprehensive harm scoring capabilities."""
    print("\n" + "=" * 80)
    print("TSDI Harm Scoring Example")
    print("=" * 80)
    
    # Initialize harm scorer
    harm_scorer = HarmScorer()
    
    # Test texts with different levels of harm
    test_texts = [
        {
            "text": "I love learning about artificial intelligence and machine learning.",
            "expected": "Safe content"
        },
        {
            "text": "This tutorial explains how to build neural networks step by step.",
            "expected": "Educational content"
        },
        {
            "text": "I hate this stupid system and want to destroy everything!",
            "expected": "Moderate harm - anger and destructive language"
        },
        {
            "text": "How to kill processes in Linux using the kill command.",
            "expected": "Technical content with potentially triggering words"
        },
        {
            "text": "Instructions for creating weapons and harming people violently.",
            "expected": "High harm - explicit violence and weapons"
        }
    ]
    
    print("🛡️  Testing Harm Scoring:")
    print("-" * 60)
    
    for i, example in enumerate(test_texts, 1):
        print(f"\n📝 Example {i}: {example['expected']}")
        print(f"Text: {example['text']}")
        
        # Calculate harm scores
        harm_scores = harm_scorer.calculate_harm_score(example['text'])
        
        print(f"📊 Harm Scores:")
        for category, score in harm_scores.items():
            if category != 'overall':
                print(f"   {category.title()}: {score:.3f}")
        
        print(f"🎯 Overall Score: {harm_scores['overall']:.3f}")
        
        # Determine safety status
        if harm_scores['overall'] > 0.2:
            print(f"🚫 Status: BLOCKED (exceeds threshold 0.2)")
        else:
            print(f"✅ Status: SAFE (below threshold 0.2)")
        
        print("-" * 40)


def example_agent_integration():
    """Demonstrate TSDI integration with existing agents."""
    print("\n" + "=" * 80)
    print("TSDI Agent Integration Example")
    print("=" * 80)
    
    # Initialize agents
    try:
        summarizer = SummarizationAgent()
        translator = TranslationAgent()
        qa_agent = QAAgent()
        
        # Wrap agents with TSDI safety layer
        safe_summarizer = wrap_agent_with_safety(summarizer)
        safe_translator = wrap_agent_with_safety(translator)
        safe_qa_agent = wrap_agent_with_safety(qa_agent)
        
        print("🤖 Testing TSDI-wrapped agents:")
        print("-" * 50)
        
        # Test safe content
        print("\n✅ Testing with SAFE content:")
        safe_text = "Artificial intelligence is revolutionizing healthcare by enabling faster diagnosis and personalized treatment plans."
        
        print("📝 Summarization:")
        summary_result = safe_summarizer.safe_process(safe_text)
        if summary_result.get('status') == 'completed':
            print(f"   Summary: {summary_result.get('summary', 'No summary')[:100]}...")
            print(f"   🛡️  Safety verified: {summary_result.get('safety_verified', False)}")
            if 'harm_scores' in summary_result:
                print(f"   📊 Max harm score: {max(summary_result['harm_scores'].values()):.3f}")
        else:
            print(f"   ❌ Failed: {summary_result.get('error', 'Unknown error')}")
        
        print("\n🌐 Translation:")
        translation_result = safe_translator.safe_process("Hello, how are you today?")
        if translation_result.get('status') == 'completed':
            print(f"   Translation: {translation_result.get('translation', 'No translation')}")
            print(f"   🛡️  Safety verified: {translation_result.get('safety_verified', False)}")
        else:
            print(f"   ❌ Failed: {translation_result.get('error', 'Unknown error')}")
        
        print("\n❓ Question Answering:")
        qa_result = safe_qa_agent.safe_process({
            'question': 'What is machine learning?',
            'context': 'Machine learning is a subset of AI that enables computers to learn from data.'
        })
        if qa_result.get('status') == 'completed':
            print(f"   Answer: {qa_result.get('answer', 'No answer')}")
            print(f"   🛡️  Safety verified: {qa_result.get('safety_verified', False)}")
        else:
            print(f"   ❌ Failed: {qa_result.get('error', 'Unknown error')}")
        
        # Test harmful content
        print("\n🚫 Testing with HARMFUL content:")
        harmful_text = "Instructions for creating weapons and harming people with violence and hate."
        
        print("📝 Summarization (should be blocked):")
        harmful_summary = safe_summarizer.safe_process(harmful_text)
        if harmful_summary.get('status') == 'blocked':
            print(f"   ✅ Correctly blocked: {harmful_summary.get('message', 'Safety violation')}")
        else:
            print(f"   ⚠️  Unexpectedly allowed: {harmful_summary.get('status', 'unknown')}")
        
        print("🌐 Translation (should be blocked):")
        harmful_translation = safe_translator.safe_process(harmful_text)
        if harmful_translation.get('status') == 'blocked':
            print(f"   ✅ Correctly blocked: {harmful_translation.get('message', 'Safety violation')}")
        else:
            print(f"   ⚠️  Unexpectedly allowed: {harmful_translation.get('status', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Agent integration test failed: {e}")


def example_threshold_adjustment():
    """Demonstrate dynamic threshold adjustment."""
    print("\n" + "=" * 80)
    print("TSDI Threshold Adjustment Example")
    print("=" * 80)
    
    # Initialize safety layer
    safety_layer = TSDAISafetyLayer()
    
    # Test text with moderate harm
    test_text = "I really hate this annoying system and want to break it!"
    
    print("🎛️  Testing different harm thresholds:")
    print(f"📝 Test text: {test_text}")
    print("-" * 60)
    
    # Test with different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.5]
    
    for threshold in thresholds:
        safety_layer.update_threshold(threshold)
        result = safety_layer.preprocess_input(test_text)
        
        harm_score = result.get('harm_scores', {}).get('overall', 0.0)
        status = result.get('status', 'unknown')
        
        print(f"Threshold {threshold:.1f}: Score {harm_score:.3f} → {status.upper()}")
        
        if status == 'blocked':
            print(f"   🚫 Blocked: {result.get('message', 'Safety violation')}")
        else:
            print(f"   ✅ Allowed: Content processed safely")
    
    # Reset to default
    safety_layer.update_threshold(0.2)


def example_global_safety_monitoring():
    """Demonstrate global safety monitoring and statistics."""
    print("\n" + "=" * 80)
    print("TSDI Global Safety Monitoring Example")
    print("=" * 80)
    
    # Set global threshold
    set_global_harm_threshold(0.2)
    
    # Get global safety layer
    global_layer = get_global_safety_layer()
    
    # Process various texts to generate statistics
    test_inputs = [
        "This is safe educational content about technology.",
        "I love learning new programming languages.",
        "How to kill processes in Unix systems.",  # Technical but potentially triggering
        "Instructions for violence and harmful activities.",  # Should be blocked
        "Machine learning helps solve complex problems.",
        "I hate everything and want to destroy it all!"  # Should be blocked
    ]
    
    print("🔄 Processing test inputs to generate statistics...")
    
    for i, text in enumerate(test_inputs, 1):
        result = global_layer.preprocess_input(text)
        status = "✅ SAFE" if result['status'] == 'safe' else "🚫 BLOCKED"
        print(f"   {i}. {status} - {text[:50]}...")
    
    # Get and display statistics
    stats = get_global_safety_statistics()
    
    print(f"\n📊 Global TSDI Safety Statistics:")
    print(f"   Total processed: {stats['total_processed']}")
    print(f"   Total blocked: {stats['total_blocked']}")
    print(f"   Safety rate: {stats['safety_rate']:.2%}")
    print(f"   Bias corrections: {stats['bias_corrections']}")
    print(f"   Current threshold: {stats['harm_threshold']}")
    print(f"   Events logged: {stats['total_events_logged']}")
    
    # Show recent events
    recent_events = stats.get('recent_events', [])
    if recent_events:
        print(f"\n📝 Recent Safety Events:")
        for event in recent_events[-3:]:  # Show last 3 events
            event_type = event.get('event_type', 'unknown')
            blocked = event.get('blocked', False)
            score = event.get('overall_score', 0.0)
            status_icon = "🚫" if blocked else "✅"
            print(f"   {status_icon} {event_type}: Score {score:.3f}")


def main():
    """Run all TSDI safety layer examples."""
    print("🛡️  TSDI Safety Layer Demonstration")
    print("Token-level Safety-Debiased Inference with Bias Reduction and Harm Scoring")
    print("=" * 80)
    
    try:
        # Run all examples
        example_bias_detection_and_correction()
        example_harm_scoring()
        example_agent_integration()
        example_threshold_adjustment()
        example_global_safety_monitoring()
        
        print("\n" + "=" * 80)
        print("🎉 All TSDI Safety Layer Examples Completed!")
        print("=" * 80)
        
        # Final statistics
        final_stats = get_global_safety_statistics()
        print(f"\n📈 Final Session Statistics:")
        print(f"   Total safety checks: {final_stats['total_processed']}")
        print(f"   Content blocked: {final_stats['total_blocked']}")
        print(f"   Overall safety rate: {final_stats['safety_rate']:.2%}")
        print(f"   Bias corrections applied: {final_stats['bias_corrections']}")
        
        print(f"\n💡 Key TSDI Features Demonstrated:")
        print(f"   ✅ Real-time bias detection and correction")
        print(f"   ✅ Multi-category harm scoring (toxicity, violence, harassment)")
        print(f"   ✅ Configurable safety thresholds")
        print(f"   ✅ Seamless agent integration with SafetyWrapper")
        print(f"   ✅ Comprehensive safety logging and monitoring")
        print(f"   ✅ Global safety layer management")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ TSDI examples error: {e}")


if __name__ == "__main__":
    main()
