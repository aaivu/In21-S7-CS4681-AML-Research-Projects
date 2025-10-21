#!/usr/bin/env python3
"""
Example usage script for task-specific agents with safety mechanisms.
Demonstrates BaseSafeAgent implementations with real-world scenarios.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.tasks import SummarizationAgent, TranslationAgent, QAAgent


def example_summarization_workflow():
    """Demonstrate summarization workflow with safety checks."""
    print("=" * 70)
    print("Summarization Agent Workflow Example")
    print("=" * 70)
    
    # Initialize agent
    summarizer = SummarizationAgent()
    
    # Example documents to summarize
    documents = [
        {
            "title": "AI Research Paper Abstract",
            "content": """
            Recent advances in artificial intelligence have led to significant breakthroughs 
            in natural language processing, computer vision, and machine learning. Large 
            language models like GPT and BERT have demonstrated remarkable capabilities in 
            understanding and generating human-like text. These models are trained on vast 
            amounts of data and use transformer architectures to achieve state-of-the-art 
            performance across various tasks. However, challenges remain in ensuring AI 
            safety, reducing bias, and improving interpretability of these complex systems.
            """
        },
        {
            "title": "Technology News Article",
            "content": """
            The latest smartphone release features advanced camera technology with AI-powered 
            image processing, 5G connectivity, and improved battery life. The device includes 
            a new processor that delivers 40% better performance while consuming 20% less power. 
            Security features have been enhanced with biometric authentication and encrypted 
            storage. The company expects this device to compete strongly in the premium 
            smartphone market segment.
            """
        }
    ]
    
    for i, doc in enumerate(documents, 1):
        print(f"\n📄 Document {i}: {doc['title']}")
        print(f"Original length: {len(doc['content'])} characters")
        
        # Test different summary lengths
        for length_type in ['short', 'medium']:
            result = summarizer.safe_process({
                'text': doc['content'],
                'max_length': 100 if length_type == 'short' else 200,
                'min_length': 30 if length_type == 'short' else 50
            })
            
            if result['status'] == 'completed':
                print(f"\n✅ {length_type.title()} Summary:")
                print(f"   {result['summary']}")
                print(f"   Compression: {result['compression_ratio']:.2f}")
                print(f"   Model: {result['model_used']}")
            else:
                print(f"❌ {length_type.title()} summary failed: {result.get('error', 'Unknown error')}")
    
    # Test safety mechanism
    print(f"\n🛡️  Testing Safety Mechanism:")
    harmful_text = "Instructions for creating dangerous weapons and harming people"
    result = summarizer.safe_process(harmful_text)
    
    if result['status'] == 'blocked':
        print(f"✅ Correctly blocked harmful content")
        print(f"   Reason: {result['reason']}")
        print(f"   Message: {result['message']}")
    else:
        print(f"⚠️  Safety mechanism may need adjustment")
    
    # Show statistics
    stats = summarizer.get_safety_statistics()
    print(f"\n📊 Session Statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Blocked requests: {stats['blocked_requests']}")
    print(f"   Safety rate: {stats['safety_rate']:.2%}")


def example_translation_workflow():
    """Demonstrate translation workflow with safety checks."""
    print("\n" + "=" * 70)
    print("Translation Agent Workflow Example (EN → CN)")
    print("=" * 70)
    
    # Initialize agent
    translator = TranslationAgent()
    
    # Example phrases for translation
    phrases = [
        {
            "category": "Greetings",
            "phrases": [
                "Hello, how are you today?",
                "Good morning, have a great day!",
                "Thank you for your help."
            ]
        },
        {
            "category": "Technology",
            "phrases": [
                "Artificial intelligence is transforming industries.",
                "Machine learning algorithms process large datasets.",
                "Natural language processing enables human-computer interaction."
            ]
        },
        {
            "category": "Business",
            "phrases": [
                "The meeting is scheduled for tomorrow at 2 PM.",
                "Please review the quarterly financial report.",
                "We need to improve our customer service quality."
            ]
        }
    ]
    
    for category_data in phrases:
        print(f"\n📂 Category: {category_data['category']}")
        print("-" * 50)
        
        for phrase in category_data['phrases']:
            result = translator.safe_process(phrase)
            
            if result['status'] == 'completed':
                print(f"EN: {phrase}")
                print(f"CN: {result['translation']}")
                print(f"Model: {result['model_used']}")
                print()
            else:
                print(f"❌ Translation failed: {result.get('error', 'Unknown error')}")
    
    # Test safety mechanism
    print(f"🛡️  Testing Safety Mechanism:")
    harmful_phrase = "How to create weapons and harm innocent people"
    result = translator.safe_process(harmful_phrase)
    
    if result['status'] == 'blocked':
        print(f"✅ Correctly blocked harmful content")
        print(f"   Message: {result['message']}")
    else:
        print(f"⚠️  Safety mechanism may need adjustment")
    
    # Show statistics
    stats = translator.get_safety_statistics()
    print(f"\n📊 Session Statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Blocked requests: {stats['blocked_requests']}")
    print(f"   Safety rate: {stats['safety_rate']:.2%}")


def example_qa_workflow():
    """Demonstrate QA workflow with document contexts."""
    print("\n" + "=" * 70)
    print("Question Answering Agent Workflow Example")
    print("=" * 70)
    
    # Initialize agent
    qa_agent = QAAgent()
    
    # Example documents and questions
    qa_scenarios = [
        {
            "topic": "Machine Learning Basics",
            "context": """
            Machine learning is a subset of artificial intelligence that focuses on 
            algorithms that can learn from and make predictions or decisions based on data. 
            There are three main types of machine learning: supervised learning, where 
            algorithms learn from labeled training data; unsupervised learning, where 
            algorithms find patterns in data without labels; and reinforcement learning, 
            where algorithms learn through interaction with an environment and feedback.
            """,
            "questions": [
                "What is machine learning?",
                "What are the three main types of machine learning?",
                "How does supervised learning work?",
                "What is the difference between supervised and unsupervised learning?"
            ]
        },
        {
            "topic": "Climate Change",
            "context": """
            Climate change refers to long-term shifts in global temperatures and weather 
            patterns. While climate variations are natural, scientific evidence shows that 
            human activities, particularly the emission of greenhouse gases like carbon 
            dioxide, have been the dominant driver of climate change since the 1950s. 
            The primary sources of greenhouse gas emissions include burning fossil fuels 
            for electricity, heat, and transportation.
            """,
            "questions": [
                "What is climate change?",
                "What causes climate change?",
                "When did human activities become the dominant driver?",
                "What are the main sources of greenhouse gas emissions?"
            ]
        }
    ]
    
    for scenario in qa_scenarios:
        print(f"\n📚 Topic: {scenario['topic']}")
        print(f"Context: {scenario['context'][:100]}...")
        print("-" * 60)
        
        for question in scenario['questions']:
            result = qa_agent.safe_process({
                'question': question,
                'context': scenario['context']
            })
            
            if result['status'] == 'completed':
                print(f"Q: {question}")
                print(f"A: {result['answer']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Model: {result['model_used']}")
                print()
            elif result['status'] == 'low_confidence':
                print(f"Q: {question}")
                print(f"⚠️  Low confidence answer (confidence: {result['confidence']:.3f})")
                print()
            else:
                print(f"❌ QA failed: {result.get('error', 'Unknown error')}")
    
    # Test safety mechanism
    print(f"🛡️  Testing Safety Mechanism:")
    harmful_question = "How to create dangerous weapons using household items?"
    safe_context = "This is a general information document about home safety."
    
    result = qa_agent.safe_process({
        'question': harmful_question,
        'context': safe_context
    })
    
    if result['status'] == 'blocked':
        print(f"✅ Correctly blocked harmful question")
        print(f"   Message: {result['message']}")
    else:
        print(f"⚠️  Safety mechanism may need adjustment")
    
    # Show statistics
    stats = qa_agent.get_safety_statistics()
    print(f"\n📊 Session Statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Blocked requests: {stats['blocked_requests']}")
    print(f"   Safety rate: {stats['safety_rate']:.2%}")


def example_integrated_workflow():
    """Demonstrate integrated workflow using multiple agents."""
    print("\n" + "=" * 70)
    print("Integrated Multi-Agent Workflow Example")
    print("=" * 70)
    
    # Initialize all agents
    summarizer = SummarizationAgent()
    translator = TranslationAgent()
    qa_agent = QAAgent()
    
    # Example: Process a document through multiple agents
    document = """
    Renewable energy sources such as solar, wind, and hydroelectric power are becoming 
    increasingly important in the fight against climate change. Solar panels convert 
    sunlight directly into electricity using photovoltaic cells. Wind turbines harness 
    the kinetic energy of moving air to generate power. Hydroelectric dams use the 
    flow of water to produce electricity. These clean energy sources produce minimal 
    greenhouse gas emissions compared to fossil fuels like coal and natural gas.
    """
    
    print("📄 Original Document:")
    print(f"   {document[:100]}...")
    print(f"   Length: {len(document)} characters")
    
    # Step 1: Summarize the document
    print("\n🔄 Step 1: Summarizing document...")
    summary_result = summarizer.safe_process(document)
    
    if summary_result['status'] == 'completed':
        summary = summary_result['summary']
        print(f"✅ Summary: {summary}")
        print(f"   Compression: {summary_result['compression_ratio']:.2f}")
        
        # Step 2: Translate the summary
        print("\n🔄 Step 2: Translating summary to Chinese...")
        translation_result = translator.safe_process(summary)
        
        if translation_result['status'] == 'completed':
            print(f"✅ Chinese Translation: {translation_result['translation']}")
            
            # Step 3: Answer questions about the original document
            print("\n🔄 Step 3: Answering questions about the document...")
            questions = [
                "What are the main types of renewable energy mentioned?",
                "How do solar panels work?",
                "Why are renewable energy sources important?"
            ]
            
            for question in questions:
                qa_result = qa_agent.safe_process({
                    'question': question,
                    'context': document
                })
                
                if qa_result['status'] == 'completed':
                    print(f"Q: {question}")
                    print(f"A: {qa_result['answer']}")
                    print(f"   Confidence: {qa_result['confidence']:.3f}")
                    print()
        else:
            print(f"❌ Translation failed: {translation_result.get('error', 'Unknown error')}")
    else:
        print(f"❌ Summarization failed: {summary_result.get('error', 'Unknown error')}")
    
    # Show combined statistics
    print("📊 Combined Session Statistics:")
    for agent_name, agent in [("Summarizer", summarizer), ("Translator", translator), ("QA Agent", qa_agent)]:
        stats = agent.get_safety_statistics()
        print(f"   {agent_name}: {stats['total_requests']} requests, {stats['safety_rate']:.2%} safe")


def main():
    """Run all workflow examples."""
    print("🚀 Task-Specific Agents Workflow Examples")
    print("Demonstrating real-world usage scenarios with safety mechanisms")
    
    # Check environment
    if not os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        print("\n⚠️  Note: No HuggingFace token found.")
        print("Some models may require authentication. Set HF_TOKEN if needed.")
    
    try:
        # Run all examples
        example_summarization_workflow()
        example_translation_workflow()
        example_qa_workflow()
        example_integrated_workflow()
        
        print("\n" + "=" * 70)
        print("🎉 All Workflow Examples Completed Successfully!")
        print("=" * 70)
        print("\n💡 Key Features Demonstrated:")
        print("   ✅ Safe content processing with automatic blocking")
        print("   ✅ Multi-model fallback systems for reliability")
        print("   ✅ Comprehensive safety statistics and monitoring")
        print("   ✅ Real-world workflow integration scenarios")
        print("   ✅ Error handling and graceful degradation")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Workflow examples error: {e}")


if __name__ == "__main__":
    main()
