"""Coordinator agent for orchestrating multi-agent tasks."""

from typing import Any, Dict, List, Optional
import logging
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class CoordinatorAgent(BaseAgent):
    """Agent responsible for coordinating tasks between different agents."""
    
    def __init__(self, **kwargs):
        """Initialize the Coordinator Agent."""
        default_system_prompt = """You are a coordinator agent responsible for orchestrating tasks between different specialized agents.

Your responsibilities include:
1. Analyzing task requirements and determining which agents to involve
2. Coordinating the workflow between safety and analysis agents
3. Synthesizing results from multiple agents into coherent summaries
4. Identifying conflicts or inconsistencies between agent outputs
5. Providing comprehensive final reports and recommendations
6. Managing task prioritization and resource allocation

When coordinating tasks:
- Consider the strengths of each agent type
- Ensure comprehensive coverage of all relevant aspects
- Synthesize findings into actionable insights
- Highlight important patterns or anomalies
- Provide clear recommendations based on combined analysis

Be systematic and thorough in your coordination approach."""
        
        kwargs.setdefault('system_prompt', default_system_prompt)
        kwargs.setdefault('name', 'CoordinatorAgent')
        super().__init__(**kwargs)
        
        self.available_agents = {}
        self.task_history = []
    
    def register_agent(self, agent_name: str, agent_instance):
        """Register an agent for coordination."""
        self.available_agents[agent_name] = agent_instance
        logger.info(f"Registered agent: {agent_name}")
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Coordinate processing across multiple agents."""
        if isinstance(input_data, dict) and 'task_type' in input_data:
            return self._handle_structured_task(input_data)
        elif isinstance(input_data, str):
            # Default to comprehensive analysis
            return self._coordinate_comprehensive_analysis(input_data)
        elif isinstance(input_data, list):
            return self._coordinate_batch_analysis(input_data)
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def _handle_structured_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a structured task with specific requirements."""
        task_type = task_data.get('task_type')
        
        if task_type == 'safety_analysis':
            return self._coordinate_safety_analysis(task_data)
        elif task_type == 'pattern_analysis':
            return self._coordinate_pattern_analysis(task_data)
        elif task_type == 'comprehensive_analysis':
            return self._coordinate_comprehensive_analysis(task_data.get('text', ''))
        elif task_type == 'dataset_evaluation':
            return self._coordinate_dataset_evaluation(task_data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _coordinate_safety_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate safety-focused analysis."""
        text = task_data.get('text', '')
        
        if 'safety_agent' not in self.available_agents:
            return {"error": "Safety agent not available"}
        
        safety_agent = self.available_agents['safety_agent']
        safety_result = safety_agent.process(text)
        
        # Generate coordination summary
        summary_prompt = f"""Based on the safety analysis results below, provide a coordination summary:

Safety Analysis Results:
{safety_result}

Provide:
1. Key safety findings
2. Risk level assessment
3. Recommended actions
4. Priority level for review"""
        
        coordination_summary = self.generate_response(summary_prompt)
        
        result = {
            "coordinator": self.name,
            "task_type": "safety_analysis",
            "safety_results": safety_result,
            "coordination_summary": coordination_summary,
            "timestamp": self._get_timestamp(),
            "agents_involved": ["safety_agent"]
        }
        
        self._log_task(result)
        return result
    
    def _coordinate_pattern_analysis(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate pattern analysis tasks."""
        text = task_data.get('text', '')
        
        if 'analysis_agent' not in self.available_agents:
            return {"error": "Analysis agent not available"}
        
        analysis_agent = self.available_agents['analysis_agent']
        analysis_result = analysis_agent.process(text)
        
        # Generate coordination summary
        summary_prompt = f"""Based on the pattern analysis results below, provide a coordination summary:

Pattern Analysis Results:
{analysis_result}

Provide:
1. Key patterns identified
2. Significance of findings
3. Implications for content understanding
4. Recommendations for further analysis"""
        
        coordination_summary = self.generate_response(summary_prompt)
        
        result = {
            "coordinator": self.name,
            "task_type": "pattern_analysis",
            "analysis_results": analysis_result,
            "coordination_summary": coordination_summary,
            "timestamp": self._get_timestamp(),
            "agents_involved": ["analysis_agent"]
        }
        
        self._log_task(result)
        return result
    
    def _coordinate_comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Coordinate comprehensive analysis using both safety and analysis agents."""
        results = {}
        agents_used = []
        
        # Run safety analysis if available
        if 'safety_agent' in self.available_agents:
            safety_agent = self.available_agents['safety_agent']
            results['safety_analysis'] = safety_agent.process(text)
            agents_used.append('safety_agent')
        
        # Run pattern analysis if available
        if 'analysis_agent' in self.available_agents:
            analysis_agent = self.available_agents['analysis_agent']
            results['pattern_analysis'] = analysis_agent.process(text)
            agents_used.append('analysis_agent')
        
        if not results:
            return {"error": "No agents available for analysis"}
        
        # Synthesize results
        synthesis_prompt = f"""Synthesize the following analysis results into a comprehensive report:

Results from multiple agents:
{results}

Provide a comprehensive synthesis including:
1. Executive Summary
2. Key Findings from Safety Analysis
3. Key Findings from Pattern Analysis
4. Cross-Analysis Insights
5. Overall Risk Assessment
6. Recommendations
7. Areas for Further Investigation"""
        
        synthesis = self.generate_response(synthesis_prompt)
        
        result = {
            "coordinator": self.name,
            "task_type": "comprehensive_analysis",
            "individual_results": results,
            "synthesis": synthesis,
            "timestamp": self._get_timestamp(),
            "agents_involved": agents_used
        }
        
        self._log_task(result)
        return result
    
    def _coordinate_batch_analysis(self, texts: List[str]) -> Dict[str, Any]:
        """Coordinate batch analysis across multiple texts."""
        batch_results = []
        
        for i, text in enumerate(texts[:10]):  # Limit to 10 for performance
            logger.info(f"Coordinating analysis for text {i+1}/{min(len(texts), 10)}")
            result = self._coordinate_comprehensive_analysis(text)
            batch_results.append(result)
        
        # Generate batch summary
        summary_prompt = f"""Analyze the batch results from {len(batch_results)} texts and provide:

1. Overall patterns across the batch
2. Common safety concerns
3. Recurring analysis themes
4. Aggregate risk assessment
5. Batch-level recommendations

Batch Results Summary:
{[r.get('synthesis', 'No synthesis available')[:200] + '...' for r in batch_results]}"""
        
        batch_summary = self.generate_response(summary_prompt)
        
        result = {
            "coordinator": self.name,
            "task_type": "batch_analysis",
            "batch_size": len(texts),
            "processed_count": len(batch_results),
            "individual_results": batch_results,
            "batch_summary": batch_summary,
            "timestamp": self._get_timestamp()
        }
        
        self._log_task(result)
        return result
    
    def _coordinate_dataset_evaluation(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate evaluation of a specific dataset."""
        dataset_name = task_data.get('dataset_name')
        sample_size = task_data.get('sample_size', 20)
        
        if not dataset_name:
            return {"error": "Dataset name required for evaluation"}
        
        results = {}
        agents_used = []
        
        # Run safety evaluation if available
        if 'safety_agent' in self.available_agents:
            safety_agent = self.available_agents['safety_agent']
            results['safety_evaluation'] = safety_agent.evaluate_dataset(dataset_name, sample_size)
            agents_used.append('safety_agent')
        
        # Run pattern analysis if available
        if 'analysis_agent' in self.available_agents:
            analysis_agent = self.available_agents['analysis_agent']
            results['pattern_evaluation'] = analysis_agent.analyze_patterns_across_dataset(dataset_name, sample_size)
            agents_used.append('analysis_agent')
        
        if not results:
            return {"error": "No agents available for dataset evaluation"}
        
        # Generate dataset evaluation summary
        summary_prompt = f"""Provide a comprehensive dataset evaluation report based on the following results:

Dataset: {dataset_name}
Sample Size: {sample_size}

Agent Results:
{results}

Include:
1. Dataset Overview
2. Safety Assessment Summary
3. Pattern Analysis Summary
4. Key Risks Identified
5. Dataset Quality Assessment
6. Recommendations for Use
7. Mitigation Strategies"""
        
        evaluation_summary = self.generate_response(summary_prompt)
        
        result = {
            "coordinator": self.name,
            "task_type": "dataset_evaluation",
            "dataset_name": dataset_name,
            "sample_size": sample_size,
            "evaluation_results": results,
            "evaluation_summary": evaluation_summary,
            "timestamp": self._get_timestamp(),
            "agents_involved": agents_used
        }
        
        self._log_task(result)
        return result
    
    def _log_task(self, task_result: Dict[str, Any]):
        """Log completed task for history tracking."""
        self.task_history.append({
            "task_id": len(self.task_history) + 1,
            "task_type": task_result.get("task_type"),
            "timestamp": task_result.get("timestamp"),
            "agents_involved": task_result.get("agents_involved", []),
            "status": "completed"
        })
        
        # Keep only last 50 tasks
        if len(self.task_history) > 50:
            self.task_history = self.task_history[-50:]
    
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get the task history."""
        return self.task_history.copy()
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents."""
        return list(self.available_agents.keys())
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        agent_status = {}
        for name, agent in self.available_agents.items():
            if hasattr(agent, 'get_info'):
                agent_status[name] = agent.get_info()
            else:
                agent_status[name] = {"status": "available"}
        
        return {
            "coordinator": self.name,
            "available_agents": list(self.available_agents.keys()),
            "total_tasks_completed": len(self.task_history),
            "agent_status": agent_status,
            "timestamp": self._get_timestamp()
        }
