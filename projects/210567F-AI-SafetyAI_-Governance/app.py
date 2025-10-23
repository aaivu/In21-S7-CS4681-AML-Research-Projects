#!/usr/bin/env python3
"""
FastAPI web application for SafetyAlignNLP system.
Provides a web interface for safe query processing with TSDI safety layer.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import system components
from utils.config import Config
from utils.safety import (
    get_global_safety_layer, wrap_agent_with_safety,
    set_global_harm_threshold, get_global_safety_statistics
)
from agents.overseer import OverseerAgent
from agents.tasks import SummarizationAgent, TranslationAgent, QAAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for system components
app_state = {
    'overseer_agent': None,
    'task_agents': {},
    'safety_layer': None,
    'config': None,
    'startup_time': None
}


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    """Request model for query processing."""
    query: str = Field(..., min_length=1, max_length=5000, description="User query to process")
    task_type: Optional[str] = Field(None, description="Specific task type (summarization, translation, qa)")
    safety_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Custom safety threshold")


class QueryResponse(BaseModel):
    """Response model for query processing."""
    status: str = Field(..., description="Processing status (completed, blocked, error)")
    response: Optional[str] = Field(None, description="Processed response")
    task_type: Optional[str] = Field(None, description="Task type used")
    safety_verified: bool = Field(False, description="Whether content passed safety checks")
    harm_scores: Optional[Dict[str, float]] = Field(None, description="Harm scores by category")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Response timestamp")
    message: Optional[str] = Field(None, description="Additional message or error details")


class SafetyStatsResponse(BaseModel):
    """Response model for safety statistics."""
    total_processed: int
    total_blocked: int
    safety_rate: float
    bias_corrections: int
    harm_threshold: float
    recent_events: List[Dict[str, Any]]


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    status: str
    uptime: str
    agents_available: List[str]
    safety_layer_active: bool
    total_requests: int
    system_load: str


# Startup and shutdown handlers
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("🚀 Starting SafetyAlignNLP Web Application")
    
    try:
        # Initialize configuration
        app_state['config'] = Config()
        app_state['startup_time'] = datetime.now()
        
        # Initialize TSDI safety layer
        logger.info("🛡️  Initializing TSDI Safety Layer...")
        app_state['safety_layer'] = get_global_safety_layer()
        set_global_harm_threshold(0.2)
        logger.info("✅ TSDI Safety Layer initialized")
        
        # Initialize OverseerAgent
        logger.info("🤖 Initializing OverseerAgent...")
        app_state['overseer_agent'] = OverseerAgent(app_state['config'])
        logger.info("✅ OverseerAgent initialized")
        
        # Initialize task-specific agents
        logger.info("🔧 Initializing task-specific agents...")
        try:
            app_state['task_agents'] = {
                'summarizer': wrap_agent_with_safety(SummarizationAgent()),
                'translator': wrap_agent_with_safety(TranslationAgent()),
                'qa_agent': wrap_agent_with_safety(QAAgent())
            }
            logger.info("✅ Task-specific agents initialized")
        except Exception as e:
            logger.warning(f"⚠️  Some task agents failed to initialize: {e}")
            app_state['task_agents'] = {}
        
        logger.info("🎉 SafetyAlignNLP Web Application started successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SafetyAlignNLP Web Application")


# Create FastAPI app
app = FastAPI(
    title="SafetyAlignNLP Web Interface",
    description="Web interface for safe AI query processing with TSDI safety layer",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates
static_dir = project_root / "static"
templates_dir = project_root / "templates"

# Create directories if they don't exist
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


# API Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "SafetyAlignNLP - Safe AI Query Processing"
    })


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a user query through the SafetyAlignNLP system."""
    start_time = datetime.now()
    
    try:
        # Set custom safety threshold if provided
        if request.safety_threshold is not None:
            set_global_harm_threshold(request.safety_threshold)
        
        # Determine processing method
        if request.task_type and request.task_type in app_state['task_agents']:
            # Use specific task agent
            agent = app_state['task_agents'][request.task_type]
            
            if request.task_type == 'qa_agent':
                # QA agent needs special handling
                result = agent.safe_process({
                    'question': request.query,
                    'context': 'Please provide an answer based on your knowledge.'
                })
            else:
                result = agent.safe_process(request.query)
            
            task_type = request.task_type
            
        else:
            # Use OverseerAgent for general processing
            if app_state['overseer_agent']:
                result = app_state['overseer_agent'].process(request.query)
                task_type = result.get('task_type', 'general')
            else:
                raise HTTPException(status_code=503, detail="OverseerAgent not available")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = QueryResponse(
            status=result.get('status', 'completed'),
            response=result.get('response', result.get('summary', result.get('translation', result.get('answer', 'No response')))),
            task_type=task_type,
            safety_verified=result.get('safety_verified', False),
            harm_scores=result.get('harm_scores'),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            message=result.get('message')
        )
        
        # Log request in background
        background_tasks.add_task(log_request, request.query, response.status, processing_time)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            status="error",
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            message=f"Processing error: {str(e)}"
        )


@app.get("/api/safety-stats", response_model=SafetyStatsResponse)
async def get_safety_stats():
    """Get current safety statistics."""
    try:
        stats = get_global_safety_statistics()
        
        return SafetyStatsResponse(
            total_processed=stats.get('total_processed', 0),
            total_blocked=stats.get('total_blocked', 0),
            safety_rate=stats.get('safety_rate', 0.0),
            bias_corrections=stats.get('bias_corrections', 0),
            harm_threshold=stats.get('harm_threshold', 0.2),
            recent_events=stats.get('recent_events', [])
        )
        
    except Exception as e:
        logger.error(f"Error getting safety stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve safety statistics")


@app.get("/api/system-status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get current system status."""
    try:
        uptime = datetime.now() - app_state['startup_time'] if app_state['startup_time'] else None
        uptime_str = str(uptime).split('.')[0] if uptime else "Unknown"
        
        # Get available agents
        agents_available = []
        if app_state['overseer_agent']:
            agents_available.append("OverseerAgent")
        agents_available.extend(app_state['task_agents'].keys())
        
        # Get safety stats for total requests
        safety_stats = get_global_safety_statistics()
        
        return SystemStatusResponse(
            status="running",
            uptime=uptime_str,
            agents_available=agents_available,
            safety_layer_active=app_state['safety_layer'] is not None,
            total_requests=safety_stats.get('total_processed', 0),
            system_load="normal"  # Could be enhanced with actual system metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system status")


@app.post("/api/safety-threshold")
async def update_safety_threshold(threshold: float = Field(..., ge=0.0, le=1.0)):
    """Update the global safety threshold."""
    try:
        set_global_harm_threshold(threshold)
        return {"message": f"Safety threshold updated to {threshold}", "threshold": threshold}
    except Exception as e:
        logger.error(f"Error updating safety threshold: {e}")
        raise HTTPException(status_code=500, detail="Failed to update safety threshold")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# Background task functions
async def log_request(query: str, status: str, processing_time: float):
    """Log request details for monitoring."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query_preview": query[:100] + "..." if len(query) > 100 else query,
        "status": status,
        "processing_time": processing_time
    }
    
    # In production, this could write to a database or monitoring system
    logger.info(f"Request logged: {log_entry}")


# Create HTML template if it doesn't exist
def create_html_template():
    """Create a basic HTML template for the web interface."""
    template_path = templates_dir / "index.html"
    
    if not template_path.exists():
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
        }
        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .header p {
            color: #7f8c8d;
            font-size: 16px;
        }
        .query-section {
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #34495e;
        }
        textarea, select, input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 14px;
            font-family: inherit;
            box-sizing: border-box;
        }
        textarea {
            height: 120px;
            resize: vertical;
        }
        button {
            background: #3498db;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #2980b9;
        }
        button:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }
        .response-section {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
        }
        .response-box {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
            margin-top: 15px;
        }
        .response-box.blocked {
            border-left-color: #e74c3c;
            background: #fdf2f2;
        }
        .response-box.error {
            border-left-color: #f39c12;
            background: #fef9e7;
        }
        .stats-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .stat-card {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 14px;
        }
        .loading {
            display: none;
            text-align: center;
            color: #7f8c8d;
        }
        .safety-indicator {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }
        .safety-safe {
            background: #d4edda;
            color: #155724;
        }
        .safety-blocked {
            background: #f8d7da;
            color: #721c24;
        }
        .safety-error {
            background: #fff3cd;
            color: #856404;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ SafetyAlignNLP</h1>
            <p>Safe AI Query Processing with TSDI Safety Layer</p>
        </div>

        <div class="query-section">
            <form id="queryForm">
                <div class="form-group">
                    <label for="query">Enter your query:</label>
                    <textarea id="query" name="query" placeholder="Type your question or request here..." required></textarea>
                </div>
                
                <div class="form-group">
                    <label for="taskType">Task Type (optional):</label>
                    <select id="taskType" name="taskType">
                        <option value="">Auto-detect</option>
                        <option value="summarizer">Summarization</option>
                        <option value="translator">Translation</option>
                        <option value="qa_agent">Question Answering</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="safetyThreshold">Safety Threshold:</label>
                    <input type="range" id="safetyThreshold" name="safetyThreshold" min="0" max="1" step="0.1" value="0.2">
                    <span id="thresholdValue">0.2</span>
                </div>
                
                <button type="submit">Process Query</button>
            </form>
            
            <div class="loading" id="loading">
                <p>🔄 Processing your query safely...</p>
            </div>
        </div>

        <div class="response-section" id="responseSection" style="display: none;">
            <h3>Response</h3>
            <div id="responseBox" class="response-box"></div>
        </div>

        <div class="stats-section" id="statsSection">
            <div class="stat-card">
                <div class="stat-value" id="totalProcessed">0</div>
                <div class="stat-label">Total Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="totalBlocked">0</div>
                <div class="stat-label">Blocked</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="safetyRate">100%</div>
                <div class="stat-label">Safety Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="biasCorrections">0</div>
                <div class="stat-label">Bias Corrections</div>
            </div>
        </div>
    </div>

    <script>
        // Update threshold display
        document.getElementById('safetyThreshold').addEventListener('input', function() {
            document.getElementById('thresholdValue').textContent = this.value;
        });

        // Handle form submission
        document.getElementById('queryForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const query = document.getElementById('query').value;
            const taskType = document.getElementById('taskType').value;
            const safetyThreshold = parseFloat(document.getElementById('safetyThreshold').value);
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('responseSection').style.display = 'none';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        task_type: taskType || null,
                        safety_threshold: safetyThreshold
                    })
                });
                
                const result = await response.json();
                
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                
                // Show response
                displayResponse(result);
                
                // Update stats
                updateStats();
                
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                displayError('Network error: ' + error.message);
            }
        });

        function displayResponse(result) {
            const responseSection = document.getElementById('responseSection');
            const responseBox = document.getElementById('responseBox');
            
            responseSection.style.display = 'block';
            
            // Set response class based on status
            responseBox.className = 'response-box';
            if (result.status === 'blocked') {
                responseBox.classList.add('blocked');
            } else if (result.status === 'error') {
                responseBox.classList.add('error');
            }
            
            // Create safety indicator
            let safetyIndicator = '';
            if (result.status === 'completed') {
                safetyIndicator = '<span class="safety-indicator safety-safe">✅ SAFE</span>';
            } else if (result.status === 'blocked') {
                safetyIndicator = '<span class="safety-indicator safety-blocked">🚫 BLOCKED</span>';
            } else if (result.status === 'error') {
                safetyIndicator = '<span class="safety-indicator safety-error">⚠️ ERROR</span>';
            }
            
            // Display response
            let content = `<strong>Status:</strong> ${result.status.toUpperCase()} ${safetyIndicator}<br>`;
            content += `<strong>Processing Time:</strong> ${result.processing_time.toFixed(3)}s<br>`;
            
            if (result.task_type) {
                content += `<strong>Task Type:</strong> ${result.task_type}<br>`;
            }
            
            if (result.response) {
                content += `<br><strong>Response:</strong><br>${result.response}`;
            }
            
            if (result.message) {
                content += `<br><br><strong>Message:</strong> ${result.message}`;
            }
            
            if (result.harm_scores) {
                content += '<br><br><strong>Harm Scores:</strong><br>';
                for (const [category, score] of Object.entries(result.harm_scores)) {
                    content += `${category}: ${score.toFixed(3)}<br>`;
                }
            }
            
            responseBox.innerHTML = content;
        }

        function displayError(message) {
            const responseSection = document.getElementById('responseSection');
            const responseBox = document.getElementById('responseBox');
            
            responseSection.style.display = 'block';
            responseBox.className = 'response-box error';
            responseBox.innerHTML = `<strong>Error:</strong> ${message}`;
        }

        async function updateStats() {
            try {
                const response = await fetch('/api/safety-stats');
                const stats = await response.json();
                
                document.getElementById('totalProcessed').textContent = stats.total_processed;
                document.getElementById('totalBlocked').textContent = stats.total_blocked;
                document.getElementById('safetyRate').textContent = (stats.safety_rate * 100).toFixed(1) + '%';
                document.getElementById('biasCorrections').textContent = stats.bias_corrections;
                
            } catch (error) {
                console.error('Failed to update stats:', error);
            }
        }

        // Update stats on page load
        updateStats();
        
        // Update stats every 30 seconds
        setInterval(updateStats, 30000);
    </script>
</body>
</html>'''
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Created HTML template at {template_path}")


# Initialize template on import
create_html_template()


# CLI function for running the web app
def run_web_app(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the FastAPI web application."""
    import uvicorn
    
    print("🌐 Starting SafetyAlignNLP Web Interface")
    print(f"📍 URL: http://{host}:{port}")
    print("🛡️  TSDI Safety Layer: Active")
    print("🤖 Agents: OverseerAgent + Task-specific agents")
    print("-" * 50)
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SafetyAlignNLP Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    run_web_app(host=args.host, port=args.port, reload=args.reload)
