#!/usr/bin/env python3
"""
Simple FastAPI web application for SafetyAlignNLP system without heavy ML dependencies.
Provides a web interface for basic query processing with pattern-based safety checks.
"""

import os
import sys
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple safety checker without ML dependencies
class SimpleSafetyChecker:
    """Simple pattern-based safety checker."""
    
    def __init__(self):
        self.harmful_patterns = [
            r'\b(kill|murder|violence|harm|hurt|attack|destroy|weapon|bomb|shoot|stab)\b',
            r'\b(death|die|dead|corpse|blood|gore|torture|abuse)\b',
            r'\b(hate|discriminat\w+|racist|sexist|homophobic|transphobic)\b',
            r'\b(stupid|idiot|moron|retard|freak|loser|worthless)\b'
        ]
        
        self.bias_patterns = [
            r'\b(he|him|his|man|men|male|boy|boys)\b.*\b(better|superior|stronger)\b',
            r'\b(she|her|hers|woman|women|female|girl|girls)\b.*\b(worse|inferior|weaker)\b'
        ]
        
        self.stats = {
            'total_processed': 0,
            'total_blocked': 0,
            'bias_corrections': 0
        }
    
    def check_safety(self, text: str) -> Dict[str, Any]:
        """Check text safety using pattern matching."""
        self.stats['total_processed'] += 1
        
        # Check for harmful patterns
        harm_score = 0.0
        for pattern in self.harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                harm_score += 0.3
        
        # Check for bias patterns
        bias_detected = False
        for pattern in self.bias_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                bias_detected = True
                self.stats['bias_corrections'] += 1
                break
        
        # Determine if content should be blocked
        blocked = harm_score > 0.2
        if blocked:
            self.stats['total_blocked'] += 1
        
        return {
            'status': 'blocked' if blocked else 'safe',
            'harm_score': min(harm_score, 1.0),
            'bias_detected': bias_detected,
            'message': 'Content blocked due to harmful patterns' if blocked else 'Content is safe'
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get safety statistics."""
        total = self.stats['total_processed']
        safety_rate = (total - self.stats['total_blocked']) / total if total > 0 else 1.0
        
        return {
            'total_processed': total,
            'total_blocked': self.stats['total_blocked'],
            'safety_rate': safety_rate,
            'bias_corrections': self.stats['bias_corrections'],
            'harm_threshold': 0.2
        }

# Global safety checker
safety_checker = SimpleSafetyChecker()

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    task_type: Optional[str] = Field(None)

class QueryResponse(BaseModel):
    status: str
    response: Optional[str] = None
    safety_verified: bool = False
    harm_score: float = 0.0
    processing_time: float = 0.0
    timestamp: str
    message: Optional[str] = None

# Create FastAPI app
app = FastAPI(
    title="SafetyAlignNLP Simple Web Interface",
    description="Simple web interface for safe AI query processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
project_root = Path(__file__).parent
templates_dir = project_root / "templates"
templates_dir.mkdir(exist_ok=True)

# Create simple HTML template
template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SafetyAlignNLP - Simple Web Interface</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        textarea, button { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        textarea { height: 100px; }
        button { background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .response { margin-top: 20px; padding: 15px; border-radius: 5px; }
        .safe { background: #d4edda; border: 1px solid #c3e6cb; }
        .blocked { background: #f8d7da; border: 1px solid #f5c6cb; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-top: 20px; }
        .stat-card { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ SafetyAlignNLP</h1>
        <p>Simple Safe AI Query Processing</p>
    </div>

    <form id="queryForm">
        <div class="form-group">
            <label for="query">Enter your query:</label>
            <textarea id="query" name="query" placeholder="Type your question or request here..." required></textarea>
        </div>
        <button type="submit">Process Query</button>
    </form>

    <div id="response" style="display: none;"></div>

    <div class="stats">
        <div class="stat-card">
            <div id="totalProcessed">0</div>
            <div>Total Processed</div>
        </div>
        <div class="stat-card">
            <div id="totalBlocked">0</div>
            <div>Blocked</div>
        </div>
        <div class="stat-card">
            <div id="safetyRate">100%</div>
            <div>Safety Rate</div>
        </div>
        <div class="stat-card">
            <div id="biasCorrections">0</div>
            <div>Bias Corrections</div>
        </div>
    </div>

    <script>
        document.getElementById('queryForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const query = document.getElementById('query').value;
            const responseDiv = document.getElementById('response');
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const result = await response.json();
                
                responseDiv.style.display = 'block';
                responseDiv.className = 'response ' + (result.status === 'blocked' ? 'blocked' : 'safe');
                responseDiv.innerHTML = `
                    <strong>Status:</strong> ${result.status.toUpperCase()}<br>
                    <strong>Harm Score:</strong> ${result.harm_score.toFixed(3)}<br>
                    <strong>Processing Time:</strong> ${result.processing_time.toFixed(3)}s<br>
                    ${result.response ? '<br><strong>Response:</strong> ' + result.response : ''}
                    <br><strong>Message:</strong> ${result.message}
                `;
                
                updateStats();
                
            } catch (error) {
                responseDiv.style.display = 'block';
                responseDiv.className = 'response blocked';
                responseDiv.innerHTML = '<strong>Error:</strong> ' + error.message;
            }
        });

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
    </script>
</body>
</html>'''

with open(templates_dir / "simple.html", "w") as f:
    f.write(template_content)

templates = Jinja2Templates(directory=templates_dir)

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("simple.html", {"request": request})

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query with simple safety checking."""
    start_time = datetime.now()
    
    try:
        # Check safety
        safety_result = safety_checker.check_safety(request.query)
        
        # Generate simple response if safe
        if safety_result['status'] == 'safe':
            response_text = f"Thank you for your query: '{request.query[:100]}...'. This is a safe response from the simple SafetyAlignNLP system."
        else:
            response_text = None
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            status=safety_result['status'],
            response=response_text,
            safety_verified=safety_result['status'] == 'safe',
            harm_score=safety_result['harm_score'],
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            message=safety_result['message']
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        return QueryResponse(
            status="error",
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            message=f"Processing error: {str(e)}"
        )

@app.get("/api/safety-stats")
async def get_safety_stats():
    """Get current safety statistics."""
    return safety_checker.get_stats()

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    
    print("🌐 Starting Simple SafetyAlignNLP Web Interface")
    print("📍 URL: http://localhost:8000")
    print("🛡️  Pattern-based Safety: Active")
    print("-" * 50)
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
