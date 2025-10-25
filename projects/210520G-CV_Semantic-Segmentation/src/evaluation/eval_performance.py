"""
Performance Evaluation Metrics
Measures: FPS, Processing Time, Memory Usage
"""

import time
import psutil
import os
import json
from datetime import datetime

class PerformanceEvaluator:
    def __init__(self, tracker_name):
        self.tracker_name = tracker_name
        self.frame_times = []
        self.memory_usage = []
        self.start_time = None
        self.end_time = None
        self.total_frames = 0
        self.process = psutil.Process(os.getpid())
        
    def start_tracking(self):
        """Call this when tracking starts"""
        self.start_time = time.time()
        self.frame_times = []
        self.memory_usage = []
        self.total_frames = 0
        print(f"[{self.tracker_name}] Performance tracking started")
        
    def start_frame(self):
        """Call this at the beginning of each frame processing"""
        return time.time()
    
    def end_frame(self, frame_start_time):
        """Call this at the end of each frame processing"""
        frame_time = time.time() - frame_start_time
        self.frame_times.append(frame_time)
        
        # Measure memory usage
        mem_info = self.process.memory_info()
        memory_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
        self.memory_usage.append(memory_mb)
        
        self.total_frames += 1
    
    def end_tracking(self):
        """Call this when tracking ends"""
        self.end_time = time.time()
        print(f"[{self.tracker_name}] Performance tracking ended")
    
    def get_results(self):
        """Calculate and return performance metrics"""
        if not self.frame_times:
            return None
        
        total_time = self.end_time - self.start_time
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # Calculate statistics
        min_frame_time = min(self.frame_times)
        max_frame_time = max(self.frame_times)
        
        avg_memory = sum(self.memory_usage) / len(self.memory_usage)
        max_memory = max(self.memory_usage)
        
        results = {
            "tracker_name": self.tracker_name,
            "timestamp": datetime.now().isoformat(),
            "total_frames": self.total_frames,
            "total_time_seconds": round(total_time, 2),
            "average_fps": round(fps, 2),
            "average_frame_time_ms": round(avg_frame_time * 1000, 2),
            "min_frame_time_ms": round(min_frame_time * 1000, 2),
            "max_frame_time_ms": round(max_frame_time * 1000, 2),
            "average_memory_mb": round(avg_memory, 2),
            "peak_memory_mb": round(max_memory, 2),
            "frames_per_second_realtime": round(self.total_frames / total_time, 2)
        }
        
        return results
    
    def print_results(self):
        """Print performance results in a readable format"""
        results = self.get_results()
        if not results:
            print("No performance data available")
            return
        
        print("\n" + "="*60)
        print(f"PERFORMANCE METRICS - {results['tracker_name']}")
        print("="*60)
        print(f"Total Frames Processed: {results['total_frames']}")
        print(f"Total Processing Time: {results['total_time_seconds']:.2f} seconds")
        print(f"\nSpeed Metrics:")
        print(f"  Average FPS: {results['average_fps']:.2f}")
        print(f"  Real-time FPS: {results['frames_per_second_realtime']:.2f}")
        print(f"  Avg Frame Time: {results['average_frame_time_ms']:.2f} ms")
        print(f"  Min Frame Time: {results['min_frame_time_ms']:.2f} ms")
        print(f"  Max Frame Time: {results['max_frame_time_ms']:.2f} ms")
        print(f"\nMemory Metrics:")
        print(f"  Average Memory: {results['average_memory_mb']:.2f} MB")
        print(f"  Peak Memory: {results['peak_memory_mb']:.2f} MB")
        print("="*60 + "\n")
    
    def save_results(self, output_file="performance_results.json"):
        """Save results to JSON file"""
        results = self.get_results()
        if not results:
            print("No results to save")
            return
        
        # Load existing results if file exists
        all_results = []
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                all_results = json.load(f)
        
        # Append new results
        all_results.append(results)
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Results saved to {output_file}")


# Example usage in your tracker
"""
# At the beginning of your tracking code:
from eval_performance import PerformanceEvaluator

evaluator = PerformanceEvaluator("CSRT_Tracker")  # or "SAM2_Tracker", "SAMTrack_Tracker"
evaluator.start_tracking()

# In your main tracking loop:
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_start = evaluator.start_frame()
    
    # Your tracking code here
    # ... track_frame(frame)
    # ... draw results
    
    evaluator.end_frame(frame_start)

# After tracking ends:
evaluator.end_tracking()
evaluator.print_results()
evaluator.save_results("performance_results.json")
"""