"""
Robustness Evaluation Metrics
Measures: Occlusion Handling, Recovery Rate, Identity Consistency
"""

import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict

class RobustnessEvaluator:
    def __init__(self, tracker_name):
        self.tracker_name = tracker_name
        self.frame_data = []
        self.occlusion_events = []
        self.recovery_events = []
        self.identity_switches = []
        self.player_appearance_history = defaultdict(list)
        
    def add_frame_data(self, frame_idx, player_data):
        """
        Add tracking data for a frame
        
        Args:
            frame_idx: int
            player_data: list of dicts with keys:
                - player_id: int
                - bbox: tuple (x, y, w, h)
                - is_active: bool
                - confidence: float (optional)
                - team_id: int (optional)
        """
        self.frame_data.append({
            'frame_idx': frame_idx,
            'players': player_data
        })
        
        # Track player appearances
        for player in player_data:
            self.player_appearance_history[player['player_id']].append({
                'frame_idx': frame_idx,
                'is_active': player['is_active'],
                'bbox': player.get('bbox'),
                'confidence': player.get('confidence', 1.0)
            })
    
    def detect_occlusion_events(self):
        """
        Detect when players are lost (potential occlusions)
        An occlusion is when a player transitions from active to inactive
        """
        self.occlusion_events = []
        
        for player_id, history in self.player_appearance_history.items():
            for i in range(len(history) - 1):
                current = history[i]
                next_frame = history[i + 1]
                
                # Detect occlusion: was active, then became inactive
                if current['is_active'] and not next_frame['is_active']:
                    self.occlusion_events.append({
                        'player_id': player_id,
                        'start_frame': current['frame_idx'],
                        'end_frame': None,  # Will be filled when recovered
                        'duration': None
                    })
    
    def detect_recovery_events(self):
        """
        Detect when lost players are recovered
        Recovery is when a player transitions from inactive to active
        """
        self.recovery_events = []
        
        for player_id, history in self.player_appearance_history.items():
            for i in range(len(history) - 1):
                current = history[i]
                next_frame = history[i + 1]
                
                # Detect recovery: was inactive, then became active
                if not current['is_active'] and next_frame['is_active']:
                    # Find the corresponding occlusion event
                    for occ in self.occlusion_events:
                        if occ['player_id'] == player_id and occ['end_frame'] is None:
                            occ['end_frame'] = next_frame['frame_idx']
                            occ['duration'] = occ['end_frame'] - occ['start_frame']
                            
                            self.recovery_events.append({
                                'player_id': player_id,
                                'lost_at_frame': occ['start_frame'],
                                'recovered_at_frame': next_frame['frame_idx'],
                                'occlusion_duration': occ['duration']
                            })
                            break
    
    def calculate_overlap_ratio(self, bbox1, bbox2):
        """Calculate how much two bboxes overlap (for detecting occlusions)"""
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        
        return intersection / area1 if area1 > 0 else 0.0
    
    def detect_potential_occlusions_by_overlap(self):
        """
        Detect frames where players likely occlude each other
        Based on bbox overlap
        """
        occlusion_frames = []
        
        for frame_data in self.frame_data:
            frame_idx = frame_data['frame_idx']
            players = [p for p in frame_data['players'] if p['is_active'] and p.get('bbox')]
            
            overlaps = []
            for i, player1 in enumerate(players):
                for player2 in players[i+1:]:
                    overlap = self.calculate_overlap_ratio(player1['bbox'], player2['bbox'])
                    if overlap > 0.3:  # 30% overlap threshold
                        overlaps.append({
                            'player1_id': player1['player_id'],
                            'player2_id': player2['player_id'],
                            'overlap_ratio': overlap
                        })
            
            if overlaps:
                occlusion_frames.append({
                    'frame_idx': frame_idx,
                    'num_overlaps': len(overlaps),
                    'overlaps': overlaps
                })
        
        return occlusion_frames
    
    def calculate_identity_consistency(self):
        """
        Measure how consistently player IDs are maintained
        Based on spatial consistency of tracks
        """
        identity_issues = []
        
        for player_id, history in self.player_appearance_history.items():
            active_frames = [h for h in history if h['is_active'] and h['bbox']]
            
            if len(active_frames) < 2:
                continue
            
            # Check for sudden large movements (potential ID switch)
            for i in range(len(active_frames) - 1):
                curr = active_frames[i]
                next_f = active_frames[i + 1]
                
                # Calculate center distance
                curr_center = (curr['bbox'][0] + curr['bbox'][2]/2, 
                              curr['bbox'][1] + curr['bbox'][3]/2)
                next_center = (next_f['bbox'][0] + next_f['bbox'][2]/2,
                              next_f['bbox'][1] + next_f['bbox'][3]/2)
                
                distance = np.sqrt((curr_center[0] - next_center[0])**2 + 
                                  (curr_center[1] - next_center[1])**2)
                
                # If distance is very large in consecutive frames (accounting for gaps)
                frame_gap = next_f['frame_idx'] - curr['frame_idx']
                expected_max_movement = 100 * frame_gap  # Assume max 100 pixels per frame
                
                if distance > expected_max_movement:
                    identity_issues.append({
                        'player_id': player_id,
                        'frame_idx': next_f['frame_idx'],
                        'distance': distance,
                        'likely_switch': True
                    })
        
        return identity_issues
    
    def calculate_metrics(self):
        """Calculate all robustness metrics"""
        # Detect events
        self.detect_occlusion_events()
        self.detect_recovery_events()
        occlusion_frames = self.detect_potential_occlusions_by_overlap()
        identity_issues = self.calculate_identity_consistency()
        
        # Calculate statistics
        total_occlusions = len(self.occlusion_events)
        total_recoveries = len(self.recovery_events)
        recovery_rate = (total_recoveries / total_occlusions * 100) if total_occlusions > 0 else 0
        
        # Average occlusion duration
        occlusion_durations = [e['duration'] for e in self.occlusion_events if e['duration'] is not None]
        avg_occlusion_duration = np.mean(occlusion_durations) if occlusion_durations else 0
        
        # Average recovery time
        recovery_times = [e['occlusion_duration'] for e in self.recovery_events]
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        
        # Calculate tracking persistence (how long tracks survive)
        track_durations = []
        for player_id, history in self.player_appearance_history.items():
            active_frames = [h for h in history if h['is_active']]
            if active_frames:
                duration = max(h['frame_idx'] for h in active_frames) - min(h['frame_idx'] for h in active_frames)
                track_durations.append(duration)
        
        avg_track_duration = np.mean(track_durations) if track_durations else 0
        
        # Occlusion handling score
        frames_with_occlusions = len(occlusion_frames)
        total_frames = len(self.frame_data)
        occlusion_frame_ratio = (frames_with_occlusions / total_frames * 100) if total_frames > 0 else 0
        
        results = {
            'tracker_name': self.tracker_name,
            'timestamp': datetime.now().isoformat(),
            'total_frames': total_frames,
            'total_players': len(self.player_appearance_history),
            
            # Occlusion metrics
            'total_occlusion_events': total_occlusions,
            'total_recovery_events': total_recoveries,
            'recovery_rate_percent': round(recovery_rate, 2),
            'average_occlusion_duration_frames': round(avg_occlusion_duration, 2),
            'average_recovery_time_frames': round(avg_recovery_time, 2),
            
            # Overlap-based occlusions
            'frames_with_overlaps': frames_with_occlusions,
            'occlusion_frame_ratio_percent': round(occlusion_frame_ratio, 2),
            
            # Identity consistency
            'potential_identity_switches': len(identity_issues),
            'identity_switch_rate_per_1000_frames': round(len(identity_issues) / total_frames * 1000, 2) if total_frames > 0 else 0,
            
            # Track persistence
            'average_track_duration_frames': round(avg_track_duration, 2),
            
            # Overall robustness score (0-100)
            'robustness_score': self._calculate_robustness_score(recovery_rate, avg_track_duration, len(identity_issues), total_frames)
        }
        
        return results
    
    def _calculate_robustness_score(self, recovery_rate, avg_track_duration, num_switches, total_frames):
        """
        Calculate overall robustness score (0-100)
        Higher is better
        """
        # Normalize components to 0-100 scale
        recovery_component = recovery_rate  # Already 0-100
        
        # Track duration component (assume 100 frames is excellent)
        duration_component = min(avg_track_duration / 100 * 100, 100)
        
        # Identity consistency component (fewer switches is better)
        switch_rate = (num_switches / total_frames * 100) if total_frames > 0 else 0
        identity_component = max(100 - switch_rate * 10, 0)
        
        # Weighted average
        score = (
            recovery_component * 0.4 +
            duration_component * 0.3 +
            identity_component * 0.3
        )
        
        return round(score, 2)
    
    def print_results(self):
        """Print robustness results in readable format"""
        results = self.calculate_metrics()
        
        print("\n" + "="*60)
        print(f"ROBUSTNESS METRICS - {results['tracker_name']}")
        print("="*60)
        print(f"Total Frames: {results['total_frames']}")
        print(f"Total Players: {results['total_players']}")
        
        print(f"\nOcclusion Handling:")
        print(f"  Total Occlusion Events: {results['total_occlusion_events']}")
        print(f"  Successful Recoveries: {results['total_recovery_events']}")
        print(f"  Recovery Rate: {results['recovery_rate_percent']:.2f}%")
        print(f"  Avg Occlusion Duration: {results['average_occlusion_duration_frames']:.2f} frames")
        print(f"  Avg Recovery Time: {results['average_recovery_time_frames']:.2f} frames")
        
        print(f"\nPlayer Overlap Detection:")
        print(f"  Frames with Overlaps: {results['frames_with_overlaps']}")
        print(f"  Occlusion Frame Ratio: {results['occlusion_frame_ratio_percent']:.2f}%")
        
        print(f"\nIdentity Consistency:")
        print(f"  Potential ID Switches: {results['potential_identity_switches']}")
        print(f"  Switch Rate: {results['identity_switch_rate_per_1000_frames']:.2f} per 1000 frames")
        
        print(f"\nTrack Persistence:")
        print(f"  Avg Track Duration: {results['average_track_duration_frames']:.2f} frames")
        
        print(f"\n{'='*20}")
        print(f"OVERALL ROBUSTNESS SCORE: {results['robustness_score']:.2f}/100")
        print(f"{'='*20}")
        print("="*60 + "\n")
    
    def save_results(self, output_file="robustness_results.json"):
        """Save results to JSON file"""
        results = self.calculate_metrics()
        
        # Load existing results
        all_results = []
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                all_results = json.load(f)
        
        all_results.append(results)
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def generate_report(self):
        """Generate detailed text report"""
        results = self.calculate_metrics()
        
        report = []
        report.append("=" * 70)
        report.append(f"ROBUSTNESS EVALUATION REPORT - {results['tracker_name']}")
        report.append("=" * 70)
        report.append(f"Generated: {results['timestamp']}")
        report.append("")
        
        report.append("SUMMARY:")
        report.append(f"  Robustness Score: {results['robustness_score']:.2f}/100")
        report.append(f"  Recovery Rate: {results['recovery_rate_percent']:.2f}%")
        report.append(f"  Identity Switches: {results['potential_identity_switches']}")
        report.append("")
        
        report.append("DETAILED METRICS:")
        report.append("")
        report.append("1. Occlusion Handling")
        report.append(f"   - Total occlusion events: {results['total_occlusion_events']}")
        report.append(f"   - Successful recoveries: {results['total_recovery_events']}")
        report.append(f"   - Average occlusion duration: {results['average_occlusion_duration_frames']:.2f} frames")
        report.append(f"   - Average recovery time: {results['average_recovery_time_frames']:.2f} frames")
        report.append("")
        
        report.append("2. Overlap Detection")
        report.append(f"   - Frames with player overlaps: {results['frames_with_overlaps']}")
        report.append(f"   - Percentage of frames: {results['occlusion_frame_ratio_percent']:.2f}%")
        report.append("")
        
        report.append("3. Identity Consistency")
        report.append(f"   - Potential identity switches: {results['potential_identity_switches']}")
        report.append(f"   - Switch rate: {results['identity_switch_rate_per_1000_frames']:.2f} per 1000 frames")
        report.append("")
        
        report.append("4. Track Persistence")
        report.append(f"   - Average track duration: {results['average_track_duration_frames']:.2f} frames")
        report.append("")
        
        # Rating interpretation
        score = results['robustness_score']
        if score >= 80:
            rating = "Excellent"
        elif score >= 60:
            rating = "Good"
        elif score >= 40:
            rating = "Fair"
        else:
            rating = "Poor"
        
        report.append(f"OVERALL RATING: {rating}")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def save_report(self, output_file="robustness_report.txt"):
        """Save detailed report to text file"""
        report = self.generate_report()
        
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {output_file}")


# Example usage in your tracker
"""
from eval_robustness import RobustnessEvaluator

evaluator = RobustnessEvaluator("CSRT_Tracker")

# In your tracking loop:
frame_idx = 0
while tracking:
    # ... your tracking code ...
    
    # Collect data for this frame
    player_data = []
    for player in self.players:
        player_data.append({
            'player_id': player.player_id,
            'bbox': player.bbox,
            'is_active': player.is_active,
            'confidence': getattr(player, 'confidence', 1.0),
            'team_id': player.team_id
        })
    
    evaluator.add_frame_data(frame_idx, player_data)
    frame_idx += 1

# After tracking:
evaluator.print_results()
evaluator.save_results("robustness_results.json")
evaluator.save_report("robustness_report.txt")
"""