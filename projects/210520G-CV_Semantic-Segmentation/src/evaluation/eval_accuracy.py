"""
Accuracy Evaluation Metrics
Measures: IoU, Center Distance Error, Identity Switches
Can work with or without ground truth
"""

import numpy as np
import json
import os
from datetime import datetime

class AccuracyEvaluator:
    def __init__(self, tracker_name):
        self.tracker_name = tracker_name
        self.frame_data = []
        self.player_tracks = {}  # player_id -> list of bboxes
        self.ground_truth = None
        
    def add_frame_prediction(self, frame_idx, player_predictions):
        """
        Add predictions for a frame
        
        Args:
            frame_idx: int - frame number
            player_predictions: list of dicts with keys:
                - player_id: int
                - bbox: tuple (x, y, w, h)
                - is_active: bool
                - team_id: int (optional)
        """
        self.frame_data.append({
            'frame_idx': frame_idx,
            'predictions': player_predictions
        })
        
        # Store tracks per player
        for pred in player_predictions:
            player_id = pred['player_id']
            if player_id not in self.player_tracks:
                self.player_tracks[player_id] = []
            self.player_tracks[player_id].append({
                'frame_idx': frame_idx,
                'bbox': pred['bbox'],
                'is_active': pred['is_active']
            })
    
    def load_ground_truth(self, gt_file):
        """
        Load ground truth annotations
        Format: JSON file with structure similar to predictions
        """
        if os.path.exists(gt_file):
            with open(gt_file, 'r') as f:
                self.ground_truth = json.load(f)
            print(f"Loaded ground truth from {gt_file}")
        else:
            print(f"Ground truth file not found: {gt_file}")
    
    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        bbox format: (x, y, w, h)
        """
        if bbox1 is None or bbox2 is None:
            return 0.0
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def calculate_center_distance(self, bbox1, bbox2):
        """Calculate Euclidean distance between bbox centers"""
        if bbox1 is None or bbox2 is None:
            return None
        
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance
    
    def calculate_metrics_with_gt(self):
        """Calculate accuracy metrics when ground truth is available"""
        if not self.ground_truth:
            print("No ground truth available")
            return None
        
        iou_scores = []
        center_errors = []
        
        for frame_data in self.frame_data:
            frame_idx = frame_data['frame_idx']
            predictions = frame_data['predictions']
            
            # Find corresponding ground truth
            gt_frame = next((f for f in self.ground_truth if f['frame_idx'] == frame_idx), None)
            if not gt_frame:
                continue
            
            # Match predictions to ground truth (simple matching by player_id)
            for pred in predictions:
                if not pred['is_active']:
                    continue
                
                player_id = pred['player_id']
                gt_pred = next((p for p in gt_frame['predictions'] if p['player_id'] == player_id), None)
                
                if gt_pred:
                    iou = self.calculate_iou(pred['bbox'], gt_pred['bbox'])
                    iou_scores.append(iou)
                    
                    dist = self.calculate_center_distance(pred['bbox'], gt_pred['bbox'])
                    if dist is not None:
                        center_errors.append(dist)
        
        if not iou_scores:
            return None
        
        return {
            'average_iou': np.mean(iou_scores),
            'median_iou': np.median(iou_scores),
            'iou_std': np.std(iou_scores),
            'average_center_error': np.mean(center_errors) if center_errors else 0,
            'median_center_error': np.median(center_errors) if center_errors else 0
        }
    
    def calculate_tracking_quality_metrics(self):
        """Calculate metrics without ground truth - tracking quality indicators"""
        
        # Track fragmentation: how often tracks are lost and recovered
        track_fragments = {}
        track_lengths = {}
        identity_switches = 0
        
        for player_id, tracks in self.player_tracks.items():
            fragments = 0
            was_active = False
            continuous_length = 0
            max_length = 0
            
            for track in tracks:
                if track['is_active']:
                    if not was_active:
                        fragments += 1
                    was_active = True
                    continuous_length += 1
                    max_length = max(max_length, continuous_length)
                else:
                    was_active = False
                    continuous_length = 0
            
            track_fragments[player_id] = fragments
            track_lengths[player_id] = max_length
        
        # Calculate bbox stability (how much bboxes change frame-to-frame)
        bbox_stabilities = []
        for player_id, tracks in self.player_tracks.items():
            active_tracks = [t for t in tracks if t['is_active'] and t['bbox'] is not None]
            
            if len(active_tracks) < 2:
                continue
            
            for i in range(len(active_tracks) - 1):
                bbox1 = active_tracks[i]['bbox']
                bbox2 = active_tracks[i + 1]['bbox']
                
                # Calculate size change ratio
                area1 = bbox1[2] * bbox1[3]
                area2 = bbox2[2] * bbox2[3]
                size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
                
                bbox_stabilities.append(size_ratio)
        
        results = {
            'tracker_name': self.tracker_name,
            'timestamp': datetime.now().isoformat(),
            'total_players': len(self.player_tracks),
            'total_frames': len(self.frame_data),
            'average_track_fragments': np.mean(list(track_fragments.values())) if track_fragments else 0,
            'average_max_track_length': np.mean(list(track_lengths.values())) if track_lengths else 0,
            'bbox_stability': np.mean(bbox_stabilities) if bbox_stabilities else 0,
            'tracking_success_rate': self._calculate_success_rate()
        }
        
        return results
    
    def _calculate_success_rate(self):
        """Calculate percentage of frames where players are successfully tracked"""
        if not self.frame_data:
            return 0.0
        
        total_possible = 0
        total_successful = 0
        
        for frame_data in self.frame_data:
            predictions = frame_data['predictions']
            total_possible += len(predictions)
            total_successful += sum(1 for p in predictions if p['is_active'])
        
        return (total_successful / total_possible * 100) if total_possible > 0 else 0
    
    def get_results(self, use_ground_truth=False):
        """Get all accuracy metrics"""
        results = self.calculate_tracking_quality_metrics()
        
        if use_ground_truth and self.ground_truth:
            gt_metrics = self.calculate_metrics_with_gt()
            if gt_metrics:
                results.update(gt_metrics)
        
        return results
    
    def print_results(self, use_ground_truth=False):
        """Print accuracy results"""
        results = self.get_results(use_ground_truth)
        
        print("\n" + "="*60)
        print(f"ACCURACY METRICS - {results['tracker_name']}")
        print("="*60)
        print(f"Total Players Tracked: {results['total_players']}")
        print(f"Total Frames: {results['total_frames']}")
        print(f"\nTracking Quality:")
        print(f"  Tracking Success Rate: {results['tracking_success_rate']:.2f}%")
        print(f"  Avg Track Fragments: {results['average_track_fragments']:.2f}")
        print(f"  Avg Max Track Length: {results['average_max_track_length']:.2f} frames")
        print(f"  BBox Stability: {results['bbox_stability']:.3f}")
        
        if 'average_iou' in results:
            print(f"\nGround Truth Comparison:")
            print(f"  Average IoU: {results['average_iou']:.3f}")
            print(f"  Median IoU: {results['median_iou']:.3f}")
            print(f"  IoU Std Dev: {results['iou_std']:.3f}")
            print(f"  Avg Center Error: {results['average_center_error']:.2f} pixels")
            print(f"  Median Center Error: {results['median_center_error']:.2f} pixels")
        
        print("="*60 + "\n")
    
    def save_results(self, output_file="accuracy_results.json"):
        """Save results to JSON file"""
        results = self.get_results(use_ground_truth=(self.ground_truth is not None))
        
        # Load existing results
        all_results = []
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                all_results = json.load(f)
        
        all_results.append(results)
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Results saved to {output_file}")


# Example usage in your tracker
"""
from eval_accuracy import AccuracyEvaluator

evaluator = AccuracyEvaluator("CSRT_Tracker")

# In your tracking loop, after updating players:
frame_idx = 0
while tracking:
    # ... your tracking code ...
    
    # Collect predictions for this frame
    predictions = []
    for player in self.players:
        predictions.append({
            'player_id': player.player_id,
            'bbox': player.bbox,
            'is_active': player.is_active,
            'team_id': player.team_id
        })
    
    evaluator.add_frame_prediction(frame_idx, predictions)
    frame_idx += 1

# After tracking:
evaluator.print_results()
evaluator.save_results("accuracy_results.json")
"""