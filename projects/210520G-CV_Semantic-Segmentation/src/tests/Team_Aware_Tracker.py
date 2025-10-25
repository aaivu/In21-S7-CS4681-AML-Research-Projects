import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from collections import deque
import sys

class PlayerTracker:
    """Individual player tracker with appearance model"""
    def __init__(self, player_id, bbox, mask, frame, team_id=None):
        self.player_id = player_id
        self.team_id = team_id
        self.bbox = bbox
        self.mask = mask
        self.tracker = None
        self.lost_frames = 0
        self.max_lost_frames = 30
        self.is_active = True
        
        # Appearance model - jersey color histogram
        self.color_histogram = self.extract_color_histogram(frame, mask)
        self.appearance_history = deque(maxlen=10)
        self.appearance_history.append(self.color_histogram)
        
        # Position history for prediction
        self.position_history = deque(maxlen=5)
        self.position_history.append(self.get_center(bbox))
        
    def extract_color_histogram(self, frame, mask, bins=32):
        """Extract color histogram from masked region (jersey colors)"""
        # Focus on upper body (jersey area) - top 60% of mask
        h, w = mask.shape
        upper_mask = mask.copy()
        upper_mask[int(h*0.6):, :] = 0
        
        # Convert to HSV for better color discrimination
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for masked region
        hist_h = cv2.calcHist([hsv], [0], upper_mask.astype(np.uint8), [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], upper_mask.astype(np.uint8), [bins], [0, 256])
        
        # Normalize
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        
        # Combine H and S channels
        histogram = np.concatenate([hist_h, hist_s])
        
        return histogram
    
    def get_center(self, bbox):
        """Get center point of bounding box"""
        x, y, w, h = bbox
        return (int(x + w/2), int(y + h/2))
    
    def update_appearance(self, frame, mask):
        """Update appearance model with new observation"""
        new_histogram = self.extract_color_histogram(frame, mask)
        self.appearance_history.append(new_histogram)
        
        # Update average histogram
        self.color_histogram = np.mean(list(self.appearance_history), axis=0)
    
    def compare_appearance(self, frame, mask):
        """Compare appearance with stored model (returns similarity 0-1)"""
        new_histogram = self.extract_color_histogram(frame, mask)
        
        # Use Bhattacharyya distance for histogram comparison
        similarity = cv2.compareHist(
            self.color_histogram.astype(np.float32),
            new_histogram.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        
        # Convert to similarity (0 = identical, 1 = completely different)
        # Invert so higher = more similar
        return 1.0 - similarity


class TeamAwareTracker:
    """Main tracking system with team awareness"""
    def __init__(self, sam_checkpoint, model_type="vit_b"):
        print("Loading SAM model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print(f"SAM loaded on {self.device}")
        
        self.players = []
        self.next_player_id = 0
        self.frame = None
        self.points = []
        self.current_team = 0
        self.team_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Green, Red, Blue
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for player selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            print(f"Selected player at ({x}, {y}) for Team {self.current_team}")
            self.draw_points()
    
    def draw_points(self):
        """Draw selection points on frame"""
        temp_frame = self.frame.copy()
        for i, point in enumerate(self.points):
            color = self.team_colors[self.current_team]
            cv2.circle(temp_frame, tuple(point), 5, color, -1)
            cv2.putText(temp_frame, f"P{i+1}", (point[0]+10, point[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('Select Players', temp_frame)
    
    def select_players(self, frame):
        """Interactive UI to select multiple players"""
        self.frame = frame.copy()
        self.points = []
        self.players = []
        self.current_team = 0
        
        cv2.namedWindow('Select Players')
        cv2.setMouseCallback('Select Players', self.mouse_callback)
        
        print("\n=== Player Selection ===")
        print("Click on players to track")
        print("Press '1', '2', '3': Switch team (affects color)")
        print("Press SPACE: Segment selected players")
        print("Press 'r': Reset selections")
        print("Press 'q': Quit")
        print("========================\n")
        
        cv2.imshow('Select Players', self.frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Segment players
                if len(self.points) > 0:
                    self.segment_players(frame)
                    break
                else:
                    print("Please select at least one player!")
            
            elif key == ord('1'):
                self.current_team = 0
                print("Switched to Team 0 (Green)")
            
            elif key == ord('2'):
                self.current_team = 1
                print("Switched to Team 1 (Red)")
            
            elif key == ord('3'):
                self.current_team = 2
                print("Switched to Team 2 (Blue/Referee)")
            
            elif key == ord('r'):
                self.points = []
                self.current_team = 0
                print("Reset selections")
                cv2.imshow('Select Players', self.frame)
            
            elif key == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)
        
        cv2.destroyAllWindows()
        return self.players
    
    def segment_players(self, frame):
        """Segment all selected players using SAM"""
        print(f"\nSegmenting {len(self.points)} players...")
        self.predictor.set_image(frame)
        
        # Track which team each point belongs to (assign teams based on order)
        team_assignments = []
        for i in range(len(self.points)):
            # Simple heuristic: ask user or assign based on selection order
            team_assignments.append(i % 2)  # Alternate teams for demo
        
        for idx, point in enumerate(self.points):
            # Get mask for this player
            input_point = np.array([point])
            input_label = np.array([1])
            
            masks, scores, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            
            # Select best mask
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]
            
            # Convert to bbox
            bbox = self.mask_to_bbox(mask)
            if bbox is None:
                print(f"Failed to segment player {idx+1}")
                continue
            
            # Create tracker for this player
            team_id = team_assignments[idx]
            player = PlayerTracker(self.next_player_id, bbox, mask, frame, team_id)
            
            # Initialize OpenCV tracker
            try:
                player.tracker = cv2.legacy.TrackerCSRT_create()
            except AttributeError:
                player.tracker = cv2.TrackerCSRT_create()
            
            player.tracker.init(frame, bbox)
            
            self.players.append(player)
            print(f"Player {self.next_player_id} (Team {team_id}): bbox={bbox}, score={scores[best_mask_idx]:.3f}")
            self.next_player_id += 1
        
        print(f"Successfully initialized {len(self.players)} players\n")
    
    def mask_to_bbox(self, mask):
        """Convert mask to bounding box"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    
    def find_best_match(self, frame, bbox, mask, lost_players):
        """Find best matching lost player using appearance similarity"""
        best_match = None
        best_similarity = 0.5  # Minimum threshold
        
        for player in lost_players:
            similarity = player.compare_appearance(frame, mask)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = player
        
        return best_match, best_similarity
    
    def recover_lost_players(self, frame):
        """Try to recover lost players using SAM re-segmentation"""
        lost_players = [p for p in self.players if not p.is_active]
        
        if len(lost_players) == 0:
            return
        
        print(f"Attempting to recover {len(lost_players)} lost players...")
        
        # Predict likely positions based on velocity
        recovery_regions = []
        for player in lost_players:
            if len(player.position_history) >= 2:
                # Simple linear prediction
                last_pos = player.position_history[-1]
                prev_pos = player.position_history[-2]
                
                vx = last_pos[0] - prev_pos[0]
                vy = last_pos[1] - prev_pos[1]
                
                predicted_x = last_pos[0] + vx * player.lost_frames
                predicted_y = last_pos[1] + vy * player.lost_frames
                
                recovery_regions.append((predicted_x, predicted_y))
        
        # Sample points in recovery regions and try SAM
        self.predictor.set_image(frame)
        
        for player, region in zip(lost_players, recovery_regions):
            # Try multiple points around predicted position
            search_radius = 50
            test_points = [
                region,
                (region[0] + search_radius, region[1]),
                (region[0] - search_radius, region[1]),
                (region[0], region[1] + search_radius),
                (region[0], region[1] - search_radius),
            ]
            
            for point in test_points:
                # Check if point is within frame
                if point[0] < 0 or point[0] >= frame.shape[1] or \
                   point[1] < 0 or point[1] >= frame.shape[0]:
                    continue
                
                input_point = np.array([point])
                input_label = np.array([1])
                
                try:
                    masks, scores, _ = self.predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )
                    
                    best_mask_idx = np.argmax(scores)
                    mask = masks[best_mask_idx]
                    
                    # Check appearance similarity
                    similarity = player.compare_appearance(frame, mask)
                    
                    if similarity > 0.6:  # Good match
                        bbox = self.mask_to_bbox(mask)
                        if bbox:
                            # Reinitialize tracker
                            try:
                                player.tracker = cv2.legacy.TrackerCSRT_create()
                            except AttributeError:
                                player.tracker = cv2.TrackerCSRT_create()
                            
                            player.tracker.init(frame, bbox)
                            player.bbox = bbox
                            player.is_active = True
                            player.lost_frames = 0
                            player.update_appearance(frame, mask)
                            
                            print(f"Recovered Player {player.player_id} (similarity: {similarity:.2f})")
                            break
                
                except Exception as e:
                    continue
    
    def track_frame(self, frame):
        """Track all players in current frame"""
        active_count = 0
        
        for player in self.players:
            if not player.is_active:
                player.lost_frames += 1
                if player.lost_frames > player.max_lost_frames:
                    print(f"Player {player.player_id} permanently lost")
                continue
            
            # Update tracker
            success, bbox = player.tracker.update(frame)
            
            if success:
                player.bbox = bbox
                player.position_history.append(player.get_center(bbox))
                active_count += 1
                
                # Optional: periodically update appearance model
                # (skipped for performance, but could re-segment every N frames)
            else:
                player.is_active = False
                player.lost_frames += 1
                print(f"Lost track of Player {player.player_id}")
        
        # Try to recover lost players every 10 frames
        if len([p for p in self.players if not p.is_active]) > 0:
            frame_count = sum([len(p.position_history) for p in self.players])
            if frame_count % 10 == 0:
                self.recover_lost_players(frame)
        
        return active_count
    
    def draw_tracking(self, frame):
        """Draw tracking results with team colors"""
        display_frame = frame.copy()
        
        for player in self.players:
            if not player.is_active:
                continue
            
            x, y, w, h = [int(v) for v in player.bbox]
            color = self.team_colors[player.team_id]
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw player ID and team
            label = f"P{player.player_id} T{player.team_id}"
            cv2.putText(display_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw trajectory
            if len(player.position_history) > 1:
                points = np.array(list(player.position_history), dtype=np.int32)
                cv2.polylines(display_frame, [points], False, color, 2)
        
        # Draw stats
        active = sum([1 for p in self.players if p.is_active])
        lost = sum([1 for p in self.players if not p.is_active])
        cv2.putText(display_frame, f"Active: {active} | Lost: {lost}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame
    
    def track_video(self, video_path, output_path=None):
        """Main tracking loop"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        
        # Select players
        self.select_players(frame)
        
        if len(self.players) == 0:
            print("No players selected!")
            return
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("\nTracking started... Press 'q' to quit\n")
        
        # Tracking loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Track all players
            active_count = self.track_frame(frame)
            
            # Draw results
            display_frame = self.draw_tracking(frame)
            
            # Show progress
            cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Team-Aware Tracking', display_frame)
            
            if writer:
                writer.write(display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nTracking stopped by user")
                break
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
            print(f"\nOutput saved to: {output_path}")
        cv2.destroyAllWindows()
        
        print(f"Tracking complete! Processed {frame_count} frames")
    
    def track_video_with_evaluation(self, video_path, output_path=None, results_dir="results"):
        """
        Track video with comprehensive evaluation metrics
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            results_dir: Directory to save evaluation results
        """
        # Import evaluators
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from evaluation.eval_performance import PerformanceEvaluator
        from evaluation.eval_accuracy import AccuracyEvaluator
        from evaluation.eval_robustnuss import RobustnessEvaluator
        
        # Initialize evaluators
        tracker_name = "CSRT_Tracker"
        perf_eval = PerformanceEvaluator(tracker_name)
        acc_eval = AccuracyEvaluator(tracker_name)
        robust_eval = RobustnessEvaluator(tracker_name)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        
        # Get segmentation mask from SAM (your existing code)
        mask = self.get_sam_segmentation(frame)
        
        # Convert mask to bounding box
        bbox = self.mask_to_bbox(mask)
        if bbox is None:
            print("Error: Could not extract bounding box from mask")
            return
        
        print(f"Initial bbox: {bbox}")
        
        # Initialize tracker (your existing code)
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            tracker = cv2.TrackerCSRT_create()
        
        tracker.init(frame, bbox)
        print(f"Initialized CSRT tracker")
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 1
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # ===== START EVALUATION TRACKING =====
        perf_eval.start_tracking()
        
        print("\nTracking with evaluation started... Press 'q' to quit\n")
        
        # Draw first frame
        display_frame = frame.copy()
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, "CSRT Tracker", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Tracking', display_frame)
        if writer:
            writer.write(display_frame)
        
        # Collect data for first frame
        predictions = [{
            'player_id': 0,  # Single player for simplicity, adjust if tracking multiple
            'bbox': bbox,
            'is_active': True,
            'team_id': 0,
            'confidence': 1.0
        }]
        acc_eval.add_frame_prediction(frame_idx, predictions)
        robust_eval.add_frame_data(frame_idx, predictions)
        frame_idx += 1
        
        # Main tracking loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # ===== START FRAME TIMING =====
            frame_start = perf_eval.start_frame()
            
            # Update tracker
            success, bbox = tracker.update(frame)
            
            # Prepare predictions for evaluation
            predictions = [{
                'player_id': 0,
                'bbox': bbox if success else None,
                'is_active': success,
                'team_id': 0,
                'confidence': 1.0 if success else 0.0
            }]
            
            # Collect evaluation data
            acc_eval.add_frame_prediction(frame_idx, predictions)
            robust_eval.add_frame_data(frame_idx, predictions)
            
            # Draw results
            display_frame = frame.copy()
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, "CSRT Tracker", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Tracking failure", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show progress
            cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Tracking', display_frame)
            
            # Write frame if output is specified
            if writer:
                writer.write(display_frame)
            
            # ===== END FRAME TIMING =====
            perf_eval.end_frame(frame_start)
            frame_idx += 1
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nTracking stopped by user")
                break
        
        # ===== END EVALUATION TRACKING =====
        perf_eval.end_tracking()
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nTracking complete! Processed {frame_count} frames")
        
        # ===== PRINT AND SAVE RESULTS =====
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Performance metrics
        perf_eval.print_results()
        perf_eval.save_results(os.path.join(results_dir, f"performance_{tracker_name}.json"))
        
        # Accuracy metrics
        acc_eval.print_results()
        acc_eval.save_results(os.path.join(results_dir, f"accuracy_{tracker_name}.json"))
        
        # Robustness metrics
        robust_eval.print_results()
        robust_eval.save_results(os.path.join(results_dir, f"robustness_{tracker_name}.json"))
        robust_eval.save_report(os.path.join(results_dir, f"robustness_{tracker_name}.txt"))
        
        print(f"\n✓ All evaluation results saved to '{results_dir}/' directory")
        print("="*70)


# ============================================================================
# FOR MULTI-PLAYER TRACKING
# ============================================================================

    def track_video_with_evaluation_multiPlayer(self, video_path, output_path=None, results_dir="results"):
        """
        Track video with multiple players and comprehensive evaluation
        Use this version if you're tracking multiple players with team-aware features
        """

        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from evaluation.eval_performance import PerformanceEvaluator
        from evaluation.eval_accuracy import AccuracyEvaluator
        from evaluation.eval_robustnuss import RobustnessEvaluator
        
        # Initialize evaluators
        tracker_name = "CSRT_TeamAware_Tracker"
        perf_eval = PerformanceEvaluator(tracker_name)
        acc_eval = AccuracyEvaluator(tracker_name)
        robust_eval = RobustnessEvaluator(tracker_name)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        
        # Select players (your existing method)
        self.select_players(frame)
        
        if len(self.players) == 0:
            print("No players selected!")
            return
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 1
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Start evaluation
        perf_eval.start_tracking()
        
        # Collect first frame data
        predictions = []
        for player in self.players:
            predictions.append({
                'player_id': player.player_id,
                'bbox': player.bbox,
                'is_active': player.is_active,
                'team_id': player.team_id,
                'confidence': getattr(player, 'confidence', 1.0)
            })
        acc_eval.add_frame_prediction(frame_idx, predictions)
        robust_eval.add_frame_data(frame_idx, predictions)
        frame_idx += 1
        
        print("\nTracking with evaluation started... Press 'q' to quit\n")
        
        # Main tracking loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Start frame timing
            frame_start = perf_eval.start_frame()
            
            # Track all players (your existing method)
            active_count = self.track_frame(frame)
            
            # Collect evaluation data
            predictions = []
            for player in self.players:
                predictions.append({
                    'player_id': player.player_id,
                    'bbox': player.bbox,
                    'is_active': player.is_active,
                    'team_id': player.team_id,
                    'confidence': getattr(player, 'confidence', 1.0)
                })
            
            acc_eval.add_frame_prediction(frame_idx, predictions)
            robust_eval.add_frame_data(frame_idx, predictions)
            
            # Draw results (your existing method)
            display_frame = self.draw_tracking(frame)
            
            # Show progress
            cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Tracking', display_frame)
            
            if writer:
                writer.write(display_frame)
            
            # End frame timing
            perf_eval.end_frame(frame_start)
            frame_idx += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nTracking stopped by user")
                break
        
        # End evaluation
        perf_eval.end_tracking()
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\nTracking complete! Processed {frame_count} frames")
        
        # Print and save results
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        os.makedirs(results_dir, exist_ok=True)
        
        perf_eval.print_results()
        perf_eval.save_results(os.path.join(results_dir, f"performance_{tracker_name}.json"))
        
        acc_eval.print_results()
        acc_eval.save_results(os.path.join(results_dir, f"accuracy_{tracker_name}.json"))
        
        robust_eval.print_results()
        robust_eval.save_results(os.path.join(results_dir, f"robustness_{tracker_name}.json"))
        robust_eval.save_report(os.path.join(results_dir, f"robustness_{tracker_name}.txt"))
        
        print(f"\n✓ All evaluation results saved to '{results_dir}/' directory")
        print("="*70)


def main():
    # Configuration
    SAM_CHECKPOINT = "../models/sam_vit_h_4b8939.pth"
    MODEL_TYPE = "vit_h"
    VIDEO_PATH = r"../clips\football\test (26).mp4"
    OUTPUT_PATH = "../videos/output_tracked_long.mp4"
    
    # Initialize and run
    tracker = TeamAwareTracker(SAM_CHECKPOINT, MODEL_TYPE)
    # tracker.track_video(VIDEO_PATH, OUTPUT_PATH)
    tracker.track_video_with_evaluation_multiPlayer(VIDEO_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()