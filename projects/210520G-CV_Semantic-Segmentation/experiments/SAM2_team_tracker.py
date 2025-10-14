import cv2
import numpy as np
import torch
from collections import deque
import sys
import os

# Try alternative SAM 2 import
try:

    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
    USE_BUILD_FUNCTION = True
    
except ImportError:
    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        SAM2_AVAILABLE = True
        USE_BUILD_FUNCTION = False
        
    except ImportError:
        print("SAM 2 not available")
        SAM2_AVAILABLE = False
        USE_BUILD_FUNCTION = False

class PlayerTrackerSAM2:
    """Individual player tracker using SAM 2"""
    def __init__(self, player_id, mask, frame, team_id=None, object_id=None):
        self.player_id = player_id
        self.team_id = team_id
        self.object_id = object_id  # SAM 2 internal tracking ID
        self.mask = mask
        self.bbox = self.mask_to_bbox(mask)
        self.is_active = True
        self.lost_frames = 0
        
        # Appearance model - jersey color histogram
        self.color_histogram = self.extract_color_histogram(frame, mask)
        self.appearance_history = deque(maxlen=10)
        self.appearance_history.append(self.color_histogram)
        
        # Position history
        self.position_history = deque(maxlen=5)
        if self.bbox:
            self.position_history.append(self.get_center(self.bbox))
    
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
    
    def mask_to_bbox(self, mask):
        """Convert mask to bounding box"""
        if mask is None or not mask.any():
            return None
            
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    
    def get_center(self, bbox):
        """Get center point of bounding box"""
        if bbox is None:
            return None
        x, y, w, h = bbox
        return (int(x + w/2), int(y + h/2))
    
    def update_appearance(self, frame, mask):
        """Update appearance model with new observation"""
        if mask is None or not mask.any():
            return
        new_histogram = self.extract_color_histogram(frame, mask)
        self.appearance_history.append(new_histogram)
        self.color_histogram = np.mean(list(self.appearance_history), axis=0)
    
    def compare_appearance(self, frame, mask):
        """Compare appearance with stored model (returns similarity 0-1)"""
        if mask is None or not mask.any():
            return 0.0
        new_histogram = self.extract_color_histogram(frame, mask)
        
        similarity = cv2.compareHist(
            self.color_histogram.astype(np.float32),
            new_histogram.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        
        return 1.0 - similarity


class TeamAwareTrackerSAM2:
    """Main tracking system using SAM 2 with team awareness"""
    def __init__(self, sam2_checkpoint, model_cfg="sam2_hiera_l.yaml"):
        print("Loading SAM 2 model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        config_name = model_cfg
        
        print(f"Using config: {config_name}")
        print(f"Checkpoint: {sam2_checkpoint}")
        
        try:
            if USE_BUILD_FUNCTION:
                # Method 1: Using build function
                self.predictor = build_sam2_video_predictor(config_name, sam2_checkpoint, device=self.device)
            else:
                # Method 2: Direct initialization (alternative)
                self.predictor = SAM2VideoPredictor.from_pretrained(
                    model_id=f"{config_name}",
                    device=self.device
                )
            print(f"SAM 2 loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading SAM 2: {e}")
            raise
        
        self.players = []
        self.next_player_id = 0
        self.next_object_id = 1  # SAM 2 object IDs start from 1
        self.frame = None
        self.points = []
        self.current_team = 0
        self.team_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Green, Red, Blue
        
        self.inference_state = None
        self.video_segments = {}
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for player selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append({'pos': [x, y], 'team': self.current_team})
            print(f"Selected player at ({x}, {y}) for Team {self.current_team}")
            self.draw_points()
    
    def draw_points(self):
        """Draw selection points on frame"""
        temp_frame = self.frame.copy()
        for i, point_data in enumerate(self.points):
            point = point_data['pos']
            team = point_data['team']
            color = self.team_colors[team]
            cv2.circle(temp_frame, tuple(point), 5, color, -1)
            cv2.putText(temp_frame, f"P{i+1}", (point[0]+10, point[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow('Select Players', temp_frame)
    
    def select_players(self, first_frame):
        """Interactive UI to select multiple players"""
        self.frame = first_frame.copy()
        self.points = []
        
        cv2.namedWindow('Select Players')
        cv2.setMouseCallback('Select Players', self.mouse_callback)
        
        print("\n=== Player Selection ===")
        print("Click on players to track")
        print("Press '1', '2', '3': Switch team (affects color)")
        print("Press SPACE: Start tracking")
        print("Press 'r': Reset selections")
        print("Press 'q': Quit")
        print("========================\n")
        
        cv2.imshow('Select Players', self.frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Start tracking
                if len(self.points) > 0:
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
    
    def initialize_sam2_tracking(self, video_path, frame_idx=0):
        """Initialize SAM 2 with selected points"""
        print(f"\nInitializing SAM 2 tracking with {len(self.points)} players...")
        
        # Initialize inference state
        self.inference_state = self.predictor.init_state(video_path=video_path)
        
        # Add prompts for each selected player
        for point_data in self.points:
            point = point_data['pos']
            team = point_data['team']
            
            # Add point prompt to SAM 2
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=self.next_object_id,
                points=np.array([point], dtype=np.float32),
                labels=np.array([1], np.int32),
            )
            
            # Get the mask from logits
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            
            # Create player tracker
            player = PlayerTrackerSAM2(
                player_id=self.next_player_id,
                mask=mask,
                frame=self.frame,
                team_id=team,
                object_id=self.next_object_id
            )
            
            self.players.append(player)
            print(f"Player {self.next_player_id} (Team {team}, ObjID {self.next_object_id}): bbox={player.bbox}")
            
            self.next_player_id += 1
            self.next_object_id += 1
        
        print(f"Successfully initialized {len(self.players)} players\n")
    
    def propagate_masks(self):
        """Propagate masks through video using SAM 2"""
        print("Propagating masks through video...")
        
        # Run propagation
        self.video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        print(f"Propagated masks for {len(self.video_segments)} frames")
    
    def update_players(self, frame_idx, frame):
        """Update player information from propagated masks"""
        if frame_idx not in self.video_segments:
            # Mark all players as lost
            for player in self.players:
                player.is_active = False
                player.lost_frames += 1
            return
        
        frame_masks = self.video_segments[frame_idx]
        
        # Update each player
        for player in self.players:
            if player.object_id in frame_masks:
                mask = frame_masks[player.object_id]
                
                # Check if mask is valid (has content)
                if mask.any():
                    player.mask = mask
                    player.bbox = player.mask_to_bbox(mask)
                    player.is_active = True
                    player.lost_frames = 0
                    
                    if player.bbox:
                        center = player.get_center(player.bbox)
                        if center:
                            player.position_history.append(center)
                    
                    # Update appearance model
                    player.update_appearance(frame, mask)
                else:
                    player.is_active = False
                    player.lost_frames += 1
            else:
                player.is_active = False
                player.lost_frames += 1
    
    def draw_tracking(self, frame):
        """Draw tracking results with team colors"""
        display_frame = frame.copy()
        
        # Draw masks and boxes
        for player in self.players:
            if not player.is_active or player.mask is None:
                continue
            
            color = self.team_colors[player.team_id]
            
            # Draw semi-transparent mask overlay
            mask_overlay = np.zeros_like(display_frame)
            mask_overlay[player.mask] = color
            display_frame = cv2.addWeighted(display_frame, 1.0, mask_overlay, 0.3, 0)
            
            # Draw bounding box
            if player.bbox:
                x, y, w, h = [int(v) for v in player.bbox]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw player ID and team
                label = f"P{player.player_id} T{player.team_id}"
                cv2.putText(display_frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw trajectory
            if len(player.position_history) > 1:
                points = [p for p in player.position_history if p is not None]
                if len(points) > 1:
                    points = np.array(points, dtype=np.int32)
                    cv2.polylines(display_frame, [points], False, color, 2)
        
        # Draw stats
        active = sum([1 for p in self.players if p.is_active])
        lost = sum([1 for p in self.players if not p.is_active])
        cv2.putText(display_frame, f"Active: {active} | Lost: {lost} | SAM 2", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame
    
    def track_video(self, video_path, output_path=None):
        """Main tracking pipeline using SAM 2"""
        # Read first frame for player selection
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        cap.release()
        
        # Select players on first frame
        self.select_players(first_frame)
        
        if len(self.points) == 0:
            print("No players selected!")
            return
        
        # Initialize SAM 2 tracking
        self.initialize_sam2_tracking(video_path, frame_idx=0)
        
        # Propagate masks through entire video
        self.propagate_masks()
        
        # Now display results
        cap = cv2.VideoCapture(video_path)
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("\nDisplaying tracking results... Press 'q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update player information
            self.update_players(frame_idx, frame)
            
            # Draw results
            display_frame = self.draw_tracking(frame)
            
            # Show progress
            cv2.putText(display_frame, f"Frame: {frame_idx + 1}/{total_frames}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('SAM 2 Team-Aware Tracking', display_frame)
            
            if writer:
                writer.write(display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nDisplay stopped by user")
                break
            
            frame_idx += 1
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
            print(f"\nOutput saved to: {output_path}")
        cv2.destroyAllWindows()
        
        print(f"Tracking complete! Processed {frame_idx} frames")


def main():
    # Configuration
    SAM2_CHECKPOINT = r"../models/sam2.1_hiera_large.pt"
    
    MODEL_CFG = "D:\Campus\CSE Life\S7\Advanced Machine Learning\Research Paper\Segment Anything\segment-anything-2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml"  # Try 2.1 version first
    
    # If 2.1 doesn't work, try original SAM 2:
    # MODEL_CFG = "sam2_hiera_l"
    
    VIDEO_PATH = r"..\clips\football\0bfacc_0.mp4"
    OUTPUT_PATH = r"../videos/output_sam2_team_tracking.mp4"
    
    if not SAM2_AVAILABLE:
        print("ERROR: SAM 2 not available!")
        return
    
    # Check if checkpoint exists
    if not os.path.exists(SAM2_CHECKPOINT):
        print(f"Error: SAM 2 checkpoint not found at {SAM2_CHECKPOINT}")
        return
    
    print("=== SAM 2 Configuration ===")
    print(f"Checkpoint: {SAM2_CHECKPOINT}")
    print(f"Config: {MODEL_CFG}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("===========================\n")
    
    # Initialize and run
    try:
        tracker = TeamAwareTrackerSAM2(SAM2_CHECKPOINT, MODEL_CFG)
        tracker.track_video(VIDEO_PATH, OUTPUT_PATH)
    except Exception as e:
        print("Failed to run SAM 2 tracker")
        import traceback
        traceback.print_exc()
    
    # Initialize and run
    tracker = TeamAwareTrackerSAM2(SAM2_CHECKPOINT, MODEL_CFG)
    tracker.track_video(VIDEO_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()