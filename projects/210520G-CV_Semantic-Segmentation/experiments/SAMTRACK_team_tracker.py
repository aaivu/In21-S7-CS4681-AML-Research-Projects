import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from collections import deque
import sys
import os

# Note: You'll need to clone Segment-and-Track-Anything repository
# Installation:
# git clone https://github.com/z-x-yang/Segment-and-Track-Anything.git
# cd Segment-and-Track-Anything
# bash script/install.sh
# bash script/download_ckpt.sh

import sys
# Add the Segment-and-Track-Anything path
SAM_TRACK_PATH = "../Segment-and-Track-Anything"
sys.path.append(SAM_TRACK_PATH)

try:
    from model_args import segtracker_args, sam_args, aot_args
    from SegTracker import SegTracker
    SAMTRACK_AVAILABLE = True
except ImportError:
    print("Warning: SAM-Track not found.")
    print("Install from: https://github.com/z-x-yang/Segment-and-Track-Anything")
    SAMTRACK_AVAILABLE = False


class PlayerTrackerSAMTrack:
    """Individual player tracker using SAM + AOT"""
    def __init__(self, player_id, mask, frame, team_id=None, tracker_id=None):
        self.player_id = player_id
        self.team_id = team_id
        self.tracker_id = tracker_id  # AOT internal tracking ID
        self.mask = mask
        self.bbox = self.mask_to_bbox(mask)
        self.is_active = True
        self.lost_frames = 0
        self.confidence = 1.0
        
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
        if mask is None or not mask.any():
            return np.zeros(bins * 2)
            
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


class TeamAwareTrackerSAMTrack:
    """Main tracking system using SAM-Track with team awareness"""
    def __init__(self, sam_checkpoint="./Segment-and-Track-Anything/ckpt/sam_vit_b_01ec64.pth",
                 aot_checkpoint="./Segment-and-Track-Anything/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth"):
        
        if not SAMTRACK_AVAILABLE:
            print("ERROR: SAM-Track not available!")
            self.segtracker = None
            return
            
        print("Loading SAM-Track...")
        
        # Initialize SegTracker with default args
        self.segtracker = SegTracker(segtracker_args, sam_args, aot_args)
        self.segtracker.restart_tracker()
        
        print(f"SAM-Track loaded successfully")
        
        self.players = []
        self.next_player_id = 0
        self.next_tracker_id = 1
        self.frame = None
        self.points = []
        self.current_team = 0
        self.team_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        
        self.frame_idx = 0
    
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
    
    def initialize_tracking(self, first_frame):
        """Initialize SAM-Track segmentation and tracking"""
        if not SAMTRACK_AVAILABLE or self.segtracker is None:
            print("ERROR: Cannot initialize tracking without SAM-Track!")
            return False
        
        print(f"\nInitializing SAM-Track with {len(self.points)} players...")
        
        # Process each selected point
        for point_data in self.points:
            point = point_data['pos']
            team = point_data['team']
            
            # Add point to SegTracker
            predicted_mask, _ = self.segtracker.seg_acc_click(
                origin_frame=first_frame,
                coords=np.array([point], dtype=np.float32),
                modes=np.array([1], dtype=np.int32),  # 1 for positive point
                multimask=True
            )
            
            # Create player tracker
            player = PlayerTrackerSAMTrack(
                player_id=self.next_player_id,
                mask=predicted_mask,
                frame=first_frame,
                team_id=team,
                tracker_id=self.next_tracker_id
            )
            
            self.players.append(player)
            print(f"Player {self.next_player_id} (Team {team}): bbox={player.bbox}")
            
            self.next_player_id += 1
            self.next_tracker_id += 1
        
        print(f"Successfully initialized {len(self.players)} players\n")
        return True
    
    def track_frame(self, frame):
        """Track all players in current frame using SAM-Track"""
        if not SAMTRACK_AVAILABLE or self.segtracker is None:
            return False
        
        # Track with SegTracker
        pred_mask = self.segtracker.track(frame, update_memory=True)
        
        # Update each player based on mask
        all_obj_ids = np.unique(pred_mask)
        all_obj_ids = all_obj_ids[all_obj_ids != 0]  # Remove background
        
        # Map object IDs to players
        for player in self.players:
            # Find if this player's ID exists in current frame
            if player.tracker_id in all_obj_ids:
                player_mask = (pred_mask == player.tracker_id).astype(np.uint8)
                
                if player_mask.any():
                    player.mask = player_mask
                    player.bbox = player.mask_to_bbox(player_mask)
                    player.is_active = True
                    player.lost_frames = 0
                    
                    if player.bbox:
                        center = player.get_center(player.bbox)
                        if center:
                            player.position_history.append(center)
                    
                    # Update appearance
                    player.update_appearance(frame, player_mask)
                    player.confidence = 1.0
                else:
                    player.is_active = False
                    player.lost_frames += 1
                    player.confidence = 0.0
            else:
                player.is_active = False
                player.lost_frames += 1
                player.confidence = 0.0
        
        self.frame_idx += 1
        return True
    
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
            mask_overlay[player.mask > 0] = color
            display_frame = cv2.addWeighted(display_frame, 1.0, mask_overlay, 0.3, 0)
            
            # Draw bounding box
            if player.bbox:
                x, y, w, h = [int(v) for v in player.bbox]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw player ID, team, and confidence
                label = f"P{player.player_id} T{player.team_id} ({player.confidence:.2f})"
                cv2.putText(display_frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory
            if len(player.position_history) > 1:
                points = [p for p in player.position_history if p is not None]
                if len(points) > 1:
                    points = np.array(points, dtype=np.int32)
                    cv2.polylines(display_frame, [points], False, color, 2)
        
        # Draw stats
        active = sum([1 for p in self.players if p.is_active])
        lost = sum([1 for p in self.players if not p.is_active])
        cv2.putText(display_frame, f"Active: {active} | Lost: {lost} | SAM-Track", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame
    
    def track_video(self, video_path, output_path=None):
        """Main tracking pipeline using SAM-Track"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Read first frame
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        
        # Select players
        self.select_players(first_frame)
        
        if len(self.points) == 0:
            print("No players selected!")
            return
        
        # Initialize tracking
        if not self.initialize_tracking(first_frame):
            print("Failed to initialize tracking!")
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
        
        # Draw first frame
        display_frame = self.draw_tracking(first_frame)
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('SAM-Track Team-Aware Tracking', display_frame)
        if writer:
            writer.write(display_frame)
        
        print("\nTracking started... Press 'q' to quit\n")
        
        # Main tracking loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Track frame
            self.track_frame(frame)
            
            # Draw results
            display_frame = self.draw_tracking(frame)
            
            # Show progress
            cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('SAM-Track Team-Aware Tracking', display_frame)
            
            if writer:
                writer.write(display_frame)
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                active = sum([1 for p in self.players if p.is_active])
                print(f"Frame {frame_count}/{total_frames} - Active: {active}/{len(self.players)}")
            
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


def main():
    # Configuration - paths relative to Segment-and-Track-Anything installation
    SAM_CHECKPOINT = "./Segment-and-Track-Anything/ckpt/sam_vit_b_01ec64.pth"
    AOT_CHECKPOINT = "./Segment-and-Track-Anything/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth"
    VIDEO_PATH = "football_match.mp4"
    OUTPUT_PATH = "output_samtrack_team_tracking.mp4"
    
    # Check if SAM-Track is available
    if not SAMTRACK_AVAILABLE:
        print("\nERROR: SAM-Track not installed!")
        print("\nInstallation steps:")
        print("1. git clone https://github.com/z-x-yang/Segment-and-Track-Anything.git")
        print("2. cd Segment-and-Track-Anything")
        print("3. bash script/install.sh")
        print("4. bash script/download_ckpt.sh")
        print("\nThen run this script from the parent directory of Segment-and-Track-Anything")
        return
    
    # Initialize and run
    tracker = TeamAwareTrackerSAMTrack(SAM_CHECKPOINT, AOT_CHECKPOINT)
    if tracker.segtracker is None:
        return
    tracker.track_video(VIDEO_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()