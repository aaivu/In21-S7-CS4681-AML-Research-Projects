import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import sys

class SAMTracker:
    def __init__(self, sam_checkpoint, model_type="vit_b"):
        """Initialize SAM model"""
        print("Loading SAM model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print(f"SAM loaded on {self.device}")
        
        # Variables for user interaction
        self.points = []
        self.labels = []
        self.drawing = False
        self.bbox = None
        self.frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: positive point (object)
            self.points.append([x, y])
            self.labels.append(1)
            print(f"Added positive point: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: negative point (background)
            self.points.append([x, y])
            self.labels.append(0)
            print(f"Added negative point: ({x}, {y})")
            
        # Redraw frame with points
        if len(self.points) > 0:
            self.draw_points()
    
    def draw_points(self):
        """Draw points on the frame"""
        temp_frame = self.frame.copy()
        for i, (point, label) in enumerate(zip(self.points, self.labels)):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(temp_frame, tuple(point), 5, color, -1)
            cv2.putText(temp_frame, str(i+1), (point[0]+10, point[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Select Object', temp_frame)
    
    def get_sam_segmentation(self, frame):
        """Interactive UI to get object segmentation from SAM"""
        self.frame = frame.copy()
        self.points = []
        self.labels = []
        
        cv2.namedWindow('Select Object')
        cv2.setMouseCallback('Select Object', self.mouse_callback)
        
        print("\n=== Object Selection ===")
        print("Left Click: Add positive point (on object)")
        print("Right Click: Add negative point (on background)")
        print("Press SPACE/ENTER: Confirm selection")
        print("Press 'r': Reset points")
        print("Press 'q': Quit")
        print("========================\n")
        
        cv2.imshow('Select Object', self.frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') or key == 13:  # SPACE or ENTER
                if len(self.points) > 0:
                    break
                else:
                    print("Please select at least one point!")
                    
            elif key == ord('r'):  # Reset
                self.points = []
                self.labels = []
                print("Points reset")
                cv2.imshow('Select Object', self.frame)
                
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                sys.exit(0)
        
        # Generate mask using SAM
        print("Generating segmentation...")
        self.predictor.set_image(frame)
        
        input_points = np.array(self.points)
        input_labels = np.array(self.labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        # Select the best mask (highest score)
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        
        print(f"Segmentation complete (score: {scores[best_mask_idx]:.3f})")
        
        # Show the mask
        mask_overlay = self.frame.copy()
        mask_overlay[mask] = mask_overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.imshow('Select Object', mask_overlay.astype(np.uint8))
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        return mask
    
    def mask_to_bbox(self, mask):
        """Convert binary mask to bounding box"""
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)
    
    def track_video(self, video_path, tracker_type='CSRT', output_path=None):
        """Main tracking pipeline"""
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
        
        # Get segmentation mask from SAM
        mask = self.get_sam_segmentation(frame)
        
        # Convert mask to bounding box
        bbox = self.mask_to_bbox(mask)
        if bbox is None:
            print("Error: Could not extract bounding box from mask")
            return
        
        print(f"Initial bbox: {bbox}")
        
        # Initialize tracker (compatible with different OpenCV versions)
        try:
            # Try new API first (OpenCV 4.5.1+)
            if tracker_type == 'CSRT':
                tracker = cv2.legacy.TrackerCSRT_create()
            elif tracker_type == 'KCF':
                tracker = cv2.legacy.TrackerKCF_create()
            elif tracker_type == 'MOSSE':
                tracker = cv2.legacy.TrackerMOSSE_create()
            else:
                print(f"Unknown tracker type: {tracker_type}")
                print(f"Available: CSRT, KCF, MOSSE")
                return
        except AttributeError:
            # Fallback to old API (OpenCV < 4.5.1)
            if tracker_type == 'CSRT':
                tracker = cv2.TrackerCSRT_create()
            elif tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            elif tracker_type == 'MOSSE':
                tracker = cv2.TrackerMOSSE_create()
            else:
                print(f"Unknown tracker type: {tracker_type}")
                print(f"Available: CSRT, KCF, MOSSE")
                return
        tracker.init(frame, bbox)
        print(f"Initialized {tracker_type} tracker")
        
        # Setup video writer if output path is provided
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
            
            # Update tracker
            success, bbox = tracker.update(frame)
            
            # Draw results
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{tracker_type} Tracker", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking failure", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show progress
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (50, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Tracking', frame)
            
            # Write frame if output is specified
            if writer:
                writer.write(frame)
            
            # Check for quit
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
    # Configuration
    # SAM_CHECKPOINT = "../models/sam_vit_b_01ec64.pth"
    SAM_CHECKPOINT = "../models/sam_vit_h_4b8939.pth"
    MODEL_TYPE = "vit_h"
    VIDEO_PATH = r"..\clips\football\0bfacc_0.mp4"
    OUTPUT_PATH = "../videos/output_tracked.mp4"  # Optional: set to None to disable saving
    TRACKER_TYPE = "CSRT"  # Options: CSRT, KCF, MOSSE
    
    # Initialize and run
    sam_tracker = SAMTracker(SAM_CHECKPOINT, model_type=MODEL_TYPE)
    sam_tracker.track_video(VIDEO_PATH, TRACKER_TYPE, OUTPUT_PATH)


if __name__ == "__main__":
    main()