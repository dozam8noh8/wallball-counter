import cv2
import torch
import numpy as np
import argparse
from yolov5.utils.torch_utils import select_device

"""
This counter doesn't work well because the 
Yolov5 model doesnt do so well at detecting the ball.
A more pronounced ball or better model would help.
"""



# Parse command-line arguments
parser = argparse.ArgumentParser(description='Count wallball reps from a video file.')
parser.add_argument('video_path', type=str, help='Path to the input video file (e.g., .mp4, .mov)')
args = parser.parse_args()

video_path = args.video_path
cap = cv2.VideoCapture(video_path)

# Load YOLOv5 model (use yolov5m for better accuracy)
device = select_device('')
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.conf = 0.2  # lower confidence threshold

ball_positions = []
positions_history = []  # Will now store (frame_index, cy)
reps = 0
frame_count = 0

# Rep detection parameters
MIN_FRAMES_BETWEEN_REPS = 10  # Only count a new rep if at least 10 frames since last
MIN_HEIGHT_DIFF = 5           # Lowered for more sensitivity
last_rep_frame = -MIN_FRAMES_BETWEEN_REPS

# Tracker variables
tracker = None
tracking = False
track_fail_count = 0
MAX_TRACK_FAIL = 5
last_bbox = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only process every 5th frame
    if frame_count % 3 != 0:
        continue

    # Rotate frame if needed (if width > height)
    if frame.shape[1] > frame.shape[0]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Inference with higher input size
    results = model(frame, size=960)
    detections = results.xyxy[0].cpu().numpy()

    # Find the ball (class 32 is 'sports ball' in COCO)
    balls = [det for det in detections if int(det[5]) == 32]
    if balls:
        print(f"Ball detected at frame {frame_count}")
        # Take the largest ball (in case of multiple)
        ball = max(balls, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
        x1, y1, x2, y2, conf, cls = ball
        h, w = frame.shape[:2]
        box_w = int(x2 - x1)
        box_h = int(y2 - y1)
        # Check bbox validity
        if (
            box_w > 0 and box_h > 0 and
            0 <= int(x1) < w and 0 <= int(y1) < h and
            int(x1) + box_w <= w and int(y1) + box_h <= h
        ):
            bbox = (int(x1), int(y1), box_w, box_h)
            tracker = cv2.TrackerCSRT_create()
            tracking = tracker.init(frame, bbox)
            last_bbox = bbox
            track_fail_count = 0
        else:
            tracker = None
            tracking = False
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        ball_positions.append(cy)
        positions_history.append((frame_count, cy))
        # Only keep the last 3 detected positions
        if len(positions_history) > 3:
            positions_history.pop(0)
        # Rep detection with gaps allowed
        if len(positions_history) == 3:
            f0, y0 = positions_history[0]
            f1, y1 = positions_history[1]
            f2, y2 = positions_history[2]
            if (
                y1 < y0 and y1 < y2 and
                (y0 - y1 > MIN_HEIGHT_DIFF) and (y2 - y1 > MIN_HEIGHT_DIFF) and
                (f1 - last_rep_frame >= MIN_FRAMES_BETWEEN_REPS)
            ):
                reps += 1
                last_rep_frame = f1
                print(f"Rep {reps} at frame {f1}")
        # Draw detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
    else:
        ball_positions.append(None)
        # Use tracker if available
        if tracker is not None and tracking:
            ok, bbox = tracker.update(frame)
            if ok:
                x, y, box_w, box_h = [int(v) for v in bbox]
                h, w = frame.shape[:2]
                # Check bbox validity for tracked box
                if (
                    box_w > 0 and box_h > 0 and
                    0 <= x < w and 0 <= y < h and
                    x + box_w <= w and y + box_h <= h
                ):
                    cx = int(x + box_w/2)
                    cy = int(y + box_h/2)
                    print(f"Ball tracked at frame {frame_count}")
                    positions_history.append((frame_count, cy))
                    if len(positions_history) > 3:
                        positions_history.pop(0)
                    if len(positions_history) == 3:
                        f0, y0 = positions_history[0]
                        f1, y1 = positions_history[1]
                        f2, y2 = positions_history[2]
                        if (
                            y1 < y0 and y1 < y2 and
                            (y0 - y1 > MIN_HEIGHT_DIFF) and (y2 - y1 > MIN_HEIGHT_DIFF) and
                            (f1 - last_rep_frame >= MIN_FRAMES_BETWEEN_REPS)
                        ):
                            reps += 1
                            last_rep_frame = f1
                            print(f"Rep {reps} at frame {f1}")
                    # Draw tracked box
                    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    last_bbox = bbox
                    track_fail_count = 0
                else:
                    tracker = None
                    tracking = False
            else:
                track_fail_count += 1
                if track_fail_count > MAX_TRACK_FAIL:
                    tracker = None
                    tracking = False
        if frame_count % 10 == 0:
            print(f"Did not detect rep (frame {frame_count})")

    # Display rep count on frame
    cv2.putText(frame, f"Reps: {reps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Wallball Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()