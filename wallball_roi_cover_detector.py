import cv2
import numpy as np
import argparse

"""
This wallball counter works well in the video I tested it on but 
it is specifically configured for a wall with a bright patch and a dark ball.
It works by selecting a Region of Interest (ROI) and then detecting when it is covered and uncovered.
It requires a minimum number of frames between reps to avoid counting the same rep multiple times.
This means lower frame rate devices may have to adjust the MIN_FRAMES_BETWEEN_REPS parameter.
NOTE: IT will not work if the first frame already has the ROI covered.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Detect when a region is covered in a video (e.g., by a ball).')
parser.add_argument('video_path', type=str, help='Path to the input video file (e.g., .mp4, .mov)')
args = parser.parse_args()

video_path = args.video_path
cap = cv2.VideoCapture(video_path)

ret, first_frame = cap.read()
if not ret:
    print('Could not read first frame from video.')
    exit(1)

# Resize for ROI selection
scale = 0.3  # 30% of original size
small_frame = cv2.resize(first_frame, (0, 0), fx=scale, fy=scale)
roi_small = cv2.selectROI('Select ROI', small_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select ROI')
# Scale ROI back to original size
x, y, w, h = [int(coord / scale) for coord in roi_small]
print(f'Selected ROI: x={x}, y={y}, w={w}, h={h}')

# Rewind video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Get baseline (uncovered) mean intensity from the first frame
first_roi = first_frame[y:y+h, x:x+w]
first_mean = np.mean(cv2.cvtColor(first_roi, cv2.COLOR_BGR2GRAY))
print(f'Baseline (uncovered) mean intensity: {first_mean:.2f}')

# The threshold to detect if the selected spot is covered.
COVERED_THRESHOLD = first_mean - 12

covered = False
frame_count = 0
rep_count = 0
previous_rep_frame = None

# Get FPS from video
fps = cap.get(cv2.CAP_PROP_FPS)
desired_interval_sec = 0.15  # process every 0.15 seconds
frame_skip = max(1, int(round(fps * desired_interval_sec)))
MIN_FRAMES_BETWEEN_REPS = fps * 1.25 # 1.25 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # Skip frames to speed up processing
    if frame_count % frame_skip != 0:
        continue
    roi_frame = frame[y:y+h, x:x+w]
    mean_intensity = np.mean(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY))
    print(f'Mean intensity: {mean_intensity:.2f}')

    # Detect cover/uncover transitions
    if not covered and mean_intensity < COVERED_THRESHOLD:
        print(f'ROI covered at frame {frame_count} (mean intensity: {mean_intensity:.2f})')
        covered = True
    elif covered and mean_intensity >= COVERED_THRESHOLD:
        print(f'ROI uncovered at frame {frame_count} (mean intensity: {mean_intensity:.2f})')
        covered = False
        if previous_rep_frame is None or frame_count - previous_rep_frame > MIN_FRAMES_BETWEEN_REPS:
            rep_count += 1
            previous_rep_frame = frame_count
            print(f'Rep {rep_count} at frame {frame_count}')

    # Optional: show the ROI on the frame
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Add big overlay for rep count in top left
    cv2.putText(frame, f'Reps: {rep_count}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
    cv2.imshow('ROI Cover Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
print(f'Total reps: {rep_count}')