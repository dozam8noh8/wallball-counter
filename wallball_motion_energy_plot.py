import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Live plot motion energy signal from a video file.')
parser.add_argument('video_path', type=str, help='Path to the input video file (e.g., .mp4, .mov)')
args = parser.parse_args()

video_path = args.video_path
cap = cv2.VideoCapture(video_path)

motion_energies = []
prev_gray = None
frame_count = 0

# Live plotting setup
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
line, = ax.plot([], [])
ax.set_title('Motion Energy Signal (Live)')
ax.set_xlabel('Frame')
ax.set_ylabel('Motion Energy (sum of pixel differences)')

# Peak detection parameters
buffer_size = 7  # Number of frames to look for a peak (centered)
motion_buffer = []
rep_count = 0
last_rep_frame = -10  # To avoid double-counting
min_frames_between_reps = 30
peak_threshold_factor = 0.65  # Fraction of max motion energy to consider a valid peak

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 5 != 0:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        motion_energy = np.sum(diff)
        motion_energies.append(motion_energy)
        motion_buffer.append(motion_energy)
        if len(motion_buffer) > buffer_size:
            motion_buffer.pop(0)
        # Peak detection: check if the center of the buffer is a local maximum
        if len(motion_buffer) == buffer_size:
            center = buffer_size // 2
            center_val = motion_buffer[center]
            # Dynamic threshold: at least a fraction of the max seen so far
            dynamic_threshold = peak_threshold_factor * (np.max(motion_energies) if motion_energies else 1)
            if (
                center_val == max(motion_buffer) and
                center_val > dynamic_threshold and
                (frame_count - min_frames_between_reps) > last_rep_frame
            ):
                rep_count += 1
                last_rep_frame = frame_count
                print(f"Rep {rep_count} at frame {frame_count - (buffer_size - center - 1)} (motion energy: {center_val})")
    prev_gray = gray

    # Update live plot
    line.set_data(range(len(motion_energies)), motion_energies)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)

    # Overlay rep count on video
    cv2.putText(frame, f'Reps: {rep_count}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
    cv2.imshow('Motion Energy Rep Counter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
plt.ioff()
plt.show()
cv2.destroyAllWindows()

# Optional: smooth the signal with a moving average
window_size = 5
smoothed = np.convolve(motion_energies, np.ones(window_size)/window_size, mode='valid')

# Find peaks (reps)
peaks, _ = find_peaks(smoothed, distance=10)  # distance=10: minimum frames between reps, tune as needed

print(f"Detected {len(peaks)} reps.")
# Optionally, plot the peaks
plt.figure(figsize=(12, 6))
plt.plot(smoothed, label='Smoothed Motion Energy')
plt.plot(peaks, smoothed[peaks], 'rx', label='Detected Reps')
plt.legend()
plt.title('Motion Energy with Detected Reps')
plt.xlabel('Frame')
plt.ylabel('Motion Energy')
plt.show() 