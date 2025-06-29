# wallball-counter
Exploring different motion detection methods to count crossfit wallballs

## 1. Object detection
The limitation is the model that detects the object (ball). I used Yolov5 small and medium models and it didn't detect the ball in every frame, leading to poor counting.
This method tries to find a local minima (the frame between two other frames where the balls y position was at its lowest) - Because we're hitting the bottom and then bouncing back up.
## 2. Region of interest detection
This approach sets a region of interest (ROI) manually by the user - i.e. where the ball is expected to hit the wall. The pixels of the region of interest will change and be "covered" and "uncovered" when the ball crosses onto and off the ROI. This counts a rep.

## 3. Cycle detection with smoothing and peak detection
This method converts the whole image to a bunch of greyscale pixels and then detects the changes between the pixels, plotting them on a graph. Because a wallball is cyclical and the camera doesnt move, the plot on the graph becomes a wave. The wave is processed to determine peaks, with a minimum amount of frames required between peaks. This allows us to detect reps.
