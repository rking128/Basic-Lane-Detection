# Basic-Lane-Detection
This project implements a basic lane detection algorithm using Python and OpenCV to process dashcam or road footage. It identifies lane lines based on color and orientation, filtering out noise and drawing clear, visual overlays of the detected driving path. This was developed as part of coursework for ENME435 - Remote Sensing at the University of Maryland, College Park.

#Features
  Detects white and yellow lane lines using color filtering in HSV space
  Uses Canny edge detection and Hough Transform for line identification
  Converts polar coordinates (ρ, θ) to cartesian lines for drawing
  Filters out horizontal or irrelevant lines using angle thresholds

#Draws:
  True lane lines in blue (left) and red (right)
  A semi-transparent overlay indicating the possible road area

#Goals
  Provide a basic, visually-interpretable representation of lane lines
  Handle video frames where no lines are detected gracefully (i.e., no crashing)
  Offer a fallback visual in the case of overlay rendering issues

#How It Works
  Color Filtering: Extracts yellow and white lane lines from the image using specfifc HSV mask
  Edge Detection: Canny is used to highlight edges within the filtered regions
  Line Detection: Hough Transform is applied to detect straight lines
  Filtering & Drawing:  Angles are filtered to ensure only realistic lane lines are drawn; Left and right lines are                                 separated based on angle
                        A polygon is drawn between the endpoints of valid lane lines
                        Overlay Creation: A transparent region highlights the road ahead.

#Files
  Basic_Lane_Detection.py – main script with all functions and frame processing
  README.md – this file
  videos/ – test_video.mp4
            test_video_02.mp4
            30.mp4

#Requirements
  Python 3.8+
  OpenCV (cv2)
  NumPy

#How to Run
  python Basic_Lane_Detection.py

Note: You can change the input video path or output options directly in the script

#Sample Output
![image](https://github.com/user-attachments/assets/24ece44d-806b-44eb-822c-0ebceeccc26d)
