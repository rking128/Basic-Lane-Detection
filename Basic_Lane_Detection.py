# Basic Lane Detection Algorithm_v1

import numpy as np
import cv2

# fxn that takes the image in that frame, and snips and masks just the lane lines (usable for both yellow and white lanes)
    # Takes the image thats meant to be processed
def clean(img):

    # Isolating Lane segment of the frame ; using a snippiet of the frame to limit processing noise
    snip = img[(img.shape[0]-300):(img.shape[0]-50),(0):(img.shape[1])]
    mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8") 
    
    # Marking trapazoid of points where lanes should be located
    pts = np.array([[80,snip.shape[0]-25],
                    [80,snip.shape[0]-35], 
                    [int(snip.shape[1]/2)-140, 165], 
                    [int(snip.shape[1]/2)+145, 165], 
                    [snip.shape[1]-80,snip.shape[0]-35], 
                    [snip.shape[1]-80,snip.shape[0]-25]], 
                    dtype=np.int32)  

    cv2.fillConvexPoly(mask,pts,255)    
    mask = cv2.bitwise_and(snip,snip,mask=mask)

    return mask

#fxn that takes the image in that frame and thresholds it in HSV, blurs it, and identifies the egdes
    # Takes the image to be thresholded and the name of the video relating to it's specfic HSV mask
def edge_process(img, str):

    # Convert to HSV Space and threshold image for lane lines; Used 'colorpicker.py' to identify HSV range for lanes (based on 3 video stills)
    img_hsv1 = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2HSV)
    img_hsv2 = img_hsv1.copy()

    # Based on video name, selects HSV mask (up to two versions for white and yellow respecfully)
    if str == "test_video.mp4":
    
        white_lane_Lower = np.array([0,0,56])	
        white_lane_Upper = np.array([255,100,255])

        white_lane_Lower2 = np.array([0,0,50])	
        white_lane_Upper2 = np.array([255,255,255])

        white_lane_mask = cv2.inRange(img_hsv1, white_lane_Lower, white_lane_Upper)
        white_lane_mask2 = cv2.inRange(img_hsv1, white_lane_Lower2, white_lane_Upper2)
        yellow_lane_mask= white_lane_mask
        yellow_lane_mask2 = white_lane_mask2      
    elif str == "30.mp4":

        white_lane_Lower = np.array([14,15,87])	
        white_lane_Upper = np.array([43,91,140]) 
        
        white_lane_Lower2 = np.array([22,22,174])	
        white_lane_Upper2 = np.array([255,60,255]) 

        yellow_lane_Lower = np.array([12,59,192])	   
        yellow_lane_Upper = np.array([255,135,225]) 
        
        yellow_lane_Lower2 = np.array([12,105,125])	   
        yellow_lane_Upper2 = np.array([255,150,225])

        white_lane_mask = cv2.inRange(img_hsv1, white_lane_Lower, white_lane_Upper)
        white_lane_mask2 = cv2.inRange(img_hsv1, white_lane_Lower2, white_lane_Upper2) 
        yellow_lane_mask = cv2.inRange(img_hsv2, yellow_lane_Lower, yellow_lane_Upper)  
        yellow_lane_mask2 = cv2.inRange(img_hsv2, yellow_lane_Lower2, yellow_lane_Upper2)

        #yellow_lane_mask2 = yellow_lane_mask
        #white_lane_mask2 = white_lane_mask''
    elif str == "test_video_02.mp4":

        white_lane_Lower = np.array([12,45,145])	
        white_lane_Upper = np.array([255,85,255])
        
        white_lane_Lower2 = np.array([17,50,130])	
        white_lane_Upper2 = np.array([255,105,255])

        white_lane_mask = cv2.inRange(img_hsv1, white_lane_Lower, white_lane_Upper) 
        white_lane_mask2 = cv2.inRange(img_hsv1, white_lane_Lower2, white_lane_Upper2)
        yellow_lane_mask = white_lane_mask
        yellow_lane_mask2 = white_lane_mask2
    else:  
        print('unknown video name error, add the video name and and HSV mask for it in the "clean()" fxn')

    combined_white_mask = cv2.bitwise_or(white_lane_mask, white_lane_mask2)
    combined_yellow_mask = cv2.bitwise_or(yellow_lane_mask, yellow_lane_mask2)
    
    white_mask = cv2.bitwise_and(img_hsv1, img_hsv1, mask = combined_white_mask)
    yellow_mask = cv2.bitwise_and(img_hsv2, img_hsv2, mask = combined_yellow_mask)

    # Identify edge cases:  using a blur as a low pass filter to ignore noise
    img_blur_white = cv2.GaussianBlur(white_mask,(17,17),0) 
    img_blur_yellow = cv2.GaussianBlur(yellow_mask,(17,17),0)

    # using Canny() fxn to compare pixels in blurred thresholded image and identify steep transitions (possible edge points) -- based on min and max values
    edge_max = 70
    edge_min = 55
    img_edges_white = cv2.Canny(img_blur_white, edge_min, edge_max) 
    img_edges_yellow = cv2.Canny(img_blur_yellow,edge_min, edge_max)

    return img_edges_white, img_edges_yellow

# a fxn that takes identifed edges in the image and uses Hough Transforms to identifies lines in the image
    # Takes the image with possible edge points identified
def line_process(img):

    # an openCV fxn that using properties of Hough space finds a line at best fits the edge points in the img
    img_hough = cv2.HoughLines(img, 1, np.pi/180, 30)

    return img_hough

# a fxn used to troublshoot Hough transform possible lane lines -- converts possible hough line segments into lines
    # To use -- uncomment neccessary components
def hough(img_hough, img):
    
    img_test = img.copy()

    # snip corrections for plotting Hough Lines:
    y_offset = img_test.shape[0] - 300
    x_offset = 0

    if img_hough is not None:   # avoid error for no lines detected
        for lines in img_hough:
            rho, theta = lines[0]   # properly unpack HoughLines
        
            # Convert polar (rho, theta) to Cartesian (x1, y1, x2, y2)
            cosine = np.cos(theta)
            sine = np.sin(theta)
            x0 = cosine * rho
            y0 = sine * rho
                
            x1 = int(x0 + 1000 * (-sine)) + x_offset
            y1 = int(y0 + 1000 * (cosine)) + y_offset
            x2 = int(x0 - 1000 * (-sine)) + x_offset
            y2 = int(y0 - 1000 * (cosine)) + y_offset
                
            cv2.line(img_test, (x1,y1), (x2,y2), (255,0,0), 2)

    return img_test

# a fxn to convert from polar coordinates in Hough transform to cartesian x-y space
    # Takes the hough transform array with polar coordinates and the img they'll be drawn on
def polar_to_cart(array, img):

    # snip corrections for plotting Hough Lines:
    y_offset = img.shape[0] - 300
    x_offset = 0

    coordinates = []
    line_length = 1000

    if array is not None and array.size > 0:   # avoid error for no lines detected or empty arrays
        for rho,theta in array:

            # last check for undesired false postive true lines
            if np.pi / 3 <= theta <= 2 * np.pi / 3:
             continue

            # to move past any "NaN" values that would get past "is not None"
            if np.isnan(rho) and np.isnan(theta):  
                continue    #skip that value

            # Convert polar (rho, theta) to Cartesian (x1, y1, x2, y2)
            cosine = np.cos(theta)
            sine = np.sin(theta)
            x0 = cosine * rho
            y0 = sine * rho
                
            x1 = int(x0 + line_length * (-sine)) + x_offset
            y1 = int(y0 + line_length * (cosine)) + y_offset
            x2 = int(x0 - line_length * (-sine)) + x_offset
            y2 = int(y0 - line_length * (cosine)) + y_offset

            new_line = [x1, y1, x2, y2]
            coordinates.append(new_line)  

    return coordinates

# a fxn that draws the 'true lane lines' on the frame as well as the 'possible road' display 
    # Takes the left/right sorted lane line arrays and the image they'll be drawn on
def overlay(left, right, img): 

    img_lined = img.copy()
    img_overlayed = img.copy()

    left_start = left_end = right_start = right_end = None  # assigned a value in case no lines were dtected

    # Drawing lanes on video frame:
    coordinates_left = polar_to_cart(left, img_lined)
    for x1, y1, x2, y2 in coordinates_left:
        cv2.line(img_lined, (x1,y1), (x2,y2), (255,0,0), 7)
        left_start = [x1,y1]
        left_end = [x2,y2]

    coordinates_right = polar_to_cart(right, img_lined)
    for x1, y1, x2, y2 in coordinates_right:
        cv2.line(img_lined, (x1,y1), (x2,y2), (0,0,255), 7)
        right_start = [x1,y1]
        right_end = [x2,y2]

    # Drawing possible road path on frame:
    alpha = 0.3

    if None not in [left_start, left_end, right_start, right_end]:
        pts = np.array([left_start, left_end, right_end, right_start], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))  # reshaping points into acceptable array for polylines()
        cv2.polylines(img_lined, pts, isClosed=True, color=(255, 255, 0), thickness=3)
    else:
        print("None was detected in one of the lane line start/end points --- so the lane overlay was skipped")
    
    cv2.addWeighted(img_lined, alpha, img_overlayed, 1 - alpha, 0, img_overlayed)

    return img_overlayed

# Main Algorith:
def main():

    # Accessing video information and frames:
    name = "test_video_02.mp4"
    video = cv2.VideoCapture(name)   
    fps = int(video.get(cv2.CAP_PROP_FPS))  
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Define how video is stored/played (codec) and video out file 
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")    # codec for an mp4
    output_video = cv2.VideoWriter("Lane_Detcetion_Results.mp4", fourcc, fps, (frame_w, frame_h)) 

    while video.isOpened(): 
        # uses the read() fxn to read the each frame in the video
            # here "ret" is the boolian that holds the answer to wether or not there is a "next" frame
        ret, frame = video.read()
        if not ret: 
            break

        # Editing the images in the video frame by frame:
        lane_display = frame.copy()
        
        image = clean(frame)   
        lane_edge_white, lane_edge_yellow = edge_process(image, name) 
        lane_lines_white = line_process(lane_edge_white)
        lane_lines_yellow = line_process(lane_edge_yellow)

         # Debugging Hough Transform Lines:
        '''
        hough_test1 = hough(lane_lines_white, frame)
        hough_test2 = hough(lane_lines_yellow, frame)
        '''

        # Displaying legend:
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        font_scale = 0.7
        thickness = 1

        cv2.putText(frame, 'Left Lane', (10, 30), font, font_scale, (255, 0, 0), thickness)
        cv2.putText(frame, 'Right Lane', (10, 60), font, font_scale, (0, 0, 255), thickness)

        # Defining "TRUE LANE LINES":
        left_lines = []
        right_lines = []
        left_rho = []
        left_angles = []
        right_rho =[]
        right_angles = []
        left_mean = []
        right_mean = []
    
        #Sorting lines based on reference angles:
        if lane_lines_white is not None:   # avoid error for no lines detected
            for lines in lane_lines_white:
                rho, theta = lines[0]   # properly unpack HoughLines 

                if theta < (np.pi/6):
                    left_lines.append((rho,theta))
                elif theta > ((2*np.pi)/3):
                    right_lines.append((rho,theta))
                else:
                    continue
  
        if lane_lines_yellow is not None:   # avoid error for no lines detected
            for lines in lane_lines_yellow:
                rho, theta = lines[0]  

                if theta < ((2*np.pi)/6):
                    left_lines.append((rho,theta))
                elif theta > (np.pi/2):
                    right_lines.append((rho,theta))
                else:
                    continue
        
        # Finding mean rho and theta values for left and right lines respecfully -- these will define the "true lane lines"
            for rho, theta in left_lines:
                left_angles.append(theta)
                left_rho.append(rho)
            left_angle_array = np.array(left_angles)
            left_rho_array = np.array(left_rho)
            left_mean.append((np.mean(left_rho_array), np.mean(left_angle_array)))
            
            for rho, theta in right_lines:
                right_angles.append(theta)
                right_rho.append(rho)
            right_angle_array = np.array(right_angles)
            right_rho_array = np.array(right_rho)
            right_mean.append((np.mean(right_rho_array), np.mean(right_angle_array)))
        
            lane_display = overlay(np.array(left_mean), np.array(right_mean), frame)  

        output_video.write(lane_display) 
    
        #Display Video frame-by-frame:

        #cv2.imshow("Original Video", frame)
        #cv2.imshow("Cleaned Video", image)
        #cv2.imshow("Edge Video White Mask", lane_edge_white)
        #cv2.imshow("Edge Video Yellow Mask", lane_edge_yellow)

        # Troubleshooting Hough Possible Lines:
        '''
        cv2.imshow("Possible White Lines1", hough_test1)
        cv2.imshow("Possible Yellow Lines", hough_test2)
        '''

        cv2.imshow("Identified Lane Lines", lane_display)
        cv2.waitKey(1)


    # Properly clean up resources:
    video.release()
    output_video.release()
    cv2.destroyAllWindows()
    

main()