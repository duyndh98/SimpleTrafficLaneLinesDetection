# The following techniques are used:

# => 1. Color Selection
# => 2. Canny Edge Detection
# => 3. Region of Interest Selection
# => 4. Hough Transform Line Detection

import cv2
import numpy as np
from matplotlib import pyplot as plt

# threshold
low_red1 = np.array([0, 100, 80])
up_red1 = np.array([15, 255, 255])
low_red2 = np.array([145, 100, 80])
up_red2 = np.array([180, 255, 255])

low_blue = np.array([95, 40, 40])
up_blue = np.array([110, 255, 255])

low_black = np.array([0, 0, 0])
up_black = np.array([180, 255, 30])

low_white = np.array([0, 0, 205])
up_white = np.array([180, 20, 255])

# kernel size for smooothing/morphologyEx
kernel_size = 15

# region const
region_bottom = 1
region_top = 0.6
region_bottom_left = 0
region_top_left = 0.4

# drawing parameters
draw_color = [0, 0, 255]
draw_thickness = 10

def ycrcb_equalize(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4,4))
    y = clahe.apply(y)

    ycrcb = cv2.merge([y, cr, cb])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
def color_selection(image, colors):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = np.zeros(image.shape[:2], np.uint8)
    if 'blue' in colors:
        mask = mask | cv2.inRange(hsv, low_blue, up_blue)
    if 'red' in colors:
        mask = mask | cv2.inRange(hsv, low_red1, up_red1) | cv2.inRange(hsv, low_red2, up_red2)
    if 'white' in colors:
        mask = mask | cv2.inRange(hsv, low_white, up_white)
    if 'black' in colors:
        mask = mask | cv2.inRange(hsv, low_black, up_black)
    
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations = 1)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)    
    return cv2.bitwise_and(image, image, mask = mask)
    
def filter_region(image, vertices):
    '''
    Create the mask using the vertices and apply it to the input image
    '''
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)
    
def select_region(image):
    '''
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    '''
    # first, define the polygon by vertices
    height, width = image.shape[:2]
    bottom_left  = [width * region_bottom_left, height * region_bottom]
    top_left     = [width * region_top_left, height * region_top]
    bottom_right = [width * (1 - region_bottom_left), height * region_bottom]
    top_right    = [width * (1 - region_top_left), height * region_top] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # cv2.polylines(image, [vertices], True, (255, 255, 255), 2)

    return filter_region(image, vertices)

def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            '''
			We want two lane lines: one for the left and the other for the right. 
			The left lane should have a positive slope, and the right lane should have a negative slope. 
			Therefore, we'll collect positive slope lines and negative slope lines separately and take averages.

			Note: in the image, y coordinate is reversed. 
			The higher y value is actually lower in the image. 
			Therefore, the slope is negative for the left lane, and the slope is positive for the right lane.
			
			** My description **

			O--------------------------------> x
			|     /               \
			|    / => slope < 0    \ => slope > 0
			|   /                   \
			V
			y

			y = ax + b
			(x1, y1), (x2, y2)
			=> slope:		a = (y2 - y1) / (x2 - x1)
			   intercept:	b = y1 - a * x1 = y2 - a * x2

            '''
            if x2 == x1:
            	# ignore a vertical line (x2 - x1 must not be zero)
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1 
            
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            # left lane line
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else: # right lane line
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)

def make_line_points(y1, y2, line):
    '''
    Convert a line represented in slope and intercept into pixel points
    '''
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    '''
    O--------------> x
	|      y2
	|      /
	|     / 
	|    /  
	|   y1
	V
	y
	
	'''
    y1 = image.shape[0] # bottom of the image
    y2 = image.shape[0] * region_top # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line

def draw_lane_lines(image, lines):
	# make a separate image to draw lines and combine with the orignal later
	line_image = np.zeros_like(image)
	for line in lines:
		if line is not None:
			cv2.line(line_image, *line, draw_color, draw_thickness)
    # image1 * α + image2 * β + λ
	return cv2.addWeighted(image, 1, line_image, 1, 0)

def edge_detection(image):
	# convert image to grayscale
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# smooth out rough edges
	cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
	# edge detection
	return cv2.Canny(gray_image, 50, 150)
	
def line_detection(image, region_image):
	lines = cv2.HoughLinesP(region_image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
	# Averaging and Extrapolating Lines
	left_line, right_line = lane_lines(region_image, lines)
	return draw_lane_lines(image, (left_line, right_line))

def process_image(image):
	orginal = image
	image = ycrcb_equalize(image)
	
	image = color_selection(image, ['white'])
	
	edges_image = edge_detection(image)

	region_image = select_region(edges_image)
	
	orginal = line_detection(orginal, region_image)

	cv2.imshow('region', region_image)
	cv2.imshow('orginal', orginal)
	
def process_video(video_input, video_output):
	inp = cv2.VideoCapture(video_input)
	
	video_width, video_height = int(inp.get(cv2.CAP_PROP_FRAME_WIDTH)), int(inp.get(cv2.CAP_PROP_FRAME_HEIGHT))
	video_fps = inp.get(cv2.CAP_PROP_FPS)
	
	print('Video resolution: (' + str(video_width) + ', ' + str(video_height) + ')')
	print('Video fps:', video_fps)

	out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'X264'), video_fps, (video_width, video_height))
	
	print('Video is running')
	while inp.isOpened():
		ret, frame = inp.read()
		
		if (not ret) or (cv2.waitKey(1) & 0xFF == ord('q')):
			break
	
		process_image(frame)
		out.write(frame)

	inp.release()
	out.release()

if __name__ == '__main__':

	process_video('video_input.mp4', 'video_output.mp4')

	cv2.destroyAllWindows()