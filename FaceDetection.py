'''
@author Ryan Chung

This program is used to train face images of size 225x225

This program first divides an image into 25 windows
Each window will calculate values based on 5 haar-like features
After the calculations are performed for each image, the mean values are taken from every image
Images will vote on the features that brought them the closest to the mean values
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

# rect reshape for same dimension arithmetic
def reshape(white_rect, black_rect):
    white_rect_rows = np.shape(white_rect)[0]
    white_rect_cols = np.shape(white_rect)[1]
    black_rect_rows = np.shape(black_rect)[0]
    black_rect_cols = np.shape(black_rect)[1]
    if white_rect_rows > black_rect_rows:
        white_rect = white_rect[:black_rect_rows, :]
    if white_rect_cols > black_rect_cols:
        white_rect = white_rect[:, :black_rect_cols]
    if black_rect_rows > white_rect_rows:
        black_rect = black_rect[:white_rect_rows, :]
    if black_rect_cols > white_rect_cols:
        black_rect = black_rect[:, :white_rect_cols]
    
    return (white_rect, black_rect)
        
# left-right rectangles
def left_right(image, startx, endx, starty, endy):
    midpointx = ((endx - startx) // 2) + startx
    midpointy = ((endy - starty) // 2) + starty

    white_rect = image[starty:midpointy, startx:endx]
    black_rect = image[midpointy:endy, startx:endx]
    
    rects = reshape(white_rect, black_rect)
    white_rect = rects[0]
    black_rect = rects[1]
        
    return np.subtract(white_rect, black_rect)

# top-bottom rectangles
def top_bottom(image, startx, endx, starty, endy):
    midpointx = ((endx - startx) // 2) + startx
    midpointy = ((endy - starty) // 2) + starty

    white_rect = image[starty:endy, startx:midpointx]
    black_rect = image[starty:endy, midpointx:endx]
    
    rects = reshape(white_rect, black_rect)
    white_rect = rects[0]
    black_rect = rects[1]

    return np.subtract(white_rect, black_rect)

# horizontal-middle rectangles
def horizontal_middle(image, startx, endx, starty, endy):
    thirdx = (endx - startx) // 3
    thirdy = (endy - starty) // 3
    thirdpointx = thirdx + startx
    thirdpointy = thirdy + starty
    
    white_rect1 = image[starty:thirdpointy, startx:endx]
    black_rect = image[thirdpointy:thirdpointy + thirdy, startx:endx]
    white_rect2 = image[thirdpointy + thirdy:endy, startx:endx]
    white_rect = np.add(white_rect1, white_rect2)
    
    return np.subtract(white_rect, black_rect)

# vertical-middle rectangles
def vertical_middle(image, startx, endx, starty, endy):
    thirdx = (endx - startx) // 3
    thirdy = (endy - starty) // 3
    thirdpointx = thirdx + startx
    thirdpointy = thirdy + starty

    white_rect1 = image[starty:endy, startx:thirdpointx]
    black_rect = image[starty:endy, thirdpointx:thirdpointx + thirdx]
    white_rect2 = image[starty:endy, thirdpointx + thirdx:endx]
    white_rect = np.add(white_rect1, white_rect2)

    return np.subtract(white_rect, black_rect)

# diagonal rectangles
def diagonal(image, startx, endx, starty, endy):
    midpointx = ((endx - startx) // 2) + startx
    midpointy = ((endy - starty) // 2) + starty

    white_rect1 = image[starty:midpointy, startx:midpointx]
    black_rect1 = image[midpointy:endy, startx:midpointx]
    black_rect2 = image[starty:midpointy, midpointx:endx]
    white_rect2 = image[midpointy:endy, midpointx:endx]
    
    white_rects = reshape(white_rect1, white_rect2)
    white_rect1 = white_rects[0]
    white_rect2 = white_rects[1]
    
    black_rects = reshape(black_rect1, black_rect2)
    black_rect1 = black_rects[0]
    black_rect2 = black_rects[1]
    
    white_rect = np.add(white_rect1, white_rect2)
    black_rect = np.add(black_rect1, black_rect2)
    
    rects = reshape(white_rect, black_rect)
    white_rect = rects[0]
    black_rect = rects[1]

    return np.subtract(white_rect, black_rect)

'''
# scan rectangular differences in each window of an image
# each image will contain 25 windows
# each window will contain 5 values representing different rectangular differences
'''
def scan(image, size):
    window_size = size//5

    windows = []
    startx = 0
    starty = 0
    endx = window_size
    endy= window_size
    for row in range(0, 5):
        for col in range(0, 5):
            rectangular_differences = []
            rectangular_differences.append( left_right(image, startx, endx, starty, endy) )
            rectangular_differences.append( top_bottom(image, startx, endx, starty, endy) )
            rectangular_differences.append( horizontal_middle(image, startx, endx, starty, endy) )
            rectangular_differences.append( vertical_middle(image, startx, endx, starty, endy) )
            rectangular_differences.append( diagonal(image, startx, endx, starty, endy) )
            windows.append(rectangular_differences)
            startx += window_size
            endx += window_size
        startx = 0
        endx = window_size
        starty += window_size
        endy += window_size
    
    return windows

'''
# performs window scans for each image in a path
# results of window scans are stored in image_data list
# PREREQ - each image in the path must be size 225x225
#        - each image in the path must be named face#.jpg where # is max 57
'''
def scan_images(path):
    image_data = []
    for i in range(1, 58):
        image_path = path + "/face" + str(i) + ".jpg"
        print("Scanning image: " + image_path)
        image = img.imread(image_path)
        rgb_weights = [0.2989, 0.5870, 0.1140]
        image = np.dot(image[..., :3], rgb_weights) # grayscale
        image_data.append(scan(image, 225))
    return image_data

'''
# compares the results from feature scan data across all images
# values that are within close range will add a point to the feature
# features with the highest number of points are good face detection features
# returns a list of windows that each contain a list of votes for the features
'''
def boost(image_data):
    # init windows each containing an array of votes, to represent the common features found across all images scanned
    window_votes = []
    threshold = 10 # threshold for a vote to pass
    for i in range(0,25): # 25 windows
        votes = []
        for j in range(0,5): # 5 features in the window
            votes.append(0)
        window_votes.append(votes)

    # init windows each containing an array of means, to represent the common values found across all images scanned
    window_means = []
    for i in range(0, 25):
        means = []
        for j in range(0, 5):
            means.append(0)
        window_means.append(means)

    # obtain mean values for each feature in every image's windows
    for data in image_data:
        for i in range(0, 25):
            for j in range(0, 5):
                window_means[i][j] += data[i][j]
    num_images = len(image_data)
    for i in range(0, 25):
        for j in range(0, 5):
            window_means[i][j] /= num_images

    # compare the windows of data from each image
    for data in image_data:
        for i in range(0, 25):
            for j in range(0, 5):
                # this window's feature compared to the mean is within the threshold, increase feature vote
                diff = np.abs(np.subtract(data[i][j], window_means[i][j]))
                thr = np.ones(np.shape(diff)) * threshold
                truth_values = np.less(diff, thr)
                # count number of trues
                trues = 0
                for t in truth_values:
                    for val in t:
                        if val == True:
                            trues += 1           
                if trues >= 300:
                    window_votes[i][j] += 1

    return window_votes


image_data = scan_images("faces")
window_votes = boost(image_data)

for i in range(0, len(window_votes)):
    print("Votes for window " + str(i) + ":")
    for j in range(0, 5):
        print("Votes for feature " + str(j+1) + ": " + str(window_votes[i][j]))