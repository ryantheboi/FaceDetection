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
import matplotlib.patches as patches


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
    window_size = size // 5

    windows = []
    startx = 0
    starty = 0
    endx = window_size
    endy = window_size
    for row in range(0, 5):
        for col in range(0, 5):
            rectangular_differences = []
            rectangular_differences.append(left_right(image, startx, endx, starty, endy))
            rectangular_differences.append(top_bottom(image, startx, endx, starty, endy))
            rectangular_differences.append(horizontal_middle(image, startx, endx, starty, endy))
            rectangular_differences.append(vertical_middle(image, startx, endx, starty, endy))
            rectangular_differences.append(diagonal(image, startx, endx, starty, endy))
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
        image = np.dot(image[..., :3], rgb_weights)  # grayscale
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
    threshold = 10  # threshold for a vote to pass
    for i in range(0, 25):  # 25 windows
        votes = []
        for j in range(0, 5):  # 5 features in the window
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

    return (window_votes, window_means)


image_data = scan_images("faces")
window_votes_means = boost(image_data)
window_votes = window_votes_means[0]
window_means = window_votes_means[1]

# mapping of (int, int[]) to represent (window, feature list)
features_dict = {}
num_features = 0
n = 20 # vote threshold
for i in range(0, len(window_votes)):
    features_dict[i] = []
    print("Votes for window " + str(i) + ":")
    for j in range(0, 5):
        print("Votes for feature " + str(j+1) + ": " + str(window_votes[i][j]))
        # only features with above n votes
        if window_votes[i][j] > n:
            features_dict[i].append(j)
            num_features += 1

print("BEST WINDOWS AND FEATURES COUNT: " + str(num_features))
for entry in features_dict:
    print(str(entry) + ": " + str(features_dict[entry]))

'''
# classifies a 225x225 subwindow as a face by running the subwindow through the features dictionary
# returns true if the subwindow contains a face, false if it doesn't
'''
def classify(image, features_dict, num_features, size):
    window_size = size // 5

    windows = []
    window_count = 0

    passed_features = 0
    threshold = 10  # threshold for a feature to pass
    startx = 0
    starty = 0
    endx = window_size
    endy = window_size
    for window_num in range(0, 25):
        window_features = features_dict[window_num]
        if len(window_features) > 0:
            for feature_num in window_features:
                feature_mean = window_means[window_num][feature_num]
                rectangular_difference = 0
                if feature_num == 0:
                    rectangular_difference = left_right(image, startx, endx, starty, endy)
                elif feature_num == 1:
                    rectangular_difference = top_bottom(image, startx, endx, starty, endy)
                elif feature_num == 2:
                    rectangular_difference = horizontal_middle(image, startx, endx, starty, endy)
                elif feature_num == 3:
                    rectangular_difference = vertical_middle(image, startx, endx, starty, endy)
                else:  # feature_num == 4
                    rectangular_difference = diagonal(image, startx, endx, starty, endy)

                # this window's feature compared to the mean is within the threshold, it passes this feature
                diff = np.abs(np.subtract(rectangular_difference, feature_mean))
                thr = np.ones(np.shape(diff)) * threshold
                truth_values = np.less(diff, thr)
                # count number of trues
                trues = 0
                for t in truth_values:
                    for val in t:
                        if val == True:
                            trues += 1
                if trues >= 300:
                    passed_features += 1

        # move to next subwindow
        window_count += 1
        startx += window_size
        endx += window_size
        if window_count == 5:
            startx = 0
            endx = window_size
            starty += window_size
            endy += window_size
            window_count = 0

    if passed_features >= num_features // 2:
        return True
    else:
        return False

'''
# divides image of size larger than 225x225 into smaller 225x225 subwindows, with each window containing 25 subwindows
# uses the features dictionary to analyze features at the subwindows
# if the majority of features analyzed at the 225x225 window pass the threshold, then this window contains a face
# returns list of coordinates where face was detected
'''
def detect(image, features_dict, num_features):
    window_size = 225
    startx = 0
    endx = window_size
    starty = 0
    endy = window_size
    image_rows = np.shape(image)[0]
    image_cols = np.shape(image)[1]
    detected_faces = []  # list of coordinates where faces were detected
    while (endy < image_rows):
        while (endx < image_cols):
            i = image[starty:endy, startx:endx]
            classification = classify(i, features_dict, num_features, window_size)
            if classification == True:
                detected_faces.append((starty, startx))
                print("face found at (" + str(starty) + ", " + str(startx) + ")")

            startx += window_size
            endx += window_size
        startx = 0
        endx = window_size
        starty += window_size
        endy += window_size
    return detected_faces


'''
# finds possible faces in an image and draws a red rectangle around it
'''
def find_face(img_path):
    image = img.imread(img_path)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    image = np.dot(image[..., :3], rgb_weights)  # grayscale
    coordinates = detect(image, features_dict, num_features)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    image = img.imread(img_path)
    count = 0
    for c in coordinates:
        coord_row = c[0]
        coord_col = c[1]

        # Display the image
        ax.imshow(image)

        # Create a Rectangle patch
        rect = patches.Rectangle((coord_col, coord_row), 225, 225, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

find_face("classifyphotos/test.jpg")
find_face("classifyphotos/test2.jpg")
find_face("classifyphotos/test3.jpg")
find_face("classifyphotos/test4.jpg")
