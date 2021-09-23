# Import the modules
import cv2
import joblib
from skimage.feature import hog
import numpy as np
import Hog
from numpy import *


model = joblib.load("model.npy")

# Read the input image 
im = cv2.imread("./test_photo_1.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
count = 1
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 3)
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    image = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    image = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
    image = cv2.dilate(image, (3, 3))
    #cv2.imwrite(str(count)+".png", image)

    
    feature_vector = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    hog_prediction_number = model.predict(np.array([feature_vector], 'float64'))
    cv2.putText(im, str(int(hog_prediction_number[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    count = count + 1

cv2.imwrite('output.png', im)
print("Done")
print("image saved")

