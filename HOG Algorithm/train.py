# Import the modules
import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from numpy import *
import tensorflow as tf
#from scipy.misc import imresize

# Load the dataset
import Hog
import cv2
(train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data()

'''
    reshape all dataset
'''
print(train_X.shape) # (60000, 28, 28)
print(test_X.shape) # (60000, 28, 28)

train_X = np.expand_dims(train_X, axis=-1)
train_X = tf.image.resize(train_X, [64,128]) # if we want to resize

test_X = np.expand_dims(test_X, axis=-1)
test_X = tf.image.resize(test_X, [64,128]) # if we want to resize

train_features = train_X[:10000]
train_labels = train_y[:10000]
test_features = test_X[:5000]
test_labels = test_y[:5000]

#print(train_features.shape)
#print(test_features.shape)



#Save Sample From Training Images and Testing Images
'''
for i in range(len(train_features)):
	filename = "./Train_dataset/" + str(i) + ".png"
	img = train_features[i]
	cv2.imwrite(filename, img)

for i in range(len(test_features)):
	filename = "./Test_dataset/" + str(i) + ".png"
	img = test_features[i]
	cv2.imwrite(filename, img)
'''

# Extract the hog features
list_hog_fd = []
print("Train Case")
for i in range(len(train_features)):
    img = train_features[i]
    feature_vector = np.ravel(Hog.calculate_hog_features(img))
    print("image number " + str(i) + " Processes is Done\n")
    where_are_NaNs = isnan(feature_vector)
    feature_vector[where_are_NaNs] = 0
    #print(feature_vector)

	#fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    list_hog_fd.append(feature_vector)
hog_features = np.array(list_hog_fd)


'''
	we use the following part to get test features to use it to calculate accuracy of code 
'''
test_list_hog_fd = []
print("\nTest Case")

for i in range(len(test_features)):
    img = test_features[i]
    feature_vector = np.ravel(Hog.calculate_hog_features(img))
    print("image number " + str(i) + " Processes is Done\n")
    where_are_NaNs = isnan(feature_vector)
    feature_vector[where_are_NaNs] = 0
    #test_fd = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1))
    test_list_hog_fd.append(feature_vector)
test_hog_features = np.array(test_list_hog_fd)


print ("Count of training digits in dataset", Counter(train_labels))
print ("Count of testing digits in dataset", Counter(test_labels))

# Create an linear SVM object
model = LinearSVC()

# Perform the training
model.fit(hog_features, train_labels)

# Evaluate the classifier
print(" Evaluating classifier on test data Please wait ...")
predictions = model.predict(test_hog_features)


print(classification_report(test_labels, predictions))

print("\nModel Accuracy Score : ",accuracy_score(test_labels, predictions), "\n")

# Save the classifier
joblib.dump(model, "model.npy")
