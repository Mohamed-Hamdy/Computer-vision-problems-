# Computer-vision-problems

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#General">Computer Vision Problems</a>
      <ul>
        <li><a href="#problem_1">problem 1 Histogram of Oriented Gradients Algorithm</a></li>
      </ul>
      <ul>
        <li><a href="#problem_2">problem 2 Applying Convolutional Neural Network on mnist dataset  </a></li>
      </ul>
      <ul>
        <li><a href="#problem_3">problem 3 Panorama Stitching</a></li>
      </ul>
    </li>
  </ol>
</details>




<!-- General C++ Problems -->
## General
there is decription file for each problem, it explain details of each problem expain what problem should take as input and there is examples to test functions so check it before test code.

### problem_1
<h3>Histogram of Oriented Gradients Algorithm (HOG is built from scratch)</h3>
problem Decription:
The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image. This method is similar to that of edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts, but differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy.<br>

problem Requirment:
* 1-  Apply HOG on MNIST dataset (train & test).
* 2-  Save the Features to be used later 
* 3-  Use SVM classifier (built-in in sklearn python library) to train on the training set.   
* 4- Save the SVM model so that it can be used later.
* 5- Use the saved model on the test set.
* 6- Print the training accuracy and testing accuracy (# of correctly classified samples / total samples).

problem Output:
<img src="https://github.com/Mohamed-Hamdy/Computer-vision-problems-/blob/master/images/Hog%20output.png">

<h4><a href="https://learnopencv.com/histogram-of-oriented-gradients/">Explanation of the algorithm</a>:</h4>

<h4><a href="https://github.com/Mohamed-Hamdy/Computer-vision-problems-/tree/master/HOG%20Algorithm">
  Problem Implementation</a>:</h4>

<hr>

### problem_2
<h3>Applying Convolutional Neural Network on mnist dataset</h3>
problem Decription:
CNN is basically a model known to be Convolutional Neural Network and in the recent time it has gained a lot of popularity because of it’s usefullness. CNN uses multilayer perceptrons to do computational works. CNNs use relatively little pre-processing compared to other image classification algorithms. This means the network learns through filters that in traditional algorithms were hand-engineered. So, for image processing task CNNs are the best-suited option.

MNIST dataset: 
mnist dataset is a dataset of handwritten images as shown below in image.<br>

problem Output:
<img src="https://github.com/Mohamed-Hamdy/Computer-vision-problems-/blob/master/images/CNN%20Prediction_image.jpg">

<h4><a href="https://www.geeksforgeeks.org/applying-convolutional-neural-network-on-mnist-dataset/">Explanation of the algorithm</a>:</h4>

<h4><a href="https://github.com/Mohamed-Hamdy/Computer-vision-problems-/tree/master/Applying%20CNN%20on%20mnist%20dataset">
  Problem Implementation</a>:</h4>
<hr>

### problem_3
<h3>Panorama Stitching</h3>
problem Decription:
1. What is a panoramic image?
Panoramic photography is a technique that combines multiple images from the same rotating camera to form a single, wide photo. It captures images with horizontally or vertically elongated fields. This process of combining multiple photos to produce a panorama is called image stitching. So, after we rotate a camera to produce a full 360 or less degree effect we will stitch those images together to get a panoramic photo.

What is image stitching?
At the beginning of the stitching process, as input, we have several images with overlapping areas. The output is a unification of these images. It is important to note that a full scene from the input image must be preserved in the process.

problem Output:
<img src="https://github.com/Mohamed-Hamdy/Computer-vision-problems-/blob/master/images/panorama%20image.png">

<h4><a href="http://datahacker.rs/005-how-to-create-a-panorama-image-using-opencv-with-python/">Explanation of the algorithm</a>:</h4>

<h4><a href="https://github.com/Mohamed-Hamdy/Computer-vision-problems-/tree/master/Panorama%20Stitching">
  Problem Implementation</a>:</h4>
<hr>
