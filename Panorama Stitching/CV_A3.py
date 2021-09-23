
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import imutils
import numpy as np

def RGBTOGrayScale(Image1, Image2):
    Image1 = imutils.resize(Image1, width=400)
    Image2 = imutils.resize(Image2, width=400)

    Image1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2GRAY)
    Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2GRAY)

    return Image1 , Image2

def detectKeyPoints(image1 , image2):
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)

    keypoints_image_1 = cv2.drawKeypoints(image1, keypoints_1, ImageA)
    keypoints_image_2 = cv2.drawKeypoints(image2, keypoints_2, ImageB)
    return keypoints_image_1 , keypoints_image_2, descriptors_1,descriptors_2,keypoints_1,keypoints_2

def MatchingFeature(keypoints_1,keypoints_2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
    # draw Matches on images
    Matching_Feature_image = cv2.drawMatchesKnn(ImageA, keypoints_1, ImageB, keypoints_2, matches[:50], None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return  matches , Matching_Feature_image

def FindBestMatches(ImageA,ImageB , keypoints_1,keypoints_2,matches):
    good = []
    temp = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            temp.append([m])

    # draw The Best matching in image 1
    Goog_for_ImageA = cv2.drawKeypoints(ImageA, [keypoints_1[m.queryIdx] for m in good], None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('image 1 best Matching', Goog_for_ImageA)
    cv2.imwrite('Best Matching Feature for image 1.png', Goog_for_ImageA)
    cv2.waitKey(0)
    # draw The Best matching in image 2
    Goog_for_ImageB = cv2.drawKeypoints(ImageB, [keypoints_2[m.trainIdx] for m in good], None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('image 2 best Matching', Goog_for_ImageB)
    cv2.imwrite('Best Matching Feature for image 2.png', Goog_for_ImageB)
    cv2.waitKey(0)

    # cv.drawMatchesKnn expects list of lists as matches.
    best_Matches_img = cv2.drawMatchesKnn(ImageA, keypoints_1, ImageB, keypoints_1, temp[:50], None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Best Matching between two images', best_Matches_img)
    cv2.imwrite('Best Matching Feature for two images.png', best_Matches_img)
    cv2.waitKey(0)

    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Establish a homography
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def warpImages(ImageA ,ImageB, h):
    rows1, cols1 = ImageA.shape[:2]
    rows2, cols2 = ImageB.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # transformation matrix
    list_of_points_2 = cv2.perspectiveTransform(temp_points, h)
    #print("transformation matrix as np array : \n", list_of_points_2)
    print("\n\ntransformation matrix as 1D array : \n", list_of_points_2.flatten())

    # use function concatenate to make on np array have all images
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    Panorama_image = cv2.warpPerspective(ImageA, H_translation.dot(h), (x_max - x_min, y_max - y_min))
    Panorama_image[translation_dist[1]:rows1 + translation_dist[1],
    translation_dist[0]:cols1 + translation_dist[0]] = ImageB

    return Panorama_image

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    Image1 = cv2.imread("Picture1.jpg")
    Image2 = cv2.imread("Picture2.jpg")

    # Step 1 Load our images
    ImageA , ImageB = RGBTOGrayScale(Image1, Image2)

    # Step 2 Dectet Keypoints of input images
    keypoints_img_1, keypoints_img_2,descriptors_1,descriptors_2,keypoints_1,keypoints_2 = detectKeyPoints(ImageA,ImageB)

    cv2.imshow('keypoints_img_1' ,keypoints_img_1)
    cv2.imwrite('keypoints image 1.png', keypoints_img_1)
    cv2.waitKey(0)
    cv2.imshow('keypoints_img_2' , keypoints_img_2)
    cv2.imwrite('keypoints image 2.png', keypoints_img_2)
    cv2.waitKey(0)


    # Step 3 feature matching
    matches , Matching_Feature_image = MatchingFeature(keypoints_1,keypoints_2)
    cv2.imshow('Matching Feature between two input images' , Matching_Feature_image)
    cv2.imwrite('Matching Feature image.png', Matching_Feature_image)
    cv2.waitKey(0)

    # Step 4 Find Best Matches Between two images
    h = FindBestMatches(ImageA,ImageB , keypoints_1,keypoints_2,matches)


    # final step warpimages and concatenate two arrays of images to get final image (panorama image)
    Final_image = warpImages(ImageA,ImageB,h)
    cv2.imshow('Final image (panorama image)',Final_image)
    cv2.imwrite('panorama image.png', Final_image)
    cv2.waitKey(0)
