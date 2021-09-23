import numpy as np
from PIL import Image
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy
import matplotlib.pyplot
from sklearn.preprocessing import normalize


def preprocess(image):
    
    image = np.array(image)
    #print(image.shape)

    return image


def calculate_gradients(image_array):

    # Calculate gradient

    gradient_kernel_x = np.array([[-1, 0, 1]])

    gradient_kernel_y = np.reshape(gradient_kernel_x, (3, 1))

    gradient_x = cv2.filter2D(image_array, -1, gradient_kernel_x)
    gradient_y = cv2.filter2D(image_array, -1, gradient_kernel_y)

    #gradient_x = cv2.Sobel(image_array, cv2.CV_32F, 1, 0, ksize=1)
    #gradient_y = cv2.Sobel((image_array), cv2.CV_32F, 0, 1, ksize=1)

    '''
    gradientx = ndimage.convolve(image_array, -filter_, mode='constant', cval=0)
    gradienty = ndimage.convolve(np.transpose(image_array), filter_, mode='constant', cval=0)
    gradienty = -np.transpose(gradient_y)
    '''

    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow((gradient_x + 255) / 2, cmap='gray');
    ax1.set_xlabel("Gx")
    ax2.imshow((gradient_x + 255) / 2, cmap='gray');
    ax2.set_xlabel("Gy")
    plt.show()
    '''
    return gradient_x, gradient_y

def magnitude_and_direction( gradient_x, gradient_y):

    h_gradient_square = np.power(gradient_x, 2)
    v_gradient_square = np.power(gradient_y, 2)
    sum = h_gradient_square + v_gradient_square
    magnitude = np.sqrt(sum)

    direction = np.arctan(gradient_y / (gradient_x + 0.00000001))

    direction = np.rad2deg(direction)

    new_direction = direction % 180
    '''
    print(mag)
    print("//////////////////////////////////////////////////////////////////////////////////////")
    print(new_angle)
    exit(0)
    '''
    return magnitude , new_direction

def histogram_image(magnitude_cell,direction_cell):
    '''
        Returns:-
            total_cells: Numpy array of size ((magnitude_cell.shape[0]//8 , magnitude_cell.shape[1]//8,9)).
            This array stores the histogram of each 8x8 cell .
    '''

    total_cells = np.zeros((magnitude_cell.shape[0] // 8 * magnitude_cell.shape[1] // 8, 9))
    #print(total_cells.shape)

    cell_counter = 0
    for i in range(0, magnitude_cell.shape[0], 8):
        for j in range(0, magnitude_cell.shape[1], 8):
            total_cells[cell_counter] = HOG_cell_histogram(magnitude_cell[i:i + 8, j:j + 8],
                                                       direction_cell[i:i + 8, j:j + 8])
            cell_counter += 1


    #print(magnitude_cell.shape[0] // 8 , magnitude_cell.shape[1] // 8) # value = 128
    #final total_cells Shape is (128,9)
    total_cells = total_cells.reshape((magnitude_cell.shape[0] // 8 , magnitude_cell.shape[1] // 8, 9))
    hist, bin_edges = np.histogram(total_cells, density=True)

    #print(total_cells)
    return total_cells

def HOG_cell_histogram(cell_direction, cell_magnitude):
    '''
        Returns: bins: Nummpy array of size 9. This array stores the histogram of a single 8x8 cell.
    '''

    hist_bins = numpy.array([10, 30, 50, 70, 90, 110, 130, 150, 170])

    HOG_cell_hist = numpy.zeros(shape=(hist_bins.size))
    cell_size = cell_direction.shape[0]

    for row_index in range(cell_size):
        for col_idx in range(cell_size):
            current_direction = cell_direction[row_index, col_idx]
            curr_magnitude = cell_magnitude[row_index, col_idx]

            diff = numpy.abs(current_direction - hist_bins)

            if current_direction < hist_bins[0]:
                first_bin_idx = 0
                second_bin_idx = hist_bins.size - 1
            elif current_direction > hist_bins[-1]:
                first_bin_idx = hist_bins.size - 1
                second_bin_idx = 0
            else:
                first_bin_idx = numpy.where(diff == numpy.min(diff))[0][0]
                temp = hist_bins[[(first_bin_idx - 1) % hist_bins.size, (first_bin_idx + 1) % hist_bins.size]]
                temp2 = numpy.abs(current_direction - temp)
                res = numpy.where(temp2 == numpy.min(temp2))[0][0]
                if res == 0 and first_bin_idx != 0:
                    second_bin_idx = first_bin_idx - 1
                else:
                    second_bin_idx = first_bin_idx + 1

            first_bin_value = hist_bins[first_bin_idx]
            second_bin_value = hist_bins[second_bin_idx]
            HOG_cell_hist[first_bin_idx] = HOG_cell_hist[first_bin_idx] + (
                        numpy.abs(current_direction - first_bin_value) / (180.0 / hist_bins.size)) * curr_magnitude
            HOG_cell_hist[second_bin_idx] = HOG_cell_hist[second_bin_idx] + (
                        numpy.abs(current_direction - second_bin_value) / (180.0 / hist_bins.size)) * curr_magnitude

    #print(HOG_cell_hist.shape)

    return HOG_cell_hist



def block_normalization(unnormalized_blocks):
    '''
        Parameters:-
            unnormalized_blocks: A numpy array of size ((8 , 16 ,9)) .
                                 Stores the 9 bins of each 8x8 cell.
        Returns:-
            normalized_block: A numpy array of size ((105,36)) 105 is number of values inside array.
                              Normalized value consisting of two 8x8 cells at a time and rolling it over the image.
    '''
    #print(unnormalized_blocks.shape)
    x_dims = unnormalized_blocks.shape[0] - 1
    y_dims = unnormalized_blocks.shape[1] - 1
    normalized_block = np.zeros((x_dims * y_dims, 36))
    cell_count = 0
    for i in range(unnormalized_blocks.shape[0] - 1):
        for j in range(unnormalized_blocks.shape[1] - 1):
            x = unnormalized_blocks[i:i + 2, j:j + 2, :]
            normalized_block[cell_count] = calculate_norm(unnormalized_blocks[i:i + 2, j:j + 2, :])
            cell_count += 1
    #print(len(normalized_block))
    #print(normalized_block.shape)

    return normalized_block


def calculate_norm(mini_block):
    '''
        Parameters:-
            mini_block: A numpy array of size (2,2,9) .
        Returns:-
            normed_vector: (36,1) shaped normalized vector.
    '''
    # it return 1 D Array for example
    '''
    input: np.array([[1, 2, 3], [4, 5, 6]])
    output: array([1, 2, 3, 4, 5, 6])
    '''
    unnormed_vector = np.ravel(mini_block)
    normed_value = np.linalg.norm(unnormed_vector)
    normed_vector = np.divide(unnormed_vector, normed_value)
    #print(unnormed_vector.shape)

    return normed_vector


def hog_feature_vector(normalized_block):
    '''
               Parameters:-
                  normalized_block: A numpy array of size (105,36) 
                  Represents all (36,1) vectors from normalized block
               Returns:-
                   feature_vector: A numpy array of size (105*36).
                    Unrolled feature vector by concatenating all (36,1) vectors.
    '''
    
    feature_vector = np.ravel(normalized_block)
    return feature_vector.reshape(-1, 1)





def calculate_hog_features(image):

    processed_image = preprocess(image)
    gradient_x, gradient_y = calculate_gradients(processed_image)
    magnitude_matrix , direction_matrix = magnitude_and_direction(gradient_x, gradient_y)


    unnormalized_blocks = histogram_image(magnitude_matrix,direction_matrix)
    #print(unnormalized_blocks)

    total_normalized_blocks = block_normalization(unnormalized_blocks)

    feature_vector = hog_feature_vector(total_normalized_blocks).reshape(-1,1)
    #print(feature_vector.shape)
    return feature_vector
