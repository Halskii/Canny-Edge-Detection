# NOTE: This program will write the resulting images to it's local directory.
#       Run in an isolated folder to avoid clutter.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Performs convolution on a image with a 1-D kernel (filter).
def apply_filter_1D(img, kernel):
    img = img.astype('float')
    img_height, img_width = img.shape
    filtered_img_x = np.zeros((img_height, img_width))
    filtered_img_y = np.zeros((img_height, img_width))
    kernel_radius = int(np.floor(kernel.shape[0] / 2.))

    # Manual convolution requires kernel flipping
    kernel = kernel[::-1]

    for row in range(kernel_radius, img_height - kernel_radius):
        for col in range(kernel_radius, img_width - kernel_radius):
            image_patch_x = img[row, col - kernel_radius:col + kernel_radius + 1]
            image_patch_y = img[row - kernel_radius:row + kernel_radius + 1, col]

            conv_x = np.sum(image_patch_x * kernel)
            conv_y = np.sum(image_patch_y * kernel)

            filtered_img_x[row, col] = conv_x
            filtered_img_y[row, col] = conv_y

    return filtered_img_x, filtered_img_y

# Performs convolution on an image with a 2-D kernel (filter).
def apply_filter_2D(img, kernel):
    img = img.astype(np.float32)
    img_height, img_width = img.shape
    filtered_img = np.zeros((img_height, img_width))
    kernel_radius_height = int(np.floor(kernel.shape[0] / 2.))
    kernel_radius_width = int(np.floor(kernel.shape[1] / 2.))

    # Manual convolution requires kernel flipping
    kernel = np.flipud(np.fliplr(kernel))

    for row in range(kernel_radius_width, img_height - kernel_radius_width):
        for col in range(kernel_radius_height, img_width - kernel_radius_height):
            image_patch = img[row - kernel_radius_width: row + kernel_radius_width + 1,
                          col - kernel_radius_height: col + kernel_radius_height + 1]

            # Write the new value to an output matrix
            filtered_img[row, col] = np.sum(np.multiply(image_patch, kernel))

    return filtered_img

# Simple function to find the weak / strong pixels in an image.
def simple_threshold(img, lowRatio, highRatio):
    upperbound = img.max() * highRatio
    lowerbound = upperbound * lowRatio

    return lowerbound, upperbound

# Scale the image
def scale(value):
    return 255 * (value - np.min(value)) / (np.max(value) - np.min(value))

#========================================#
#____________Preprocessing_______________#
#========================================#
# Loading in the images
img1 = cv2.imread('image1.png', 0)
img2 = cv2.imread('image2.png', 0)
img3 = cv2.imread('image3.png', 0)
img4 = cv2.imread('image4.png', 0)
canny1 = cv2.imread('canny1.jpg', 0)
canny2 = cv2.imread('canny2.jpg', 0)
output_canny1 = cv2.imread('output_canny1.png', 0)
output_canny2 = cv2.imread('output_canny2.png', 0)
output_image1 = cv2.imread('output_image.png', 0)
output_image2 = cv2.imread('output_image2.png', 0)

# Box Filtering
#========================================#
#______________Question 1________________#
#========================================#

print("Now Running Question 1...")
# Defining our box filters
box_filter3x3 = np.ones((3,3),np.float32)/9
box_filter5x5 = np.ones((5,5),np.float32)/25

# Applies our box filters to images 1 and 2.
question1_img1_3x3 = apply_filter_2D(img1, box_filter3x3)
question1_img1_5x5 = apply_filter_2D(img1, box_filter5x5)
question1_img2_3x3 = apply_filter_2D(img2, box_filter3x3)
question1_img2_5x5 = apply_filter_2D(img2, box_filter5x5)

# Shows box filtering of image 1 and saves results.
cv2.imshow("Input Image 1", img1)
cv2.imshow("Image 1 Filtered With 3x3 Box Filter", question1_img1_3x3)
cv2.imshow("Image 1 Filtered With 5x5 Box Filter", question1_img1_5x5)
cv2.imwrite("1_img1_3x3.png", question1_img1_3x3)
cv2.imwrite("1_img1_5x5.png", question1_img1_5x5)
cv2.waitKey(0)

# Shows box filtering of image 2 and saves results.
cv2.imshow("Input Image 2", img2)
cv2.imshow("Image 2 Filtered With 3x3 Box Filter", question1_img2_3x3)
cv2.imshow("Image 2 Filtered With 5x5 Box Filter", question1_img2_5x5)
cv2.imwrite("1_img2_3x3.png", question1_img2_3x3)
cv2.imwrite("1_img2_5x5.png", question1_img2_5x5)
cv2.waitKey(0)

# # # Note the differences between the original and the subsequent images,
# # # the box filters remove sharpness in the image and cause them to get
# # # 'smoother' as the filter size increases.

# Median Filtering
# ========================================#
# ______________Question 2________________#
# ========================================#

print("Now Running Question 2...")
# Function applies a median filter to a passed numpy array
def apply_median_filter(img, filter_size):
    buffer = []
    indexer = filter_size // 2
    filtered_image = np.zeros((len(img), len(img[0])))

    for i in range(len(img)):
        for j in range(len(img[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(img) - 1:
                    for c in range(filter_size):
                        buffer.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(img[0]) - 1:
                        buffer.append(0)
                    else:
                        for k in range(filter_size):
                            buffer.append(img[i + z - indexer][j + k - indexer])

            buffer.sort()
            filtered_image[i][j] = buffer[len(buffer) // 2]
            buffer = []
    return filtered_image


# Applies our median filter to images 1 and 2.
question2_img1_3x3 = apply_median_filter(img1, 3)
question2_img1_5x5 = apply_median_filter(img1, 5)
question2_img1_7x7 = apply_median_filter(img1, 7)
question2_img2_3x3 = apply_median_filter(img2, 3)
question2_img2_5x5 = apply_median_filter(img2, 5)
question2_img2_7x7 = apply_median_filter(img2, 7)

# Shows median filtering of image 1 and saves results.
cv2.imshow("Input Image 1", img1)
cv2.imshow("Image 1 Filtered With 3x3 Median Filter", question2_img1_3x3)
cv2.imshow("Image 1 Filtered With 5x5 Median Filter", question2_img1_5x5)
cv2.imshow("Image 1 Filtered With 5x5 Median Filter", question2_img1_7x7)
cv2.imwrite("2_img1_3x3.png", question2_img1_3x3)
cv2.imwrite("2_img1_5x5.png", question2_img1_5x5)
cv2.imwrite("2_img1_7x7.png", question2_img1_7x7)
cv2.waitKey(0)

# Shows median filtering of image 2 and saves results.
cv2.imshow("Input Image 2", img2)
cv2.imshow("Image 2 Filtered With 3x3 Median Filter", question2_img2_3x3)
cv2.imshow("Image 2 Filtered With 5x5 Median Filter", question2_img2_5x5)
cv2.imshow("Image 2 Filtered With 5x5 Median Filter", question2_img2_7x7)
cv2.imwrite("2_img2_3x3.png", question2_img2_3x3)
cv2.imwrite("2_img2_5x5.png", question2_img2_5x5)
cv2.imwrite("2_img2_7x7.png", question2_img2_7x7)
cv2.waitKey(0)

# # # Note the differences between the original and the subsequent images,
# # # the median filters denoise the image

# Gaussian Smoothing
# ========================================#
# ______________Question 3________________#
# ========================================#

print("Now Running Question 3...")
# Returns a 2D Gaussian Kernel
def gaussian_kernel(size, sigma):
    mean = 0
    kernel_radius = int(np.floor(size) / 2)

    gaussian_kernel_1D = np.linspace(-kernel_radius, kernel_radius, size)
    gaussian_kernel_1D = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.e ** (
                -np.power((gaussian_kernel_1D - mean) / sigma, 2) / 2)

    gaussian_kernel_1D = gaussian_kernel_1D / np.sum(gaussian_kernel_1D)

    gaussian_kernel_2D = np.outer(gaussian_kernel_1D, gaussian_kernel_1D)

    return gaussian_kernel_2D

# Performs variation of gaussian smoothing on image 1.
img1sigma3 = apply_filter_2D(img1, gaussian_kernel(size=3, sigma=3))
img1sigma5 = apply_filter_2D(img1, gaussian_kernel(size=3, sigma=5))
img1sigma10 = apply_filter_2D(img1, gaussian_kernel(size=3, sigma=10))

# Displays and saves image 1 results.
cv2.imshow("Input Image 1", img1)
cv2.imshow("Image 1 Filtered With Sigma 3 Gaussian Filter", img1sigma3)
cv2.imshow("Image 1 Filtered With Sigma 5 Gaussian Filter", img1sigma5)
cv2.imshow("Image 1 Filtered With Sigma 10 Gaussian Filter", img1sigma10)
cv2.imwrite("3_img1_3.png", img1sigma3)
cv2.imwrite("3_img1_5.png", img1sigma5)
cv2.imwrite("3_img1_10.png", img1sigma10)
cv2.waitKey(0)

# Performs variation of gaussian smoothing on image 2.
img2sigma3 = apply_filter_2D(img2, gaussian_kernel(size=3, sigma=3))
img2sigma5 = apply_filter_2D(img2, gaussian_kernel(size=3, sigma=5))
img2sigma10 = apply_filter_2D(img2, gaussian_kernel(size=3, sigma=10))

# Displays and saves image 2 results.
cv2.imshow("Input Image 2", img2)
cv2.imshow("Image 2 Filtered With Sigma 3 Gaussian Filter", img2sigma3)
cv2.imshow("Image 2 Filtered With Sigma 5 Gaussian Filter", img2sigma5)
cv2.imshow("Image 2 Filtered With Sigma 10 Gaussian Filter", img2sigma10)
cv2.imwrite("3_img2_3.png", img2sigma3)
cv2.imwrite("3_img2_5.png", img2sigma5)
cv2.imwrite("3_img2_10.png", img2sigma10)
cv2.waitKey(0)

# NOTE: Increasing the sigma value essentially increases the variance that's allowed around the mean,
#      as sigma becomes larger the image will become blurrier,likewise if sigma is decreased the image
#      will be less blurred because there will be less variance in the kernel values.
#
#      As for the differences between this filter and the ones from question 1/2/3:
#
#      Question 1 implemented a box filter, and question 2 implemented a median filter, which are both similar to Gaussian
#      filters min that they are intended to smooth/denoise the passed image, however, they have slightly different results.
#
#      When the box filter is applied it takes the mean of a neighborhood of pixels and replaces pixels with it to achieve
#      its smoothing effect, the median filter does this but with the median of the neighborhood of pixels. Gaussian filtering
#      takes the weighted mean of each pixels neighborhood where the mean is weighted more towards the value of the central pixels.
#      Because of this Gaussian filtering took noticably more time than the box filtering, however, it also preserves the images edges
#      better.

# Gradient Operations
# ========================================#
# ______________Question 4________________#
# ========================================#
print("Now Running Question 4...")

def derivatives(img, filter, save):
    f_x, f_y = np.array(apply_filter_1D(img, filter))

    # Calculates the magnitude.
    magnitude = np.sqrt((f_x * f_x) + (f_y * f_y))

    # Scales image with magnitude.
    image_magnitude = scale(magnitude).astype(np.uint8)

    # Prepping derivatives for images.
    f_x = np.abs(f_x)
    f_y = np.abs(f_y)
    f_x = scale(f_x).astype(np.uint8)
    f_y = scale(f_y).astype(np.uint8)

    # Displays the derivative and magnitude images.
    cv2.imshow("Img F-x ", f_x)
    cv2.imshow("Img F_y ", f_y)
    cv2.imshow("Magnitude", image_magnitude)
    cv2.waitKey(0)

    if (save == 1):
        cv2.imwrite("4_img3_Fx_backward.png", f_x)
        cv2.imwrite("4_img3_Fy_backward.png", f_y)
        cv2.imwrite("4_img3_mag_backward.png", image_magnitude)

    elif (save == 2):
        cv2.imwrite("4_img3_Fx_forward.png", f_x)
        cv2.imwrite("4_img3_Fy_forward.png", f_y)
        cv2.imwrite("4_img3_mag_forward.png", image_magnitude)

    elif (save == 3):
        cv2.imwrite("4_img3_Fx_central.png", f_x)
        cv2.imwrite("4_img3_Fy_central.png", f_y)
        cv2.imwrite("4_img3_mag_central.png", image_magnitude)


# Making the 3 1D Filters
backward_difference = np.array([-1, 1, 0], 'float')
forward_difference  = np.array([1, -1, 0], 'float')
central_difference  = np.array([-1, 0, 1], 'float')

derivatives(img3, backward_difference, 1)
derivatives(img3, forward_difference, 2)
derivatives(img3, central_difference, 3)

# Sobel Filtering
# ========================================#
# ______________Question 5________________#
# ========================================#
print("Now Running Question 5...")

# Creates a sobel kernel, and performs/shows filtering on the passed image.
def sobel(img):
    filter_x, filter_y = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]]), \
                         np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]]).T

    s_x = np.array(apply_filter_2D(img, filter_x))
    s_y = np.array(apply_filter_2D(img, filter_y))

    # Gets the final gradient for our sobel filter
    s_gradient = np.sqrt((s_x * s_x) + (s_y * s_y))
    s_gradient = scale(s_gradient).astype(np.uint8)

    s_x = np.abs(s_x)
    s_y = np.abs(s_y)
    s_x = scale(s_x).astype(np.uint8)
    s_y = scale(s_y).astype(np.uint8)

    return s_x, s_y, s_gradient

# Performs sobel filtering on the passed images and shows resulting images.
s1_x, s1_y, s1_gradient = sobel(img1)
s2_x, s2_y, s2_gradient = sobel(img2)

# Displays the derivative and magnitude images.
cv2.imshow("Img x-dim W/ Sobel ", s1_x)
cv2.imshow("Img y-dim W/ Sobel", s1_y)
cv2.imshow("Gradient W/ Sobel", s1_gradient)
cv2.imwrite("5_img1_Sobel_X.png", s1_x)
cv2.imwrite("5_img1_Sobel_Y.png", s1_y)
cv2.imwrite("5_img1_Sobel_Grad.png", s1_gradient)
cv2.waitKey(0)

# Displays the derivative and magnitude images.
cv2.imshow("Img x-dim W/ Sobel ", s2_x)
cv2.imshow("Img y-dim W/ Sobel", s2_y)
cv2.imshow("Gradient W/ Sobel", s2_gradient)
cv2.imwrite("5_img2_Sobel_X.png", s2_x)
cv2.imwrite("5_img2_Sobel_Y.png", s2_y)
cv2.imwrite("5_img2_Sobel_Grad.png", s2_gradient)
cv2.waitKey(0)

# NOTE: Note how in the final images after the sobel filter is applied the edges
#       have become obvious in the images. This is the point of sobel filtering,
#       it is primarily used for edge detection within images.

# Faster Gaussian Filtering
# ========================================#
# ______________Question 6________________#
# ========================================#
print("Now Running Question 6...")

def pad_image(img, pad):
    return img[pad:img.shape[0] - pad, pad:img.shape[1] - pad]

# Returns a 1D Gaussian Kernel
def gaussian_1D_kernel(size, sigma):
    mean = 0
    kernel_radius = int(np.floor(size) / 2)

    gaussian_kernel_1D = np.linspace(-kernel_radius, kernel_radius, size)
    gaussian_kernel_1D = (1 / (np.sqrt(2 * np.pi) * sigma)) * \
                         np.exp(-np.power((gaussian_kernel_1D - mean) / sigma, 2) / 2)

    gaussian_kernel_1D = gaussian_kernel_1D / np.sum(gaussian_kernel_1D)

    return gaussian_kernel_1D

# Performs a fast gaussian smoothing on a passed image.
def fast_gaussian(img, size, sigma):

    # performs our 1D convolutions for x and y directions.
    x_direction, y_direction = apply_filter_1D(img, gaussian_1D_kernel(size, sigma))

    # combines the x and y filtered images.
    filtered_image = np.sqrt((x_direction * y_direction) + (y_direction * y_direction))


    x_direction = pad_image(x_direction, pad=size)
    y_direction = pad_image(y_direction, pad=size)
    filtered_image = pad_image(filtered_image, pad=size)

    return x_direction, y_direction, filtered_image

x_direction1, y_direction1, filtered_image1 = fast_gaussian(img1, 3, 3)
x_direction2, y_direction2, filtered_image2 = fast_gaussian(img2, 3, 3)

# displays and saves the gradient images.
cv2.imshow("Unmodified Img1", img1)
cv2.imshow("FG Img1 X Direction", scale(x_direction1).astype(np.uint8))
cv2.imshow("FG Img1 Y Direction", scale(y_direction1).astype(np.uint8))
cv2.imshow("FG Filtered Img1", scale(filtered_image1).astype(np.uint8))

cv2.imwrite("6_img1_FG_X.png", scale(x_direction1).astype(np.uint8))
cv2.imwrite("6_img1_FG_Y.png", scale(y_direction1).astype(np.uint8))
cv2.imwrite("6_img1_FG_Final.png", scale(filtered_image1).astype(np.uint8))
cv2.waitKey(0)

# displays and saves the gradient images.
cv2.imshow("Unmodified Img2", img2)
cv2.imshow("FG Img2 X Direction", scale(x_direction2).astype(np.uint8))
cv2.imshow("FG Img2 Y Direction", scale(y_direction2).astype(np.uint8))
cv2.imshow("FG Filtered Img2", scale(filtered_image2).astype(np.uint8))

cv2.imwrite("6_img2_FG_X.png", scale(x_direction2).astype(np.uint8))
cv2.imwrite("6_img2_FG_Y.png", scale(y_direction2).astype(np.uint8))
cv2.imwrite("6_img2_FG_Final.png", scale(filtered_image2).astype(np.uint8))
cv2.waitKey(0)

# NOTE: Note that this method is much faster than the original gaussian filter
#       operation. This is mainly due to us avoiding the nested for loop in the
#       apply_filter_2D function by using 1D gaussian kernels and combining them
#       through magnitude.

# Custom Histogram Function
# ========================================#
# ______________Question 7________________#
# ========================================#
print("Now Running Question 7...")
def plot_hist(img, bins):

    mat_range = np.max(img) - (np.min(img)) + 1
    bin_size = int(mat_range / bins)
    bin_container = np.zeros(shape=(bins, 1))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
                this_bin = img[i, j] // bin_size
                bin_container[this_bin, 0] += 1

    plt.figure()
    plt.plot(bin_container)
    plt.show()

plot_hist(img4, 256)
plot_hist(img4, 128)
plot_hist(img4, 64)

# NOTE: For each of these histograms, despite us changing the bins, they look very similar.
#       This is because we are not changing the overall distribution of pixel values,
#       we are simply condensing their value to fit the specified number of bins.

# Canny Edge Detection
# ========================================#
# ______________Question 8________________#
# ========================================#
print("Now Running Question 8...")

# Performs non-max suppression on an image with a given direction.
def non_max_suppression(img, orientation):
    s1, s2 = img.shape
    map = np.zeros((s1, s2), dtype=np.int32)

    for i in range(1, s1 - 1):
        for j in range(1, s2 - 1):
            try:
                q, r = 255, 255

                # 0 degree angle.
                if (0 <= orientation[i, j] < 22.5) or (157.5 <= orientation[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # 45 degree angle.
                elif (22.5 <= orientation[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # 90 degree angle.
                elif (67.5 <= orientation[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # 135 degree angle.
                elif (112.5 <= orientation[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]
                if (img[i, j] >= q) and (img[i, j] >= r):
                    map[i, j] = img[i, j]
                else:
                    map[i, j] = 0

            except IndexError as index_err:
                pass

    return map

# Uses hysteris thresholding algorithm to further enhance our edgemap
def hysteresis_thresholding(img, weak, strong):
    s1, s2 = img.shape
    for i in range(1, s1 - 1):
        for j in range(1, s1 - 1):
            if (img[i,j] == weak):
                try:
                    if ((img[i + 1, j - 1] == strong)
                        or (img[i + 1, j] == strong)
                        or (img[i + 1, j + 1] == strong)
                        or (img[i, j - 1] == strong)
                        or (img[i, j + 1] == strong)
                        or (img[i - 1, j - 1] == strong)
                        or (img[i - 1, j] == strong)
                        or (img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as index_error:
                    pass
    return img


# Function that takes two images and produces canny edgemaps.
def make_edgemaps(image1, image2, size, sigma, skipOutputs):

    # Obtains smoothed or blurred images from our fast gaussian filter.
    x_direction1, y_direction1, blurred_image1 = fast_gaussian(image1, size, sigma)
    x_direction2, y_direction2, blurred_image2 = fast_gaussian(image2, size, sigma)

    # If we're not skipping outputs, we print out the gaussian smoothing.
    if (skipOutputs != True):
        cv2.imshow("Unmodified Image1", image1)
        cv2.imshow("FG In X Direction", scale(x_direction1).astype(np.uint8))
        cv2.imshow("FG In Y Direction", scale(y_direction1).astype(np.uint8))
        cv2.imshow("FG Filtered Image", scale(blurred_image1).astype(np.uint8))
        cv2.imwrite("8_c1_FG_X.png", scale(x_direction1).astype(np.uint8))
        cv2.imwrite("8_c1_FG_Y.png", scale(y_direction1).astype(np.uint8))
        cv2.imwrite("8_c1_FG_Final.png", scale(blurred_image1).astype(np.uint8))
        cv2.waitKey(0)

        cv2.imshow("Unmodified Image2", image2)
        cv2.imshow("FG In X Direction", scale(x_direction2).astype(np.uint8))
        cv2.imshow("FG In Y Direction", scale(y_direction2).astype(np.uint8))
        cv2.imshow("FG Filtered Image", scale(blurred_image2).astype(np.uint8))
        cv2.imwrite("8_c2_FG_X.png", scale(x_direction2).astype(np.uint8))
        cv2.imwrite("8_c2_FG_Y.png", scale(y_direction2).astype(np.uint8))
        cv2.imwrite("8_c2_FG_Final.png", scale(blurred_image2).astype(np.uint8))
        cv2.waitKey(0)

    # Obtains the resulting gradient images from our sobel filter.
    i_x_c1, i_y_c1, gradient_magnitude_c1 = sobel(blurred_image1)
    gradient_orient_c1 = np.degrees(np.arctan2(i_y_c1, i_x_c1))
    i_x_c2, i_y_c2, gradient_magnitude_c2 = sobel(blurred_image2)
    gradient_orient_c2 = np.degrees(np.arctan2(i_y_c2, i_x_c2))

    # If skip isn't true we're printing out the gradient images.
    if (skipOutputs != True):
        cv2.imshow("Img x-dim W/ Sobel ", i_x_c1)
        cv2.imshow("Img y-dim W/ Sobel", i_y_c1)
        cv2.imshow("Gradient W/ Sobel", gradient_magnitude_c1)
        cv2.imwrite("8_c1_x-dim_sobel.png", i_x_c1)
        cv2.imwrite("8_c1_y_dim_sobel.png", i_y_c1)
        cv2.imwrite("8_c1_gradient.png", gradient_magnitude_c1)
        cv2.waitKey(0)

        cv2.imshow("Img x-dim W/ Sobel ", i_x_c2)
        cv2.imshow("Img y-dim W/ Sobel", i_y_c2)
        cv2.imshow("Gradient W/ Sobel", gradient_magnitude_c2)
        cv2.imwrite("8_c2_x-dim_sobel.png", i_x_c2)
        cv2.imwrite("8_c2_y_dim_sobel.png", i_y_c2)
        cv2.imwrite("8_c2_gradient.png", gradient_magnitude_c2)
        cv2.waitKey(0)

    # Obtains edge mappings using a non max suppression algorithm.
    edge_map_c1 = non_max_suppression(gradient_magnitude_c1, gradient_orient_c1)
    edge_map_c2 = non_max_suppression(gradient_magnitude_c2, gradient_orient_c2)

    # If skip isn't true we're printing NMS maps.
    if (skipOutputs != True):
        cv2.imshow("Post NMS C1 Map", scale(edge_map_c1).astype(np.uint8))
        cv2.imshow("Post NMS C2 Map", scale(edge_map_c2).astype(np.uint8))
        cv2.imwrite("8_c1_NMS_edgemap.png", scale(edge_map_c1).astype(np.uint8))
        cv2.imwrite("8_c2_NMS_edgemap.png", scale(edge_map_c2).astype(np.uint8))

        cv2.waitKey(0)

    # Grabs the weaker and strong pixel bounds.
    weak1, strong1 = simple_threshold(edge_map_c1, lowRatio=.05, highRatio=.09)
    weak2, strong2 = simple_threshold(edge_map_c2, lowRatio=.05, highRatio=.09)

    # Improves the edge maps with hyteresis_thresholding.
    ht_edge_map_c1 = hysteresis_thresholding(edge_map_c1, weak=weak1, strong=strong1)
    ht_edge_map_c2 = hysteresis_thresholding(edge_map_c2, weak=weak2, strong=strong2)

    if (skipOutputs != True):
        # Shows the final, canny edge maps.
        cv2.imshow("Canny Edge Map 1", scale(ht_edge_map_c1).astype(np.uint8))
        cv2.imshow("Canny Edge Map 2", scale(ht_edge_map_c2).astype(np.uint8))
        cv2.imwrite("8_c1_edgemap.png", scale(ht_edge_map_c1).astype(np.uint8))
        cv2.imwrite("8_c2_edgemap.png", scale(ht_edge_map_c2).astype(np.uint8))
        cv2.waitKey(0)

    # Returns the edge maps in case they need to be printed later for comparisons.
    # Added padding for these to alleviate rough borders.
    return_img1 = scale(ht_edge_map_c1).astype(np.uint8)
    return_img2 = scale(ht_edge_map_c2).astype(np.uint8)

    return return_img1, return_img2

# Makes and shows refined edge maps for canny1 and canny2.
sigma3_1, sigma3_2 = make_edgemaps(canny1, canny2, size=9, sigma=3, skipOutputs=False)

# Now we will explore the effect of different gaussian kernels on edge-mapping.
# For comparison let's use sigma values: 1, 3, 6. For 1D size we will use about
# 3 times the standard deviation sigma.

# NOTE: Our default sigma3 was saved above to alleviate extra runtime.

sigma1_1, sigma1_2 = make_edgemaps(canny1, canny2,size=3, sigma=1, skipOutputs=True)
sigma6_1, sigma6_2 = make_edgemaps(canny1, canny2, size=19, sigma=6, skipOutputs=True)

cv2.imshow("Sigma 1 Canny1 Edgemap", sigma1_1)
cv2.imshow("Sigma 3 Canny1 Edgemap", sigma3_1)
cv2.imshow("Sigma 6 Canny1 Edgemap", sigma6_1)
cv2.imwrite("8_c1sigma1.png", sigma1_1)
cv2.imwrite("8_c1sigma3.png", sigma3_1)
cv2.imwrite("8_c1sigma6.png", sigma6_1)
cv2.waitKey(0)

cv2.imshow("Sigma 1 Canny2 Edgemap", sigma1_2)
cv2.imshow("Sigma 3 Canny2 Edgemap", sigma3_2)
cv2.imshow("Sigma 6 Canny2 Edgemap", sigma6_2)
cv2.imwrite("8_c2sigma1.png", sigma1_2)
cv2.imwrite("8_c2sigma3.png", sigma3_2)
cv2.imwrite("8_c2sigma6.png", sigma6_2)
cv2.waitKey(0)

# NOTE: Which sigma value works best depends on the image and the desired outcome.
#       A large sigma value detects large scale edges better, whereas a smaller
#       sigma will result in finer features being detected. So i would say for something
#       like facial recognition in unlocking a phone, it would require a very small sigma
#       value to confirm the identity of the person unlocking it. Whereas if you just want
#       something to be able to detect if theres a building or a vehicle in an area, depending
#       on the quality of the camera you should have a bit higher of a sigma value.

# For similar results to the output-canny images let's try different sigma values.
# NOTE: I tried a bunch of different size/sigmas and I couldn't get anything near
#       the output images.
cannyoutput1, cannyoutput2 = make_edgemaps(canny1, canny2,size=3, sigma=1, skipOutputs=True)

cv2.imshow("Sigma 1 Canny1 Edgemap", cannyoutput1)
cv2.imshow("Sigma 1 Canny2 Edgemap", cannyoutput2)
cv2.imshow("output_canny1.png", output_canny1)
cv2.imshow("output_canny2.png", output_canny2)
cv2.waitKey(0)

# Image Segmentation
# ========================================#
# ______________Question 9________________#
# ========================================#
print("Now Running Question 9...")
def binarization(img, lower, upper):

    hist = cv2.calcHist([img], [0], None, [255], [0, 255])
    plt.plot(hist)
    plt.show()

    result = np.zeros((img.shape), dtype=np.int32)
    strong_i, strong_j = np.where(img >= upper)
    weak_i, weak_j = np.where((img <= upper) & (img >= lower))

    result[strong_i, strong_j] = lower
    result[weak_i, weak_j] = 255

    return scale(result).astype(np.uint8)

# Grabs the lower bound for binarization using otsu method.
def otsu(img):
    img_flat = np.reshape(img, (img.shape[0] * img.shape[1]))
    [hist, _] = np.histogram(img_flat, bins=256, range=(0,255))

    # Normalizes the pixel values from histogram.
    hist = 1.0 * hist / np.sum(hist)
    val_max = -999
    lower = -1

    # Calculates the optimal lower threshold bound.
    for t in range(1, 255):
        q1 = np.sum(hist[:t])
        q2 = np.sum(hist[t:])
        m1 = np.sum(np.array([i for i in range(t)]) * hist[:t]) / q1
        m2 = np.sum(np.array([i for i in range(t, 256)]) * hist[t:]) / q2
        val = q1 * (1 - q1) * np.power(m1 - m2, 2)
        if val_max < val:
            val_max = val
            lower = t

    return lower

# Preprocessing for segmentation.
q9pic1 = cv2.imread('q9pic1.PNG')
q9pic1 = cv2.resize(q9pic1, (800, 600))
q9pic1 = cv2.cvtColor(q9pic1, cv2.COLOR_BGR2GRAY)
q9pic2 = cv2.imread('q9pic2.PNG')
q9pic2 = cv2.resize(q9pic2, (800, 600))
q9pic2 = cv2.cvtColor(q9pic2, cv2.COLOR_BGR2GRAY)
q9pic3 = cv2.imread('q9pic3.JPG')
q9pic3 = cv2.resize(q9pic3, (800, 600))
q9pic3 = cv2.cvtColor(q9pic3, cv2.COLOR_BGR2GRAY)

# Writes the original grayscale images.
cv2.imwrite("9_original1.png", q9pic1)
cv2.imwrite("9_original2.png", q9pic2)
cv2.imwrite("9_original3.png", q9pic3)

# Calculates the lower and upper bounds for our simple threshold.
lower_1, upper_1 = simple_threshold(q9pic1, lowRatio=.7, highRatio=.9)
lower_2, upper_2 = simple_threshold(q9pic2, lowRatio=.3, highRatio=.9)
lower_3, upper_3 = simple_threshold(q9pic3, lowRatio=.55, highRatio=.9)

# Shows results of simple thresholding binarization on picture 1.
cv2.imshow("Original Picture 1",q9pic1)
bin_pic1 = binarization(q9pic1, lower_1, upper_1)
cv2.imshow("Picture Post Simple Thresholding", bin_pic1)
cv2.waitKey(0)

# Shows results of simple thresholding binarization on picture 2.
cv2.imshow("Original Picture 2",q9pic2)
bin_pic2 = binarization(q9pic2, lower_2, upper_2)
cv2.imshow("Picture Post Simple Thresholding", bin_pic2)
cv2.waitKey(0)

# Shows results of simple thresholding binarization on picture 3.
cv2.imshow("Original Picture 3",q9pic3)
bin_pic3 = binarization(q9pic3, lower_3, upper_3)
cv2.imshow("Picture Post Simple Thresholding", bin_pic3)
cv2.waitKey(0)

# Writes all the simple binarization images.
cv2.imwrite("9_simple1.png", bin_pic1)
cv2.imwrite("9_simple2.png", bin_pic2)
cv2.imwrite("9_simple3.png", bin_pic3)

# Shows results of otsu thresholding binarization on picture 1.
cv2.imshow("Original Picture 1",q9pic1)
otsu_pic1 = binarization(q9pic1, otsu(q9pic1), 255)
cv2.imshow("Picture Post Otsu Thresholding", otsu_pic1)
cv2.waitKey(0)

# Shows results of otsu thresholding binarization on picture 2.
cv2.imshow("Original Picture 2",q9pic2)
otsu_pic2 = binarization(q9pic2, otsu(q9pic2), 255)
cv2.imshow("Picture Post Otsu Thresholding", otsu_pic2)
cv2.waitKey(0)

# Shows results of otsu thresholding binarization on picture 3.
cv2.imshow("Original Picture 3",q9pic3)
otsu_pic3 = binarization(q9pic3, otsu(q9pic3), 255)
cv2.imshow("Picture Post Otsu Thresholding", otsu_pic3)
cv2.waitKey(0)

# Writes all the Otsu binarization images.
cv2.imwrite("9_otsu1.png", otsu_pic1)
cv2.imwrite("9_otsu2.png", otsu_pic2)
cv2.imwrite("9_otsu3.png", otsu_pic3)

print("Project Run Completed: See report for additional findings")
