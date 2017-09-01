import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import glob
import numpy as np

from image_utils import ImageUtils
from image_utils import PerspectiveTransform
from image_utils import color_and_gradient_transform
from image_utils import abs_sobel_thresh
from image_utils import mag_thresh
from image_utils import dir_threshold

def plot_two_images(distorted_image, straight_image):
    # Plot calibration image undistorted
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    f.tight_layout()
    ax1.imshow(distorted_image)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(straight_image, cmap='gray')
    ax2.set_title('Undistorted image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def fit_poly(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

IMAGE_TEST_PATH = "test_images"
CAMERA_CALIBRATION_PATH = "camera_cal"

# Load informations
calibration_image = mpimg.imread("../" + CAMERA_CALIBRATION_PATH +"/calibration1.jpg")
road_images = glob.glob("../" + IMAGE_TEST_PATH + "/test1.jpg")

image_manager = ImageUtils()

# Calibrate the image_manager
image_manager.calibrate_camera(calibration_image)
image_straight = image_manager.undistort_image(calibration_image)

# plot_two_images(calibration_image, image_straight)

# Plot road image undistorted

for idx, fname in enumerate(road_images):
    image_distorted = mpimg.imread(fname)
    road_staight = image_manager.undistort_image(image_distorted)
    # plot_two_images(image_distorted, road_staight)

# Perspective transform - Calibration

straight_lines_image = mpimg.imread("../" + IMAGE_TEST_PATH + "/straight_lines1.jpg")

src = np.float32([(527.0, 498.0), (759.0, 498.0), (1018.0, 663.0), (291.0, 663.0)])
dst = np.float32([(480.0, 470.0), (860.0, 470.0), (860.0, 640.0), (480.0, 640.0)])
transform = PerspectiveTransform(src, dst)

# Plot the binary version on the images

kernel_size = 9
s_thresh = (170, 255)

for idx, fname in enumerate(road_images):
    image_distorted = mpimg.imread(fname)
    road_staight = image_manager.undistort_image(image_distorted)

    # binary_image = color_and_gradient_transform(image_distorted)
    # plot_two_images(image_distorted, binary_image)

    gradx = abs_sobel_thresh(road_staight, orient='x', sobel_kernel=kernel_size, thresh=(20, 100))
    grady = abs_sobel_thresh(road_staight, orient='y', sobel_kernel=kernel_size, thresh=(20, 100))
    mag_binary = mag_thresh(road_staight, sobel_kernel=kernel_size, thresh=(30, 100))
    dir_binary = dir_threshold(road_staight, sobel_kernel=kernel_size, thresh=(0.7, 1.3))

    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    hsv = cv2.cvtColor(road_staight, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hsv[:, :, 2]

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    color_binary = np.dstack((np.zeros_like(combined), combined, s_binary))

    # plot_two_images(road_staight, color_binary)

    warped = transform.perspective_transform(color_binary)
    # plot_two_images(color_binary, warped)

    fit_poly(warped)


# fit poly