import cv2
import numpy as np


class ImageUtils:
    def __init__(self):
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = None, None, None, None, None

    def undistort_image(self, image, nx=9, ny=5):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def calibrate_camera(self, image, nx=9, ny=5):
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            print("Found the corners")
            objpoints.append(objp)
            imgpoints.append(corners)
            # return cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, (image.shape[1], image.shape[0]), None, None)
        else:
            print("Didn't find the corners! D=")


class PerspectiveTransform:
    def __init__(self, src=np.float32([(611.0, 440.0), (670.0, 440.0), (1018.0, 663.0), (291.0, 663.0)]), dst=np.float32([(480.0, 430.0), (860.0, 430.0), (860.0, 640.0), (480.0, 640.0)])):
        self.src, self.dst = src, dst

    def perspective_transform(self, image):
        img_size = (image.shape[1], image.shape[0])
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        return cv2.warpPerspective(image, M, img_size)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(s, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    else:
        abs_sobel = np.absolute(cv2.Sobel(s, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sxbinary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]

    sobelx = cv2.Sobel(s, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(s, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    abs_sobel = (abs_sobelx ** 2 + abs_sobely ** 2) ** 0.5

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sxbinary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:, :, 2]
    sobelx = cv2.Sobel(s, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(s, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    direction = np.arctan(abs_sobely / abs_sobelx)

    sxbinary = np.zeros_like(direction)
    sxbinary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1

    return sxbinary


def color_and_gradient_transform(image, ksize=9):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=(130, 150))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1