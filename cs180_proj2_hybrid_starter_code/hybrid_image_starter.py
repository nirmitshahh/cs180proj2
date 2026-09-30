import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal

from align_image_code import align_images

# First load images

# high sf
im1 = plt.imread('./DerekPicture.jpg') / 255.

# low sf
im2 = plt.imread('./nutmeg.jpg') / 255.

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)



## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

def hybrid_image(imhigh, imlow, sigmahigh, sigmalow):

    oned = cv2.getGaussianKernel(31, sigmalow)
    twod = oned @ oned.T

    low = np.zeros(imlow.shape)

    for i in range(3):
        low[:, :, i] = scipy.signal.convolve2d(imlow[:, :, i], twod, mode="same")


    oned = cv2.getGaussianKernel(31, sigmahigh)
    twod = oned @ oned.T

    blurred = np.zeros(imhigh.shape)

    for i in range(3):
        blurred[:, :, i] = scipy.signal.convolve2d(imhigh[:, :, i], twod, mode="same")


    high = imhigh - blurred

    out = low + high

    return out

sigma1 = 5
sigma2 = 8
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

plt.imshow(hybrid)
plt.show()