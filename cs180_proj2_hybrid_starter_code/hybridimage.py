import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.signal
from skimage import io

from align_image_code import align_images

# First load images

# high sf
im1 = io.imread('./DerekPicture.jpg', as_gray=True)

# low sf
im2 = io.imread('./nutmeg.jpg', as_gray=True)



# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

def hybrid_image(imhigh, imlow, sigmahigh, sigmalow):

    oned = cv2.getGaussianKernel(31, sigmalow)
    twod = oned @ oned.T

    low = scipy.signal.convolve2d(imlow, twod, mode="same")

    oned = cv2.getGaussianKernel(31, sigmahigh)
    twod = oned @ oned.T

    blurred = scipy.signal.convolve2d(imhigh, twod, mode="same")

    high = imhigh - blurred

    out = low + high

    return out

sigma1 = 8
sigma2 = 4
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

plt.imshow(hybrid, cmap="gray")
plt.show()