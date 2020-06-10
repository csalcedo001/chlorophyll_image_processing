import cv2
import numpy as np
import scipy.stats as st

def gkern(kernlen, nsig):
	"""Returns a 2D Gaussian kernel."""
	
	x = np.linspace(-nsig, nsig, kernlen+1)
	kern1d = np.diff(st.norm.cdf(x))
	kern2d = np.outer(kern1d, kern1d)
	return kern2d/kern2d.sum()

image = cv2.imread("data/input/test_images/test_image.png")

kernel_index = 10
kernel_size = 2 * kernel_index + 1
kernel_reshape_factor = 5

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# First method
kernel_sharp = -np.ones((kernel_size, kernel_size))
kernel_sharp[kernel_index, kernel_index] = kernel_size ** 2

# Second method
kernel_sharp = kernel_reshape_factor * gkern(kernel_size, 10)
kernel_sharp -= (kernel_reshape_factor - 1) / kernel_size ** 2
print(np.sum(kernel_sharp))
print(gkern(kernel_size, 10))

sharpened = cv2.filter2D(blurred, -1, kernel_sharp)

cv2.imwrite("data/gray/test_images/test_image.png", gray)
cv2.imwrite("data/blurred/test_images/test_image.png", blurred)
cv2.imwrite("data/sharpened/test_images/test_image.png", sharpened)
