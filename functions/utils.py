import numpy as np

def full_image_contour(image):
	"""
	Computes contour of all the image by including every corner
	in the contour

	Arguments:
	image -- from where the contour is taken.

	Returns:
	contour -- list of corners of image
	"""

	(width, height, _) = image.shape

	width -= 1
	height -= 1

	contour = np.array([[[0, 0]], [[0, width]], [[height, width]], [[height, 0]]])

	return contour
