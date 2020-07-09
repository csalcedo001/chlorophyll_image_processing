import numpy as np
import cv2
import os
import pathlib
import wx

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

	contour = np.array([[[0, 0]], [[0, height]], [[width, height]], [[width, 0]]])

	return contour

def box_contour(x, y, w, h):
	"""
	Outputs the contour of the box in origin (x, y) and dimensions (w, h)

	Arguments:
	x -- x-axis origin of box.
	y -- y-axis origin of box.
	w -- width of box.
	h -- height of box.

	Returns:
	contour -- list of corners of image
	"""

	contour = np.array([[[x, y]], [[x, y + h]], [[x + w, y + h]], [[x + w, y]]])

	return contour
	
def delete_unwanted_files(root):
	"""
	Remove all unwanted files from given path recursivelly.

	Arguments:
	root -- directory from which files are deleted.
	"""

	print(root)

	for path, subdirs, files in os.walk(root):
		print(path)
		if ".DS_Store" in files:
			os.system("rm " + os.path.join(path, ".DS_Store"))

def image_resize(image):
	"""
	Resize image to make it fit on screen.

	Arguments:
	image -- image RGB matrix.
	"""

	wx.App(False)
	screen_width, screen_height = wx.GetDisplaySize().Get()

	screen_width -= 100
	screen_height -= 100

	image_height, image_width = image.shape[:2]

	if image_width < screen_width and image_height < screen_height:
		return
	
	width_factor = screen_width / image_width
	height_factor = screen_height / image_height

	if width_factor < height_factor:
		factor = width_factor
	else:
		factor = height_factor
	
	resized_image = cv2.resize(image, (round(factor * image_width), round(factor * image_height)))

	return resized_image
