"""
Find LAB color of image from IMAGE_PATH.
"""

import sys

if len(sys.argv) != 2:
	print("Usage: python3 image_recoloring.py <image_path>")
	exit()

import cv2
import numpy as np
from skimage import color

from functions.main import *

image_path = sys.argv[1]

image = cv2.imread(image_path)

contour = np.array([[[[0, 0]], [[image.shape[1] - 1, image.shape[0] - 1]]]])

colors = get_colors(image, contour, filter_out_of_range=False)

print(color.rgb2lab(colors[0][::-1]) / 100)
