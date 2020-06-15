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

from functions.utils import full_image_contour
from functions.main import get_colors
from functions import choose_valid_points

image_path = sys.argv[1]

image = cv2.imread(image_path)

contour = full_image_contour(image)

colors = get_colors(image, [contour],
	choose_valid_points=choose_valid_points.all,
	filter_out_of_range=False
)

print(color.rgb2lab(colors[0][::-1]) / 100)
