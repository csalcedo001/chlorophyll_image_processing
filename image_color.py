"""
Find LAB color of image from IMAGE_PATH.
"""

import sys

if len(sys.argv) not in [2, 3]:
	print("Usage: python3 image_color.py <image_path> [<color_format>]")
	exit()

import cv2
import numpy as np
from skimage import color

from functions.utils import full_image_contour
from functions.main import get_colors
from functions import choose_valid_points

# Arguments
if len(sys.argv) == 3:
	if sys.argv[2] not in ["RGB", "BGR", "LAB"]:
		raise Exception("Unsupported color format " + str(sys.argv[2]))
	
	color_format = sys.argv[2]
else:
	color_format = "LAB"

image_path = sys.argv[1]

# Program
image = cv2.imread(image_path)

contour = full_image_contour(image)

colors = get_colors(image, [contour],
	choose_valid_points=choose_valid_points.all,
	filter_out_of_range=False
)["object_colors"]

print(colors[0].to(color_format))
