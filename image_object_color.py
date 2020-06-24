"""
Load image from IMAGE_PATH, detect the objects in it and print their colors
as COLOR_FORMAT.
"""

import sys

if len(sys.argv) not in [2, 3, 4]:
	raise Exception("Usage: python3 image_object_color.py <image_path> [<color_format>]")

import cv2
from functions.main import *


# Arguments
## Image path
image_path = sys.argv[1]

## Color format
if len(sys.argv) == 3:
	if sys.argv[2] not in ["RGB", "BGR", "LAB"]:
		raise Exception("Unsupported color format " + str(sys.argv[2]))

	color_format = sys.argv[2]
else:
	color_format = "LAB"


image = cv2.imread(image_path)

contours = detect_objects(image)
colors = get_colors(image, contours, choose_color = choose_color.biggest_colored_cluster)["object_colors"]

for color in colors:
	print(color.to(color_format))
