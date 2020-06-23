"""
Load image from IMAGE_PATH, detect the objects in it and print their colors.
"""

import sys

if len(sys.argv) not in [2, 3]:
	print("Usage: python3 image_object_color.py <image_path> [<result_path>]")
	exit()

import cv2
from functions.main import *

image_path = sys.argv[1]

if len(sys.argv) == 3:
	result_path = sys.argv[2]
else:
	result_path = "result.png"

image = cv2.imread(image_path)

contours = detect_objects(image)
colors = get_colors(image, contours, choose_color = choose_color.biggest_colored_cluster)

print("RGB color of objects:")
print([object_color_data["color"].tolist() for object_color_data in colors["object_colors"]])
