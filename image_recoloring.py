"""
Use image from REFERENCE_IMAGE_PATH to adjust color of image
from TARGET_IMAGE_PATH saving the result image in RESULT_PATH
"""

import sys

if len(sys.argv) not in [3, 4]:
	print("Usage: python3 image_recoloring.py <reference_image_path> <target_image_path> [<result_path>]")
	exit()

import cv2
from functions.main import *

reference_image_path = sys.argv[1]
target_image_path = sys.argv[2]

if len(sys.argv) == 4:
	result_path = sys.argv[3]
else:
	result_path = "result.png"

reference_image = cv2.imread(reference_image_path)
target_image = cv2.imread(target_image_path)

reference_contours = detect_objects(reference_image)
reference_colors = get_colors(reference_image, reference_contours)

target_contours = detect_objects(target_image)
target_colors = get_colors(target_image, target_contours)

recolored_image = image_recoloring(target_image, target_colors, reference_colors)

cv2.imwrite(result_path, recolored_image)
