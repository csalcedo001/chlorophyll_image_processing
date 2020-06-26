"""
Load image from IMAGE_PATH and print the color from the box
with origin (X, Y) and dimensions (W ,H) as COLOR_FORMAT.
"""

import sys

if len(sys.argv) not in [6, 7]:
	raise Exception("Usage: python3 image_box_color.py <image_path> <x> <y> <w> <h> [<color_format>]")

import cv2
from functions.main import get_colors
from functions.utils import box_contour
from functions import choose_valid_points


# Arguments
## Image path
image_path = sys.argv[1]

## Box
x = int(sys.argv[2])
y = int(sys.argv[3])
w = int(sys.argv[4])
h = int(sys.argv[5])

## Color format
if len(sys.argv) == 7:
	if sys.argv[6] not in ["RGB", "BGR", "LAB"]:
		raise Exception("Unsupported color format " + str(sys.argv[6]))

	color_format = sys.argv[6]
else:
	color_format = "LAB"

# Configuration
box_color = [36, 255, 12]
rescale_factor = 0.2

# Program
image = cv2.imread(image_path)

contour = box_contour(x, y, w, h)

colors = get_colors(image, [contour],
	choose_valid_points=choose_valid_points.all,
	filter_out_of_range=False
)["object_colors"]

print(colors[0].to(color_format))

cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 5)

width = int(image.shape[1] * rescale_factor)
height = int(image.shape[0] * rescale_factor)

image = cv2.resize(image, (width, height))

cv2.imshow("image", image)
cv2.waitKey(0)
