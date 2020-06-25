import cv2
import json
import numpy as np
from sklearn.cluster import KMeans
import skimage

from functions import choose_color
from functions import choose_valid_points
from functions import lab_processing
from functions.color import Color

def detect_objects(
	original_image,
	lab_processing_function = lab_processing.none
):
	"""
	Finds objects in ORIGINAL_IMAGE.

	Arguments:
	original_image -- image from which objects are recognized.

	Returns:
	contours -- contours of detected objects.
	"""
	image = original_image.copy()

	lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	hue_image = lab_processing_function(lab_image);
	hue_image = cv2.cvtColor(hue_image, cv2.COLOR_LAB2BGR)

	gray = cv2.cvtColor(hue_image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	canny = cv2.Canny(blurred, 120, 255, 1)
	kernel = np.ones((5,5),np.uint8)
	dilate = cv2.dilate(canny, kernel, iterations=1)

	# Find contours
	contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]

	return contours

# TODO: Return RGB instead of BGR from image colors
def get_colors(
	image,
	contours,
	number_of_clusters=3,
	choose_valid_points=choose_valid_points.ellipse,
	choose_color=choose_color.biggest_colored_cluster,
	filter_out_of_range=True
):
	"""
	Calculates colors in IMAGE from each object in CONTOURS.

	Arguments:
	image -- image from which objects are recognized.
	contours -- list of contours of detected objects.
	number_of_clusters -- used in k-means clustering.
	choose_color -- function to obtain the representative color of an object.
	choose_valid_points -- function to obtain the set of valid points from an object.
	filter_out_of_range -- if True, prevents small objects to be considered.

	Returns
	lab_colors -- list of colors associated with objects.
	"""

	lab_colors = []

	for c in contours:
		x, y, w, h = cv2.boundingRect(c)

		if filter_out_of_range and (w < 100 or h < 100):
			continue

		valid_points = choose_valid_points(x, y, w, h)
	
		# Group selected points into clusters
		clusters = KMeans(n_clusters=number_of_clusters, random_state=0).fit(np.array([image[p[0]][p[1]] for p in valid_points]))
	
		# Select index of cluster
		index = choose_color(clusters)
		object_color = Color(clusters.cluster_centers_[index], "BGR")

		lab_colors.append(object_color)
	
	image_colors = {}

	lab_colors = np.array(lab_colors)

	for lab_color in lab_colors[::-1]:
		if lab_color.label() in ["red", "blue"]:
			image_colors[lab_color.label()] = lab_color

	image_color_data = {
		"object_colors": lab_colors,
		"image_colors": image_colors
	}

	return image_color_data

def image_recoloring(target_image, target_colors, reference_colors):
	"""
	Recolors TARGET_IMAGE to approximate TARGET_COLORS as much as
	possible to REFERENCE_COLORS.

	Attributes:
	target_image -- image whose colors will be updated
	target_colors -- colors that represent the palette of target_image.
	reference_colors -- colors to which target_colors must approximate.

	Returns:
	recolored_image -- image whose colors are as close as possible to those of reference_colors
	"""

	Color.load_cluster()

	rate = 0
	rate_count = 1

	color_weights_total = np.array([0., 0., 0.])

	for color in ["red", "blue"]:
		if color in target_colors and color in reference_colors:
			color_index = Color.labels.index(color)
			cluster_color = Color(Color.clusters.cluster_centers_[color_index], Color.format).array("RGB")

			color_weights = cluster_color

			color_weights_total += color_weights

			rate += color_weights * reference_colors[color].array("RGB") / target_colors[color].array("RGB")
			rate_count += 1
	
	rate = rate / color_weights_total

	recolored_image = target_image * rate
	
	return recolored_image
