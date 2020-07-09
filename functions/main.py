import cv2
import json
import numpy as np
from sklearn.cluster import KMeans
import skimage

from functions import choose_color
from functions import choose_valid_points
from functions import lab_processing
from functions import recoloring_functions
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
	canny = cv2.Canny(blurred, 25, 100, 5)
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
	filter_out_of_range=True,
	draw_box=False,
	draw_points=False,
	stats_format=None
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
	draw_box -- if True, draw box in image.
	draw_points -- if True, paint valid points in image.
	stats_format -- prints object stats in the given format. If None is given, stats are not printed.

	Returns
	lab_colors -- list of colors associated with objects.
	"""

	print_stats = stats_format is not None
	
	if not print_stats:
		stats_format = "LAB"

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


		# The following lines 
		box_color = [36, 255, 12]
	
		if draw_box:
			cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
	
		if draw_points or print_stats:
			cluster_points = []
	
			for i in range(len(valid_points)):
				if clusters.labels_[i] == index:
					cluster_points.append(Color.transform(image[valid_points[i][0], valid_points[i][1]], "BGR", to=stats_format))
	
					if draw_points:
						image[valid_points[i][0], valid_points[i][1]] = box_color
	
			if print_stats:
				print(object_color.to(stats_format))
				print("    Average: " + str(np.average(cluster_points,axis=0)))
				print("    Standard deviation: " + str(np.std(cluster_points,axis=0)))
	
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

def image_recoloring(target_image, target_colors, reference_colors, recoloring_function = recoloring_functions.rgb_weighted_average):
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

	recolored_image = recoloring_function(target_image, target_colors, reference_colors)
	
	return recolored_image
