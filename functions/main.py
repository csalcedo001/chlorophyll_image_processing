import cv2
import numpy as np
from sklearn.cluster import KMeans

from functions import choose_color
from functions import lab_processing

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

	return cnts

def get_colors(
	image,
	contours,
	number_of_clusters=3,
	choose_color=choose_color.biggest_cluster,
	filter_out_of_range=True
):
	"""
	Calculates colors in IMAGE from each object in CONTOURS.

	Arguments:
	image -- image from which objects are recognized.
	contours -- list of contours of detected objects.
	number_of_clusters -- used in k-means clustering.
	choose_color -- function to obtain the representative color of an object.
	filter_our_of_range -- if True, prevents small objects to be considered.

	Returns
	lab_colors -- list of colors associated with objects.
	"""

	lab_colors = []

	for c in contours:
		x, y, w, h = cv2.boundingRect(c)

		if filter_out_of_range and (w < 100 or h < 100):
			continue
		
		valid_points = []
	
		# Evaluate only points inside ellipse
		for row in range(y, y + h):
			for col in range(x, x + w):
				# Points to be transformed in the plane
				y_tran = row
				x_tran = col
	
				# Move center of coordinates to center of object
				y_tran = y_tran - y - h / 2
				x_tran = x_tran - x - w / 2
	
				# Scale plane to match radius = width of image
				y_tran = y_tran * w / h
				
				if 2 * np.sqrt(y_tran ** 2 + x_tran ** 2) <= w:
					valid_points.append([row, col])
	
		# Group selected points into clusters
		clusters = KMeans(n_clusters=number_of_clusters, random_state=0).fit(np.array([image[p[0]][p[1]] for p in valid_points]))
	
	
		# Select color from clustered points
		object_color = choose_color(clusters)

		lab_colors.append(object_color)

	return lab_colors

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

	target_colors = np.array(target_colors)
	reference_colors = np.array(reference_colors)

	rate = np.sum(reference_colors / target_colors, axis=0) / len(target_colors)
	
	recolored_image = target_image * rate

	return recolored_image
