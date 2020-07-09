"""
Module of functions that recolor a target image based on
a reference image. Recolors TARGET_IMAGE to approximate
TARGET_COLORS as much as possible to REFERENCE_COLORS.

Attributes:
target_image -- image whose colors will be updated
target_colors -- colors that represent the palette of target_image.
reference_colors -- colors to which target_colors must approximate.

Returns:
recolored_image -- image whose colors are as close as possible to those of reference_colors
"""

import numpy as np
import cv2

from functions.color import Color

def rgb_log_weighted_average(target_image, target_colors, reference_colors):
	"""
	Computes a log weighted average over reference-target color ratio
	in RGB format using preprocessed clusters centers as weights.
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

			rate += color_weights * np.log(reference_colors[color].array("RGB") / target_colors[color].array("RGB"))
	
	rate = np.exp(rate / color_weights_total)

	recolored_image = target_image * rate

	
	return recolored_image

def rgb_weighted_average(target_image, target_colors, reference_colors):
	"""
	Recolors target image by performing a weighted average over color
	differences with reference image using preprocessed cluster centers
	as weights.
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

			rate += color_weights * (reference_colors[color].array("RGB") - target_colors[color].array("RGB"))
	
	rate = rate / color_weights_total

	recolored_image = target_image + rate

	
	return recolored_image

def l_log_simple_average(target_image, target_colors, reference_colors):
	"""
	Calculate the average log L factor from image colors to use
	as a multiplier for the image L channel.
	"""

	number_of_factors = 0
	factor_total = 0
	for color in ["red", "blue"]:
		if color in target_colors and color in reference_colors:
			factor_total += np.log(reference_colors[color].array("LAB")[0] / target_colors[color].array("LAB")[0])

			number_of_factors += 1
	
	if number_of_factors == 0:
		return target_image
	
	factor = np.exp(factor_total / number_of_factors)

	lab_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB)

	lab_image[:,:,0] = lab_image[:,:,0] * factor

	recolored_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

	return recolored_image

def l_simple_average(target_image, target_colors, reference_colors):
	"""
	Calculates term that sums to image L channel as LAB format using
	a simple average over target and reference color differences.
	"""

	number_of_factors = 0
	factor_total = 0
	for color in ["red", "blue"]:
		if color in target_colors and color in reference_colors:
			factor_total += reference_colors[color].array("LAB")[0] - target_colors[color].array("LAB")[0]

			number_of_factors += 1
	
	if number_of_factors == 0:
		return target_image
	
	factor = factor_total / number_of_factors

	lab_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2LAB)

	lab_image[:,:,0] = lab_image[:,:,0] + factor

	recolored_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

	return recolored_image
