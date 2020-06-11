"""
Module of functions used to select a group of valid pixels from a
rectangle: those that will be used for clustering. Each function should
have the folowwing structure:

Arguments:
x -- x-axis origin of rectangle.
y -- y-axis origin of rectangle.
w -- width of rectangle.
h -- height of rectangle.

Returns:
valid_points -- list of valid points.
"""

import numpy as np
import itertools

def all(x, y, w, h):
	"""
	No points are filtered; all pixels selected.
	"""

	valid_points = np.array(list(itertools.product(
		[col for col in range(x, w)],
		[row for row in range(y, h)]
	)))
	
	return valid_points

def ellipse(x, y, w, h):
	"""
	Select only points that belong to the ellipse inscribed
	in the rectangle with origin (x, y) and dimensions (w, h).
	"""

	valid_points = []

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
	
	return valid_points
