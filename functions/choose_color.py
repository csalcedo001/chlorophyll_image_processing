"""
Module of functions used to choose the representative color of an object
from the k-means clusters created from its pixels. Each function should 
have the following structure:

Arguments:
clusters -- made up of pixels from some image.

Returns:
  index -- index of selected cluster.
"""

from sklearn.cluster import KMeans
from skimage import color
import json

from functions.color import Color

def biggest_cluster(clusters):
	"""
	Select the biggest cluster's mean color as the object's color.
	"""

	mean_count = [0] * len(clusters.cluster_centers_)
	
	for label in clusters.labels_:
		mean_count[label] += 1
	
	index = mean_count.index(max(mean_count))
	
	return index

def biggest_colored_cluster(clusters):
	"""
	Select the mean color of the biggest cluster not classified as white 
	using previous color observations as a reference.
	"""

	mean_count = [0] * len(clusters.cluster_centers_)
	
	for label in clusters.labels_:
		mean_count[label] += 1
	
	indices = [index for index in range(len(mean_count))]
	
	indices.sort(key=lambda index : -mean_count[index])

	for index in indices:
		if Color(clusters.cluster_centers_[index], "BGR").label() != "white":
			break
	else:
		index = indices[0]
	
	return index
