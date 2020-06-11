"""
Module of functions used to choose the representative color of an object
from the k-means clusters created from its pixels. Each function should 
have the following structure:

Arguments:
clusters -- made up of pixels from some image.

Returns:
object_color -- chosen color for the object.
"""

def biggest_cluster(clusters):
	"""
	Select the biggest cluster's mean color as the object's color.
	"""

	mean_count = [0] * len(clusters.cluster_centers_)
	
	for label in clusters.labels_:
		mean_count[label] += 1
	
	index = mean_count.index(max(mean_count))
	
	object_color = clusters.cluster_centers_[index]

	return object_color
