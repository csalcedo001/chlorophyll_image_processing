"""
Module of functions used to calculate the "real" color of an object
from the k-means cluster created from its pixels. Each function should 
have the following structure:

Arguments:
kmeans -- computed clusters from pixels of some image.

Returns:
object_color -- color of the biggest cluster.
"""

def biggest_cluster(kmeans):
	"""
	Select the biggest cluster's mean color as the object color.
	"""

	mean_count = [0] * len(kmeans.cluster_centers_)
	
	for label in kmeans.labels_:
		mean_count[label] += 1
	
	index = mean_count.index(max(mean_count))
	
	object_color = kmeans.cluster_centers_[index]

	return object_color
