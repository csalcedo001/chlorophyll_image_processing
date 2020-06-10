def biggest_cluster(kmeans):
	"""
	Computes color of the largest cluster.

	Arguments:
	kmeans -- computed clusters from pixels of some image.

	Returns:
	object_color -- color of the biggest cluster.
	"""
	mean_count = [0] * len(kmeans.cluster_centers_)
	
	for label in kmeans.labels_:
		mean_count[label] += 1
	
	index = mean_count.index(max(mean_count))
	
	object_color = kmeans.cluster_centers_[index]

	return object_color
