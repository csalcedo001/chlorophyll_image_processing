"""
Module of functions used to choose the representative color of an object
from the k-means clusters created from its pixels. Each function should 
have the following structure:

Arguments:
clusters -- made up of pixels from some image.

Returns:
object_color -- chosen color for the object.
"""

from sklearn.cluster import KMeans
from skimage import color
import json

def biggest_cluster(clusters):
	"""
	Select the biggest cluster's mean color as the object's color.
	"""

	mean_count = [0] * len(clusters.cluster_centers_)
	
	for label in clusters.labels_:
		mean_count[label] += 1
	
	index = mean_count.index(max(mean_count))
	
	object_color = clusters.cluster_centers_[index]

	return object_color, index

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

	lab_color_data = None

	with open("data/lab_cluster_colors.json") as input_file:
		lab_color_data = json.load(input_file)
	
	reference_labels = lab_color_data["labels"]
	reference_clusters = KMeans(n_clusters=4, random_state=0).fit(lab_color_data["colors"])


	for index in indices:
		print(reference_labels[reference_clusters.predict([color.rgb2lab(clusters.cluster_centers_[index])])[0]])
		if reference_labels[reference_clusters.predict([color.rgb2lab(clusters.cluster_centers_[index])])[0]] != "none":
			return clusters.cluster_centers_[index], index
	
	index = indices[0]
	
	object_color = clusters.cluster_centers_[index]

	return object_color, index
