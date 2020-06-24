"""
Loads all images from data/input/plant_images, detect their colors
and clusterize them. By giving the labels of each color as input, these
clusters are saved in data/lab_cluster_colors.json.
"""

# TODO: Receive as input the directory and number of clusters

import os
import numpy as np
import cv2
import json
from sklearn.cluster import KMeans
from skimage import color

from functions.main import detect_objects
from functions import choose_valid_points
from functions.color import Color

colors = []

directory = "plant_images"
number_of_clusters = 4
random_state = 0
choose_valid_points = choose_valid_points.ellipse
clustering_format = "LAB"

for filename in os.listdir("data/input/" + directory):
	input_path = "data/input/" + directory + "/" + filename

	print(input_path)

	image = cv2.imread(input_path)

	contours = detect_objects(image)

	for c in contours:
		x, y, w, h = cv2.boundingRect(c)

		if w < 100 or h < 100:
			continue

		valid_points = choose_valid_points(x, y, w, h)
		valid_colors = np.array([image[p[0], p[1]] for p in valid_points])
	
		# Group selected points into clusters
		clusters = KMeans(n_clusters=number_of_clusters, random_state=0).fit(valid_colors)

		colors.append([Color(color, "BGR").array(clustering_format) for color in clusters.cluster_centers_])

lab_colors = np.concatenate(np.array(colors), axis=0)

clusters = KMeans(n_clusters=number_of_clusters, random_state=random_state).fit(lab_colors)
print(clusters.cluster_centers_)

labels = ["red", "blue", "green", "white"]
color_labels = []

for cluster_center in clusters.cluster_centers_:
	print("Labels (red, blue, green or white):")

	color_label = None

	cluster_color = Color(cluster_center, clustering_format).array("RGB")

	while color_label not in labels:
		color_label = input("Cluster center " + str(cluster_color) + " color (in RGB format): ")
	
	color_labels.append(color_label)

with open("data/lab_cluster_colors.json", "w") as output_file:
	json.dump({
		"format": clustering_format,
		"number_of_clusters": number_of_clusters,
		"random_seed": random_state,
		"labels": color_labels,
		"colors": lab_colors.tolist()
	}, output_file)
