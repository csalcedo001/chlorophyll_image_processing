import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage import color
import matplotlib.pyplot as plt
import json

from functions.main import detect_objects
from functions import choose_valid_points
from mpl_toolkits.mplot3d import Axes3D

colors = []

directory = "plant_images"
number_of_clusters = 4
choose_valid_points = choose_valid_points.ellipse

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
	
		# Group selected points into clusters
		clusters = KMeans(n_clusters=number_of_clusters, random_state=0).fit(np.array([image[p[0]][p[1]] for p in valid_points]))

		colors.append(clusters.cluster_centers_)

rgb_colors = np.concatenate(colors, axis=0)[:,::-1]
lab_colors = [color.rgb2lab(pixel) / 100 for pixel in rgb_colors]

clusters = KMeans(n_clusters=4, random_state=0).fit(lab_colors)
print(clusters.cluster_centers_)

labels = ["red", "blue", "green", "white"]
color_labels = []

for cluster_center in clusters.cluster_centers_:
	print("Labels (red, blue, green or white):")

	color_input = None

	while color_input not in labels:
		color_input = input("Cluster center " + str(cluster_center) + " color: ")
	
	color_labels.append(color_input)

with open("data/lab_cluster_colors.json", "w") as output_file:
	json.dump({
		"colors": np.array(lab_colors).tolist(),
		"labels": color_labels
	}, output_file)

lab_colors = np.array(lab_colors)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(lab_colors[:,0], lab_colors[:,1], lab_colors[:,2], c=np.array([color_labels[clusters.labels_[i]] for i in range(len(lab_colors))]))
plt.show()
plt.scatter(a, b)
plt.show()
