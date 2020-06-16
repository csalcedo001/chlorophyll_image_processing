"""
Get color data from images and store it in CSV format.
"""

import sys

if len(sys.argv) not in [1, 2]:
	print("Usage: python3 color_data_to_csv.py [<directory_path>]")
	exit()

import os
import cv2
import csv
import json
import numpy as np
from sklearn.cluster import KMeans
from skimage import color

from functions.main import detect_objects
from functions import choose_color
from functions import choose_valid_points

# Arguments
if len(sys.argv) == 2:
	directory_path = sys.argv[1]
else:
	directory_path = "data/input"

# Reference clusters
lab_color_data = None

with open("data/lab_cluster_colors.json") as input_file:
	lab_color_data = json.load(input_file)

lab_colors = np.array(lab_color_data["colors"])

reference_color_labels = lab_color_data["labels"]
reference_clusters = KMeans(n_clusters=4, random_state=0).fit(lab_colors)

# Configuration
filter_out_of_range = True
choose_valid_points = choose_valid_points.ellipse
number_of_clusters = 3
choose_color = choose_color.biggest_colored_cluster

# Data
csv_data = []

for path, subdirs, files in os.walk(directory_path):
	for filename in files:
		input_path = os.path.join(path, filename)

		print(input_path)

		image = cv2.imread(input_path)

		contours = detect_objects(image)
		
		image_number = 0
		for c in contours:
			x, y, w, h = cv2.boundingRect(c)
	
			if filter_out_of_range and (w < 100 or h < 100):
				continue

			image_number += 1
	
			valid_points = choose_valid_points(x, y, w, h)
	
			# Group selected points into clusters
			clusters = KMeans(n_clusters=number_of_clusters, random_state=0).fit(np.array([image[p[0]][p[1]] for p in valid_points]))
	
	
			# Select color from clustered points
			object_color_data = choose_color(clusters)

			object_color = object_color_data["color"]
			index = object_color_data["index"]
			color_label = object_color_data["label"]

			
		
			print("  Object", image_number, "color: ", object_color)

			cluster_points = []

			for i in range(len(valid_points)):
				if clusters.labels_[i] == index:
					cluster_points.append(color.rgb2lab(image[valid_points[i][0], valid_points[i][1]][::-1]))

			average = np.average(cluster_points, axis=0)
			standard_deviation = np.std(cluster_points, axis=0)

			print("    Average: " + str(average))
			print("    Standard deviation: " + str(standard_deviation))

			csv_data.append({
				"name": input_path,
				"object_id": image_number,
				"average": average,
				"standard_deviation": standard_deviation,
				"color_label": color_label
			})

with open("data/color_data.csv", "w") as csv_file:
	field_names = ["name", "object_id", "average", "standard_deviation", "color_label"]
	writer = csv.DictWriter(csv_file, fieldnames=field_names)

	writer.writeheader()

	for row in csv_data:
		writer.writerow(row)
