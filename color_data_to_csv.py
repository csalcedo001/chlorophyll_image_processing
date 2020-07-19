"""
Get color data from images and store it in CSV format.
"""

import click

import os
import cv2
import csv
import json
import numpy as np
from skimage import color
from sklearn.cluster import KMeans

from functions.main import detect_objects
from functions import choose_color
from functions import choose_valid_points
from functions.color import Color
	
# Configuration
filter_out_of_range = True
choose_valid_points = choose_valid_points.ellipse
number_of_clusters = 3
choose_color = choose_color.biggest_colored_cluster

@click.command()
@click.argument('directory_path', type=click.Path(exists=True), default="data/input")
def main(directory_path):
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
				index = choose_color(clusters)
	
				object_color = Color(clusters.cluster_centers_[index], "BGR")
				color_label = object_color.label()
				
			
				print("  Object", image_number, "color: ", object_color.array("LAB"))
	
				cluster_points = []
	
				for i in range(len(valid_points)):
					if clusters.labels_[i] == index:
						cluster_points.append(Color.transform(image[valid_points[i][0], valid_points[i][1]], "BGR", to="LAB"))
	
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


if __name__ == "__main__":
	main()
