import numpy as np
import skimage.color
from sklearn.cluster import KMeans
import json

class Color():
	format = None
	clusters = None
	labels = None

	def __init__(self, color_, format_):
		if format_ not in ["RGB", "BGR", "LAB"]:
			raise Exception("Unsupported color format " + str(format_))

		self.format_ = format_
		self.color_ = np.array(color_)
	
	def to(self, format_):
		self.color_ = self.array(format_)
		self.format_ = format_

		return self
	
	def array(self, format_ = None):
		if format_ == None:
			return self.color_

		result = self.color_

		if format_ == "RGB":
			if self.format_ == "BGR":
				result = self.color_[::-1]
			if self.format_ == "LAB":
				result = skimage.color.lab2rgb(self.color_) * 255
		elif format_ == "BGR":
			if self.format_ == "RGB":
				result = self.color_[::-1]
			if self.format_ == "LAB":
				result = skimage.color.lab2rgb(self.color_)[::-1] * 255
		elif format_ == "LAB":
			if self.format_ == "RGB":
				result = skimage.color.rgb2lab(self.color_ / 255)
			if self.format_ == "BGR":
				result = skimage.color.rgb2lab(self.color_[::-1] / 255)
		else:
			raise Exception("Unsupported color format " + str(format_))

		return result
	
	def label(self):
		if Color.clusters == None:
			color_cluster_data = None

			with open("data/lab_cluster_colors.json") as input_file:
				color_cluster_data = json.load(input_file)
			
			Color.format = color_cluster_data["format"]
			Color.labels = color_cluster_data["labels"]
			Color.clusters = KMeans(
				n_clusters=color_cluster_data["number_of_clusters"],
				random_state=color_cluster_data["random_seed"]
			).fit(color_cluster_data["colors"])

		return Color.labels[Color.clusters.predict([self.array(Color.format)])[0]]
			
	
	def __str__(self):
		return "Color in " + str(self.format_) + " format: " + str(np.round(self.color_).tolist())