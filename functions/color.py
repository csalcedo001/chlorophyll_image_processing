import numpy as np
import skimage.color
from sklearn.cluster import KMeans
import json

class Color():
	format = None
	clusters = None
	labels = None
	formats = ["RGB", "BGR", "LAB"]

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

		color_array = Color.transform(self.color_, self.format_, to=format_)

		return color_array 
	
	def label(self):
		Color.load_cluster()

		return Color.labels[Color.clusters.predict([self.array(Color.format)])[0]]
	
	def __str__(self):
		if self.format_ == "LAB":
			print_format = "[{color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}]"
		else:
			print_format = "[{color[0]:.0f}, {color[1]:.0f}, {color[2]:.0f}]"

		return ("Color in {} format: " + print_format).format(self.format_, color=self.color_.tolist())
		return "Color in " + str(self.format_) + " format: " + str(np.round(self.color_).tolist())
	
	@classmethod
	def load_cluster(cls):
		if cls.clusters == None:
			color_cluster_data = None

			with open("data/lab_cluster_colors.json") as input_file:
				color_cluster_data = json.load(input_file)
			
			cls.format = color_cluster_data["format"]
			cls.labels = color_cluster_data["labels"]
			cls.clusters = KMeans(
				n_clusters=color_cluster_data["number_of_clusters"],
				random_state=color_cluster_data["random_seed"]
			).fit(color_cluster_data["colors"])
	
	@classmethod
	def transform(cls, color_array, from_, to):
		if from_ not in cls.formats:
			raise Exception("Unsupported color format " + str(from_))

		if to not in cls.formats:
			raise Exception("Unsupported color format " + str(to))

		result = color_array = np.array(color_array)

		if from_ == "RGB":
			if to == "BGR":
				result = color_array[::-1]
			if to == "LAB":
				result = skimage.color.rgb2lab(color_array / 255)
		elif from_ == "BGR":
			if to == "RGB":
				result = color_array[::-1]
			if to == "LAB":
				result = skimage.color.rgb2lab(color_array[::-1] / 255)
		elif from_ == "LAB":
			if to == "RGB":
				result = skimage.color.lab2rgb(color_array) * 255
			if to == "BGR":
				result = skimage.color.lab2rgb(color_array)[::-1] * 255

		return result
