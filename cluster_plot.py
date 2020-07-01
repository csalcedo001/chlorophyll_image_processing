"""
Plots pixels from data/lab_cluster_colors.json in a 3D graphic
with the colors of each pixel associated with the label color
given to its cluster. The default color format used for plotting
is RGB. The format can be specified as the first argument of
the executable. Available color formats are RGB, BGR and LAB.
"""

import click

import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

from functions.color import Color

@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--format', '-f', "plot_format",
	type=click.Choice(["RGB", "BGR", "LAB"],
	case_sensitive=False),
	default="LAB"
)
def main(image_path, plot_format):
	color_cluster_data = None
	
	with open("data/lab_cluster_colors.json") as input_file:
		color_cluster_data = json.load(input_file)
	
	colors = np.array(color_cluster_data["colors"])
	
	color_format = color_cluster_data["format"]
	color_labels = color_cluster_data["labels"]
	clusters = KMeans(
		n_clusters=color_cluster_data["number_of_clusters"],
		random_state=color_cluster_data["random_seed"]
	).fit(colors)
	
	colors = np.array([Color(color, color_format).array(plot_format) for color in colors])
	
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(colors[:,0], colors[:,1], colors[:,2], c=np.array([color_labels[clusters.labels_[i]] for i in range(len(colors))]))
	
	ax.set_xlabel(plot_format[0])
	ax.set_ylabel(plot_format[1])
	ax.set_zlabel(plot_format[2])
	
	plt.show()


if __name__ == "__main__":
	main()
