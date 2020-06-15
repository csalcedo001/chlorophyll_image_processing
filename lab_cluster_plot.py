"""
Plots pixels from data/lab_cluster_colors.json in a 3D graphic
with the colors of each pixel associated with the label color
given to its cluster.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

lab_color_data = None

with open("data/lab_cluster_colors.json") as input_file:
	lab_color_data = json.load(input_file)

lab_colors = np.array(lab_color_data["colors"])

color_labels = lab_color_data["labels"]
clusters = KMeans(n_clusters=4, random_state=0).fit(lab_colors)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(lab_colors[:,0], lab_colors[:,1], lab_colors[:,2], c=np.array([color_labels[clusters.labels_[i]] for i in range(len(lab_colors))]))
plt.show()
