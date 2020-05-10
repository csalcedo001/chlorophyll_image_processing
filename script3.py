from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

image = Image.open("data/base_image.jpg")
image_array = np.array(image)

height, width, _ = image_array.shape

color_array = image_array.copy()

# Mask of pixels
# 0: Undefined
# 1: Infected
# 2: Immune
mask_matrix = np.zeros((height, width))

mean_infected = np.array([0, 0, 0])
mean_immune = np.array([0, 0, 0])

total_infected = 0
total_immune = 0

initial_immune = [230, 230]

mask_matrix[230, 230] = 2

for row in range(height):
	for col in range(width):
		if row == 0 or row == height - 1 or col == 0 or col == width - 1:
			mask_matrix[row][col] = 1
			mean_infected = color_array[row][col]
			total_infected += 1

mean_infected  = mean_infected / total_infected

mean_immune = color_array[initial_immune[0], initial_immune[1]]

iteration = 0

def mask_to_rgb(mask):
	result = []
	for row in range(mask.shape[0]):
		new_row = []
		for col in range(mask.shape[1]):
			if mask[row][col] == 0:
				new_row.append([255, 255, 255])
			elif mask[row][col] == 1:
				new_row.append([0, 0, 0])
			else:
				new_row.append([0, 255, 0])
		result.append(new_row)
	
	return np.array(result, dtype=np.uint8)

while True:
	for row in range(1, height - 1):
		for col in range(1, width - 1):
			mask = color_array[row - 1: row + 1][col - 1: col + 1]
	
			color_sum = np.sum(mask, axis=(0,1))
			
			if mask_matrix[row][col] == 0 and np.sum(mask_matrix[row - 1: row + 1][col - 1: col + 1]) > 0:
				if np.linalg.norm(color_sum - mean_infected) < np.linalg.norm(color_sum - mean_immune):
					mask_matrix[row][col] = 1
				else:
					mask_matrix[row][col] = 2

	img = Image.fromarray(mask_to_rgb(mask_matrix), "RGB")
	img.save("result.png")

	print("Iteration ", iteration)

	iteration += 1

	# Wait for key press
	input()
					
#
#		if R > 200 and G > 200 and B > 200:
#			color_array[row][col] = [0, 0, 0]
#
#img = Image.fromarray(color_array, "RGB")
#img.save("result.png")
