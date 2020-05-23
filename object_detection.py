import os
import numpy as np
import cv2
from skimage import color
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

constant_box_color = True

for directory in os.listdir("data/input"):
	for filename in os.listdir("data/input/" + directory):
		input_path = "data/input/" + directory + "/" + filename
		output_path = "data/output/" + directory + "/" + filename

		print(input_path)

		image = cv2.imread(input_path)
		original = image.copy()

		lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
		lab_image_radius = np.sum(np.abs(lab_image.astype("float64"))**2,axis=-1)**(1./2)
		lab_image = lab_image - 128
		lab_image = np.round(lab_image * np.expand_dims(128 / lab_image_radius, axis=-1)).astype('uint8')
		lab_image = lab_image + 128
		lab_image_radius = np.sum(np.abs(lab_image.astype("float64"))**2,axis=-1)**(1./2)
		lab_image[:,:,0] = 128
		for row in range(lab_image.shape[0]):
			for col in range(lab_image.shape[1]):
				if lab_image_radius[row, col] < 255:
					lab_image[row, col] = [0, 128, 128]
		hue_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
		#hue_image = hue_image * 8
		#hue_image = np.abs(hue_image - 128)
		#hue_image = image
		
		gray = cv2.cvtColor(hue_image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (3, 3), 0)
		canny = cv2.Canny(blurred, 120, 255, 1)
		kernel = np.ones((5,5),np.uint8)
		dilate = cv2.dilate(canny, kernel, iterations=1)

		cv2.imwrite("data/lab/" + directory + "/" + filename, hue_image)
		cv2.imwrite("data/gray/" + directory + "/" + filename, gray)
		cv2.imwrite("data/blurred/" + directory + "/" + filename, gray)
		cv2.imwrite("data/canny/" + directory + "/" + filename, canny)

		# Find contours
		cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if len(cnts) == 2 else cnts[1]
		
		# Iterate thorugh contours and filter for ROI
		image_number = 0
		for c in cnts:
			x,y,w,h = cv2.boundingRect(c)

			if w < 100 or h < 100:
				continue
			
			valid_points = []
		
			# Evaluate only points inside ellipse
			for row in range(y, y + h):
				for col in range(x, x + w):
					# Points to be transformed in the plane
					y_tran = row
					x_tran = col
		
					# Move center of coordinates to center of object
					y_tran = y_tran - y - h / 2
					x_tran = x_tran - x - w / 2
		
					# Scale plane to match radius = width of image
					y_tran = y_tran * w / h
					
					if np.sqrt(y_tran ** 2 + x_tran ** 2) <= w:
						valid_points.append([row, col])
		
			# Group selected points into clusters
			number_of_clusters = 3
			
			kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(np.array([image[p[0]][p[1]] for p in valid_points]))
		
		
			# Find color from clustered points
		
			# First approach: find darkest cluster
			#index = kmeans_dists.index(min([np.linalg.norm(mean) for mean in kmeans.cluster_centers_]))
		
			# Second approach: find biggest cluster
			mean_count = [0] * number_of_clusters
		
			for label in kmeans.labels_:
				mean_count[label] += 1
			
			index = mean_count.index(max(mean_count))
		
			# Third approach: find cluster with smallest standard deviation
			# ...
		
			object_color = kmeans.cluster_centers_[index]
		
			print("Object", image_number, "color: ", color.rgb2lab(object_color))
			# print("Object", image_number, "color: ", object_color)
		
			if constant_box_color:
				box_color = (36,255,12)
			else:
				box_color = object_color

			cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
			ROI = original[y:y+h, x:x+w]
			cv2.imwrite("results/ROI_{}.png".format(image_number), ROI)
			image_number += 1

		cv2.imwrite(output_path, image)
		
		
		# cv2.imshow('canny', canny)
		# cv2.imshow('image', image)
		# cv2.waitKey(0)
