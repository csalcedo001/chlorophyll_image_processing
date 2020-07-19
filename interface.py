from tkinter import *
from tkinter.ttk import *
from PIL import ImageTk, Image 
from tkinter import filedialog 

import os
import cv2
import csv
import json
import numpy as np
from skimage import color
from sklearn.cluster import KMeans

from functions.utils import full_image_contour, find_first_image
from functions.main import detect_objects, get_colors
from functions import choose_valid_points
from functions import choose_color
from functions.color import Color

# Configuration
filter_out_of_range = True
choose_valid_points = choose_valid_points.ellipse
number_of_clusters = 3
choose_color = choose_color.biggest_colored_cluster

def setup_interface():
	global root

	global message_box
	global type_variable
	global type_selector
	global image_button
	global format_text
	global format_variable
	global format_selector

	global x_text
	global y_text
	global w_text
	global h_text
	global x_entry
	global y_entry
	global w_entry
	global h_entry

	global run_button
	global export_button

	global directory_path

	root = Tk()

	root.title("Image Loader")
	root.resizable(width=True, height=True)
	
	# Message box
	message_box = Label(root, text="<no-messages>")
	message_box.grid(column=10, row=5)
	
	# File or directory selection
	type_variable = StringVar(root)
	type_variable.set("File")
	type_selector = OptionMenu(root, type_variable, "File", "File", "Directory", command=choose_type)
	type_selector.grid(column=0, row=0)

	image_button = Button(root, text='Select directory', command=open_image)
	image_button.grid(column=1, row=0)

	# Format selection
	format_text = Label(root, text="Color format: ")
	format_text.grid(column=0, row=1)

	format_variable = StringVar(root)
	format_variable.set("LAB")
	format_selector = OptionMenu(root, format_variable, "LAB", "LAB", "RGB", command=choose_format)
	format_selector.grid(column=1, row=1)

	# Method selection
	method_text = Label(root, text="Method: ")
	method_text.grid(column=0, row=2)

	method_variable = StringVar(root)
	method_variable.set("Object detection")
	method_selector = OptionMenu(root, method_variable, "Object detection", "Object detection", "Box", command=choose_method)
	method_selector.grid(column=1, row=2)

	x_text = Label(root, text="x: ")
	y_text = Label(root, text="y: ")
	w_text = Label(root, text="w: ")
	h_text = Label(root, text="h: ")
	x_text.grid(column=2, row=2)
	y_text.grid(column=4, row=2)
	w_text.grid(column=6, row=2)
	h_text.grid(column=8, row=2)

	x_entry = Entry(root)
	y_entry = Entry(root)
	w_entry = Entry(root)
	h_entry = Entry(root)
	x_entry.grid(column=3, row=2)
	y_entry.grid(column=5, row=2)
	w_entry.grid(column=7, row=2)
	h_entry.grid(column=9, row=2)

	choose_method("Object detection")

	# Evaluation
	run_button = Button(root, text='Run', command=run)
	run_button.grid(column=0, row=3)

	export_button = Button(root, text='Export', command=export)
	export_button.grid(column=1, row=3)

	# Initialize constants
	directory_path = None

	# Run program
	root.mainloop()

def choose_method(value):
	objects = [x_text, y_text, w_text, h_text, x_entry, y_entry, w_entry, h_entry]

	if value == "Object detection":
		for obj in objects:
			obj.grid_remove()
	else:
		for obj in objects:
			obj.grid()

def choose_type(value):
	pass

def choose_format(value):
	pass

def open_image(): 
	global directory_path

	directory_path = filedialog.askdirectory(title ="", initialdir=".") 

	# opens the image 
	image = Image.open(find_first_image(directory_path)) 
	  
	# resize the image and apply a high-quality down sampling filter 
	image = image.resize((250, 250), Image.ANTIALIAS) 
	
	# PhotoImage class is used to add image to widgets, icons etc 
	image = ImageTk.PhotoImage(image) 
	
	panel = Label(root, image=image) 

	# set the image as img  
	panel.grid(column=10, row=4)
	panel.image = image

def run():
	if directory_path == None:
		message_box.configure(text="Unable to run. Choose directory first.")

		return 

	message_box.configure(text="Running...")

	print("Running...")

	for path, subdirs, files in os.walk(directory_path):
		for filename in files:
			print(filename)

			image_path = os.path.join(path, filename)
			
			image = cv2.imread(image_path)
			
			# Detect objects
			contours = detect_objects(image)

			# If no object was detected, use full image
			if len(contours) != 0:
				colors = get_colors(image, contours,
					stats_format=format_variable.get()
				)["object_colors"]
			else:
				contours = [full_image_contour(image)]
			
				colors = get_colors(image, contours,
					choose_valid_points=choose_valid_points.all,
					filter_out_of_range=False,
					stats_format=format_variable.get()
				)["object_colors"]

			message_box.configure(text="Finished execution.")

def export():
	if directory_path == None:
		message_box.configure(text="Unable to export. Choose directory first.")

		return 

	message_box.configure(text="Exporting...")

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

	message_box.configure(text="Successfully exported to data/color_data.csv")

setup_interface()
