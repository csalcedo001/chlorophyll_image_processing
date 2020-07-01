"""
Load image from IMAGE_PATH, detect the objects in it and print their colors
as COLOR_FORMAT.
"""

import click

import cv2
from functions.main import *

@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--format', '-f', "color_format",
	type=click.Choice(["RGB", "BGR", "LAB"],
	case_sensitive=False),
	default="LAB"
)
def main(image_path, color_format):
	image = cv2.imread(image_path)
	
	contours = detect_objects(image)
	colors = get_colors(image, contours, choose_color = choose_color.biggest_colored_cluster)["object_colors"]
	
	for color in colors:
		print(color.to(color_format))


if __name__ == "__main__":
	main()
