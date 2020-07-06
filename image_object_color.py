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
	default="LAB",
	show_default=True
)
@click.option('--draw-box', '-b', is_flag=True)
@click.option('--draw-points', '-p', is_flag=True)
@click.option('--print-stats', '-s', is_flag=True)
def main(image_path, color_format, draw_box, draw_points, print_stats):
	image = cv2.imread(image_path)

	stats_format = None

	if print_stats:
		stats_format = color_format
	
	contours = detect_objects(image)
	colors = get_colors(image, contours,
		choose_color=choose_color.biggest_colored_cluster,
		draw_box=draw_box,
		draw_points=draw_points,
		stats_format=stats_format
	)["object_colors"]
	
	if not print_stats:
		for color in colors:
			print(color.to(color_format))
	
	if draw_box or draw_points:
		cv2.imshow('result', image)
		cv2.waitKey(0)


if __name__ == "__main__":
	main()
