"""
Print color of image from IMAGE_PATH as COLOR_FORMAT.
"""

import click

import cv2
import numpy as np
from skimage import color

from functions.utils import full_image_contour
from functions.main import get_colors
from functions import choose_valid_points

@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--format', '-f', "color_format",
	type=click.Choice(["RGB", "BGR", "LAB"],
		case_sensitive=False),
	default="LAB",
	show_default=True
)
@click.option('--draw-points', '-p', is_flag=True,
	help="paint cluster points that contribute to color")
@click.option('--print-stats', '-s', is_flag=True,
	help="print statistical information: color, points average and standard deviation")
def main(image_path, color_format, draw_points, print_stats):
	stats_format = None

	if print_stats:
		stats_format = color_format


	image = cv2.imread(image_path)
	
	contour = full_image_contour(image)
	
	colors = get_colors(image, [contour],
		choose_valid_points=choose_valid_points.all,
		filter_out_of_range=False,
		draw_points=draw_points,
		stats_format=stats_format
	)["object_colors"]
	
	if not print_stats:
		print(colors[0].to(color_format))

	if draw_points:
		cv2.imshow('result', image)
		cv2.waitKey(0)


if __name__ == "__main__":
	main()
