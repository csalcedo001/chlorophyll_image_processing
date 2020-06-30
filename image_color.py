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
	default="LAB"
)
def main(image_path, color_format):
	image = cv2.imread(image_path)
	
	contour = full_image_contour(image)
	
	colors = get_colors(image, [contour],
		choose_valid_points=choose_valid_points.all,
		filter_out_of_range=False
	)["object_colors"]
	
	print(colors[0].to(color_format))


if __name__ == "__main__":
	print(main())
