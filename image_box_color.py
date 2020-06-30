"""
Load image from IMAGE_PATH and print the color from the box
with origin (X, Y) and dimensions (W ,H) as COLOR_FORMAT.
"""

import click

import cv2
from functions.main import get_colors
from functions.utils import box_contour
from functions import choose_valid_points

@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('x', type=int)
@click.argument('y', type=int)
@click.argument('w', type=int)
@click.argument('h', type=int)
@click.option('--format', '-f', "color_format",
	type=click.Choice(["RGB", "BGR", "LAB"],
	case_sensitive=False),
	default="LAB"
)
@click.option('--show-image', '-s', 'show_image', is_flag=True)
def main(image_path, x, y, w, h, color_format, show_image):
	box_color = [36, 255, 12]
	rescale_factor = 0.2
	
	image = cv2.imread(image_path)
	
	contour = box_contour(x, y, w, h)
	
	colors = get_colors(image, [contour],
		choose_valid_points=choose_valid_points.all,
		filter_out_of_range=False
	)["object_colors"]
	
	print(colors[0].to(color_format))
	
	cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 5)
	
	width = int(image.shape[1] * rescale_factor)
	height = int(image.shape[0] * rescale_factor)
	
	image = cv2.resize(image, (width, height))
	
	if show_image:
		cv2.imshow("image", image)
		cv2.waitKey(0)

if __name__ == "__main__":
	main()
