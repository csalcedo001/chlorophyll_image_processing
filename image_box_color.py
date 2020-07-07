"""
Load image from IMAGE_PATH and print the color from the box
with origin (X, Y) and dimensions (W ,H) as COLOR_FORMAT.
"""

import click

import cv2
from functions.main import get_colors
from functions.utils import box_contour, image_resize
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
	default="LAB",
	show_default=True
)
@click.option('--draw-box', '-b', is_flag=True,
	help="show box around detected object")
@click.option('--draw-points', '-p', is_flag=True,
	help="paint cluster points that contribute to color")
@click.option('--print-stats', '-s', is_flag=True,
	help="print statistical information: color, points average and standard deviation")
def main(image_path, x, y, w, h, color_format, draw_box, draw_points, print_stats):
	# box_color = [36, 255, 12]
	rescale_factor = 0.2
	
	image = cv2.imread(image_path)

	stats_format = None

	if print_stats:
		stats_format = color_format
	
	contour = box_contour(x, y, w, h)
	
	colors = get_colors(image, [contour],
		choose_valid_points=choose_valid_points.all,
		filter_out_of_range=False,
		draw_box=draw_box,
		draw_points=draw_points,
		stats_format=stats_format
	)["object_colors"]
	
	if not print_stats:
		print(colors[0].to(color_format))
	
	# cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 5)
	
	#width = int(image.shape[1] * rescale_factor)
	#height = int(image.shape[0] * rescale_factor)
	
	#image = cv2.resize(image, (width, height))

	image = image_resize(image)
	
	if draw_box or draw_points:
		cv2.imshow("image", image)
		cv2.waitKey(0)

if __name__ == "__main__":
	main()
