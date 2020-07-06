"""
Use image from REFERENCE_IMAGE_PATH to adjust color of image
from TARGET_IMAGE_PATH. The result image is saved in RESULT_PATH
"""

import click

import cv2
from functions.main import *

@click.command()
@click.argument('reference_image_path', type=click.Path(exists=True))
@click.argument('target_image_path', type=click.Path(exists=True))
@click.option('--result-path', '-r', type=click.Path(), default="result.jpeg")
def main(reference_image_path, target_image_path, result_path):
	reference_image = cv2.imread(reference_image_path)
	target_image = cv2.imread(target_image_path)
	
	reference_contours = detect_objects(reference_image)
	reference_colors = get_colors(reference_image, reference_contours)["image_colors"]
	
	target_contours = detect_objects(target_image)
	target_colors = get_colors(target_image, target_contours)["image_colors"]
	
	recolored_image = image_recoloring(target_image, target_colors, reference_colors)
	
	cv2.imwrite(result_path, recolored_image)


if __name__ == "__main__":
	main()
