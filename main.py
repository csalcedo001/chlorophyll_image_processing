import cv2
from functions.main import *

source_image_path = "data/input/test_images/extract_2.png"
target_image_path = "data/input/test_images/extract_3.png"
result_path = "result.png"

source_image = cv2.imread(source_image_path)
target_image = cv2.imread(target_image_path)

source_contours = detect_objects(source_image)
source_colors = get_colors(source_image, source_contours)

target_contours = detect_objects(target_image)
target_colors = get_colors(target_image, target_contours)

recolored_image = image_recoloring(target_image, target_colors, source_colors)

cv2.imwrite(result_path, recolored_image)
