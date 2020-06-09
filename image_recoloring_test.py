import cv2
from image_recoloring import image_recoloring

path = "data/input/test_images/extract_1.png"

image = cv2.imread(path)

recolored_image = image_recoloring(image, [[93, 46, 36], [42, 39, 149]], [[102, 51, 40], [46, 44, 158]])

cv2.imwrite("result.png", recolored_image)
