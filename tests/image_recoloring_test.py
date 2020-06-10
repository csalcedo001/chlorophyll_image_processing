import cv2
from image_recoloring import image_recoloring

path = "data/input/test_images/extract_1.png"

image = cv2.imread(path)

recolored_image = image_recoloring(image, [[93, 46, 36], [42, 39, 149]], [[110, 50, 35], [50, 40, 170]])

cv2.imwrite("result.png", recolored_image)
