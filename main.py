from PIL import Image
from numpy import array

image = Image.open("reference_image.jpeg")
print(array(image).shape)
