from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("data/reference_image.jpeg")
image_array = np.array(image)

R = image_array[:,:,0] * 0.299
G = image_array[:,:,1] * 0.587
B = image_array[:,:,2] * 0.114

sum_RGB = R + G + B

_, ax = plt.subplots()

ax.plot(sum_RGB[50,:], label="Y = 50")
ax.plot(sum_RGB[400,:], label="Y = 400")
ax.plot(sum_RGB[750,:], label="Y = 750")

ax.legend(loc="upper left")

plt.xlabel("X")
plt.ylabel("R + G + B")

x_range = [0, image_array.shape[1]]
y_range = [0, 300]

ax.imshow(image, extent=(x_range + y_range))
ax.set_aspect(image_array.shape[0] / (y_range[1] - y_range[0]))

plt.show()
