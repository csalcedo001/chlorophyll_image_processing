from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("data/reference_image.jpeg")
image_array = np.array(image)

dx = image_array[0,:,0]
dx = image_array[:,0,0]

R = image_array[:,:,0] * 0.299
G = image_array[:,:,1] * 0.587
B = image_array[:,:,2] * 0.114

sum_RGB = R + G + B

m_R = np.mean(R)
m_G = np.mean(G)
m_B = np.mean(R)

print(np.min(sum_RGB[400,:]))

plt.plot(sum_RGB[50,:])
plt.plot(sum_RGB[400,:])
plt.plot(sum_RGB[750,:])
plt.show()
