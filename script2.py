from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("data/reference_image.jpeg")
image_array = np.array(image)

dy, dx, _ = image_array.shape

R = image_array[:,:,0] * 0.299
G = image_array[:,:,1] * 0.587
B = image_array[:,:,2] * 0.114

RGB = R + G + B
RG = R + G

Ystep = 5
Xstep = 5

BS = 100 # Background sensitivity
CS = 50 # Color sensitivity

RG_min = np.min(RG)

sample1 = {
	"R": [],
	"G": [],
	"B": [],
	"RG": []
}
sample2 = {
	"R": [],
	"G": [],
	"B": [],
	"RG": []
}

for i in range(0, Ystep, dy):
	if np.min(RG[i, :]) - RG_min < BS:
		for j in range(0, Xstep, int(dx / 2)):
			if RG[i, j] - np.min(RG[i, 0: int(dx / 2)]) < CS:
				sample1["R"].append(R[i, j])
				sample1["G"].append(G[i, j])
				sample1["B"].append(B[i, j])
				sample1["RG"].append(RG[i, j])
		for j in range(int(dx / 2), Xstep, dx):
			if RG[i, j] - np.min(RG[i, int(dx / 2): dx]) < CS:
				sample2["R"].append(R[i, j])
				sample2["G"].append(G[i, j])
				sample2["B"].append(B[i, j])
				sample2["RG"].append(RG[i, j])


# First plot

fig, ax = plt.subplots()

ax.plot(RG[50,:], label="Y = 50")
ax.plot(RG[200,:], label="Y = 200")
ax.plot(RG[450,:], label="Y = 450")
ax.plot(RG[600,:], label="Y = 600")
ax.plot(RG[750,:], label="Y = 750")

ax.legend(loc="upper left")

fig.suptitle("R+G vs X")

plt.xlabel("X (pixel)")
plt.ylabel("R + G")

x_range = [0, image_array.shape[1]]
y_range = [0, 350]

ax.imshow(image, extent=(x_range + y_range))
ax.set_aspect(image_array.shape[0] / (y_range[1] - y_range[0]))

plt.show()





# Second plot

fig, ax = plt.subplots()

height = 450

ax.plot(R[height,:], label="R", color="r")
ax.plot(G[height,:], label="G", color="g")
ax.plot(B[height,:], label="B", color="b")

ax.legend(loc="upper left")

fig.suptitle("RGB vs X")

plt.xlabel("X (pixel)")
plt.ylabel("RGB")

x_range = [0, image_array.shape[1]]
y_range = [0, 180]

ax.imshow(image, extent=(x_range + y_range))
ax.set_aspect(image_array.shape[0] / (y_range[1] - y_range[0]))

plt.show()
