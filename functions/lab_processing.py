def none(image):
	return image

def normalized_vector(image):
		image[:,:,0] = 128
		image_radius = np.sum(np.abs(image.astype('float64'))**2,axis=-1)**(1./2)
		image = image - 128
		image = np.round(image * np.expand_dims(128 / image_radius, axis=-1)).astype('uint8')
		image = lab_image + 128
		image_radius = np.sum(np.abs(image.astype("float64"))**2,axis=-1)**(1./2)
		image[:,:,0] = 128
		return image
