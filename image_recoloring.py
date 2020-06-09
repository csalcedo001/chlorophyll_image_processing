import numpy as np
def image_recoloring(target_image, target_colors, reference_colors):
	"""
	Recolors TARGET_IMAGE to approximate TARGET_COLORS as much as
	possible to REFERENCE_COLORS.

	Attributes:
	target_image -- image whose colors will be updated
	target_colors -- colors that represent the palette of target_image.
	reference_colors -- colors to which target_colors must approximate.

	Returns:
	recolored_image -- image whose colors are as close as possible to those of reference_colors
	"""

	target_colors = np.array(target_colors)
	reference_colors = np.array(reference_colors)

	rate = np.sum(reference_colors / target_colors, axis=0) / len(target_colors)
	
	recolored_image = target_image * rate

	return recolored_image
