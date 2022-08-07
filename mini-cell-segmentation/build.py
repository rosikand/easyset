from glob import glob 
from PIL import Image
import pickle
import cv2
import numpy as np
from util import *
import jax.numpy as jnp
import numpy as np 
import rsbox
from rsbox import ml_utils


sub_dirs = glob("data/*/", recursive = True)
new_size = (128, 128) # all images will be resized to this
data_set = []

for subdir in sub_dirs:
	new_subdirs = sorted(glob(subdir + "/*/", recursive = True))

	# image 
	im_files = glob(new_subdirs[0] + '/*.' + 'png')
	img_file = im_files[0]
	image = cv2.imread(img_file)  # 0 means grayscale
	image = cv2.resize(image, new_size)
	image_array = np.array(image)
	image_array = image_array.astype(float)
	image_array = image_array/255.0
	image_array = np.moveaxis(image_array, -1, 0)

	# mask 
	mask_files = glob(new_subdirs[1] + '/*.' + 'png')
	mask_arrays = []
	spliced_segment = np.zeros((1, 128, 128), dtype=float)
	for elem in mask_files:
		img_segment = cv2.imread(elem, 0)  # 0 means grayscale
		img_segment = cv2.resize(img_segment, new_size)
		img_segment = np.expand_dims(img_segment, axis=0)
		# overlay the mask 
		spliced_segment = np.maximum(spliced_segment, img_segment)

	# binarize the mask (only 0 and 1 values)
	overlayed_mask = (spliced_segment > 0)

	# add background channel 
	background = np.where((overlayed_mask==0)|(overlayed_mask==1), overlayed_mask^1, overlayed_mask)

	mask_array_final = np.concatenate((background.astype(float), overlayed_mask.astype(float)), axis=0)

	# tuple and append 
	sample_tup = (image_array, mask_array_final)
	data_set.append(sample_tup)


# pickle up dataset 
out_file = open("mini_cell_seg.pkl", "wb")
pickle.dump(data_set, out_file)
out_file.close()
