import rsbox
from rsbox import ml 
import numpy as np
from glob import glob
import pickle


def img_dataset_from_dir(dir_path):
	"""
	Given a directory containing folders
	representing classes of images, this
	functions builds a valid numpy
	dataset distribution. 

	Input (dir_path) structure: 
	dir_path/class_1, class_n/1.png 
	Note: 'dir_path' must be the raw
	dir name (no trailing dash) 

	Output: [(x,y), ..., (x,y)]
	"""

	dir_path = dir_path + "/*/"
	class_list = glob(dir_path, recursive = True)
	
	master_list = []
	idx = 0
	for class_ in class_list:
		curr_class = ml.image_dir_to_data_norm(class_, "png")
		new_arrays = []
		for elem in curr_class:
			elem = np.moveaxis(elem, -1, 0)
			new_arrays.append(elem)

		labeled_list = ml.gen_label_pair(new_arrays, idx)
		master_list.append(labeled_list)
		idx += 1

	return ml.gen_distro(master_list)



re = img_dataset_from_dir("raw_data")
out_file = open("mini_cifar.pkl", "wb")
pickle.dump(re, out_file)
out_file.close()


	
