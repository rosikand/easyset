# Mini Cell Segmentation

This directory contains a dataset already pre-processed to test semantic segmentation models. The original data is from [here](https://www.kaggle.com/c/data-science-bowl-2018) is of nuclei in cells. Furthermore, the dataset is quite small with only 13 samples (< 21 mb).

The dataset was built by running `build.py` on the images linked above.  

## Dataset list format 

Data is a list of the form `[(x,y), (x,y),..., (x,y)]` where each `x` is normalized float array representing the image and each `y` is a float array representing the segmentation mask. 


## Dataset files

- **(Main file)** `mini_cell_segmentation.pkl`: images are of shape (3, 128, 128) and masks are of shape (2, 128, 128). There are two classes (binary segmentation) where the first mask channel represents the background and the second channel represents the cells. 
