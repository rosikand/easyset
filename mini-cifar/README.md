+++
title = "Mini CIFAR"
hascode = true
+++

# Mini CIFAR

> A small, pre-processed dataset for image classification. 

\toc 

This repo contains a dataset already pre-processed to test image classification models. It is a subset of the CIFAR-10 dataset 
containing only two classes, each with 20 samples. 

The main use case for this dataset is to quickly test that your model is working on actual image data. 


## Dataset list format 

Data is a list of the form `[(x,y), (x,y),..., (x,y)]` where each `x` is normalized float array representing the image and each `y` is a normalized float array representing the segmentation mask. 


## Dataset files

- **(Main file) `mini_cifar.pkl`**: numpy array format. 

Variations: 
