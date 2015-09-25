'''Library of functions for helping segment images of seedlings on plates'''

import skimage
import numpy as np
from skimage.morphology import reconstruction
from skimage import data, io
import matplotlib.pyplot as plt
from scipy import ndimage

def get_seedling_boxes():
    '''extracts the area of an image that fully contains a seedling.
    Provide with an image with many seedlings and can extract those.
    Returns an array of corners, [x1,x2,y1,y2] x1 <= x2, y1 <= y2'''
    seedlings = get_seedling_objects()

def get_seedling_objects(image,min_size=4000,max_size=80000):
    '''gets the seedling objects'''
    filled = erode(image)
    labelled = label_objects(filled)
    sizes = np.bincount(label_objects.ravel())
    return keep_objects_in_bracket(label_objects,sizes,min_size,max_size)

def keep_objects_in_bracket(label_objects,size_list, min_size, max_size):
    '''filters objects of min_size > size < max_size from a list'''
    mask_sizes = (sizes > min_size) & (sizes < max_size)
    mask_sizes[0] = 0
    return mask_sizes[label_objects]

def erode(image):
    '''calculates an erosion using skimage.morphology.reconstruction,
    returns the filled image'''
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    mask = image
    return reconstruction(seed, mask, method=’erosion’)

def label_objects(image):
    '''labels objects in an image using scipy.ndimage.label'''
    label_objects, nb_labels = ndimage.label(image)
    return label_objects

    mask_sizes = (sizes > 4000) & (sizes < 80000)
    mask_sizes[0] = 0
    big_objs = mask_sizes[label_objects]
