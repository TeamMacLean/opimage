'''Library of functions for helping segment images of seedlings on plates'''

import numpy as np
from skimage.morphology import reconstruction, opening, disk, skeletonize
from scipy import ndimage
import math

def get_seedling_boxes(seedling_objects):
    '''Gets boxes round seedlings.

    Extracts the extent of an area of labelled seedling object image that
    fully contains each seedling. Provide with an image with many seedlings
    and can extract those.
    Returns an array of slice objects,
    Should be used after get_seedling_objects()
    '''
    return ndimage.find_objects(seedling_objects)

def get_seedling_images(image, box_slices):
    '''returns sub images from an image, given a series of box slices.

    Should be used after get_seedling_images
    '''
    return [image[slce] for slce in box_slices]

def get_seedling_objects(image, min_size=4000, max_size=80000):
    '''cleans the image and gets the seedling objects by
    1. Performing an erosion
    2. Getting rid of very large and small objects
    3. Labels objects that are left.

    Returns a numpy.ndarray of labelled objects'''
    filled = erode(image)
    labelled = label_objects(filled)
    big_objs = keep_objects_in_bracket(labelled, min_size, max_size)
    plants, _ = ndimage.label(big_objs)
    return plants

def keep_objects_in_bracket(labelled, min_size, max_size):
    '''filters objects of min_size > size < max_size from a list'''
    sizes = np.bincount(labelled.ravel())
    mask_sizes = (sizes > min_size) & (sizes < max_size)
    mask_sizes[0] = 0
    return mask_sizes[labelled]

def erode(image):
    '''calculates an erosion using skimage.morphology.reconstruction,
    returns the filled image'''
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    return reconstruction(seed, image, method='erosion')

def label_objects(image):
    '''labels objects in an image using scipy.ndimage.label'''
    labelled, _ = ndimage.label(image)
    return labelled

def separate_root_and_hypocotyl(sub_image, selemsize=3):
    '''performs a morphological opening on a single seedling image
    returns the numpy.ndarray of objects
    '''
    selem = disk(selemsize)
    return opening(sub_image, selem)

def get_length(sub_image):
    '''returns the length and image of a single object image'''
    length_image = skeletonize(sub_image)
    length = np.bincount(length_image.ravel())[1]
    return length, length_image

def get_growth_delta(list_of_bounding_boxes):
    '''from a list of bounding boxes, ordered by time, work out the difference
    in length from time step t and time step t - 1

    length returned is an approximation of the true length - its the length of
    the diagonal of the the bounding box.

    '''
    result = [0.0]
    for i in range(0, len(list_of_bounding_boxes)):
        box_i = list_of_bounding_boxes[i]
        try:
            box_j = list_of_bounding_boxes[i + 1]
            result.append( length_from_coords(box_i[0][0].stop, box_j[0][0].stop,  box_i[0][1].stop, box_j[0][1].stop) )
        except IndexError:
            pass
    return result

def length_from_coords(x1, x2, y1, y2):
    return math.sqrt( (math.fabs(x1 - x2)**2 ) + (math.fabs(y1 - y2)**2) )

def select_root(sub_image, selem_size=3):
    '''returns the slice object that encloses the root bit'''
    opened_image = separate_root_and_hypocotyl(sub_image, selem_size)
    labelled_sep = label_objects(opened_image)
    objs = ndimage.find_objects(labelled_sep)
    coords = None
    lowest = 0
    for i in objs:
        if i[0].stop > lowest:
            lowest = i[0].stop
            coords = i
    return coords
