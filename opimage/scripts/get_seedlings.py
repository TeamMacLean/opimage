#!/usr/bin/env python

from opimage import plates, utils
import csv


def get_boxes_round_seedlings(image):
    labelled_object_image = plates.get_seedling_objects(image)
    return plates.get_seedling_boxes(labelled_object_image)


#define the folder with the images
img_dir = '/Users/macleand/Desktop/opimage/samples/images/ir_root_length/'
img_stack = utils.get_image_stack(img_dir)
#sort by date - oldest first
img_stack = utils.sort_by_datetime(img_stack)
timepoints = [utils.fname_to_date(fname) for fname in img_stack]
img_stack = zip(img_stack, timepoints)

#work out the extent of the seedlings in the image
last_image_fname = img_stack[-1][0] #use the last image taken..
#get the seedling objects out of the image
#these are the extents over which the root will grow
reference_image = utils.image_file_to_array(last_image_fname)
seedling_bounding_boxes = get_boxes_round_seedlings(reference_image)

result = []

## for every seedling in every image..
for idx, sbox in enumerate(seedling_bounding_boxes):
    # now, for each timepoint get this box, get the sub image, and get the box
    # surrounding the object in it
    #
    boxes = []
    for img_fname, date in img_stack:
        label = img_fname + "-" + str(date) + "-" + str(idx) + ".png"
        img = utils.image_file_to_array(img_fname)
        sub_img = img[sbox]
        boxes.append( get_boxes_round_seedlings(sub_img) )

    growths = plates.get_growth_delta(boxes)

    for i, (img_fname, timepoint) in enumerate(img_stack):
        result.append( [ img_fname, timepoint, idx, utils.box_to_string(sbox), utils.box_to_string(boxes[i][0]), growths[i] ] )

utils.save_csv(result, fname="data.csv", colnames=['file','datetime','seedling','seedling_area','within_seedling_area_sample_area','pixel_length'])
