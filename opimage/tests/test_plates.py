from unittest import TestCase
from opimage import plates
import numpy as np
from scipy import ndimage

class TestPlates(TestCase):
    def setUp(self):
        self.start_image = self.generate_start_image()
        self.filtered_image = self.generate_filtered_image()

    def generate_start_image(self):
        thousand_object = np.ones(1000)
        hundred_object = np.concatenate((np.ones(100), np.zeros(900)) ,axis=0 )
        one_object = np.concatenate((np.ones(1), np.zeros(999)) ,axis=0 )
        spacer = np.zeros(1000)
        return np.vstack((thousand_object, spacer, hundred_object, spacer, one_object))

    def generate_filtered_image(self):
        hundred_object = np.concatenate((np.ones(100), np.zeros(900)) ,axis=0 )
        spacer = np.zeros(1000)
        return np.vstack((spacer,spacer,hundred_object,spacer,spacer))


    def test_keep_objects_in_brackets(self):
        labelled_objects, _ = ndimage.label(self.start_image)
        filtered_list = plates.keep_objects_in_bracket(labelled_objects, 10, 900)
        assert  np.array_equal(filtered_list, self.filtered_image)
