import csv
import os
import numpy as np
import pandas as pd

# filters
from skimage.measure import label, regionprops
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import zoom
import math

class ReWrite():
    relative_path_to_data = "../input"
    relative_path_to_outputs = "../output"
    x_int_name = "train_x_int"
    filetype = ".csv"

    def __init__(self):
        self.x, y = GetData().load_training()

    def save_x_as_int(self):
        x_location = os.path.join(
            self.relative_path_to_data,
            self.x_int_name + self.filetype
        )

        data = pd.DataFrame(self.x)

        # header=False stops writing the first line as the column names
        # index=False stops writing the column names (index)
        data.to_csv(x_location, header=False, index=False)

class GetData():
    relative_path_to_data = "../input"
    web_path_to_data = "http://cs.mcgill.ca/~ksinha4/datasets/kaggle/"
    train_datasets = ["train_x", "train_y"]
    test_dataset = "test_x"
    filetype = ".csv"

    TRAIN_SAMPLES = 50000
    IMAGE_SIZE = 64

    def __init__(self):
        self.relative_path_to_datasets = os.path.join(
            os.getcwd(),
            self.relative_path_to_data
        )

    def load_training(self, dataset_modifier="", as_image=False,
                      transform=False, augment=False):

        """
        This will actually return the flattened images
            where x is an array of 50,000 by 4,096

            dataset_modifier (option): set to '_int' for speed benefits in local
            environment ONLY when you have run ReWrite().save_x_as_int()

            x_as_image (option): when you want the images to be 64 by 64.
        """

        self.as_image = as_image

        x_file = "train_x" + dataset_modifier + self.filetype
        y_file = "train_y" + self.filetype

        x = self._load(x_file, True)
        y = self._load(y_file, True)

        if augment:
            print('beginning augmentation for training set...')
            x, y = self._augment(x, y, as_image)
            print('finished augmentation for training set')

        if transform:
            print('beginning manual transformations for training set...')
            x = self._transform(x, as_image)
            print('finished manual transformation for training set')

        return x, y

    def load_test(self, as_image=True, transform=True):
        x_file = "test_x" + self.filetype
        x = self._load(x_file)

        if transform:
            print('beginning manual transformations for test set...')
            x = self._transform(x, as_image)
            print('finished manual transformation for test set')

        return x

    def _load(self, name, is_web):
        if is_web:
            location = self.web_path_to_data + name
        else:
            location = os.path.join(
                self.relative_path_to_datasets,
                name
            )

        # header=None ensures that we don't skip the first line of the file!
        print(location)
        data = pd.read_csv(location, header=None).as_matrix()
        return data.astype(np.int16)

    def _transform(self, x, as_image):
        x = x.reshape((-1, 64, 64))

        for index, image in enumerate(x):
            transformed_image = self.__transform(image)

            if transformed_image.shape != (64, 64):
                transformed_image = self._pad(transformed_image)

            x[index] = transformed_image

        if not as_image:
            x = x.reshape((-1, 4096))

        return x

    def __transform(self, x):
#        x = self._binarize(x)
        x, image_attributes = self._mask_square(x)
#        x = self._zoom_on_mask_square(x, image_attributes)

        return x

    def _binarize(self, image):
        blank_mask = image < 255
        image[blank_mask] = 0
        return image

    def _mask_square(self, image):
        label_image = label(image)

        largest_image_start = (0, 0)
        largest_image_edge = 0

        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            longest_edge = max(maxr - minr, maxc - minc)
            if longest_edge > largest_image_edge:
                largest_image_edge = longest_edge
                largest_image_start = (minc, minr)

        largest_image_extent = (largest_image_edge, largest_image_edge)
        rr, cc = self.__rectangle(
            largest_image_start,
            extent=largest_image_extent,
            shape=image.shape
        )

        isolated_image = np.zeros(image.shape)
        isolated_image[cc, rr] = image[cc, rr]
        image_coordinates = (cc, rr)

        image_attributes = {
            'cc': cc,
            'rr': rr,
            'edge': largest_image_edge
        }

        return isolated_image.astype(np.int16), image_attributes

    def _zoom_on_mask_square(self, image, attributes):
        largest_digit = image[
            (attributes['cc'], attributes['rr'])
        ]

        zoom_factor = math.floor(64.0/attributes['edge'])
        zoom_image = zoom(largest_digit, zoom_factor, prefilter=False)

        return zoom_image

    def _pad(self, image):
        x_diff, y_diff = np.array((64, 64)) - np.array(image.shape)
        x_pad = (math.floor(x_diff / 2), math.ceil(x_diff / 2))
        y_pad = (math.floor(y_diff / 2), math.ceil(y_diff / 2))
        return np.pad(image, (x_pad, y_pad), 'constant', constant_values=0)


    def __rectangle(self, start, end=None, extent=None, shape=None):
        """
        credits: @psilentp
        https://github.com/scikit-image/scikit-image/blob/master/skimage/draw/draw.py#L694
        problem getting the most up to date scikit-image function
        """
        if extent is not None:
            end = np.array(start) + np.array(extent)
        elif end is None:
            raise ValueError("Either `end` or `extent` must be given")
        tl = np.minimum(start, end)
        br = np.maximum(start, end)
        if extent is None:
            br += 1
        if shape is not None:
            br = np.minimum(shape, br)
            tl = np.maximum(np.zeros_like(shape), tl)
        coords = np.meshgrid(*[np.arange(st, en) for st, en in zip(tuple(tl),
                                                                   tuple(br))])
        return coords

    def _augment(self, x, y, as_image):
        # we want two additional photos for every one that was in x, but rotated
        # x, y are already the 'cleaned' images
        # rotating them should be simple enough
        x = x.reshape(-1, 64, 64)
        aug_x = np.zeros((x.shape[0], 64, 64))
        aug_y = np.zeros(x.shape[0])

        for index, image in enumerate(x):
            aug_i = index# * 2
            aug_x[aug_i] = self._cw_rotate(image)
            #aug_x[aug_i+1] = self._ccw_rotate(image)

            label = y[index]
            aug_y[aug_i] = label
            #aug_y[aug_i+1] = label

        new_x = np.append(x, aug_x)
        new_y = np.append(y, aug_y)

        if not as_image:
            x = x.reshape((-1, 4096))

        return new_x, new_y

    def _cw_rotate(self, image):
        return rotate(image, -25, reshape=False, cval=0, prefilter=False)

    def _ccw_rotate(self, image):
        return rotate(image, 25, reshape=False, cval=0,  prefilter=False)

class CrossValidation():
    IMAGE_LENGTH = 64

    def __init__(
        self,
        dataset_modifier="",
        as_image=False,
        transform=False,
        augment=False,
        datatype=np.int16
    ):
        self.x, self.y = GetData().load_training(
            dataset_modifier=dataset_modifier,
            as_image=as_image,
            transform=transform,
            augment=augment
        )
        self.as_image = as_image
        self.datatype = datatype

    def get_set(self):
        return self._cv()

    def _cv(self):
        print('beginning cross validation separation...')

        self.train_index = math.floor(0.8 * self.x.shape[0])

        self._transform_correct_format()
        xy = self._pair(self.x, self.y)

        xy = np.random.permutation(xy)
        train_xy = xy[:self.train_index]
        valid_xy = xy[self.train_index:]

        train_x, train_y = self._unpair(train_xy)
        valid_x, valid_y = self._unpair(valid_xy)

        if self.as_image:
            train_x = train_x.reshape((-1, self.IMAGE_LENGTH, self.IMAGE_LENGTH, 1))
            valid_x = valid_x.reshape((-1, self.IMAGE_LENGTH, self.IMAGE_LENGTH, 1))

        all_data = [train_x, train_y, valid_x, valid_y]
        all_data = self._cast_as(all_data)

        print('finished cross validation separation')

        return all_data

    def _transform_correct_format(self):
        if not self.as_image:
            return

        self.x = self.x.reshape((-1, 4096))
        self.y = self.y.reshape((-1, 1))

    def _cast_as(self, data):
        for index, datum in enumerate(data):
            data[index] = datum.astype(self.datatype)
        return data

    def _pair(self, x, y):
        return np.hstack([x, y])

    def _unpair(self, xy):
        # I can't figure out how to extract the last element using any of the
        # numpy splits so doing it manually it is
        y = xy.T[-1]
        x = xy.T[:-1].T

        return x, y

def main():
    ReWrite().save_x_as_int()

if __name__ == "__main__":
    main()
