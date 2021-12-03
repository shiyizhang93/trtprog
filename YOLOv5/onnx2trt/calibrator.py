"""Example Google style docstrings.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import os
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


def letterbox(img, new_shape=(384, 640), color=(114, 114, 114)):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # get minimum rectangle padding(inference)

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def preprocess(img, img_size=(384, 640)):
    """Example function with PEP 484 type annotations.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
    process_img = letterbox(img, new_shape=img_size)[0]
    process_img = process_img.transpose(2, 0, 1)  # to 3x416x416
    process_img = np.ascontiguousarray(process_img)
    process_img = process_img.astype(np.float32)
    process_img = process_img.reshape((1, -1))
    process_img /= 255.0
    return process_img


class YOLOEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """YOLOEntropyCalibrator
    This class implements TensorRT's IInt8EntropyCalibtrator2 interface.
    It reads all images from the specified directory and generates INT8
    calibration data for YOLO models accordingly.
    """

    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note:
        Do not include the `self` parameter in the ``Args`` section.

    Args:
        msg (str): Human readable string describing the exception.
        code (:obj:`int`, optional): Error code.

    Attributes:
        msg (str): Human readable string describing the exception.
        code (int): Exception error code.

    """
    def __init__(self, calibration_images, cache, batch_size=4, input_channel=3, img_size=(384, 640), dtype=trt.float32):

        trt.IInt8EntropyCalibrator2.__init__(self)  # super().__init__()

        self.cache_file = cache
        self.batch_size = batch_size
        self.img_channel = input_channel
        self.img_size = img_size

        self.img_dir = calibration_images
        self.jpgs = [f for f in os.listdir(calibration_images) if f.endswith('.jpg')]
        # The number "500" is NVIDIA's suggestion.  See here:
        # https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimizing_int8_c
        if len(self.jpgs) < 500:
            print('WARNING: found less than 500 images in %s!' % calibration_images)
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.data_size = trt.volume([self.batch_size, self.img_channel, self.img_size[0], self.img_size[1]]) * dtype.itemsize
        self.device_input = cuda.mem_alloc(self.data_size)

    def __del__(self):
        del self.device_input  # free CUDA memory

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.jpgs):
            return None
        current_batch = int(self.current_index / self.batch_size)

        batch = []
        for i in range(self.batch_size):
            img_path = os.path.join(
                self.img_dir, self.jpgs[self.current_index + i])
            img = cv2.imread(img_path)
            assert img is not None, 'failed to read %s' % img_path
            batch.append(preprocess(img, self.img_size))
        batch = np.stack(batch)

        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again.
        # Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)