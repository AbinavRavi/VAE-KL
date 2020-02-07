import abc
from warnings import warn

import numpy as np
import torchvision.transforms as transforms
import torch

# # class AbstractTransform(object):
# #     __metaclass__ = abc.ABCMeta

# #     @abc.abstractmethod
# #     def __call__(self, **data_dict):
# #         raise NotImplementedError("Abstract, so implement")

# #     def __repr__(self):
# #         ret_str = str(type(self).__name__) + "( " + ", ".join(
# #             [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
# #         return ret_str

# # class BlankSquareNoiseTransform(AbstractTransform):
# #     def __init__(self, squre_size=20, n_squres=1, noise_val=(0, 0), channel_wise_n_val=False, square_pos=None,
# #                  data_key="data", label_key="seg", p_per_sample=1):

# #         self.p_per_sample = p_per_sample
# #         self.data_key = data_key
# #         self.label_key = label_key
# #         self.noise_val = noise_val
# #         self.n_squres = n_squres
# #         self.squre_size = squre_size
# #         self.channel_wise_n_val = channel_wise_n_val
# #         self.square_pos = square_pos

# #     def __call__(self, **data_dict):
# #         for b in range(len(data_dict[self.data_key])):
# #             if np.random.uniform() < self.p_per_sample:
# #                 data_dict[self.data_key][b] = augment_blank_square_noise(data_dict[self.data_key][b], self.squre_size,
# #                                                                          self.n_squres, self.noise_val,
# #                                                                          self.channel_wise_n_val, self.square_pos)
# #         return data_dict

# class SquareMask(object):
#     def __init__(self,data,margin,patchsize):
#         self.data = data
#         self.margin = margin
#         self.patchsize = patchsize

#     def random_bbox(self, image, margin, patchsize):
#             """Generate a random tlhw with configuration.
#             Args:
#                 config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
#             Returns:
#                 tuple: (top, left, height, width)
#             """
#             img_height = image.shape[0]
#             img_width = image.shape[1]
#             height = patchsize[0]
#             width = patchsize[1]
#             ver_margin = margin[0]
#             hor_margin = margin[1]
#             maxt = img_height - ver_margin - height
#             maxl = img_width - hor_margin - width
#             t = np.random.randint(low = ver_margin, high = maxt)
#             l = np.random.randint(low = hor_margin, high = maxl)
#             h = height
#             w = width
#             return (t, l, h, w)

#     def bbox2mask(self, shape, margin, patchsize, times=1):
#         """Generate mask tensor from bbox.
#         Args:
#             bbox: configuration tuple, (top, left, height, width)
#             config: Config should have configuration including IMG_SHAPES,
#                 MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
#         Returns:
#             tf.Tensor: output with shape [1, H, W, 1]
#         """
#         bboxs = []
#         for i in range(times):
#             bbox = self.random_bbox(shape, margin, patchsize)
#             bboxs.append(bbox)
#         height = shape
#         width = shape
#         mask = np.zeros((height, width), np.float32)
#         for bbox in bboxs:
#             h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
#             w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
#             image[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
#         return image #mask.reshape((1, ) + mask.shape).astype(np.float32)

#     def __call__(self,image,margin,patchsize):
#         return self.bbox2mask(image,margin,patchsize,times=1)

def random_bbox(image, margin, patchsize):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = image.shape[0]
        img_width = image.shape[1]
        height = patchsize[0]
        width = patchsize[1]
        ver_margin = margin[0]
        hor_margin = margin[1]
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low = ver_margin, high = maxt)
        l = np.random.randint(low = hor_margin, high = maxl)
        h = height
        w = width
        return (t, l, h, w)

def square_mask(image, margin, patchsize):
    """Generate mask tensor from bbox.
    Args:
    bbox: configuration tuple, (top, left, height, width)
    config: Config should have configuration including IMG_SHAPES,
    MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
    Returns:
    image shape inputted with just mask
    No ------tf.Tensor: output with shape [1, H, W, 1]
    """
    bboxs = []
    # for i in range(times):
    bbox = random_bbox(image, margin, patchsize)
    bboxs.append(bbox)
    height = image.shape[0]
    width = image.shape[1]
    # mask = np.zeros((height, width), np.float32)
    for bbox in bboxs:
        h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
        w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
        image[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
    return image #mask.reshape((1, ) + mask.shape).astype(np.float32)
