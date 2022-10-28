import numpy as np
import torch
from PIL import Image


# Original implementation: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
# Here we change input / output formats and add cutout probability.
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        size (int): The length (in pixels) of each square patch.
        probability (float): Probability to apply CutOut.
    """
    def __init__(self, n_holes, size, probability):
        self.n_holes = n_holes
        self.length = size
        self.p = probability

    def __call__(self, img):
        """
        Args:
            img : PIL image of size (C, H, W).
        Returns:
            PIL image: Image with n_holes of dimension length x length cut out of it.
        """
        if torch.rand([1]).item() > self.p:
            return img

        img = np.array(img)
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        img = img * mask

        return Image.fromarray(img.astype(np.uint8))
