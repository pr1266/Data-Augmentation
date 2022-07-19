import torchvision.transforms.functional as F

class CustomTransform(object):

    def __init__(self, func, factor):
        self.factor = factor
        self.func = func

    def __call__(self, img):

        img = self.func(img, self.factor)
        return img


class CustomGaussianBlurTransform(object):

    def __init__(self, sigma, k_size):
        self.sigma = sigma
        self.k_size = k_size

    def __call__(self, img):

        img = F.gaussian_blur(img, kernel_size=(self.k_size, self.k_size), sigma = self.sigma)
        return img
