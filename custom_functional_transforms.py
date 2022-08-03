import torchvision.transforms.functional as F
import torchvision.transforms as T
import PIL

class CustomTransform(object):

    def __init__(self, func, factor):
        self.factor = factor
        self.func = func

    def __call__(self, img):

        if isinstance(img, PIL.Image.Image):
            img = self.func(img, self.factor)
        
        else:
            img = self.func(T.ToPILImage()(img), self.factor)
        
        return img

class CustomGaussianBlurTransform(object):

    def __init__(self, sigma, k_size):
        self.sigma = sigma
        self.k_size = k_size

    def __call__(self, img):
        if isinstance(img, PIL.Image.Image):
            img = F.gaussian_blur(img, kernel_size=(self.k_size, self.k_size), sigma = self.sigma)
        else:
            img = F.gaussian_blur(T.ToPILImage()(img), kernel_size=(self.k_size, self.k_size), sigma = self.sigma)
        return img