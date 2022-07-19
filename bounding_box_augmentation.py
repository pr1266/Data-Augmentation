class Sequence(object):

    def __init__(self, augmentations, probs = 1):
        #! self.augmentation stores the list of augmentation that we want to use
        #! self.probs holds the probablity of each augmentation to be applied
        self.augmentations = augmentations
        self.probs = probs
        
    def call(self, image, bbox):
        for i, augmentation in enumerate(self.augmentations):
            #! if type of self.prob is a list, each element of it shows the probablity of its corresponding element in augmentation array
            #! but if its a number, it applies for all elements of self.augmentation array
            if type(self.probs) == list:
                prob = self.prob[i]
            else:
                prob = self.probs
            
            img, bbox = augmentation(img, bbox)
        return img, bbox